"""Fase 11F: concurrency and resource-control tests.

Scope, per the Fase 11F audit: this codebase has no internal fan-out
(no asyncio.create_task/gather/wait/shield anywhere in app/) -- the only
real concurrency scenario is multiple independent run/resume executions
happening at the same time (two CLI processes, or a caller doing
asyncio.gather(orchestrator.run(a), orchestrator.run(b)) in one process).
FetchURLTool's own internal concurrency (isolated pinned backends,
isolated clients per call) is already covered by
tests/unit/test_fetch_ssrf.py::test_concurrent_fetches_use_isolated_pinned_backends
and test_separate_executes_have_separate_keep_alive_pools -- not
duplicated here.

No OpenAI/internet access anywhere in this file -- everything is
MockLLMProvider/CalculatorTool/a disk-backed FileRunRepository under
tmp_path.
"""

import asyncio

import pytest

from app.agent.agent import Agent
from app.persistence.file_repository import FileRunRepository
from app.persistence.repository import ConcurrentResumeError, RunAlreadyFinalizedError
from app.policies.execution import ExecutionPolicy
from app.providers.mock_llm import MockLLMProvider
from app.reports.markdown import MarkdownReportRenderer
from app.resume.service import ResumeService
from app.schemas.cost import TokenUsage
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.state import ResearchRequest, TerminationReason
from app.schemas.tool import ToolCall
from app.services.run_service import RunService
from app.synthesis.mock_provider import MockSynthesisProvider
from app.synthesis.service import SynthesisService
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


def _call(call_id: str, expression: str = "1+1") -> ToolCall:
    return ToolCall(call_id=call_id, tool_name="calculator", arguments={"expression": expression})


def _agent_and_provider(decisions, usages=None):
    provider = MockLLMProvider(decisions, usages=usages)
    agent = Agent(provider, ToolRegistry([CalculatorTool()]), policy=ExecutionPolicy(max_steps=5))
    return agent, provider


class TestConcurrentRunsAreIsolated:
    async def test_two_concurrent_runs_do_not_share_state(self):
        agent_a, _ = _agent_and_provider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call("a1", "1+1")]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        agent_b, _ = _agent_and_provider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call("b1", "2+2")]),
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call("b2", "3+3")]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )

        result_a, result_b = await asyncio.gather(
            agent_a.run(ResearchRequest(question="Run A?")),
            agent_b.run(ResearchRequest(question="Run B?")),
        )

        state_a, state_b = agent_a.last_state, agent_b.last_state
        assert state_a is not state_b
        assert state_a.research_id != state_b.research_id
        assert [c.call_id for c in state_a.tool_calls] == ["a1"]
        assert [c.call_id for c in state_b.tool_calls] == ["b1", "b2"]
        assert result_a.steps == 2
        assert result_b.steps == 3
        assert state_a.original_question == "Run A?"
        assert state_b.original_question == "Run B?"

    async def test_three_concurrent_runs_remain_fully_isolated(self):
        agents = []
        for i in range(3):
            agent, _ = _agent_and_provider(
                [
                    LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call(f"c{i}", f"{i}+{i}")]),
                    LLMDecision(action=DecisionAction.FINISH),
                ]
            )
            agents.append(agent)

        await asyncio.gather(*(agent.run(ResearchRequest(question=f"Q{i}")) for i, agent in enumerate(agents)))

        research_ids = {agent.last_state.research_id for agent in agents}
        assert len(research_ids) == 3  # no collisions, no shared object
        for i, agent in enumerate(agents):
            assert agent.last_state.original_question == f"Q{i}"
            assert [c.call_id for c in agent.last_state.tool_calls] == [f"c{i}"]

    async def test_token_usage_does_not_leak_between_concurrent_runs(self):
        agent_a, _ = _agent_and_provider(
            [LLMDecision(action=DecisionAction.FINISH)],
            usages=[TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150, actual=True)],
        )
        agent_b, _ = _agent_and_provider(
            [LLMDecision(action=DecisionAction.FINISH)],
            usages=[TokenUsage(input_tokens=9, output_tokens=1, total_tokens=10, actual=True)],
        )

        await asyncio.gather(
            agent_a.run(ResearchRequest(question="A")),
            agent_b.run(ResearchRequest(question="B")),
        )

        assert agent_a.last_state.token_usage.total_tokens == 150
        assert agent_b.last_state.token_usage.total_tokens == 10

    async def test_execution_policy_does_not_leak_between_concurrent_runs(self):
        provider_a = MockLLMProvider(
            [LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call("a1")])] * 3
            + [LLMDecision(action=DecisionAction.FINISH)]
        )
        agent_a = Agent(provider_a, ToolRegistry([CalculatorTool()]), policy=ExecutionPolicy(max_steps=1))
        provider_b = MockLLMProvider(
            [LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call("b1", f"{i}+1")]) for i in range(3)]
            + [LLMDecision(action=DecisionAction.FINISH)]
        )
        agent_b = Agent(provider_b, ToolRegistry([CalculatorTool()]), policy=ExecutionPolicy(max_steps=10))

        result_a, result_b = await asyncio.gather(
            agent_a.run(ResearchRequest(question="A")),
            agent_b.run(ResearchRequest(question="B")),
        )

        assert result_a.termination_reason == TerminationReason.MAX_STEPS
        assert result_b.termination_reason == TerminationReason.COMPLETED


class TestConcurrentCheckpointsAcrossRuns:
    async def test_two_runs_checkpointing_concurrently_do_not_corrupt_each_others_files(self, tmp_path):
        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)

        seen_ids: dict[str, str] = {}

        def make_callback(label: str):
            async def checkpoint(state) -> None:
                seen_ids[label] = state.research_id
                record = run_service.build_run(state, ExecutionPolicy(), research_result=None)
                run_service.save_checkpoint(record)

            return checkpoint

        agent_a, _ = _agent_and_provider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call("a1")]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        agent_b, _ = _agent_and_provider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call("b1")]),
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call("b2")]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )

        await asyncio.gather(
            agent_a.run(ResearchRequest(question="A"), checkpoint_callback=make_callback("a")),
            agent_b.run(ResearchRequest(question="B"), checkpoint_callback=make_callback("b")),
        )

        run_id_a, run_id_b = seen_ids["a"], seen_ids["b"]
        assert run_id_a != run_id_b
        record_a = repository.get(run_id_a)
        record_b = repository.get(run_id_b)
        assert [c.call_id for c in record_a.research_state.tool_calls] == ["a1"]
        assert [c.call_id for c in record_b.research_state.tool_calls] == ["b1", "b2"]


class TestConcurrentAndDuplicateResume:
    async def _persist_incomplete_checkpoint(self, tmp_path, run_id: str = "run_concurrent"):
        from datetime import UTC, datetime

        from app.schemas.state import ResearchState

        state = ResearchState(
            research_id=run_id,
            original_question="question",
            created_at=datetime.now(UTC),
            current_step=1,
        )
        run_service = RunService(FileRunRepository(tmp_path))
        run_service.save_checkpoint(run_service.build_run(state, ExecutionPolicy(max_steps=5), research_result=None))
        return run_service

    def _resume_service(self, run_service: RunService, provider: MockLLMProvider) -> ResumeService:
        def agent_factory(policy: ExecutionPolicy) -> Agent:
            return Agent(provider, ToolRegistry([CalculatorTool()]), policy=policy)  # type: ignore[list-item]

        return ResumeService(
            run_service, agent_factory, SynthesisService(MockSynthesisProvider([])), MarkdownReportRenderer()
        )

    async def test_second_concurrent_resume_of_the_same_run_is_rejected(self, tmp_path):
        run_service = await self._persist_incomplete_checkpoint(tmp_path)

        release_first = asyncio.Event()

        class BlockingProvider:
            def __init__(self) -> None:
                self.calls = 0

            async def generate_decision(self, question, context=None, available_tools=None):
                self.calls += 1
                await release_first.wait()
                return LLMDecision(action=DecisionAction.FINISH), None

        blocking = BlockingProvider()
        first_resume_service = self._resume_service(run_service, blocking)  # type: ignore[arg-type]
        second_resume_service = self._resume_service(run_service, MockLLMProvider([{"action": "finish"}]))

        first_task = asyncio.create_task(first_resume_service.resume("run_concurrent"))
        # Give the first resume a chance to acquire the lock and block
        # inside generate_decision before starting the second one.
        while blocking.calls == 0:
            await asyncio.sleep(0)

        with pytest.raises(ConcurrentResumeError):
            await second_resume_service.resume("run_concurrent")

        release_first.set()
        result = await first_task
        assert result.termination_reason == TerminationReason.FINISHED

    async def test_lock_is_released_after_resume_so_a_later_resume_can_proceed(self, tmp_path):
        """Not concurrent -- proves the lock is released on completion,
        so a *subsequent* resume attempt (e.g. after a real crash of the
        first resume attempt) is never permanently blocked."""
        run_service = await self._persist_incomplete_checkpoint(tmp_path)
        resume_service = self._resume_service(run_service, MockLLMProvider([{"action": "finish"}]))

        result = await resume_service.resume("run_concurrent")
        assert result.termination_reason == TerminationReason.FINISHED

        # The run is now finalized; resuming again is a *different*,
        # already-covered rejection (RunAlreadyFinalizedError) -- what
        # this test actually proves is that the lock file from the first
        # call was cleaned up (no leftover ConcurrentResumeError).
        with pytest.raises(RunAlreadyFinalizedError):
            await resume_service.resume("run_concurrent")

    async def test_lock_is_released_even_if_resume_raises(self, tmp_path):
        run_service = await self._persist_incomplete_checkpoint(tmp_path)

        class FailingProvider:
            async def generate_decision(self, question, context=None, available_tools=None):
                raise RuntimeError("simulated provider crash")

        resume_service = self._resume_service(run_service, FailingProvider())  # type: ignore[arg-type]

        # The Agent's own except-Exception handling converts this into a
        # controlled PROVIDER_ERROR termination, not a raised exception --
        # so this resume succeeds (finalizes with an error state) rather
        # than raising. What matters here is that the lock is released
        # afterward either way.
        result = await resume_service.resume("run_concurrent")
        assert result.termination_reason == TerminationReason.PROVIDER_ERROR

        second_resume_service = self._resume_service(run_service, MockLLMProvider([]))
        with pytest.raises(RunAlreadyFinalizedError):
            await second_resume_service.resume("run_concurrent")


class TestNoTaskLeaks:
    async def test_agent_run_leaves_no_extra_tasks_behind(self):
        """app/ never calls asyncio.create_task/gather/wait/shield -- the
        whole Agent loop is a plain sequential await chain. This is a
        smoke test proving that, structurally: whatever the task set looks
        like before a run, it's identical after (no tasks the Agent
        started are still alive)."""
        agent, _ = _agent_and_provider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[_call("c1")]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        before = asyncio.all_tasks() - {asyncio.current_task()}

        await agent.run(ResearchRequest(question="q"))

        after = asyncio.all_tasks() - {asyncio.current_task()}
        assert after == before


class TestCancellation:
    async def test_cancelling_the_run_task_leaves_no_partial_checkpoint(self, tmp_path):
        """Checkpoint writes are synchronous filesystem calls inside an
        async function body -- asyncio can only interrupt at an `await`,
        so a save_checkpoint() call, once started, always completes before
        a cancellation can take effect. This proves that property from the
        outside: cancelling mid-run leaves either no checkpoint or a fully
        valid one, never a corrupt/partial file."""
        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)
        never_returns = asyncio.Event()

        class HangingProvider:
            async def generate_decision(self, question, context=None, available_tools=None):
                await never_returns.wait()
                raise AssertionError("unreachable")

        agent = Agent(HangingProvider(), ToolRegistry([]), policy=ExecutionPolicy(max_steps=5))  # type: ignore[arg-type]

        async def checkpoint(state) -> None:
            record = run_service.build_run(state, ExecutionPolicy(), research_result=None)
            run_service.save_checkpoint(record)

        task = asyncio.create_task(
            agent.run(ResearchRequest(question="will be cancelled"), checkpoint_callback=checkpoint)
        )
        await asyncio.sleep(0)  # let it reach the hanging generate_decision call
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # No decision ever completed, so no checkpoint was ever written --
        # nothing on disk to be partial in the first place.
        assert list(tmp_path.iterdir()) == []

    async def test_cancellation_is_not_swallowed_as_a_tool_failure(self, tmp_path):
        """CancelledError must propagate, never be caught by Agent's or
        ToolRegistry's `except Exception` handling and turned into a
        seemingly-normal failed ToolResult."""
        never_returns = asyncio.Event()

        class HangingTool:
            name = "hang"
            description = "Never returns."
            from pydantic import BaseModel

            class _Input(BaseModel):
                pass

            input_schema = _Input

            async def execute(self) -> None:
                await never_returns.wait()

        provider = MockLLMProvider(
            [LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c1", tool_name="hang", arguments={})])]
        )
        agent = Agent(provider, ToolRegistry([HangingTool()]), policy=ExecutionPolicy(max_steps=5))

        task = asyncio.create_task(agent.run(ResearchRequest(question="q")))
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        # Nothing marks the run as finished/successful -- last_state is
        # never even set, since _finish() is never reached.
        assert agent.last_state is None
