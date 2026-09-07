"""Fase 11E: crash + resume simulation tests.

A real crash is never simulated by killing a process -- instead, a
checkpoint_callback wrapping the real, disk-backed FileRunRepository is
made to raise partway through an Agent run, which leaves exactly the
persisted state a genuine crash at that point would have left (the prior
checkpoint's atomic write already completed; nothing after it did). A
brand new ResumeService (fresh Agent, fresh MockLLMProvider standing in
for "the process restarted") then continues from that exact on-disk
state, proving resume works across process boundaries, not just within
one Python object graph.
"""

from datetime import UTC, datetime

import pytest

from app.agent.agent import Agent
from app.evidence.pipeline import EvidencePipeline
from app.persistence.file_repository import FileRunRepository
from app.persistence.repository import (
    RunAlreadyFinalizedError,
    RunNotFoundError,
    UnsupportedSchemaVersionError,
)
from app.policies.execution import ExecutionPolicy
from app.providers.mock_llm import MockLLMProvider
from app.replay.service import ReplayService
from app.reports.markdown import MarkdownReportRenderer
from app.resume.service import ResumeService
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.cost import TokenUsage
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.run import RunRecord
from app.schemas.state import ResearchRequest, ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall
from app.services.orchestrator import make_checkpoint_callback
from app.services.run_service import RunService
from app.synthesis.mock_provider import MockSynthesisProvider
from app.synthesis.service import SynthesisService
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class _CrashSimulated(Exception):
    """Stands in for the process dying -- never a real application error."""


def _build_agent_factory(provider: MockLLMProvider, tool_registry: ToolRegistry):
    def agent_factory(policy: ExecutionPolicy) -> Agent:
        return Agent(provider, tool_registry, policy=policy, evidence_pipeline=EvidencePipeline())

    return agent_factory


def _resume_service(tmp_path, provider: MockLLMProvider, tool_registry: ToolRegistry, answers=None) -> ResumeService:
    run_service = RunService(FileRunRepository(tmp_path))
    synthesis_service = SynthesisService(MockSynthesisProvider(answers or []))
    return ResumeService(run_service, _build_agent_factory(provider, tool_registry), synthesis_service, MarkdownReportRenderer())


class TestResumeMidLoopCrash:
    async def test_resume_continues_from_the_checkpointed_step(self, tmp_path):
        policy = ExecutionPolicy(max_steps=5, max_tool_calls=10, max_same_tool_calls=3)
        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)

        # "Session 1": runs step 1 for real, checkpoints it, then crashes
        # before step 2 -- we capture the real, Agent-assigned run_id by
        # wrapping the callback with a spy.
        seen_run_id: dict[str, str] = {}
        provider1 = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c1", tool_name="calculator", arguments={"expression": "1+1"})]),
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c2", tool_name="calculator", arguments={"expression": "2+2"})]),
            ]
        )
        agent1 = Agent(provider1, ToolRegistry([CalculatorTool()]), policy=policy)
        real_callback = make_checkpoint_callback(run_service, policy)
        calls = {"count": 0}

        async def crashing_callback(state: ResearchState) -> None:
            calls["count"] += 1
            seen_run_id["run_id"] = state.research_id
            if calls["count"] > 1:
                raise _CrashSimulated("crash before step 2's checkpoint")
            await real_callback(state)


        with pytest.raises(_CrashSimulated):
            await agent1.run(ResearchRequest(question="Two calculations"), checkpoint_callback=crashing_callback)

        run_id = seen_run_id["run_id"]
        on_disk = repository.get(run_id)
        assert on_disk.research_state.current_step == 1
        assert on_disk.research_state.termination_reason is None
        assert on_disk.research_result is None
        assert [c.call_id for c in on_disk.research_state.tool_calls] == ["c1"]

        # "Session 2" (a brand new process/provider/agent): resumes it.
        provider2 = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c2", tool_name="calculator", arguments={"expression": "2+2"})]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        resume_service = _resume_service(tmp_path, provider2, ToolRegistry([CalculatorTool()]))

        result = await resume_service.resume(run_id)

        assert result.run_id == run_id
        assert result.record.run_id == run_id
        assert result.record.research_state.original_question == "Two calculations"
        assert result.record.execution_policy == policy
        assert [c.call_id for c in result.record.research_state.tool_calls] == ["c1", "c2"]
        assert len(result.record.research_state.tool_results) == 2
        assert result.record.research_result is not None
        assert result.termination_reason == TerminationReason.FINISHED
        assert result.success is True

        # Persisted for real, at the same run_id/run.json -- reloadable.
        reloaded = repository.get(run_id)
        assert reloaded.research_result is not None
        assert reloaded.created_at == on_disk.created_at

    async def test_does_not_duplicate_decisions_tool_calls_or_results(self, tmp_path):
        policy = ExecutionPolicy(max_steps=5)
        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)
        provider1 = MockLLMProvider(
            [LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c1", tool_name="calculator", arguments={"expression": "1+1"})])]
        )
        agent1 = Agent(provider1, ToolRegistry([CalculatorTool()]), policy=policy)
        run_id_holder: dict[str, str] = {}

        async def crashing_after_first(state: ResearchState) -> None:
            run_id_holder["id"] = state.research_id
            await make_checkpoint_callback(run_service, policy)(state)
            raise _CrashSimulated("crash right after the only checkpoint")


        with pytest.raises(_CrashSimulated):
            await agent1.run(ResearchRequest(question="one calc"), checkpoint_callback=crashing_after_first)

        run_id = run_id_holder["id"]
        provider2 = MockLLMProvider([LLMDecision(action=DecisionAction.FINISH)])
        resume_service = _resume_service(tmp_path, provider2, ToolRegistry([CalculatorTool()]))

        result = await resume_service.resume(run_id)

        assert len(result.record.research_state.decisions) == 2  # step 1's TOOL_CALL + step 2's FINISH
        assert len(result.record.research_state.tool_calls) == 1
        assert len(result.record.research_state.tool_results) == 1

    async def test_token_usage_is_not_double_counted(self, tmp_path):
        policy = ExecutionPolicy(max_steps=5)
        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)
        provider1 = MockLLMProvider(
            [LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c1", tool_name="calculator", arguments={"expression": "1+1"})])],
            usages=[TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150, actual=True)],
        )
        agent1 = Agent(provider1, ToolRegistry([CalculatorTool()]), policy=policy)
        run_id_holder: dict[str, str] = {}

        async def crashing_after_first(state: ResearchState) -> None:
            run_id_holder["id"] = state.research_id
            await make_checkpoint_callback(run_service, policy)(state)
            raise _CrashSimulated("crash after checkpointing the first (billed) decision")


        with pytest.raises(_CrashSimulated):
            await agent1.run(ResearchRequest(question="one calc"), checkpoint_callback=crashing_after_first)

        run_id = run_id_holder["id"]
        assert repository.get(run_id).research_state.token_usage.total_tokens == 150

        provider2 = MockLLMProvider(
            [LLMDecision(action=DecisionAction.FINISH)],
            usages=[TokenUsage(input_tokens=20, output_tokens=10, total_tokens=30, actual=True)],
        )
        resume_service = _resume_service(tmp_path, provider2, ToolRegistry([CalculatorTool()]))

        result = await resume_service.resume(run_id)

        # 150 (checkpointed before the crash) + 30 (the resumed decision)
        # -- never 150 re-added, never silently dropped.
        assert result.record.research_state.token_usage.total_tokens == 180

    async def test_replay_succeeds_after_resume(self, tmp_path):
        """Connects Fase 11D and 11E: a run that crashed mid-loop, was
        resumed to completion, must then be replayable and equivalent."""
        policy = ExecutionPolicy(max_steps=5)
        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)
        provider1 = MockLLMProvider(
            [LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c1", tool_name="calculator", arguments={"expression": "1+1"})])]
        )
        agent1 = Agent(provider1, ToolRegistry([CalculatorTool()]), policy=policy)
        run_id_holder: dict[str, str] = {}

        async def crashing_after_first(state: ResearchState) -> None:
            run_id_holder["id"] = state.research_id
            await make_checkpoint_callback(run_service, policy)(state)
            raise _CrashSimulated("crash")


        with pytest.raises(_CrashSimulated):
            await agent1.run(ResearchRequest(question="one calc"), checkpoint_callback=crashing_after_first)

        run_id = run_id_holder["id"]
        provider2 = MockLLMProvider([LLMDecision(action=DecisionAction.FINISH)])
        resume_service = _resume_service(tmp_path, provider2, ToolRegistry([CalculatorTool()]))
        result = await resume_service.resume(run_id)

        comparison = await ReplayService().replay(result.record)

        assert comparison.equivalent is True
        assert comparison.differences == []


def _evidence_ready_state(run_id: str, *, termination_reason) -> ResearchState:
    timestamp = datetime.now(UTC)
    source = Source(source_id="src_1", url="https://example.com/a", tool_call_id="c1", timestamp=timestamp)
    evidence = Evidence(evidence_id="ev_1", content="Evidence text", source_id="src_1", tool_call_id="c1", timestamp=timestamp)
    return ResearchState(
        research_id=run_id,
        original_question="Question needing synthesis",
        created_at=timestamp,
        current_step=2,
        decisions=[LLMDecision(action=DecisionAction.SYNTHESIZE)],
        tool_calls=[ToolCall(call_id="c1", tool_name="web_search", arguments={"query": "x"})],
        tool_results=[],
        sources=[source],
        evidence=[evidence],
        termination_reason=termination_reason,
    )


class TestResumeSynthesisOnlyCrash:
    async def test_finalizes_without_reentering_the_agent_loop(self, tmp_path):
        """Crash happened after the Agent loop already reached
        SYNTHESIS_REQUESTED and checkpointed it, but before synthesis (and
        the final save) completed -- resume must go straight to synthesis,
        never re-invoking the agent_factory at all."""
        run_id = "run_pending_synthesis"
        state = _evidence_ready_state(run_id, termination_reason=TerminationReason.SYNTHESIS_REQUESTED)
        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)
        run_service.save_checkpoint(run_service.build_run(state, ExecutionPolicy(), research_result=None))

        def fail_if_called(policy: ExecutionPolicy) -> Agent:
            raise AssertionError("resume must not re-enter the Agent loop when only synthesis is pending")

        answer = FinalAnswer(
            answer_text="Answer [1].",
            claims=[Claim(claim_id="cl_1", text="Answer", evidence_ids=["ev_1"])],
            citations=[Citation(claim="Answer", evidence_ids=["ev_1"])],
            is_complete=True,
        )
        synthesis_service = SynthesisService(MockSynthesisProvider([answer]))
        resume_service = ResumeService(run_service, fail_if_called, synthesis_service, MarkdownReportRenderer())

        result = await resume_service.resume(run_id)

        assert result.record.research_result is not None
        assert result.record.research_state.final_answer == answer
        assert result.termination_reason == TerminationReason.FINISHED

    async def test_finalize_only_crash_needs_no_provider_calls(self, tmp_path):
        """Crash happened after Agent AND synthesis both concluded (e.g.
        MAX_STEPS, no synthesis requested) but before the final
        persistence -- resume is a pure finalize with zero provider calls."""
        run_id = "run_pending_finalize"
        state = _evidence_ready_state(run_id, termination_reason=TerminationReason.MAX_STEPS)
        state.final_answer = None
        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)
        run_service.save_checkpoint(run_service.build_run(state, ExecutionPolicy(), research_result=None))

        def fail_if_called(policy: ExecutionPolicy) -> Agent:
            raise AssertionError("no Agent should be constructed for a finalize-only resume")

        class FailingSynthesisProvider:
            async def generate_answer(self, *args, **kwargs):
                raise AssertionError("synthesis must not run when termination_reason != SYNTHESIS_REQUESTED")

        synthesis_service = SynthesisService(FailingSynthesisProvider())
        resume_service = ResumeService(run_service, fail_if_called, synthesis_service, MarkdownReportRenderer())

        result = await resume_service.resume(run_id)

        assert result.record.research_result is not None
        assert result.termination_reason == TerminationReason.MAX_STEPS


def _build_finalized_record(run_id: str) -> RunRecord:
    state = ResearchState(
        research_id=run_id,
        original_question="Already done",
        created_at=datetime.now(UTC),
        current_step=1,
        termination_reason=TerminationReason.FINISHED,
    )
    return RunRecord(
        run_id=run_id,
        created_at=state.created_at,
        question=state.original_question,
        execution_policy=ExecutionPolicy(),
        research_state=state,
        research_result=ResearchResult.from_state(state),
    )


class _AgentWithoutState:
    """Mirrors the equivalent double in test_orchestrator.py -- proves
    ResumeService raises the same defensive RuntimeError Agent's real
    contract (last_state always set before returning) exists to prevent
    silently continuing with no state."""

    def __init__(self) -> None:
        self.last_state = None

    async def run(self, request, initial_state=None, checkpoint_callback=None) -> None:
        self.last_state = None


class TestResumeGuards:
    async def test_agent_missing_state_raises_runtime_error(self, tmp_path):
        run_id = "run_missing_state"
        state = ResearchState(
            research_id=run_id,
            original_question="question",
            created_at=datetime.now(UTC),
            current_step=1,
        )
        run_service = RunService(FileRunRepository(tmp_path))
        run_service.save_checkpoint(run_service.build_run(state, ExecutionPolicy(), research_result=None))

        resume_service = ResumeService(
            run_service,
            lambda policy: _AgentWithoutState(),
            SynthesisService(MockSynthesisProvider([])),
            MarkdownReportRenderer(),
        )

        with pytest.raises(RuntimeError, match="resumed Agent finished without recording last_state"):
            await resume_service.resume(run_id)


    async def test_refuses_to_resume_an_already_finalized_run(self, tmp_path):
        repository = FileRunRepository(tmp_path)
        repository.save(_build_finalized_record("run_done"))
        resume_service = _resume_service(tmp_path, MockLLMProvider([]), ToolRegistry([]))

        with pytest.raises(RunAlreadyFinalizedError, match="already finalized"):
            await resume_service.resume("run_done")

    async def test_missing_run_raises_run_not_found(self, tmp_path):
        resume_service = _resume_service(tmp_path, MockLLMProvider([]), ToolRegistry([]))

        with pytest.raises(RunNotFoundError):
            await resume_service.resume("missing_run")

    async def test_incompatible_schema_version_is_rejected_before_touching_agent(self, tmp_path):
        import json

        run_dir = tmp_path / "run_future"
        run_dir.mkdir(parents=True)
        record = _build_finalized_record("run_future")
        record.research_result = None
        payload = json.loads(record.model_dump_json())
        payload["schema_version"] = "2.0"
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

        def fail_if_called(policy: ExecutionPolicy) -> Agent:
            raise AssertionError("resume must never construct an Agent for an incompatible schema version")

        resume_service = ResumeService(
            RunService(FileRunRepository(tmp_path)),
            fail_if_called,
            SynthesisService(MockSynthesisProvider([])),
            MarkdownReportRenderer(),
        )

        with pytest.raises(UnsupportedSchemaVersionError):
            await resume_service.resume("run_future")

    async def test_resume_uses_the_injected_provider_not_a_new_one(self, tmp_path):
        """Proves resume drives the agent_factory's own MockLLMProvider
        instance (i.e. whatever composition.py injected) rather than
        constructing a fresh/real one internally."""
        run_id = "run_injected"
        state = ResearchState(
            research_id=run_id,
            original_question="question",
            created_at=datetime.now(UTC),
            current_step=1,
        )
        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)
        run_service.save_checkpoint(run_service.build_run(state, ExecutionPolicy(max_steps=3), research_result=None))

        provider = MockLLMProvider([LLMDecision(action=DecisionAction.FINISH)])
        resume_service = _resume_service(tmp_path, provider, ToolRegistry([]))

        await resume_service.resume(run_id)

        assert len(provider.calls) == 1
        assert provider.calls[0]["question"] == "question"
