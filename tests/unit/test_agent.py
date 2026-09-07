import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from app.agent.agent import Agent
from app.core.exceptions import ProviderError
from app.policies.execution import ExecutionPolicy
from app.providers.mock_llm import MockLLMProvider
from app.providers.search import MockSearchProvider, SearchHit
from app.schemas.cost import TokenUsage
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.state import ResearchRequest, ResearchState, TerminationReason
from app.schemas.tool import ToolCall
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


class EchoInput(BaseModel):
    value: str


class RecordingTool:
    name = "record"
    description = "Records execution order."
    input_schema = EchoInput
    executions: list[str] = []

    async def execute(self, value: str) -> dict[str, str]:
        self.executions.append(value)
        return {"value": value}


class FailingTool:
    name = "failing"
    description = "Raises a controlled tool error."
    input_schema = EchoInput

    async def execute(self, value: str) -> str:
        raise RuntimeError(f"failed: {value}")


class StaticFetchTool:
    name = "fetch_url"
    description = "Returns a deterministic page."
    input_schema = EchoInput

    async def execute(self, value: str) -> dict[str, Any]:
        return {"url": value, "text": "Fetched page text", "truncated": False}


class FailingLLMProvider:
    async def generate_decision(
        self,
        question: str,
        context: list[dict[str, Any]] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> LLMDecision:
        raise ProviderError("provider unavailable")


def call(call_id: str, tool_name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(call_id=call_id, tool_name=tool_name, arguments=arguments)


class TestAgentLoop:
    async def test_search_and_fetch_results_are_added_to_evidence_state(self):
        search = WebSearchTool(
            MockSearchProvider(
                [SearchHit("https://example.com/page", "Page", "Search snippet")]
            )
        )
        provider = MockLLMProvider(
            [
                LLMDecision(
                    action=DecisionAction.TOOL_CALL,
                    tool_calls=[call("c1", "web_search", {"query": "page"})],
                ),
                LLMDecision(
                    action=DecisionAction.TOOL_CALL,
                    tool_calls=[call("c2", "fetch_url", {"value": "https://example.com/page/"})],
                ),
                {"action": "finish"},
            ]
        )
        agent = Agent(provider, ToolRegistry([search, StaticFetchTool()]))

        result = await agent.run(ResearchRequest(question="find page"))

        assert result.termination_reason == TerminationReason.COMPLETED
        assert len(result.sources) == 1
        assert len(result.evidence) == 2
        assert {item.content for item in result.evidence} == {"Search snippet", "Fetched page text"}

    async def test_finish_immediately(self):
        provider = MockLLMProvider([{"action": "finish", "rationale": "Done."}])
        agent = Agent(provider, ToolRegistry([]))

        result = await agent.run(ResearchRequest(question="question"))

        assert result.termination_reason == TerminationReason.COMPLETED
        assert result.steps == 1
        assert result.tool_call_count == 0
        assert agent.last_state is not None
        assert agent.last_state.decisions[0].action == DecisionAction.FINISH

    async def test_tool_call_then_finish_updates_state_and_context(self):
        provider = MockLLMProvider(
            [
                LLMDecision(
                    action=DecisionAction.TOOL_CALL,
                    rationale="Calculate.",
                    tool_calls=[call("c1", "calculator", {"expression": "2 + 3"})],
                ),
                {"action": "finish", "rationale": "Complete."},
            ]
        )
        agent = Agent(provider, ToolRegistry([CalculatorTool()]))

        result = await agent.run(ResearchRequest(question="What is 2 + 3?"))

        assert result.termination_reason == TerminationReason.COMPLETED
        assert result.steps == 2
        assert result.tool_call_count == 1
        assert agent.last_state is not None
        assert len(agent.last_state.decisions) == 2
        assert len(agent.last_state.tool_results) == 1
        assert agent.last_state.tool_results[0].output == 5
        assert len(agent.last_state.observations) == 1
        assert provider.calls[1]["context"][0]["tool_results"][0]["output"] == 5

    async def test_multiple_tool_calls_execute_in_order(self):
        RecordingTool.executions = []
        provider = MockLLMProvider(
            [
                LLMDecision(
                    action=DecisionAction.TOOL_CALL,
                    tool_calls=[
                        call("c1", "record", {"value": "first"}),
                        call("c2", "record", {"value": "second"}),
                    ],
                ),
                {"action": "finish"},
            ]
        )
        agent = Agent(provider, ToolRegistry([RecordingTool()]))

        result = await agent.run(ResearchRequest(question="record values"))

        assert result.termination_reason == TerminationReason.COMPLETED
        assert RecordingTool.executions == ["first", "second"]
        assert [result.call_id for result in agent.last_state.tool_results] == ["c1", "c2"]

    async def test_unknown_tool_is_recorded_and_loop_continues(self):
        provider = MockLLMProvider(
            [
                LLMDecision(
                    action=DecisionAction.TOOL_CALL,
                    tool_calls=[call("c1", "missing", {})],
                ),
                {"action": "finish"},
            ]
        )
        agent = Agent(provider, ToolRegistry([]))

        result = await agent.run(ResearchRequest(question="question"))

        assert result.termination_reason == TerminationReason.COMPLETED
        assert agent.last_state is not None
        assert agent.last_state.tool_results[0].success is False
        assert "Unknown tool" in agent.last_state.errors[0]

    async def test_tool_error_is_recorded_and_loop_continues(self):
        provider = MockLLMProvider(
            [
                LLMDecision(
                    action=DecisionAction.TOOL_CALL,
                    tool_calls=[call("c1", "failing", {"value": "x"})],
                ),
                {"action": "finish"},
            ]
        )
        agent = Agent(provider, ToolRegistry([FailingTool()]))

        result = await agent.run(ResearchRequest(question="question"))

        assert result.termination_reason == TerminationReason.COMPLETED
        assert agent.last_state.tool_results[0].success is False
        assert "RuntimeError: failed: x" in agent.last_state.errors[0]

    async def test_llm_error_returns_controlled_result(self):
        agent = Agent(FailingLLMProvider(), ToolRegistry([]))

        result = await agent.run(ResearchRequest(question="question"))

        assert result.termination_reason == TerminationReason.PROVIDER_ERROR
        assert result.errors == ["ProviderError: provider unavailable"]
        assert result.tool_call_count == 0

    async def test_invalid_decision_returns_controlled_result(self):
        provider = MockLLMProvider([LLMDecision()])
        agent = Agent(provider, ToolRegistry([]))

        result = await agent.run(ResearchRequest(question="question"))

        assert result.termination_reason == TerminationReason.INVALID_OUTPUT
        assert "no actionable operation" in result.errors[0]

    async def test_synthesize_ends_research_without_synthesis(self):
        provider = MockLLMProvider([{"action": "synthesize"}])
        agent = Agent(provider, ToolRegistry([]))

        result = await agent.run(ResearchRequest(question="question"))

        assert result.termination_reason == TerminationReason.SYNTHESIS_REQUESTED
        assert result.final_answer is None
        assert agent.last_state.observations == []

    async def test_minimal_step_guard_terminates_without_infinite_loop(self):
        provider = MockLLMProvider(
            [LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c1", "record", {"value": "x"})])]
        )
        agent = Agent(provider, ToolRegistry([RecordingTool()]), max_steps=1)

        result = await agent.run(ResearchRequest(question="question"))

        assert result.termination_reason == TerminationReason.MAX_STEPS_REACHED
        assert result.steps == 1

    async def test_state_round_trips_after_run(self):
        provider = MockLLMProvider([{"action": "finish"}])
        agent = Agent(provider, ToolRegistry([]))

        await agent.run(ResearchRequest(question="question"))

        restored = type(agent.last_state).model_validate_json(agent.last_state.model_dump_json())
        assert restored.termination_reason == TerminationReason.COMPLETED
        assert len(restored.decisions) == 1


class RawRaisingToolRegistry:
    """Duck-types ToolRegistry but raises directly from execute(), unlike
    the real ToolRegistry (which always catches and returns a failed
    ToolResult) -- exercises Agent's own defensive except-branch around
    tool_registry.execute()."""

    def describe(self) -> list[dict[str, Any]]:
        return []

    async def execute(self, call: ToolCall, timeout_seconds: float | None = None) -> Any:
        raise RuntimeError("registry itself blew up")


class TestAgentLogging:
    async def test_emits_run_started_decision_and_run_lifecycle_events(self, caplog):

        caplog.set_level(logging.INFO, logger="ai_research_agent")
        provider = MockLLMProvider(
            [
                LLMDecision(
                    action=DecisionAction.TOOL_CALL,
                    rationale="Look it up.",
                    tool_calls=[call("c1", "record", {"value": "x"})],
                ),
                {"action": "finish"},
            ]
        )
        agent = Agent(provider, ToolRegistry([RecordingTool()]))

        result = await agent.run(ResearchRequest(question="question"))
        run_id = agent.last_state.research_id

        events = [record.event for record in caplog.records]
        assert events == [
            "run_started",
            "decision",
            "tool_started",
            "tool_completed",
            "decision",
        ]
        assert all(record.run_id == run_id for record in caplog.records)

        run_started = caplog.records[0]
        assert run_started.question == "question"

        first_decision = caplog.records[1]
        assert first_decision.action == "tool_call"
        assert first_decision.rationale == "Look it up."
        assert first_decision.tool_call_count == 1
        assert first_decision.step == 1

        tool_started = caplog.records[2]
        assert tool_started.call_id == "c1"
        assert tool_started.tool_name == "record"

        tool_completed = caplog.records[3]
        assert tool_completed.call_id == "c1"
        assert tool_completed.tool_name == "record"

        assert result.termination_reason == TerminationReason.COMPLETED

    async def test_emits_tool_failed_on_unsuccessful_tool_result(self, caplog):

        caplog.set_level(logging.INFO, logger="ai_research_agent")
        provider = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c1", "failing", {"value": "x"})]),
                {"action": "finish"},
            ]
        )
        agent = Agent(provider, ToolRegistry([FailingTool()]))

        await agent.run(ResearchRequest(question="question"))

        tool_failed = next(r for r in caplog.records if r.event == "tool_failed")
        assert tool_failed.levelno == logging.WARNING
        assert tool_failed.call_id == "c1"
        assert "RuntimeError" in tool_failed.error

    async def test_emits_tool_failed_when_registry_itself_raises(self, caplog):

        caplog.set_level(logging.INFO, logger="ai_research_agent")
        provider = MockLLMProvider(
            [LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c1", "anything", {"value": "x"})])]
        )
        agent = Agent(provider, RawRaisingToolRegistry())  # type: ignore[arg-type]

        result = await agent.run(ResearchRequest(question="question"))

        tool_failed = next(r for r in caplog.records if r.event == "tool_failed")
        assert tool_failed.levelno == logging.WARNING
        assert "registry itself blew up" in tool_failed.error
        assert result.termination_reason == TerminationReason.TOOL_ERROR

    async def test_no_events_emitted_below_configured_log_level(self, caplog):

        caplog.set_level(logging.WARNING, logger="ai_research_agent")
        provider = MockLLMProvider([{"action": "finish"}])
        agent = Agent(provider, ToolRegistry([]))

        await agent.run(ResearchRequest(question="question"))

        assert [r for r in caplog.records if r.event in {"run_started", "decision"}] == []


class TestAgentTokenUsageAggregation:
    async def test_accumulates_usage_across_multiple_decisions(self):
        provider = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c1", "record", {"value": "a"})]),
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c2", "record", {"value": "b"})]),
                LLMDecision(action=DecisionAction.FINISH),
            ],
            usages=[
                TokenUsage(input_tokens=60, output_tokens=40, total_tokens=100, actual=True),
                TokenUsage(input_tokens=120, output_tokens=80, total_tokens=200, actual=True),
                TokenUsage(input_tokens=180, output_tokens=120, total_tokens=300, actual=True),
            ],
        )
        agent = Agent(provider, ToolRegistry([RecordingTool()]))

        await agent.run(ResearchRequest(question="question"))

        usage = agent.last_state.token_usage
        assert usage.total_tokens == 600
        assert usage.input_tokens == 360
        assert usage.output_tokens == 240
        # MockLLMProvider forces actual=False on any configured usage,
        # regardless of what the test passed in -- a mock never claims to
        # have measured real OpenAI consumption.
        assert usage.actual is False

    async def test_skips_accumulation_when_usage_is_none_without_double_counting(self):
        provider = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c1", "record", {"value": "a"})]),
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c2", "record", {"value": "b"})]),
                LLMDecision(action=DecisionAction.FINISH),
            ],
            usages=[
                TokenUsage(total_tokens=100),
                None,
                TokenUsage(total_tokens=200),
            ],
        )
        agent = Agent(provider, ToolRegistry([RecordingTool()]))

        await agent.run(ResearchRequest(question="question"))

        assert agent.last_state.token_usage.total_tokens == 300

    async def test_no_usage_configured_leaves_token_usage_empty(self):
        provider = MockLLMProvider([{"action": "finish"}])
        agent = Agent(provider, ToolRegistry([]))

        await agent.run(ResearchRequest(question="question"))

        usage = agent.last_state.token_usage
        assert usage.total_tokens is None
        assert usage.input_tokens is None
        assert usage.output_tokens is None


class TestAgentCheckpointing:
    """Fase 11E: checkpoint_callback fires once per fully-completed step
    and once more at termination -- never mid-step -- and `initial_state`
    lets a checkpointed (not yet terminated) ResearchState be continued."""

    async def test_checkpoint_fires_once_per_completed_step_and_once_at_finish(self):
        provider = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c1", "record", {"value": "a"})]),
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c2", "record", {"value": "b"})]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        agent = Agent(provider, ToolRegistry([RecordingTool()]))
        checkpoints: list[ResearchState] = []

        async def checkpoint_callback(state: ResearchState) -> None:
            checkpoints.append(state.model_copy(deep=True))

        result = await agent.run(ResearchRequest(question="question"), checkpoint_callback=checkpoint_callback)

        # Step 1 (continues) + step 2 (continues) + the terminal FINISH
        # (its own step) -- never once per tool call or per LLM call
        # within a step.
        assert len(checkpoints) == 3
        assert [c.current_step for c in checkpoints] == [1, 2, 3]
        assert checkpoints[0].termination_reason is None
        assert checkpoints[1].termination_reason is None
        assert checkpoints[2].termination_reason == TerminationReason.FINISHED
        assert result.termination_reason == TerminationReason.COMPLETED

    async def test_checkpoint_receives_elapsed_seconds(self):
        provider = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c1", "record", {"value": "a"})]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        agent = Agent(provider, ToolRegistry([RecordingTool()]))
        checkpoints: list[ResearchState] = []

        async def checkpoint_callback(state: ResearchState) -> None:
            checkpoints.append(state.model_copy(deep=True))

        await agent.run(ResearchRequest(question="question"), checkpoint_callback=checkpoint_callback)

        assert all(c.elapsed_seconds is not None and c.elapsed_seconds >= 0 for c in checkpoints)

    async def test_no_checkpoint_callback_means_no_checkpointing(self):
        provider = MockLLMProvider([{"action": "finish"}])
        agent = Agent(provider, ToolRegistry([]))

        # Default parameter: existing callers unaffected.
        result = await agent.run(ResearchRequest(question="question"))

        assert result.termination_reason == TerminationReason.COMPLETED

    async def test_emits_checkpoint_saved_log_event(self, caplog):
        caplog.set_level(logging.INFO, logger="ai_research_agent")
        provider = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c1", "record", {"value": "a"})]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        agent = Agent(provider, ToolRegistry([RecordingTool()]))

        async def checkpoint_callback(state: ResearchState) -> None:
            return None

        await agent.run(ResearchRequest(question="question"), checkpoint_callback=checkpoint_callback)

        events = [r for r in caplog.records if r.event == "checkpoint_saved"]
        assert len(events) == 2
        assert events[0].step == 1
        assert events[0].termination_reason is None
        assert events[1].step == 2
        assert events[1].termination_reason == "finished"
        # Never the prompt/completion/tool payload -- only diagnostic fields.
        assert not hasattr(events[0], "tool_results")

    async def test_resumes_from_checkpointed_current_step_without_new_research_id(self):
        checkpointed = ResearchState(
            research_id="run_resume_1",
            original_question="question",
            created_at=datetime.now(UTC),
            current_step=1,
        )
        provider = MockLLMProvider([{"action": "finish"}])
        agent = Agent(provider, ToolRegistry([]))

        result = await agent.run(ResearchRequest(question="question"), initial_state=checkpointed)

        assert agent.last_state is checkpointed
        assert agent.last_state.research_id == "run_resume_1"
        # Continues from step 1 -> step 2, not restarted at step 1.
        assert agent.last_state.current_step == 2
        assert result.termination_reason == TerminationReason.COMPLETED

    async def test_resuming_an_already_terminated_state_raises(self):
        terminated = ResearchState(
            research_id="run_done",
            original_question="question",
            created_at=datetime.now(UTC),
            termination_reason=TerminationReason.MAX_STEPS,
        )
        provider = MockLLMProvider([{"action": "finish"}])
        agent = Agent(provider, ToolRegistry([]))

        try:
            await agent.run(ResearchRequest(question="question"), initial_state=terminated)
            raised = False
        except ValueError:
            raised = True
        assert raised

    async def test_resume_preserves_loop_detection_across_the_crash(self):
        # Already made two identical calculator calls before the crash --
        # default max_same_tool_calls is >= 2, so a resumed run must not
        # reset the count to zero and allow the policy to be exceeded.
        prior_call_1 = call("c1", "calculator", {"expression": "1+1"})
        prior_call_2 = call("c2", "calculator", {"expression": "1+1"})
        checkpointed = ResearchState(
            research_id="run_resume_loop",
            original_question="question",
            created_at=datetime.now(UTC),
            current_step=1,
            tool_calls=[prior_call_1, prior_call_2],
            tool_results=[],
        )
        provider = MockLLMProvider(
            [LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[call("c3", "calculator", {"expression": "1+1"})])]
        )
        agent = Agent(
            provider,
            ToolRegistry([CalculatorTool()]),
            policy=ExecutionPolicy(max_same_tool_calls=2),
        )

        result = await agent.run(ResearchRequest(question="question"), initial_state=checkpointed)

        assert result.termination_reason == TerminationReason.LOOP_DETECTED

    async def test_resume_accumulates_elapsed_seconds_on_top_of_prior(self):
        checkpointed = ResearchState(
            research_id="run_resume_time",
            original_question="question",
            created_at=datetime.now(UTC),
            current_step=1,
            elapsed_seconds=1000.0,
        )
        provider = MockLLMProvider([{"action": "finish"}])
        agent = Agent(provider, ToolRegistry([]))

        result = await agent.run(ResearchRequest(question="question"), initial_state=checkpointed)

        # The pre-crash 1000s is folded in, not discarded and not
        # fabricated further -- the continuation itself takes ~0s here.
        assert result.elapsed_seconds >= 1000.0
