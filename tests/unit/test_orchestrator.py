import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from app.agent.agent import Agent
from app.core.exceptions import (
    ProviderError,
    ReportRenderingError,
)
from app.persistence.repository import RepositoryError
from app.policies.execution import ExecutionPolicy
from app.providers.mock_llm import MockLLMProvider
from app.reports.markdown import MarkdownReportRenderer
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.cost import TokenUsage
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.run import RunRecord
from app.schemas.state import ResearchRequest, ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall
from app.services.orchestrator import OrchestratorResult, ResearchOrchestrator
from app.services.run_service import RunService
from app.synthesis.mock_provider import MockSynthesisProvider
from app.synthesis.service import SynthesisService
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class InMemoryRepository:
    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.save_count = 0

    def save(self, run: RunRecord) -> None:
        self.save_count += 1
        self.runs[run.run_id] = run

    def save_checkpoint(self, run: RunRecord) -> None:
        self.save_count += 1
        self.runs[run.run_id] = run

    def get(self, run_id: str) -> RunRecord:
        return self.runs[run_id]

    def exists(self, run_id: str) -> bool:
        return run_id in self.runs

    def list_runs(self) -> list[RunRecord]:
        return list(self.runs.values())

    def acquire_resume_lock(self, run_id: str) -> None:
        pass

    def release_resume_lock(self, run_id: str) -> None:
        pass


class FailingRepository:
    def save(self, run: RunRecord) -> None:
        raise RepositoryError("disk full / db failure")

    def save_checkpoint(self, run: RunRecord) -> None:
        raise RepositoryError("disk full / db failure")

    def get(self, run_id: str) -> RunRecord:
        raise RepositoryError("disk failure")

    def exists(self, run_id: str) -> bool:
        return False

    def list_runs(self) -> list[RunRecord]:
        return []

    def acquire_resume_lock(self, run_id: str) -> None:
        pass

    def release_resume_lock(self, run_id: str) -> None:
        pass


class RecordingRenderer(MarkdownReportRenderer):
    def __init__(self, should_fail: bool = False) -> None:
        self.received_records: list[RunRecord] = []
        self.should_fail = should_fail

    def render(self, run: RunRecord) -> str:
        self.received_records.append(run)
        if self.should_fail:
            raise RuntimeError("renderer crash")
        return super().render(run)


def create_evidence_state(
    run_id: str = "run_test_1",
    question: str = "What is the answer?",
) -> tuple[Source, Evidence, Claim, FinalAnswer]:
    timestamp = datetime.now(UTC)
    source = Source(
        source_id="src_1",
        url="https://example.com/source",
        title="Source 1",
        tool_call_id="call_1",
        timestamp=timestamp,
    )
    evidence = Evidence(
        evidence_id="ev_1",
        content="Evidence content",
        source_id="src_1",
        tool_call_id="call_1",
        timestamp=timestamp,
    )
    claim = Claim(claim_id="cl_1", text="Claim text", evidence_ids=["ev_1"])
    answer = FinalAnswer(
        answer_text="Claim text [1].",
        claims=[claim],
        citations=[Citation(claim="Claim text", evidence_ids=["ev_1"])],
        is_complete=True,
    )
    return source, evidence, claim, answer


class MockAgent(Agent):
    def __init__(
        self,
        termination_reason: TerminationReason,
        sources: list[Source] | None = None,
        evidence: list[Evidence] | None = None,
        claims: list[Claim] | None = None,
        errors: list[str] | None = None,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self.termination_reason = termination_reason
        self.sources = sources or []
        self.evidence = evidence or []
        self.claims = claims or []
        self.errors = errors or []
        self._policy = policy or ExecutionPolicy()
        self.last_state: ResearchState | None = None
        self.calls: list[ResearchRequest] = []

    def effective_policy(self, request: ResearchRequest) -> ExecutionPolicy:
        return self._policy

    async def run(
        self,
        request: ResearchRequest,
        initial_state: ResearchState | None = None,
        checkpoint_callback: object = None,
    ) -> ResearchResult:
        self.calls.append(request)
        state = ResearchState(
            research_id="run_mock_123",
            original_question=request.question,
            created_at=datetime.now(UTC),
            sources=self.sources,
            evidence=self.evidence,
            claims=self.claims,
            errors=self.errors,
            termination_reason=self.termination_reason,
            current_step=1,
        )
        self.last_state = state
        return ResearchResult.from_state(state)


class TestResearchOrchestrator:
    # 1. Successful research (full flow)
    async def test_successful_research(self) -> None:
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )
        synthesis = SynthesisService(MockSynthesisProvider([answer]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="What is the answer?"))

        assert isinstance(out, OrchestratorResult)
        assert out.success is True
        assert out.termination_reason == TerminationReason.FINISHED
        assert out.run_id == "run_mock_123"
        assert out.result.final_answer == answer
        assert out.record.research_result == out.result
        assert "# Research Report" in out.report_markdown
        assert repo.save_count == 1
        assert repo.runs[out.run_id] == out.record

    # 2. Agent finished without synthesis
    async def test_agent_finished_without_synthesis(self) -> None:
        agent = MockAgent(termination_reason=TerminationReason.FINISHED)
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Quick query"))

        assert out.success is True
        assert out.termination_reason == TerminationReason.FINISHED
        assert out.result.final_answer is None
        assert repo.save_count == 1

    # 3. Agent requests synthesis
    async def test_agent_requests_synthesis(self) -> None:
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )
        mock_provider = MockSynthesisProvider([answer])
        synthesis = SynthesisService(mock_provider)
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Question"))

        assert len(mock_provider.calls) == 1
        assert mock_provider.calls[0]["question"] == "Question"
        assert out.termination_reason == TerminationReason.FINISHED
        assert out.result.final_answer == answer

    # 4. Synthesis succeeds
    async def test_synthesis_succeeds_updates_state_and_claims(self) -> None:
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )
        synthesis = SynthesisService(MockSynthesisProvider([answer]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Question"))

        assert out.record.research_state.final_answer == answer
        assert len(out.record.research_state.claims) == 1
        assert out.record.research_state.claims[0].claim_id == claim.claim_id

    # 5. Insufficient evidence
    async def test_insufficient_evidence_returns_incomplete_status(self) -> None:
        incomplete_answer = FinalAnswer(
            answer_text="Not enough evidence.",
            claims=[],
            citations=[],
            is_complete=False,
            caveats="No verified sources found.",
        )
        agent = MockAgent(termination_reason=TerminationReason.SYNTHESIS_REQUESTED)
        synthesis = SynthesisService(MockSynthesisProvider([incomplete_answer]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Obscure query"))

        assert out.success is False
        assert out.result.final_answer is not None
        assert out.result.final_answer.is_complete is False
        assert "Caveats: No verified sources found." in out.report_markdown

    # 6. Agent max_steps
    async def test_agent_max_steps_halts_without_synthesis(self) -> None:
        agent = MockAgent(termination_reason=TerminationReason.MAX_STEPS)
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Long task"))

        assert out.success is False
        assert out.termination_reason == TerminationReason.MAX_STEPS
        assert out.result.final_answer is None
        assert repo.save_count == 1

    # 7. Agent max_tool_calls
    async def test_agent_max_tool_calls_halts_without_synthesis(self) -> None:
        agent = MockAgent(termination_reason=TerminationReason.MAX_TOOL_CALLS)
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Spamming tool"))

        assert out.success is False
        assert out.termination_reason == TerminationReason.MAX_TOOL_CALLS
        assert repo.save_count == 1

    # 8. Loop detected
    async def test_loop_detected_halts_without_synthesis(self) -> None:
        agent = MockAgent(
            termination_reason=TerminationReason.LOOP_DETECTED,
            errors=["Repeated tool call detected: web_search"],
        )
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Loop query"))

        assert out.success is False
        assert out.termination_reason == TerminationReason.LOOP_DETECTED
        assert "Repeated tool call" in out.errors[0]

    # 9. Timeout
    async def test_agent_timeout_halts_without_synthesis(self) -> None:
        agent = MockAgent(
            termination_reason=TerminationReason.TIMEOUT,
            errors=["Agent run timed out after 30s"],
        )
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Slow query"))

        assert out.success is False
        assert out.termination_reason == TerminationReason.TIMEOUT
        assert "timed out" in out.errors[0]

    # 10. Provider error
    async def test_agent_provider_error_halts_without_synthesis(self) -> None:
        agent = MockAgent(
            termination_reason=TerminationReason.PROVIDER_ERROR,
            errors=["ProviderError: OpenAI request failed"],
        )
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="API down"))

        assert out.success is False
        assert out.termination_reason == TerminationReason.PROVIDER_ERROR
        assert repo.save_count == 1

    # 11. Tool error
    async def test_agent_tool_error_halts_without_synthesis(self) -> None:
        agent = MockAgent(
            termination_reason=TerminationReason.TOOL_ERROR,
            errors=["Tool execution failed: registry failure"],
        )
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Crash tool"))

        assert out.success is False
        assert out.termination_reason == TerminationReason.TOOL_ERROR

    # 12. Synthesis error
    async def test_synthesis_error_recorded_and_persisted(self) -> None:
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )

        class FailingSynthesisProvider:
            async def generate_answer(self, *args: Any, **kwargs: Any) -> tuple[FinalAnswer, TokenUsage | None]:
                raise ProviderError("synthesis model timed out")

        synthesis = SynthesisService(FailingSynthesisProvider())
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Synthesis crash"))

        assert out.success is False
        assert out.termination_reason == TerminationReason.INVALID_OUTPUT
        assert any("Synthesis failed" in err for err in out.errors)
        assert out.result.final_answer is None
        assert repo.save_count == 1

    # 13. Persistence error
    async def test_persistence_error_is_propagated_and_not_swallowed(self) -> None:
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )
        synthesis = SynthesisService(MockSynthesisProvider([answer]))
        run_service = RunService(FailingRepository())
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        with pytest.raises(RepositoryError, match="disk full"):
            await orchestrator.run(ResearchRequest(question="Question"))

    # 14. Report rendering error
    async def test_report_rendering_error_is_propagated_with_clear_message(self) -> None:
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )
        synthesis = SynthesisService(MockSynthesisProvider([answer]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer(should_fail=True)

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        with pytest.raises(ReportRenderingError, match="completed but report rendering failed"):
            await orchestrator.run(ResearchRequest(question="Question"))
        # Run was saved prior to renderer crash
        assert repo.save_count == 1

    # 15. Correct run_id propagation
    async def test_correct_run_id_propagation(self) -> None:
        agent = MockAgent(termination_reason=TerminationReason.FINISHED)
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="ID test"))

        assert out.run_id == "run_mock_123"
        assert out.result.research_id == out.run_id
        assert out.record.run_id == out.run_id
        assert out.record.research_state.research_id == out.run_id
        assert out.record.research_result is not None
        assert out.record.research_result.research_id == out.run_id

    # 16. State/Result consistency
    async def test_state_and_result_consistency(self) -> None:
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )
        synthesis = SynthesisService(MockSynthesisProvider([answer]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Consistency test"))

        state = out.record.research_state
        result = out.result
        assert result.research_id == state.research_id
        assert result.question == state.original_question
        assert result.final_answer == state.final_answer
        assert result.termination_reason == state.termination_reason
        assert len(result.sources) == len(state.sources)
        assert len(result.evidence) == len(state.evidence)

    # 17. Persistence called exactly once
    async def test_persistence_called_exactly_once(self) -> None:
        agent = MockAgent(termination_reason=TerminationReason.FINISHED)
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        await orchestrator.run(ResearchRequest(question="Count check"))

        assert repo.save_count == 1

    # 18. Renderer receives correct data
    async def test_renderer_receives_correct_data(self) -> None:
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )
        synthesis = SynthesisService(MockSynthesisProvider([answer]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Renderer test"))

        assert len(renderer.received_records) == 1
        assert renderer.received_records[0] == out.record
        assert renderer.received_records[0].run_id == out.run_id
        assert renderer.received_records[0].research_state.final_answer == answer

    async def test_agent_missing_state_raises_runtime_error(self) -> None:
        class AgentWithoutState:
            def __init__(self) -> None:
                self.last_state = None

            async def run(self, request: ResearchRequest) -> None:
                self.last_state = None

        agent = AgentWithoutState()
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="Agent finished without recording last_state"):
            await orchestrator.run(ResearchRequest(question="Missing state"))

    async def test_policy_fallback_resolution(self) -> None:
        class AgentWithPrivatePolicyOnly:
            def __init__(self, policy: ExecutionPolicy) -> None:
                self._policy = policy
                self.last_state: ResearchState | None = None

            async def run(self, request: ResearchRequest) -> None:
                self.last_state = ResearchState(
                    research_id="req-1",
                    original_question=request.question,
                    created_at=datetime.now(UTC),
                    termination_reason=TerminationReason.FINISHED,
                )

        custom_policy = ExecutionPolicy(max_steps=42)
        agent = AgentWithPrivatePolicyOnly(custom_policy)
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)  # type: ignore[arg-type]
        out = await orchestrator.run(ResearchRequest(question="Test policy"))
        assert out.record.execution_policy.max_steps == 42

    async def test_default_policy_fallback_resolution(self) -> None:
        class AgentWithoutAnyPolicy:
            def __init__(self) -> None:
                self.last_state: ResearchState | None = None

            async def run(self, request: ResearchRequest) -> None:
                self.last_state = ResearchState(
                    research_id="req-2",
                    original_question=request.question,
                    created_at=datetime.now(UTC),
                    termination_reason=TerminationReason.FINISHED,
                )

        agent = AgentWithoutAnyPolicy()
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)  # type: ignore[arg-type]
        out = await orchestrator.run(ResearchRequest(question="Default policy"))
        assert out.record.execution_policy.max_steps == 8  # default ExecutionPolicy


def _field(record: logging.LogRecord, name: str) -> Any:
    """LogRecord attributes attached via `extra=` are dynamic -- mypy's
    stub for LogRecord doesn't know them, so tests read them through this
    instead of `record.<name>` to stay type-clean."""
    return getattr(record, name)


class TestOrchestratorLogging:
    async def test_emits_synthesis_started_completed_and_run_finished(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="ai_research_agent")
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )
        synthesis = SynthesisService(MockSynthesisProvider([answer]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="What is the answer?"))

        events = [_field(record, "event") for record in caplog.records]
        assert events == ["synthesis_started", "synthesis_completed", "run_finished"]
        assert all(_field(record, "run_id") == out.run_id for record in caplog.records)

        run_finished = caplog.records[-1]
        assert _field(run_finished, "termination_reason") == "finished"
        assert _field(run_finished, "success") is True

    async def test_emits_synthesis_failed_and_run_finished_with_success_false(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="ai_research_agent")
        source, evidence, claim, answer = create_evidence_state()
        agent = MockAgent(
            termination_reason=TerminationReason.SYNTHESIS_REQUESTED,
            sources=[source],
            evidence=[evidence],
        )

        class FailingSynthesisProvider:
            async def generate_answer(self, *args: Any, **kwargs: Any) -> tuple[FinalAnswer, TokenUsage | None]:
                raise ProviderError("synthesis model timed out")

        synthesis = SynthesisService(FailingSynthesisProvider())
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Synthesis crash"))

        events = [_field(record, "event") for record in caplog.records]
        assert events == ["synthesis_started", "synthesis_failed", "run_finished"]

        synthesis_failed = caplog.records[1]
        assert synthesis_failed.levelno == logging.WARNING
        assert "ProviderError" in _field(synthesis_failed, "error")

        run_finished = caplog.records[-1]
        assert _field(run_finished, "success") is False
        assert out.success is False

    async def test_no_synthesis_events_when_synthesis_not_requested(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="ai_research_agent")
        agent = MockAgent(termination_reason=TerminationReason.FINISHED)
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        await orchestrator.run(ResearchRequest(question="Quick query"))

        events = [_field(record, "event") for record in caplog.records]
        assert events == ["run_finished"]


class TestOrchestratorCheckpointing:
    """Fase 11E: `checkpoints_enabled` gates whether the Agent is wired
    with a checkpoint callback. Uses the real Agent (not MockAgent, which
    doesn't have a loop to checkpoint) driven by a scripted MockLLMProvider.
    """

    async def test_disabled_by_default_persists_exactly_once(self) -> None:
        provider = MockLLMProvider([{"action": "finish"}])
        agent = Agent(provider, ToolRegistry([]))
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer)
        out = await orchestrator.run(ResearchRequest(question="Quick query"))

        assert repo.save_count == 1
        assert out.record.research_result is not None

    async def test_enabled_checkpoints_each_step_then_finalizes(self) -> None:
        provider = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c1", tool_name="calculator", arguments={"expression": "1+1"})]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        agent = Agent(provider, ToolRegistry([CalculatorTool()]))  # type: ignore[list-item]
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer, checkpoints_enabled=True)
        out = await orchestrator.run(ResearchRequest(question="Checkpointed query"))

        # step 1 (continues) + step 2 (FINISH, agent-terminal checkpoint)
        # + the final orchestrator persistence == 3 writes, all to the
        # same run_id/run.json (no separate checkpoint file).
        assert repo.save_count == 3
        assert out.record.research_result is not None
        assert out.success is True

    async def test_intermediate_checkpoints_never_carry_a_research_result(self) -> None:
        provider = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c1", tool_name="calculator", arguments={"expression": "1+1"})]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        agent = Agent(provider, ToolRegistry([CalculatorTool()]))  # type: ignore[list-item]
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        checkpoints_seen: list[bool] = []
        original_save_checkpoint = repo.save_checkpoint

        def spying_save_checkpoint(run: RunRecord) -> None:
            checkpoints_seen.append(run.research_result is not None)
            original_save_checkpoint(run)

        repo.save_checkpoint = spying_save_checkpoint  # type: ignore[method-assign]

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer, checkpoints_enabled=True)
        await orchestrator.run(ResearchRequest(question="Checkpointed query"))

        # Every write except the last (the final RunRecord) is a
        # checkpoint with no research_result yet.
        assert checkpoints_seen[:-1] == [False] * (len(checkpoints_seen) - 1)
        assert checkpoints_seen[-1] is True

    async def test_checkpoint_uses_the_same_resolved_policy_as_the_final_record(self) -> None:
        provider = MockLLMProvider(
            [
                LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c1", tool_name="calculator", arguments={"expression": "1+1"})]),
                LLMDecision(action=DecisionAction.FINISH),
            ]
        )
        agent = Agent(provider, ToolRegistry([CalculatorTool()]), policy=ExecutionPolicy(max_steps=9))  # type: ignore[list-item]
        synthesis = SynthesisService(MockSynthesisProvider([]))
        repo = InMemoryRepository()
        run_service = RunService(repo)
        renderer = RecordingRenderer()

        seen_max_steps: list[int] = []
        original_save_checkpoint = repo.save_checkpoint

        def spying_save_checkpoint(run: RunRecord) -> None:
            seen_max_steps.append(run.execution_policy.max_steps)
            original_save_checkpoint(run)

        repo.save_checkpoint = spying_save_checkpoint  # type: ignore[method-assign]

        orchestrator = ResearchOrchestrator(agent, synthesis, run_service, renderer, checkpoints_enabled=True)
        out = await orchestrator.run(ResearchRequest(question="Checkpointed query"))

        assert seen_max_steps and all(value == 9 for value in seen_max_steps)
        assert out.record.execution_policy.max_steps == 9


