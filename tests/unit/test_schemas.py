from datetime import UTC, datetime

import pytest

from app.schemas.answer import Citation, FinalAnswer
from app.schemas.cost import TokenUsage
from app.schemas.decision import LLMDecision, Observation, ResearchPlan
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.state import ResearchRequest, ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall, ToolResult


class TestToolCallAndResult:
    def test_tool_call_defaults_to_empty_arguments(self):
        call = ToolCall(call_id="c1", tool_name="calculator")
        assert call.arguments == {}

    def test_tool_result_success_has_no_error(self):
        result = ToolResult(call_id="c1", tool_name="calculator", success=True, output=4)
        assert result.error is None

    def test_tool_result_failure_carries_error(self):
        result = ToolResult(call_id="c1", tool_name="calculator", success=False, error="division by zero")
        assert result.success is False
        assert result.output is None


class TestTokenUsageAdd:
    def test_add_sums_both_present(self):
        a = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, estimated_cost_usd=0.01)
        b = TokenUsage(input_tokens=20, output_tokens=10, total_tokens=30, estimated_cost_usd=0.02)
        total = a.add(b)
        assert total.input_tokens == 30
        assert total.output_tokens == 15
        assert total.total_tokens == 45
        assert total.estimated_cost_usd == pytest.approx(0.03)

    def test_add_both_none_stays_none(self):
        a = TokenUsage()
        b = TokenUsage()
        total = a.add(b)
        assert total.input_tokens is None
        assert total.estimated_cost_usd is None

    def test_add_one_side_none_treated_as_zero_not_dropped(self):
        a = TokenUsage(input_tokens=10)
        b = TokenUsage(input_tokens=None)
        total = a.add(b)
        assert total.input_tokens == 10

    def test_add_actual_is_and_of_both_sides(self):
        actual = TokenUsage(input_tokens=1, actual=True)
        estimated = TokenUsage(input_tokens=1, actual=False)
        assert actual.add(estimated).actual is False
        assert actual.add(actual).actual is True


class TestLLMDecision:
    def test_defaults_are_empty_and_not_finished(self):
        decision = LLMDecision()
        assert decision.tool_calls == []
        assert decision.should_finish is False

    def test_can_carry_multiple_tool_calls(self):
        decision = LLMDecision(
            tool_calls=[
                ToolCall(call_id="c1", tool_name="web_search", arguments={"query": "x"}),
                ToolCall(call_id="c2", tool_name="calculator", arguments={"expression": "1+1"}),
            ]
        )
        assert len(decision.tool_calls) == 2


class TestObservationAndPlan:
    def test_observation_defaults(self):
        obs = Observation(step=1, summary="found nothing useful")
        assert obs.tool_results == []
        assert obs.new_evidence_ids == []

    def test_plan_tracks_step(self):
        plan = ResearchPlan(summary="search for X", generated_at_step=2)
        assert plan.generated_at_step == 2


class TestEvidenceAndSource:
    def test_source_minimal(self):
        source = Source(source_id="s1", url="https://example.com", tool_call_id="c1", timestamp=datetime.now(UTC))
        assert source.url == "https://example.com"

    def test_evidence_links_to_one_source(self):
        evidence = Evidence(
            evidence_id="e1", content="X is true", source_id="s1", tool_call_id="c1", timestamp=datetime.now(UTC)
        )
        assert evidence.source_id == "s1"

    def test_claim_links_to_evidence(self):
        claim = Claim(claim_id="cl1", text="X is true", evidence_ids=["e1"])
        assert claim.evidence_ids == ["e1"]


class TestFinalAnswer:
    def test_incomplete_answer_can_have_no_citations(self):
        answer = FinalAnswer(answer_text="Not enough evidence to answer confidently.", is_complete=False)
        assert answer.citations == []

    def test_citation_links_claim_to_sources(self):
        answer = FinalAnswer(
            answer_text="X is true [1].",
            citations=[Citation(claim="X is true", source_ids=["s1"])],
            is_complete=True,
        )
        assert answer.citations[0].source_ids == ["s1"]


class TestResearchRequest:
    def test_minimal_request_has_no_overrides(self):
        request = ResearchRequest(question="What is X?")
        assert request.max_steps is None
        assert request.allowed_tools is None


class TestResearchStateSerialization:
    def test_round_trips_through_json(self):
        state = ResearchState(
            research_id="r1",
            original_question="What is X?",
            created_at=datetime.now(UTC),
            current_step=2,
            tool_calls=[ToolCall(call_id="c1", tool_name="web_search", arguments={"query": "X"})],
            sources=[Source(source_id="s1", url="https://example.com", tool_call_id="c1", timestamp=datetime.now(UTC))],
        )
        restored = ResearchState.model_validate_json(state.model_dump_json())
        assert restored.research_id == "r1"
        assert restored.current_step == 2
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].tool_name == "web_search"

    def test_default_state_has_no_final_answer_or_termination(self):
        state = ResearchState(research_id="r1", original_question="Q", created_at=datetime.now(UTC))
        assert state.final_answer is None
        assert state.termination_reason is None


class TestResearchResultFromState:
    def test_raises_when_state_has_no_termination_reason(self):
        state = ResearchState(research_id="r1", original_question="Q", created_at=datetime.now(UTC))
        with pytest.raises(ValueError, match="no termination_reason"):
            ResearchResult.from_state(state)

    def test_builds_from_completed_state(self):
        state = ResearchState(
            research_id="r1",
            original_question="Q",
            created_at=datetime.now(UTC),
            current_step=3,
            tool_calls=[ToolCall(call_id="c1", tool_name="calculator", arguments={})],
            elapsed_seconds=1.5,
            final_answer=FinalAnswer(answer_text="42", is_complete=True),
            termination_reason=TerminationReason.COMPLETED,
        )
        result = ResearchResult.from_state(state)
        assert result.steps == 3
        assert result.tool_call_count == 1
        assert result.termination_reason == TerminationReason.COMPLETED
        assert result.final_answer.answer_text == "42"

    def test_missing_elapsed_seconds_defaults_to_zero_not_none(self):
        state = ResearchState(
            research_id="r1",
            original_question="Q",
            created_at=datetime.now(UTC),
            termination_reason=TerminationReason.NO_EVIDENCE,
        )
        result = ResearchResult.from_state(state)
        assert result.elapsed_seconds == 0.0
