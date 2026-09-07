from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.policies.execution import ExecutionPolicy
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.decision import DecisionAction, LLMDecision, Observation, ResearchPlan
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.run import RunRecord
from app.schemas.state import ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall, ToolResult

TIMESTAMP = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)


def build_state() -> ResearchState:
    source = Source(
        source_id="src_1",
        url="https://example.com/report",
        title="Example report",
        source_type="search_result",
        retrieved_at=TIMESTAMP,
        tool_call_id="call_1",
        timestamp=TIMESTAMP,
    )
    evidence = Evidence(
        evidence_id="ev_1",
        content="The report contains the observed result.",
        source_id="src_1",
        tool_call_id="call_1",
        timestamp=TIMESTAMP,
    )
    claim = Claim(claim_id="claim_1", text="The result was observed.", evidence_ids=["ev_1"])
    answer = FinalAnswer(
        answer_text="The result was observed [1].",
        claims=[claim],
        citations=[Citation(claim=claim.text, evidence_ids=["ev_1"])],
        is_complete=True,
    )
    return ResearchState(
        research_id="run_1",
        original_question="What was observed?",
        created_at=TIMESTAMP,
        current_step=2,
        current_plan=ResearchPlan(summary="Review the report", generated_at_step=1),
        decisions=[
            LLMDecision(
                action=DecisionAction.TOOL_CALL,
                rationale="Search the report.",
                tool_calls=[
                    ToolCall(
                        call_id="call_1",
                        tool_name="web_search",
                        arguments={"query": "observed result"},
                    )
                ],
            ),
            LLMDecision(action=DecisionAction.FINISH, rationale="Enough evidence."),
        ],
        tool_calls=[
            ToolCall(
                call_id="call_1",
                tool_name="web_search",
                arguments={"query": "observed result"},
            )
        ],
        tool_results=[
            ToolResult(
                call_id="call_1",
                tool_name="web_search",
                success=True,
                output=[
                    {
                        "url": "https://example.com/report",
                        "title": "Example report",
                        "snippet": "The report contains the observed result.",
                    }
                ],
                latency_ms=12.5,
            )
        ],
        observations=[
            Observation(
                step=1,
                summary="Search completed.",
                tool_results=[
                    ToolResult(
                        call_id="call_1",
                        tool_name="web_search",
                        success=True,
                        output=[{"url": "https://example.com/report", "snippet": "The report contains the observed result."}],
                    )
                ],
                new_evidence_ids=["ev_1"],
            )
        ],
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        errors=[],
        elapsed_seconds=1.25,
        final_answer=answer,
        termination_reason=TerminationReason.FINISHED,
    )


def build_record(result: ResearchResult | None = None) -> RunRecord:
    state = build_state()
    return RunRecord(
        run_id=state.research_id,
        created_at=TIMESTAMP,
        question=state.original_question,
        execution_policy=ExecutionPolicy(
            max_steps=8,
            max_tool_calls=12,
            max_same_tool_calls=3,
            allowed_tools={"web_search", "fetch_url"},
            global_timeout_seconds=120,
        ),
        research_state=state,
        research_result=result or ResearchResult.from_state(state),
    )


class TestRunRecord:
    def test_creates_valid_record_with_complete_execution(self):
        record = build_record()

        assert record.run_id == "run_1"
        assert record.schema_version == "1.0"
        assert record.research_result.termination_reason == TerminationReason.FINISHED

    def test_required_fields_are_validated(self):
        with pytest.raises(ValidationError):
            RunRecord.model_validate({"run_id": "run_1"})

    @pytest.mark.parametrize(
        "field, value, message",
        [
            ("run_id", "other", "run_id must match"),
            ("question", "other question", "question must match"),
        ],
    )
    def test_rejects_inconsistent_identity(self, field, value, message):
        payload = build_record().model_dump()
        payload[field] = value

        with pytest.raises(ValidationError, match=message):
            RunRecord.model_validate(payload)

    @pytest.mark.parametrize(
        "field, value, message",
        [
            ("research_id", "other", "research_result.research_id must match"),
            ("question", "other question", "research_result.question must match"),
        ],
    )
    def test_rejects_inconsistent_result(self, field, value, message):
        payload = build_record().model_dump()
        payload["research_result"][field] = value

        with pytest.raises(ValidationError, match=message):
            RunRecord.model_validate(payload)

    def test_json_round_trip_preserves_nested_execution(self):
        original = build_record()

        restored = RunRecord.model_validate_json(original.model_dump_json())

        assert restored == original
        assert restored.created_at == TIMESTAMP
        assert restored.execution_policy.allowed_tools == {"web_search", "fetch_url"}
        assert restored.research_state.decisions[0].tool_calls[0].arguments == {"query": "observed result"}
        assert restored.research_state.tool_results[0].output[0]["url"] == "https://example.com/report"
        assert restored.research_state.observations[0].new_evidence_ids == ["ev_1"]

    def test_preserves_claim_evidence_source_answer_and_citation_links(self):
        restored = RunRecord.model_validate_json(build_record().model_dump_json())
        state = restored.research_state
        answer = state.final_answer

        assert state.claims[0].evidence_ids == ["ev_1"]
        assert state.evidence[0].source_id == state.sources[0].source_id
        assert answer is not None
        assert answer.citations[0].evidence_ids == ["ev_1"]

    def test_preserves_replay_information_without_runtime_objects(self):
        record = build_record()
        payload = record.model_dump(mode="json")

        assert [item.call_id for item in record.research_state.tool_calls] == ["call_1"]
        assert payload["research_state"]["tool_results"][0]["output"][0]["title"] == "Example report"
        assert "AsyncClient" not in record.model_dump_json()

    def test_persisted_json_does_not_contain_credentials_or_environment(self):
        serialized = build_record().model_dump_json().lower()

        for secret_marker in (
            "sk-test-secret",
            "authorization: bearer",
            "password=",
            "super-secret-credential",
            "complete_environment_dump",
        ):
            assert secret_marker not in serialized

    def test_record_can_exist_before_result_is_available(self):
        record = build_record(result=None)
        record.research_result = None

        restored = RunRecord.model_validate_json(record.model_dump_json())

        assert restored.research_result is None

    def test_old_run_record_missing_token_usage_actual_field_still_loads(self):
        """Backward compatibility (Fase 11B): a RunRecord persisted before
        token usage instrumentation existed never wrote `actual` at all
        (Pydantic didn't serialize a field that was never touched
        differently from its old True default in older payload shapes we
        might encounter). Loading such a payload today must not fail, and
        must land on the corrected default (actual=False) rather than the
        old, semantically-wrong True default -- there was never a real
        measurement in that old record, so it must not claim there was.
        """
        payload = build_record().model_dump(mode="json")
        del payload["research_state"]["token_usage"]["actual"]

        restored = RunRecord.model_validate(payload)

        usage = restored.research_state.token_usage
        assert usage.total_tokens is None
        assert usage.actual is False