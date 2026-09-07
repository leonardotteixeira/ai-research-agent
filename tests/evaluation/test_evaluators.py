from datetime import UTC, datetime

from app.evaluation.evaluators import (
    AnswerEvaluator,
    CitationEvaluator,
    ErrorEvaluator,
    EvaluationCase,
    EvidenceEvaluator,
    ExecutionEvaluator,
    GroundingEvaluator,
    PolicyEvaluator,
)
from app.evaluation.schemas import EvaluationExpected
from app.policies.execution import ExecutionPolicy
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.evidence import Evidence, Source
from app.schemas.run import RunRecord
from app.schemas.state import ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall
from tests.unit.test_run_record import build_record


def case(expected: EvaluationExpected | None = None) -> EvaluationCase:
    return EvaluationCase(id="case_1", question="What was observed?", expected=expected)


def test_answer_evaluator_passes_complete_answer_with_expected_text():
    result = AnswerEvaluator().evaluate(
        case(EvaluationExpected(answer_contains=["result was observed"])),
        build_record(),
    )

    assert result.passed is True
    assert result.score == 1.0


def test_answer_evaluator_fails_incomplete_answer():
    record = build_record()
    incomplete = FinalAnswer(answer_text="Not enough evidence.", is_complete=False)
    record.research_state.final_answer = incomplete
    record.research_result = record.research_result.model_copy(update={"final_answer": incomplete})

    result = AnswerEvaluator().evaluate(case(), record)

    assert result.passed is False
    assert "final answer is marked incomplete" in result.violations


def test_answer_evaluator_fails_missing_expected_text():
    result = AnswerEvaluator().evaluate(
        case(EvaluationExpected(answer_contains=["absent phrase"])),
        build_record(),
    )

    assert result.passed is False
    assert "absent phrase" in result.violations[0]


def test_citation_evaluator_passes_valid_citations():
    result = CitationEvaluator().evaluate(case(), build_record())

    assert result.passed is True
    assert result.details["citation_count"] == 1


def test_citation_evaluator_fails_unknown_source_id():
    record = build_record()
    answer = record.research_state.final_answer.model_copy(
        update={"citations": [Citation(claim="The result was observed.", source_ids=["missing"])]}
    )
    record.research_state.final_answer = answer
    record.research_result = record.research_result.model_copy(update={"final_answer": answer})

    result = CitationEvaluator().evaluate(case(), record)

    assert result.passed is False
    assert "unknown source" in result.violations[0]


def test_citation_evaluator_fails_unknown_evidence_id():
    record = build_record()
    answer = record.research_state.final_answer.model_copy(
        update={"citations": [Citation(claim="The result was observed.", evidence_ids=["missing"])]}
    )
    record.research_state.final_answer = answer
    record.research_result = record.research_result.model_copy(update={"final_answer": answer})

    result = CitationEvaluator().evaluate(case(), record)

    assert result.passed is False
    assert "unknown evidence" in result.violations[0]


def test_evidence_evaluator_checks_expected_sources_and_evidence():
    result = EvidenceEvaluator().evaluate(
        case(
            EvaluationExpected(
                source_urls=["https://example.com/report"],
                evidence_contains=["observed result"],
                claims=["The result was observed"],
            )
        ),
        build_record(),
    )

    assert result.passed is True
    assert result.details["claims_with_valid_evidence"] == 1


def test_evidence_evaluator_fails_missing_expected_source_and_evidence():
    result = EvidenceEvaluator().evaluate(
        case(
            EvaluationExpected(
                source_urls=["https://example.com/missing"],
                evidence_contains=["missing snippet"],
                claims=["missing claim"],
            )
        ),
        build_record(),
    )

    assert result.passed is False
    assert len(result.violations) == 3


def test_evidence_evaluator_detects_invalid_graph_from_persisted_artifact():
    timestamp = datetime.now(UTC)
    source = Source(source_id="src_1", url="https://example.com", tool_call_id="c1", timestamp=timestamp)
    evidence = Evidence(
        evidence_id="ev_1",
        content="text",
        source_id="missing",
        tool_call_id="c1",
        timestamp=timestamp,
    )
    state = ResearchState.model_construct(
        research_id="bad",
        original_question="Q",
        created_at=timestamp,
        current_step=1,
        sources=[source],
        evidence=[evidence],
        claims=[],
        errors=[],
        elapsed_seconds=0.1,
        final_answer=None,
        termination_reason=TerminationReason.FINISHED,
    )
    record = RunRecord.model_construct(
        run_id="bad",
        created_at=timestamp,
        schema_version="1.0",
        question="Q",
        execution_policy=ExecutionPolicy(),
        research_state=state,
        research_result=ResearchResult.from_state(state),
    )

    result = EvidenceEvaluator().evaluate(case(), record)

    assert result.passed is False
    assert "unknown sources" in result.violations[0]


def test_execution_evaluator_passes_finished_within_limits():
    result = ExecutionEvaluator().evaluate(case(), build_record())

    assert result.passed is True
    assert result.details["termination_reason"] == "finished"


def test_execution_evaluator_fails_timeout_loop_and_policy_block():
    for reason in (
        TerminationReason.TIMEOUT,
        TerminationReason.LOOP_DETECTED,
        TerminationReason.POLICY_BLOCKED,
    ):
        record = build_record()
        record.research_state.termination_reason = reason
        record.research_result = ResearchResult.from_state(record.research_state)

        result = ExecutionEvaluator().evaluate(case(), record)

        assert result.passed is False
        assert result.violations


def test_error_evaluator_flags_errors_and_failure_reason():
    record = build_record()
    record.research_state.errors = ["ToolExecutionError: boom"]
    record.research_state.termination_reason = TerminationReason.TOOL_ERROR
    record.research_result = ResearchResult.from_state(record.research_state)

    result = ErrorEvaluator().evaluate(case(), record)

    assert result.passed is False
    assert result.details["error_count"] == 1


def test_policy_evaluator_detects_forbidden_tool_and_repetition():
    record = build_record()
    repeated = ToolCall(call_id="call_2", tool_name="web_search", arguments={"query": "observed result"})
    forbidden = ToolCall(call_id="call_3", tool_name="fetch_url", arguments={"url": "https://example.com"})
    record.research_state.tool_calls = [record.research_state.tool_calls[0], repeated, forbidden]
    record.execution_policy = ExecutionPolicy(
        max_steps=8,
        max_tool_calls=12,
        max_same_tool_calls=1,
        allowed_tools={"web_search"},
    )

    result = PolicyEvaluator().evaluate(case(), record)

    assert result.passed is False
    assert any("outside allowed_tools" in violation for violation in result.violations)
    assert any("repeated tool calls" in violation for violation in result.violations)


def test_grounding_evaluator_reports_structural_grounding_only():
    result = GroundingEvaluator().evaluate(case(), build_record())

    assert result.passed is True
    assert result.details["grounded_claims"] == 1
    assert "does not prove semantic factuality" in result.details["limitation"]
