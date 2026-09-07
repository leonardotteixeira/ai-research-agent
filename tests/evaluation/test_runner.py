import pytest

from app.evaluation.runner import (
    EvaluationRunner,
    MissingRunRecordError,
    UnknownCaseRunRecordError,
)
from app.evaluation.schemas import (
    EvaluationCase,
    EvaluationDataset,
    MetricThreshold,
    ThresholdOperator,
)
from app.schemas.state import ResearchResult, TerminationReason
from tests.unit.test_run_record import build_record


def dataset() -> EvaluationDataset:
    return EvaluationDataset(
        id="research_quality",
        version="1",
        name="Research Quality",
        cases=[
            EvaluationCase(id="case_1", question="What was observed?"),
            EvaluationCase(id="case_2", question="What else was observed?"),
        ],
    )


def test_runner_evaluates_records_by_case_id_not_position():
    first = build_record()
    second = build_record()
    second.run_id = "run_2"
    second.research_state.research_id = "run_2"
    second.research_result = ResearchResult.from_state(second.research_state).model_copy(
        update={"research_id": "run_2"}
    )

    result = EvaluationRunner().evaluate_records(
        dataset(),
        {"case_2": second, "case_1": first},
        evaluation_id="research_quality_v1",
        candidate_id="candidate",
        metadata={"source": "test"},
    )

    assert result.overall_passed is True
    assert [case.case_id for case in result.case_results] == ["case_1", "case_2"]
    assert result.metadata == {"source": "test"}
    assert len(result.metrics) == 11


def test_runner_rejects_missing_run_record():
    with pytest.raises(MissingRunRecordError):
        EvaluationRunner().evaluate_records(
            dataset(),
            {"case_1": build_record()},
            evaluation_id="research_quality_v1",
            candidate_id="candidate",
        )


def test_runner_rejects_unknown_case_key():
    with pytest.raises(UnknownCaseRunRecordError):
        EvaluationRunner().evaluate_records(
            dataset(),
            {
                "case_1": build_record(),
                "case_2": build_record(),
                "case_3": build_record(),
            },
            evaluation_id="research_quality_v1",
            candidate_id="candidate",
        )


def test_runner_overall_fails_when_case_fails_or_threshold_fails():
    data = EvaluationDataset(
        id="research_quality",
        version="1",
        name="Research Quality",
        cases=[EvaluationCase(id="case_1", question="What was observed?")],
    )
    record = build_record()
    record.research_state.termination_reason = TerminationReason.TIMEOUT
    record.research_state.errors = ["timed out"]
    record.research_result = ResearchResult.from_state(record.research_state)

    result = EvaluationRunner().evaluate_records(
        data,
        {"case_1": record},
        evaluation_id="research_quality_v1",
        candidate_id="candidate",
        thresholds=[MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, value=1.0)],
    )

    assert result.overall_passed is False
    assert result.case_results[0].passed is False
