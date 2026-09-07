"""Permanent guard against silent JSON round-trip breakage for the
evaluation schemas: model_dump(mode="json") -> model_validate() must
reproduce an equal model. Verified manually during the Phase 10 audit;
this makes that guarantee a regression test instead of a one-off check.
"""

from app.evaluation.comparison import EvaluationComparator
from app.evaluation.datasets import EvaluationDatasetLoader
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import (
    EvaluationCaseResult,
    EvaluationComparison,
    EvaluationDataset,
    MetricResult,
    MetricThreshold,
    ThresholdOperator,
)
from tests.unit.test_run_record import build_record


def _round_trips(model):
    dumped = model.model_dump(mode="json")
    restored = type(model).model_validate(dumped)
    assert restored == model
    return dumped


def test_evaluation_dataset_round_trips():
    dataset = EvaluationDatasetLoader().load_json("evals/datasets/research_quality_v1.json")
    _round_trips(dataset)


def test_metric_threshold_round_trips_for_both_operators():
    _round_trips(MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, value=0.9))
    _round_trips(MetricThreshold(metric_name="error_rate", operator=ThresholdOperator.LTE, value=0.05))
    _round_trips(
        MetricThreshold(
            metric_name="error_rate", operator=ThresholdOperator.LTE, value=0.05, delta_value=0.02
        )
    )


def test_metric_result_round_trips_available_and_unavailable():
    _round_trips(MetricResult(name="pass_rate", value=0.9, unit="ratio", available=True, passed=True))
    _round_trips(MetricResult(name="pass_rate", value=None, unit="ratio", available=False, passed=None))


def test_evaluation_run_result_and_comparison_round_trip():
    dataset = EvaluationDatasetLoader().load_json("evals/datasets/research_quality_v1.json")
    thresholds = [
        MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, value=0.5),
        MetricThreshold(metric_name="error_rate", operator=ThresholdOperator.LTE, value=0.5),
    ]
    result = EvaluationRunner().evaluate_records(
        EvaluationDataset(
            id=dataset.id,
            version=dataset.version,
            name=dataset.name,
            cases=[dataset.cases[0].model_copy(update={"id": "case_1"})],
        ),
        {"case_1": build_record()},
        evaluation_id="e1",
        candidate_id="c1",
        thresholds=thresholds,
    )
    _round_trips(result)
    assert isinstance(result.case_results[0].evaluator_results[0], EvaluationCaseResult)
    _round_trips(result.case_results[0].evaluator_results[0])

    comparison = EvaluationComparator().compare(result, result, thresholds)
    dumped = _round_trips(comparison)
    assert isinstance(EvaluationComparison.model_validate(dumped), EvaluationComparison)
