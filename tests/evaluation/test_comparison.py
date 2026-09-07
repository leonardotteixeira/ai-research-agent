from app.evaluation.comparison import EvaluationComparator
from app.evaluation.schemas import (
    EvaluationRunResult,
    MetricResult,
    MetricThreshold,
    ThresholdOperator,
)


def run_result(candidate_id: str, metrics: list[MetricResult], overall_passed: bool = True):
    return EvaluationRunResult(
        evaluation_id="research_quality_v1",
        dataset_id="research_quality",
        dataset_version="1",
        candidate_id=candidate_id,
        metrics=metrics,
        overall_passed=overall_passed,
    )


def metric(name: str, value: float | None, available: bool = True):
    return MetricResult(name=name, value=value, unit="ratio", available=available)


def test_comparison_identifies_improvement_regression_and_no_change():
    baseline = run_result(
        "baseline",
        [metric("pass_rate", 0.8), metric("error_rate", 0.1), metric("completion_rate", 1.0)],
    )
    candidate = run_result(
        "candidate",
        [metric("pass_rate", 0.9), metric("error_rate", 0.2), metric("completion_rate", 1.0)],
    )

    comparison = EvaluationComparator().compare(baseline, candidate)

    assert "pass_rate" in comparison.improvements
    assert "error_rate" in comparison.regressions
    unchanged = next(delta for delta in comparison.metric_deltas if delta.metric_name == "completion_rate")
    assert unchanged.delta == 0.0


def test_delta_threshold_allows_tolerance_at_boundary():
    baseline = run_result("baseline", [metric("pass_rate", 0.9)])
    candidate = run_result("candidate", [metric("pass_rate", 0.88)])

    comparison = EvaluationComparator().compare(
        baseline,
        candidate,
        [MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, delta_value=-0.02)],
    )

    assert comparison.overall_passed is True
    assert comparison.metric_deltas[0].passed is True


def test_delta_threshold_fails_regression_beyond_tolerance():
    baseline = run_result("baseline", [metric("pass_rate", 0.9)])
    candidate = run_result("candidate", [metric("pass_rate", 0.85)])

    comparison = EvaluationComparator().compare(
        baseline,
        candidate,
        [MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, delta_value=-0.02)],
    )

    assert comparison.overall_passed is False
    assert comparison.metric_deltas[0].passed is False


def test_absolute_threshold_uses_candidate_metric():
    baseline = run_result("baseline", [metric("citation_validity", 1.0)])
    candidate = run_result("candidate", [metric("citation_validity", 0.5)])

    comparison = EvaluationComparator().compare(
        baseline,
        candidate,
        [MetricThreshold(metric_name="citation_validity", operator=ThresholdOperator.GTE, value=1.0)],
    )

    assert comparison.overall_passed is False
    assert comparison.metric_deltas[0].passed is False


def test_unavailable_metric_fails_threshold_by_default():
    baseline = run_result("baseline", [metric("evidence_coverage", 1.0)])
    candidate = run_result("candidate", [metric("evidence_coverage", None, available=False)])

    comparison = EvaluationComparator().compare(
        baseline,
        candidate,
        [MetricThreshold(metric_name="evidence_coverage", operator=ThresholdOperator.GTE, value=1.0)],
    )

    assert comparison.overall_passed is False
    assert comparison.metric_deltas[0].passed is False


def test_lower_is_better_delta_threshold_direction():
    """error_rate getting worse (increasing) by more than the tolerance fails,
    matching its LTE (lower-is-better) direction -- not the GTE-shaped
    delta_minimum semantics the old schema silently assumed for every metric.
    """
    baseline = run_result("baseline", [metric("error_rate", 0.05)])
    tolerable_candidate = run_result("candidate", [metric("error_rate", 0.06)])
    excessive_candidate = run_result("candidate", [metric("error_rate", 0.10)])
    threshold = [MetricThreshold(metric_name="error_rate", operator=ThresholdOperator.LTE, delta_value=0.02)]

    tolerable = EvaluationComparator().compare(baseline, tolerable_candidate, threshold)
    excessive = EvaluationComparator().compare(baseline, excessive_candidate, threshold)

    assert tolerable.overall_passed is True
    assert tolerable.metric_deltas[0].passed is True
    assert excessive.overall_passed is False
    assert excessive.metric_deltas[0].passed is False


def test_delta_is_none_when_metric_missing_from_one_side():
    baseline = run_result("baseline", [metric("pass_rate", 0.8)])
    candidate = run_result("candidate", [])  # pass_rate absent entirely on the candidate side

    comparison = EvaluationComparator().compare(baseline, candidate)

    delta = next(d for d in comparison.metric_deltas if d.metric_name == "pass_rate")
    assert delta.delta is None
    assert delta.candidate_value is None


def test_delta_is_none_when_a_metric_value_is_none_despite_available_true():
    # Defensive case: a MetricResult marked available=True but with no value
    # (not produced by MetricCalculator today, but not excluded by the
    # schema either) must still short-circuit to delta=None, never a
    # TypeError from subtracting None.
    baseline = run_result("baseline", [MetricResult(name="pass_rate", value=None, unit="ratio", available=True)])
    candidate = run_result("candidate", [metric("pass_rate", 0.9)])

    comparison = EvaluationComparator().compare(baseline, candidate)

    delta = next(d for d in comparison.metric_deltas if d.metric_name == "pass_rate")
    assert delta.delta is None


def test_delta_only_threshold_fails_by_default_when_delta_unavailable():
    baseline = run_result("baseline", [])  # metric absent -> delta will be None
    candidate = run_result("candidate", [metric("pass_rate", 0.9)])
    threshold = MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, delta_value=-0.1)

    comparison = EvaluationComparator().compare(baseline, candidate, [threshold])

    delta = next(d for d in comparison.metric_deltas if d.metric_name == "pass_rate")
    assert delta.passed is False
    assert comparison.overall_passed is False


def test_delta_only_threshold_passes_when_unavailable_passes_is_set():
    baseline = run_result("baseline", [])
    candidate = run_result("candidate", [metric("pass_rate", 0.9)])
    threshold = MetricThreshold(
        metric_name="pass_rate", operator=ThresholdOperator.GTE, delta_value=-0.1, unavailable_passes=True
    )

    comparison = EvaluationComparator().compare(baseline, candidate, [threshold])

    delta = next(d for d in comparison.metric_deltas if d.metric_name == "pass_rate")
    assert delta.passed is True


def test_compare_does_not_share_mutable_objects_with_caller():
    baseline = run_result("baseline", [metric("pass_rate", 0.8)])
    candidate = run_result("candidate", [metric("pass_rate", 0.9)])
    original_baseline_value = baseline.metrics[0].value

    comparison = EvaluationComparator().compare(baseline, candidate)

    assert comparison.baseline is not baseline
    assert comparison.candidate is not candidate

    # Mutate the caller's own objects after compare() returns.
    baseline.metrics[0].value = 0.1
    candidate.candidate_id = "mutated"

    assert comparison.baseline.metrics[0].value == original_baseline_value
    assert comparison.candidate.candidate_id == "candidate"
