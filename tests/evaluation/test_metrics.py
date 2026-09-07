from app.evaluation.evaluators import default_evaluators
from app.evaluation.metrics import MetricCalculator, thresholds_pass
from app.evaluation.schemas import (
    EvaluatedCase,
    EvaluationCase,
    EvaluationDataset,
    EvaluationExpected,
    MetricResult,
    MetricThreshold,
    ThresholdOperator,
)
from app.schemas.state import ResearchResult, TerminationReason
from tests.unit.test_run_record import build_record


def _dataset(expected_sources: list[str] | None = None) -> EvaluationDataset:
    source_urls = ["https://example.com/report"] if expected_sources is None else expected_sources
    expected = EvaluationExpected(source_urls=source_urls)
    return EvaluationDataset(
        id="research_quality",
        version="1",
        name="Research Quality",
        cases=[EvaluationCase(id="case_1", question="What was observed?", expected=expected)],
    )


def _evaluated_case(record=None, dataset=None) -> EvaluatedCase:
    record = record or build_record()
    case = (dataset or _dataset()).cases[0]
    results = [evaluator.evaluate(case, record) for evaluator in default_evaluators()]
    return EvaluatedCase(
        case_id=case.id,
        run_id=record.run_id,
        passed=all(result.passed for result in results),
        evaluator_results=results,
    )


def _metrics(dataset=None, record=None, thresholds=None):
    dataset = dataset or _dataset()
    record = record or build_record()
    return MetricCalculator().calculate(
        dataset,
        {dataset.cases[0].id: record},
        [_evaluated_case(record, dataset)],
        thresholds=thresholds,
    )


def _metric(metrics, name):
    return next(metric for metric in metrics if metric.name == name)


def test_calculates_core_rates_and_averages():
    metrics = _metrics()

    assert _metric(metrics, "pass_rate").value == 1.0
    assert _metric(metrics, "completion_rate").value == 1.0
    assert _metric(metrics, "citation_validity").value == 1.0
    assert _metric(metrics, "evidence_coverage").value == 1.0
    assert _metric(metrics, "expected_source_coverage").value == 1.0
    assert _metric(metrics, "error_rate").value == 0.0
    assert _metric(metrics, "timeout_rate").value == 0.0
    assert _metric(metrics, "policy_violation_rate").value == 0.0
    assert _metric(metrics, "average_steps").value == 2.0
    assert _metric(metrics, "average_tool_calls").value == 1.0
    assert _metric(metrics, "average_elapsed_seconds").value == 1.25


def test_failure_rates_reflect_failed_run():
    record = build_record()
    record.research_state.termination_reason = TerminationReason.TIMEOUT
    record.research_state.errors = ["Agent run timed out after 1s"]
    record.research_result = ResearchResult.from_state(record.research_state)

    metrics = _metrics(record=record)

    assert _metric(metrics, "pass_rate").value == 0.0
    assert _metric(metrics, "completion_rate").value == 0.0
    assert _metric(metrics, "error_rate").value == 1.0
    assert _metric(metrics, "timeout_rate").value == 1.0


def test_zero_denominator_metrics_are_unavailable_not_zero():
    dataset = _dataset(expected_sources=[])
    record = build_record()
    record.research_state.claims = []
    record.research_state.final_answer = record.research_state.final_answer.model_copy(update={"claims": []})
    record.research_result = ResearchResult.from_state(record.research_state)

    metrics = _metrics(dataset=dataset, record=record)

    assert _metric(metrics, "evidence_coverage").available is False
    assert _metric(metrics, "expected_source_coverage").available is False


def test_missing_elapsed_time_is_unavailable():
    dataset = _dataset()
    record = build_record()
    record.research_state.elapsed_seconds = None

    metrics = MetricCalculator().calculate(dataset, {dataset.cases[0].id: record}, [_evaluated_case(record, dataset)])

    assert _metric(metrics, "average_elapsed_seconds").available is False


def test_policy_violation_rate_reflects_a_violating_case():
    from app.policies.execution import ExecutionPolicy
    from app.schemas.tool import ToolCall

    dataset = _dataset()
    record = build_record()
    forbidden = ToolCall(call_id="call_x", tool_name="fetch_url", arguments={"url": "https://example.com"})
    record.research_state.tool_calls = [*record.research_state.tool_calls, forbidden]
    record.execution_policy = ExecutionPolicy(
        max_steps=record.execution_policy.max_steps,
        max_tool_calls=record.execution_policy.max_tool_calls,
        max_same_tool_calls=record.execution_policy.max_same_tool_calls,
        allowed_tools={"web_search"},
    )

    metrics = _metrics(dataset=dataset, record=record)

    assert _metric(metrics, "policy_violation_rate").value == 1.0


def test_thresholds_pass_fail_and_exact_boundary():
    thresholds = [
        MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, value=1.0),
        MetricThreshold(metric_name="citation_validity", operator=ThresholdOperator.GTE, value=1.0),
    ]
    metrics = _metrics(thresholds=thresholds)

    assert _metric(metrics, "pass_rate").passed is True
    assert _metric(metrics, "citation_validity").passed is True
    assert thresholds_pass(metrics, thresholds) is True

    # average_steps is lower-is-better: its guardrail is naturally an upper
    # bound (LTE), not a lower bound -- record has average_steps == 2.0.
    failing = [MetricThreshold(metric_name="average_steps", operator=ThresholdOperator.LTE, value=1.0)]
    failing_metrics = _metrics(thresholds=failing)
    assert _metric(failing_metrics, "average_steps").passed is False
    assert thresholds_pass(failing_metrics, failing) is False

    passing = [MetricThreshold(metric_name="average_steps", operator=ThresholdOperator.LTE, value=2.0)]
    passing_metrics = _metrics(thresholds=passing)
    assert _metric(passing_metrics, "average_steps").passed is True
    assert thresholds_pass(passing_metrics, passing) is True


def test_all_rate_and_average_metrics_are_unavailable_with_zero_records():
    dataset = _dataset()
    metrics = MetricCalculator().calculate(dataset, {}, [])

    for name in [
        "pass_rate",
        "completion_rate",
        "citation_validity",
        "error_rate",
        "timeout_rate",
        "policy_violation_rate",
        "average_steps",
        "average_tool_calls",
    ]:
        metric = _metric(metrics, name)
        assert metric.available is False, f"{name} should be unavailable with zero records"
        assert metric.value is None


def test_thresholds_pass_skips_delta_only_threshold_and_honors_unavailable_passes():
    metrics = [MetricResult(name="pass_rate", value=None, unit="ratio", available=False)]

    # A delta-only threshold (no `value`) is irrelevant to a single-run
    # metrics list; it must be skipped, not treated as an unmet absolute check.
    delta_only = MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, delta_value=-0.1)
    assert thresholds_pass(metrics, [delta_only]) is True

    # An unavailable metric with unavailable_passes=True must not fail the run.
    lenient = MetricThreshold(
        metric_name="pass_rate", operator=ThresholdOperator.GTE, value=1.0, unavailable_passes=True
    )
    assert thresholds_pass(metrics, [lenient]) is True


def test_unavailable_metric_does_not_pass_threshold_by_default():
    dataset = _dataset(expected_sources=[])
    threshold = MetricThreshold(metric_name="expected_source_coverage", operator=ThresholdOperator.GTE, value=1.0)
    metrics = _metrics(dataset=dataset, thresholds=[threshold])

    assert _metric(metrics, "expected_source_coverage").passed is None
    assert thresholds_pass(metrics, [threshold]) is False


def _ratio_metric(name: str, value: float) -> MetricResult:
    return MetricResult(name=name, value=value, unit="ratio", available=True)


def test_higher_is_better_threshold_boundaries():
    """pass_rate >= 0.90: 0.95 -> PASS, 0.90 -> PASS (exact boundary), 0.89 -> FAIL."""
    threshold = MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, value=0.90)
    for value, expect_pass in [(0.95, True), (0.90, True), (0.89, False)]:
        result = MetricCalculator._apply_threshold(_ratio_metric("pass_rate", value), [threshold])
        assert result.passed is expect_pass, f"pass_rate={value} expected passed={expect_pass}"


def test_lower_is_better_threshold_boundaries():
    """error_rate <= 0.05: 0.00 -> PASS, 0.05 -> PASS (exact boundary), 0.06 -> FAIL."""
    error_threshold = MetricThreshold(metric_name="error_rate", operator=ThresholdOperator.LTE, value=0.05)
    for value, expect_pass in [(0.00, True), (0.05, True), (0.06, False)]:
        result = MetricCalculator._apply_threshold(_ratio_metric("error_rate", value), [error_threshold])
        assert result.passed is expect_pass, f"error_rate={value} expected passed={expect_pass}"

    # timeout_rate <= 0.10: 0.05 -> PASS, 0.11 -> FAIL.
    timeout_threshold = MetricThreshold(metric_name="timeout_rate", operator=ThresholdOperator.LTE, value=0.10)
    for value, expect_pass in [(0.05, True), (0.11, False)]:
        result = MetricCalculator._apply_threshold(_ratio_metric("timeout_rate", value), [timeout_threshold])
        assert result.passed is expect_pass, f"timeout_rate={value} expected passed={expect_pass}"

    # policy_violation_rate <= 0.00: 0.00 -> PASS, 0.01 -> FAIL.
    policy_threshold = MetricThreshold(
        metric_name="policy_violation_rate", operator=ThresholdOperator.LTE, value=0.00
    )
    for value, expect_pass in [(0.00, True), (0.01, False)]:
        result = MetricCalculator._apply_threshold(_ratio_metric("policy_violation_rate", value), [policy_threshold])
        assert result.passed is expect_pass, f"policy_violation_rate={value} expected passed={expect_pass}"


def test_threshold_epsilon_is_consistent_across_engines():
    """The same near-boundary value must be judged identically whether
    checked via MetricCalculator._apply_threshold/thresholds_pass (used by
    EvaluationRunner) or via EvaluationComparator._passes (used for
    baseline/candidate comparisons) -- both delegate to the same
    MetricThreshold.passes_absolute, so there is exactly one floating-point
    tolerance rule, not two.
    """
    from app.evaluation.comparison import EvaluationComparator
    from app.evaluation.schemas import EvaluationRunResult

    just_inside_epsilon = 0.05 + 1e-13  # within THRESHOLD_EPSILON=1e-12 of the boundary
    threshold = MetricThreshold(metric_name="error_rate", operator=ThresholdOperator.LTE, value=0.05)

    calculator_result = MetricCalculator._apply_threshold(_ratio_metric("error_rate", just_inside_epsilon), [threshold])
    assert calculator_result.passed is True
    assert thresholds_pass([calculator_result], [threshold]) is True

    run_result = EvaluationRunResult(
        evaluation_id="e",
        dataset_id="d",
        dataset_version="1",
        candidate_id="c",
        metrics=[_ratio_metric("error_rate", just_inside_epsilon)],
        overall_passed=True,
    )
    baseline = run_result.model_copy(deep=True)
    comparison = EvaluationComparator().compare(baseline, run_result, [threshold])
    assert comparison.metric_deltas[0].passed is True
