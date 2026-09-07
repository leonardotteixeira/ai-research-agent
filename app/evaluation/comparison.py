"""Simple baseline/candidate comparison for evaluation runs."""

from app.evaluation.schemas import (
    THRESHOLD_EPSILON,
    EvaluationComparison,
    EvaluationRunResult,
    MetricDelta,
    MetricResult,
    MetricThreshold,
)

# Used only to label a delta as a "regression"/"improvement" for humans
# reading the report. This is independent of pass/fail gating: gating is
# governed entirely by each MetricThreshold's own explicit `operator`
# (see schemas.py), never by this or any other metric-name lookup table.
_LOWER_IS_BETTER = {
    "error_rate",
    "timeout_rate",
    "policy_violation_rate",
    "average_steps",
    "average_tool_calls",
    "average_elapsed_seconds",
}


class EvaluationComparator:
    def compare(
        self,
        baseline: EvaluationRunResult,
        candidate: EvaluationRunResult,
        thresholds: list[MetricThreshold] | None = None,
    ) -> EvaluationComparison:
        active_thresholds = thresholds or candidate.thresholds
        baseline_metrics = {metric.name: metric for metric in baseline.metrics}
        candidate_metrics = {metric.name: metric for metric in candidate.metrics}
        metric_names = sorted(baseline_metrics.keys() | candidate_metrics.keys())

        deltas: list[MetricDelta] = []
        regressions: list[str] = []
        improvements: list[str] = []
        overall_passed = candidate.overall_passed

        for name in metric_names:
            baseline_metric = baseline_metrics.get(name)
            candidate_metric = candidate_metrics.get(name)
            threshold = next((item for item in active_thresholds if item.metric_name == name), None)
            delta = self._delta(baseline_metric, candidate_metric)
            passed = self._passes(candidate_metric, delta, threshold)
            if passed is False:
                overall_passed = False
            if delta is not None:
                if self._is_regression(name, delta):
                    regressions.append(name)
                if self._is_improvement(name, delta):
                    improvements.append(name)
            deltas.append(
                MetricDelta(
                    metric_name=name,
                    baseline_value=baseline_metric.value if baseline_metric else None,
                    candidate_value=candidate_metric.value if candidate_metric else None,
                    delta=delta,
                    threshold=threshold,
                    passed=passed,
                )
            )

        return EvaluationComparison(
            evaluation_id=candidate.evaluation_id,
            # Deep-copied so a caller mutating their own baseline/candidate
            # object after compare() returns can never retroactively change
            # an already-built EvaluationComparison.
            baseline=baseline.model_copy(deep=True),
            candidate=candidate.model_copy(deep=True),
            metric_deltas=deltas,
            regressions=regressions,
            improvements=improvements,
            overall_passed=overall_passed,
        )

    @staticmethod
    def _delta(baseline: MetricResult | None, candidate: MetricResult | None) -> float | None:
        if baseline is None or candidate is None:
            return None
        if not baseline.available or not candidate.available:
            return None
        if baseline.value is None or candidate.value is None:
            return None
        return candidate.value - baseline.value

    @staticmethod
    def _passes(
        candidate: MetricResult | None,
        delta: float | None,
        threshold: MetricThreshold | None,
    ) -> bool | None:
        if threshold is None:
            return None
        if threshold.value is not None:
            if candidate is None or not candidate.available or candidate.value is None:
                if not threshold.unavailable_passes:
                    return False
            elif not threshold.passes_absolute(candidate.value):
                return False
        if threshold.delta_value is not None:
            if delta is None:
                if not threshold.unavailable_passes:
                    return False
            elif not threshold.passes_delta(delta):
                return False
        return True

    @staticmethod
    def _is_regression(metric_name: str, delta: float) -> bool:
        if abs(delta) <= THRESHOLD_EPSILON:
            return False
        return delta > 0 if metric_name in _LOWER_IS_BETTER else delta < 0

    @staticmethod
    def _is_improvement(metric_name: str, delta: float) -> bool:
        if abs(delta) <= THRESHOLD_EPSILON:
            return False
        return delta < 0 if metric_name in _LOWER_IS_BETTER else delta > 0
