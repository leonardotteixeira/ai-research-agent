"""Metric aggregation and absolute threshold evaluation."""

from app.evaluation.schemas import (
    EvaluatedCase,
    EvaluationDataset,
    MetricResult,
    MetricThreshold,
)
from app.schemas.run import RunRecord
from app.schemas.state import TerminationReason


class MetricCalculator:
    def calculate(
        self,
        dataset: EvaluationDataset,
        records_by_case_id: dict[str, RunRecord],
        case_results: list[EvaluatedCase],
        thresholds: list[MetricThreshold] | None = None,
    ) -> list[MetricResult]:
        metrics = [
            self._pass_rate(case_results),
            self._completion_rate(records_by_case_id),
            self._citation_validity(case_results),
            self._evidence_coverage(case_results),
            self._expected_source_coverage(dataset, case_results),
            self._error_rate(records_by_case_id),
            self._termination_rate(records_by_case_id, TerminationReason.TIMEOUT, "timeout_rate"),
            self._policy_violation_rate(case_results, records_by_case_id),
            self._average_steps(records_by_case_id),
            self._average_tool_calls(records_by_case_id),
            self._average_elapsed_seconds(records_by_case_id),
        ]
        return [self._apply_threshold(metric, thresholds or []) for metric in metrics]

    @staticmethod
    def _available(name: str, value: float, unit: str, **details: object) -> MetricResult:
        return MetricResult(name=name, value=value, unit=unit, available=True, details=details)

    @staticmethod
    def _unavailable(name: str, unit: str, reason: str, **details: object) -> MetricResult:
        return MetricResult(
            name=name,
            value=None,
            unit=unit,
            available=False,
            passed=None,
            details={"reason": reason, **details},
        )

    def _pass_rate(self, case_results: list[EvaluatedCase]) -> MetricResult:
        total = len(case_results)
        if total == 0:
            return self._unavailable("pass_rate", "ratio", "no evaluated cases")
        passed = sum(case.passed for case in case_results)
        return self._available("pass_rate", passed / total, "ratio", passed=passed, total=total)

    def _completion_rate(self, records: dict[str, RunRecord]) -> MetricResult:
        total = len(records)
        if total == 0:
            return self._unavailable("completion_rate", "ratio", "no run records")
        completed = sum(
            record.research_state.termination_reason == TerminationReason.FINISHED
            and record.research_state.final_answer is not None
            and record.research_state.final_answer.is_complete
            for record in records.values()
        )
        return self._available("completion_rate", completed / total, "ratio", completed=completed, total=total)

    def _citation_validity(self, case_results: list[EvaluatedCase]) -> MetricResult:
        total = 0
        valid = 0
        for case in case_results:
            for evaluator_result in case.evaluator_results:
                if evaluator_result.evaluator == "citation":
                    total += int(evaluator_result.details.get("citation_count", 0))
                    valid += int(evaluator_result.details.get("valid_citation_count", 0))
        if total == 0:
            return self._unavailable("citation_validity", "ratio", "no citations to evaluate")
        return self._available("citation_validity", valid / total, "ratio", valid=valid, total=total)

    def _evidence_coverage(self, case_results: list[EvaluatedCase]) -> MetricResult:
        total = 0
        covered = 0
        for case in case_results:
            for evaluator_result in case.evaluator_results:
                if evaluator_result.evaluator == "grounding":
                    total += int(evaluator_result.details.get("total_claims", 0))
                    covered += int(evaluator_result.details.get("grounded_claims", 0))
        if total == 0:
            return self._unavailable("evidence_coverage", "ratio", "no claims to evaluate")
        return self._available("evidence_coverage", covered / total, "ratio", covered=covered, total=total)

    def _expected_source_coverage(
        self,
        dataset: EvaluationDataset,
        case_results: list[EvaluatedCase],
    ) -> MetricResult:
        total_expected = sum(len(case.expected.source_urls) for case in dataset.cases if case.expected is not None)
        if total_expected == 0:
            return self._unavailable("expected_source_coverage", "ratio", "no expected sources defined")
        missing = 0
        for case in case_results:
            for evaluator_result in case.evaluator_results:
                if evaluator_result.evaluator == "evidence":
                    missing += len(evaluator_result.details.get("missing_expected_sources", []))
        found = total_expected - missing
        return self._available(
            "expected_source_coverage",
            found / total_expected,
            "ratio",
            found=found,
            total=total_expected,
        )

    def _error_rate(self, records: dict[str, RunRecord]) -> MetricResult:
        total = len(records)
        if total == 0:
            return self._unavailable("error_rate", "ratio", "no run records")
        errored = sum(bool(record.research_state.errors) for record in records.values())
        return self._available("error_rate", errored / total, "ratio", errored=errored, total=total)

    def _termination_rate(
        self,
        records: dict[str, RunRecord],
        termination_reason: TerminationReason,
        name: str,
    ) -> MetricResult:
        total = len(records)
        if total == 0:
            return self._unavailable(name, "ratio", "no run records")
        count = sum(record.research_state.termination_reason == termination_reason for record in records.values())
        return self._available(name, count / total, "ratio", count=count, total=total)

    def _policy_violation_rate(
        self,
        case_results: list[EvaluatedCase],
        records: dict[str, RunRecord],
    ) -> MetricResult:
        total = len(records)
        if total == 0:
            return self._unavailable("policy_violation_rate", "ratio", "no run records")
        violated = 0
        for case in case_results:
            if any(result.evaluator == "policy" and not result.passed for result in case.evaluator_results):
                violated += 1
        return self._available(
            "policy_violation_rate",
            violated / total,
            "ratio",
            violated=violated,
            total=total,
        )

    def _average_steps(self, records: dict[str, RunRecord]) -> MetricResult:
        if not records:
            return self._unavailable("average_steps", "count", "no run records")
        return self._available(
            "average_steps",
            sum(record.research_state.current_step for record in records.values()) / len(records),
            "count",
        )

    def _average_tool_calls(self, records: dict[str, RunRecord]) -> MetricResult:
        if not records:
            return self._unavailable("average_tool_calls", "count", "no run records")
        return self._available(
            "average_tool_calls",
            sum(len(record.research_state.tool_calls) for record in records.values()) / len(records),
            "count",
        )

    def _average_elapsed_seconds(self, records: dict[str, RunRecord]) -> MetricResult:
        elapsed = [
            record.research_state.elapsed_seconds
            for record in records.values()
            if record.research_state.elapsed_seconds is not None
        ]
        if not elapsed:
            return self._unavailable("average_elapsed_seconds", "seconds", "elapsed time unavailable")
        return self._available("average_elapsed_seconds", sum(elapsed) / len(elapsed), "seconds", total=len(elapsed))

    @staticmethod
    def _apply_threshold(metric: MetricResult, thresholds: list[MetricThreshold]) -> MetricResult:
        threshold = next((item for item in thresholds if item.metric_name == metric.name), None)
        if threshold is None or threshold.value is None:
            return metric
        if not metric.available or metric.value is None:
            return metric.model_copy(update={"passed": True if threshold.unavailable_passes else None})
        return metric.model_copy(update={"passed": threshold.passes_absolute(metric.value)})


def thresholds_pass(metrics: list[MetricResult], thresholds: list[MetricThreshold]) -> bool:
    metrics_by_name = {metric.name: metric for metric in metrics}
    for threshold in thresholds:
        if threshold.value is None:
            continue
        metric = metrics_by_name.get(threshold.metric_name)
        if metric is None or not metric.available or metric.value is None:
            if not threshold.unavailable_passes:
                return False
            continue
        if not threshold.passes_absolute(metric.value):
            return False
    return True
