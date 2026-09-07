"""Stable JSON and deterministic Markdown reports for evaluation results."""

import json

from app.evaluation.schemas import EvaluationComparison, EvaluationRunResult, MetricResult


class EvaluationJSONRenderer:
    def render(self, result: EvaluationRunResult | EvaluationComparison) -> str:
        return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)


class EvaluationMarkdownRenderer:
    def render(self, result: EvaluationRunResult) -> str:
        passed_cases = sum(case.passed for case in result.case_results)
        failed_cases = len(result.case_results) - passed_cases
        lines = [
            "# Evaluation Report",
            "",
            "## Evaluation",
            "",
            f"- Evaluation ID: `{_inline(result.evaluation_id)}`",
            f"- Dataset: `{_inline(result.dataset_id)}`",
            f"- Dataset version: `{_inline(result.dataset_version)}`",
            f"- Candidate: `{_inline(result.candidate_id)}`",
            "",
            "## Cases",
            "",
            f"- Cases: `{len(result.case_results)}`",
            f"- Passed: `{passed_cases}`",
            f"- Failed: `{failed_cases}`",
            "",
            "## Metrics",
            "",
            "| Metric | Value | Threshold | Status |",
            "|---|---:|---|---|",
        ]
        thresholds = {threshold.metric_name: threshold for threshold in result.thresholds}
        for metric in result.metrics:
            threshold = thresholds.get(metric.name)
            lines.append(
                f"| {_inline(metric.name)} | {_metric_value(metric)} | "
                f"{_inline(_threshold_text(threshold))} | {_status(metric.passed)} |"
            )
        lines.extend(["", f"Overall: {'PASS' if result.overall_passed else 'FAIL'}"])
        return "\n".join(lines)

    def render_comparison(self, comparison: EvaluationComparison) -> str:
        lines = [
            "# Evaluation Comparison",
            "",
            "## Evaluation",
            "",
            f"- Evaluation ID: `{_inline(comparison.evaluation_id)}`",
            f"- Baseline: `{_inline(comparison.baseline.candidate_id)}`",
            f"- Candidate: `{_inline(comparison.candidate.candidate_id)}`",
            "",
            "## Metric Deltas",
            "",
            "| Metric | Baseline | Candidate | Delta | Status |",
            "|---|---:|---:|---:|---|",
        ]
        for delta in comparison.metric_deltas:
            lines.append(
                f"| {_inline(delta.metric_name)} | {_optional_number(delta.baseline_value)} | "
                f"{_optional_number(delta.candidate_value)} | {_optional_number(delta.delta)} | "
                f"{_status(delta.passed)} |"
            )
        lines.extend(
            [
                "",
                "## Regressions",
                "",
                *_list_or_none(comparison.regressions),
                "",
                "## Improvements",
                "",
                *_list_or_none(comparison.improvements),
                "",
                f"Overall: {'PASS' if comparison.overall_passed else 'FAIL'}",
            ]
        )
        return "\n".join(lines)


def _inline(value: str | None) -> str:
    if value is None:
        return "Not available"
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", "").replace("\n", " ")


def _metric_value(metric: MetricResult) -> str:
    if not metric.available or metric.value is None:
        return "N/A"
    return _optional_number(metric.value)


def _optional_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6g}"


def _threshold_text(threshold: object) -> str:
    if threshold is None:
        return ""
    parts = []
    operator = getattr(threshold, "operator", None)
    operator_text = operator.value if operator is not None else ">="
    value = getattr(threshold, "value", None)
    delta_value = getattr(threshold, "delta_value", None)
    if value is not None:
        parts.append(f"{operator_text} {_optional_number(value)}")
    if delta_value is not None:
        parts.append(f"delta {operator_text} {_optional_number(delta_value)}")
    return "; ".join(parts)


def _status(passed: bool | None) -> str:
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "N/A"


def _list_or_none(values: list[str]) -> list[str]:
    if not values:
        return ["No data."]
    return [f"- {_inline(value)}" for value in values]
