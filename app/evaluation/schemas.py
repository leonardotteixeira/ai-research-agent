"""Pydantic contracts for offline research evaluation."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class EvaluationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"


class EvaluationExpected(BaseModel):
    answer_contains: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    evidence_contains: list[str] = Field(default_factory=list)


class EvaluationCasePolicy(BaseModel):
    max_steps: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_same_tool_calls: int | None = Field(default=None, ge=1)
    total_timeout_seconds: float | None = Field(default=None, gt=0)
    allowed_tools: list[str] | None = None


class EvaluationCase(BaseModel):
    id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    expected: EvaluationExpected | None = None
    policy: EvaluationCasePolicy | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationDataset(BaseModel):
    id: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    cases: list[EvaluationCase] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_case_ids_are_unique(self) -> "EvaluationDataset":
        case_ids = [case.id for case in self.cases]
        duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate evaluation case id(s): {duplicates}")
        return self


class EvaluationCaseResult(BaseModel):
    case_id: str
    run_id: str
    evaluator: str
    passed: bool
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)


class EvaluatedCase(BaseModel):
    case_id: str
    run_id: str
    passed: bool
    evaluator_results: list[EvaluationCaseResult] = Field(default_factory=list)


class MetricResult(BaseModel):
    name: str
    value: float | None = None
    unit: str
    deterministic: bool = True
    available: bool = True
    passed: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ThresholdOperator(str, Enum):
    """The relation a metric value (or delta) must satisfy to pass.

    There is deliberately no default and no metric-name-based lookup table:
    every MetricThreshold states its own direction, so a lower-is-better
    metric (error_rate, timeout_rate, ...) is declared with LTE and a
    higher-is-better metric (pass_rate, citation_validity, ...) with GTE.
    Nothing in the evaluation engine assumes ">=" is universally "better".
    """

    GTE = ">="
    LTE = "<="


THRESHOLD_EPSILON = 1e-12


def _satisfies(value: float, operator: ThresholdOperator, target: float, epsilon: float = THRESHOLD_EPSILON) -> bool:
    """The single floating-point-tolerant comparison used everywhere a
    threshold is checked (metrics.py's per-run gate, comparison.py's
    baseline/candidate gate) -- one epsilon, one rule, no drift between
    the two call sites.
    """
    if operator == ThresholdOperator.GTE:
        return value + epsilon >= target
    return value - epsilon <= target


class MetricThreshold(BaseModel):
    metric_name: str
    operator: ThresholdOperator
    value: float | None = None
    delta_value: float | None = None
    unavailable_passes: bool = False

    @model_validator(mode="after")
    def validate_at_least_one_limit(self) -> "MetricThreshold":
        if self.value is None and self.delta_value is None:
            raise ValueError("threshold requires value or delta_value")
        return self

    def passes_absolute(self, metric_value: float) -> bool | None:
        """Whether `metric_value` itself satisfies this threshold, or None
        if this threshold declares no absolute (`value`) check."""
        if self.value is None:
            return None
        return _satisfies(metric_value, self.operator, self.value)

    def passes_delta(self, delta: float) -> bool | None:
        """Whether `delta` (candidate - baseline) satisfies this threshold's
        tolerance, or None if this threshold declares no delta check.

        Uses the same `operator` as the absolute check: for a GTE
        (higher-is-better) metric, `delta_value` is the most the metric may
        drop (e.g. -0.02); for an LTE (lower-is-better) metric, it is the
        most the metric may rise (e.g. +0.02). One field, direction-correct
        for either kind of metric.
        """
        if self.delta_value is None:
            return None
        return _satisfies(delta, self.operator, self.delta_value)


class EvaluationRunResult(BaseModel):
    evaluation_id: str
    dataset_id: str
    dataset_version: str
    candidate_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    case_results: list[EvaluatedCase] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)
    thresholds: list[MetricThreshold] = Field(default_factory=list)
    overall_passed: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricDelta(BaseModel):
    metric_name: str
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    threshold: MetricThreshold | None = None
    passed: bool | None


class EvaluationComparison(BaseModel):
    evaluation_id: str
    baseline: EvaluationRunResult
    candidate: EvaluationRunResult
    metric_deltas: list[MetricDelta] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    overall_passed: bool
