"""Evaluation runner for already-persisted research RunRecords."""

from app.evaluation.evaluators import Evaluator, default_evaluators
from app.evaluation.metrics import MetricCalculator, thresholds_pass
from app.evaluation.schemas import (
    EvaluatedCase,
    EvaluationDataset,
    EvaluationRunResult,
    MetricThreshold,
)
from app.schemas.run import RunRecord


class EvaluationRunnerError(Exception):
    """Base error for evaluation runner failures."""


class MissingRunRecordError(EvaluationRunnerError):
    """A dataset case has no matching RunRecord."""


class UnknownCaseRunRecordError(EvaluationRunnerError):
    """A supplied RunRecord is keyed to no case in the dataset."""


class EvaluationRunner:
    def __init__(
        self,
        evaluators: list[Evaluator] | None = None,
        metric_calculator: MetricCalculator | None = None,
    ) -> None:
        self._evaluators = evaluators if evaluators is not None else default_evaluators()
        self._metric_calculator = metric_calculator or MetricCalculator()

    def evaluate_records(
        self,
        dataset: EvaluationDataset,
        records_by_case_id: dict[str, RunRecord],
        evaluation_id: str,
        candidate_id: str,
        thresholds: list[MetricThreshold] | None = None,
        metadata: dict | None = None,
    ) -> EvaluationRunResult:
        case_ids = {case.id for case in dataset.cases}
        missing = sorted(case_ids - records_by_case_id.keys())
        if missing:
            raise MissingRunRecordError(f"missing RunRecord for evaluation case id(s): {missing}")
        unknown = sorted(records_by_case_id.keys() - case_ids)
        if unknown:
            raise UnknownCaseRunRecordError(f"RunRecord supplied for unknown case id(s): {unknown}")

        evaluated_cases: list[EvaluatedCase] = []
        for case in dataset.cases:
            record = records_by_case_id[case.id]
            evaluator_results = [evaluator.evaluate(case, record) for evaluator in self._evaluators]
            evaluated_cases.append(
                EvaluatedCase(
                    case_id=case.id,
                    run_id=record.run_id,
                    passed=all(result.passed for result in evaluator_results),
                    evaluator_results=evaluator_results,
                )
            )

        active_thresholds = thresholds or []
        metrics = self._metric_calculator.calculate(
            dataset,
            records_by_case_id,
            evaluated_cases,
            thresholds=active_thresholds,
        )
        overall_passed = all(case.passed for case in evaluated_cases) and thresholds_pass(
            metrics,
            active_thresholds,
        )
        return EvaluationRunResult(
            evaluation_id=evaluation_id,
            dataset_id=dataset.id,
            dataset_version=dataset.version,
            candidate_id=candidate_id,
            case_results=evaluated_cases,
            metrics=metrics,
            thresholds=active_thresholds,
            overall_passed=overall_passed,
            metadata=metadata or {},
        )
