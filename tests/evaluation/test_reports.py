import json

from app.evaluation.comparison import EvaluationComparator
from app.evaluation.reports import EvaluationJSONRenderer, EvaluationMarkdownRenderer
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import (
    EvaluationCase,
    EvaluationDataset,
    MetricThreshold,
    ThresholdOperator,
)
from tests.unit.test_run_record import build_record


def _result():
    dataset = EvaluationDataset(
        id="research_quality",
        version="1",
        name="Research Quality",
        cases=[EvaluationCase(id="case_1", question="What was observed?")],
    )
    return EvaluationRunner().evaluate_records(
        dataset,
        {"case_1": build_record()},
        evaluation_id="research_quality_v1",
        candidate_id="candidate",
        thresholds=[MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, value=1.0)],
    )


def test_json_renderer_is_valid_and_deterministic():
    result = _result()
    renderer = EvaluationJSONRenderer()

    first = renderer.render(result)
    second = renderer.render(result)

    assert first == second
    payload = json.loads(first)
    assert payload["evaluation_id"] == "research_quality_v1"
    assert payload["overall_passed"] is True


def test_markdown_renderer_is_deterministic():
    result = _result()
    renderer = EvaluationMarkdownRenderer()

    markdown = renderer.render(result)

    assert markdown == renderer.render(result)
    assert markdown.startswith("# Evaluation Report")
    assert "Candidate: `candidate`" in markdown
    assert "| pass_rate | 1 | >= 1 | PASS |" in markdown
    assert "Overall: PASS" in markdown


def test_comparison_markdown_lists_regressions_and_improvements():
    baseline = _result()
    candidate = baseline.model_copy(deep=True, update={"candidate_id": "candidate_2"})
    pass_rate_index = next(
        index for index, metric in enumerate(candidate.metrics) if metric.name == "pass_rate"
    )
    candidate.metrics[pass_rate_index] = candidate.metrics[pass_rate_index].model_copy(
        update={"value": 0.5}
    )
    comparison = EvaluationComparator().compare(
        baseline,
        candidate,
        [MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, delta_value=-0.1)],
    )

    markdown = EvaluationMarkdownRenderer().render_comparison(comparison)

    assert markdown.startswith("# Evaluation Comparison")
    assert "Baseline: `candidate`" in markdown
    assert "Candidate: `candidate_2`" in markdown
    assert "- pass_rate" in markdown
    assert "Overall: FAIL" in markdown
