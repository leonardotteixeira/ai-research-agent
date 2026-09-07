from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.datasets import EvaluationDatasetLoader
from app.evaluation.reports import EvaluationJSONRenderer, EvaluationMarkdownRenderer
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import MetricThreshold, ThresholdOperator
from app.persistence.file_repository import FileRunRepository
from app.policies.execution import ExecutionPolicy
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.run import RunRecord
from app.schemas.state import ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall, ToolResult
from app.services.run_service import RunService


def _pgvector_record() -> RunRecord:
    timestamp = datetime(2026, 9, 6, tzinfo=UTC)
    call = ToolCall(
        call_id="c_search_1",
        tool_name="web_search",
        arguments={"query": "vector database postgres"},
    )
    source = Source(
        source_id="src_pgvector",
        url="https://fixture.example/pgvector-overview",
        title="pgvector: Open-source vector similarity search for Postgres",
        source_type="search_result",
        retrieved_at=timestamp,
        tool_call_id=call.call_id,
        timestamp=timestamp,
    )
    evidence = Evidence(
        evidence_id="ev_pgvector",
        content=(
            "pgvector adds vector columns and similarity search directly to PostgreSQL. "
            "It is useful when teams already run Postgres."
        ),
        source_id=source.source_id,
        tool_call_id=call.call_id,
        timestamp=timestamp,
    )
    claim = Claim(
        claim_id="claim_pgvector",
        text="pgvector adds vector columns and similarity search directly to PostgreSQL",
        evidence_ids=[evidence.evidence_id],
    )
    answer = FinalAnswer(
        answer_text="pgvector works directly in PostgreSQL by adding vector columns and similarity search [1].",
        claims=[claim],
        citations=[Citation(claim=claim.text, evidence_ids=[evidence.evidence_id])],
        is_complete=True,
    )
    state = ResearchState(
        research_id="run_pgvector",
        original_question="Which vector database works directly in Postgres?",
        created_at=timestamp,
        current_step=2,
        tool_calls=[call],
        tool_results=[
            ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                output=[{"url": source.url, "title": source.title, "snippet": evidence.content}],
            )
        ],
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        errors=[],
        elapsed_seconds=0.5,
        final_answer=answer,
        termination_reason=TerminationReason.FINISHED,
    )
    return RunRecord(
        run_id=state.research_id,
        created_at=timestamp,
        question=state.original_question,
        execution_policy=ExecutionPolicy(
            max_steps=5,
            max_tool_calls=5,
            max_same_tool_calls=3,
            allowed_tools={"web_search"},
        ),
        research_state=state,
        research_result=ResearchResult.from_state(state),
    )


def test_core_evaluation_e2e_offline_from_persisted_run(tmp_path: Path):
    dataset = EvaluationDatasetLoader().load_json("evals/datasets/research_quality_v1.json")
    service = RunService(FileRunRepository(tmp_path / "runs"))
    record = _pgvector_record()
    service.save_run(record)
    loaded = service.load_run(record.run_id)

    result = EvaluationRunner().evaluate_records(
        dataset,
        {"vector_db_postgres": loaded},
        evaluation_id="research_quality_v1",
        candidate_id="offline_candidate",
        thresholds=[
            MetricThreshold(metric_name="pass_rate", operator=ThresholdOperator.GTE, value=1.0),
            MetricThreshold(metric_name="citation_validity", operator=ThresholdOperator.GTE, value=1.0),
            MetricThreshold(metric_name="evidence_coverage", operator=ThresholdOperator.GTE, value=1.0),
            MetricThreshold(metric_name="expected_source_coverage", operator=ThresholdOperator.GTE, value=1.0),
        ],
    )
    json_report = EvaluationJSONRenderer().render(result)
    markdown_report = EvaluationMarkdownRenderer().render(result)

    assert result.overall_passed is True
    assert result.case_results[0].passed is True
    assert '"candidate_id": "offline_candidate"' in json_report
    assert "# Evaluation Report" in markdown_report
    assert "Overall: PASS" in markdown_report
