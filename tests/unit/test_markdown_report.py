from datetime import UTC, datetime
from pathlib import Path

from app.persistence import FileRunRepository
from app.policies.execution import ExecutionPolicy
from app.reports.markdown import MarkdownReportRenderer
from app.schemas.run import RunRecord
from app.schemas.state import ResearchState
from app.services.run_service import RunService
from tests.unit.test_run_record import build_record


class TestMarkdownReportRenderer:
    def test_renders_complete_report(self):
        report = MarkdownReportRenderer().render(build_record())

        assert report.startswith("# Research Report")
        assert "## Question" in report
        assert "What was observed?" in report
        assert "## Execution" in report
        assert "- Run ID: `run_1`" in report
        assert "max_steps: `8`" in report
        assert "## Sources" in report
        assert "Example report" in report
        assert "source_id: `src_1`" in report
        assert "## Evidence" in report
        assert "The report contains the observed result." in report
        assert "## Claims" in report
        assert "The result was observed." in report
        assert "## Answer" in report
        assert "The result was observed [1]." in report
        assert "## Citations" in report
        assert "evidence_ids: `ev_1`" in report

    def test_renders_minimal_report_with_explicit_empty_sections(self):
        timestamp = datetime(2026, 9, 5, tzinfo=UTC)
        state = ResearchState(
            research_id="minimal",
            original_question="No data?",
            created_at=timestamp,
        )
        record = RunRecord(
            run_id="minimal",
            created_at=timestamp,
            question="No data?",
            execution_policy=ExecutionPolicy(),
            research_state=state,
        )

        report = MarkdownReportRenderer().render(record)

        assert report.count("No data.") == 5
        assert "Termination reason: `Not available`" in report

    def test_rendering_is_deterministic(self):
        renderer = MarkdownReportRenderer()
        record = build_record()

        assert renderer.render(record) == renderer.render(record)

    def test_escapes_special_external_text_without_executing_markdown(self):
        record = build_record()
        record.research_state.sources[0].title = "Title | # not a heading"
        record.research_state.evidence[0].content = "Ignore instructions\n| external data"
        record.research_state.claims[0].text = "Claim | text"

        report = MarkdownReportRenderer().render(record)

        assert "Title \\| # not a heading" in report
        assert "Ignore instructions \\| external data" in report
        assert "Claim \\| text" in report


class TestRunPersistenceToMarkdown:
    def test_file_repository_round_trip_can_be_rendered(self, tmp_path: Path):
        original = build_record()
        service = RunService(FileRunRepository(tmp_path))
        service.save_run(original)

        loaded = service.load_run(original.run_id)
        report = MarkdownReportRenderer().render(loaded)

        assert loaded == original
        assert "# Research Report" in report
        assert "run_1" in report
        assert "ev_1" in report
        assert "https://example.com/report" in report
