"""Offline end-to-end CLI test.

Drives the actual `python -m app.cli` Typer app (via CliRunner) through
run -> runs -> show -> report -> replay against one real persisted run,
using real Agent/ToolRegistry/EvidencePipeline/SynthesisService/RunService/
FileRunRepository/MarkdownReportRenderer components -- the only doubles are
MockLLMProvider, MockSearchProvider, and MockSynthesisProvider, exactly like
tests/integration/test_orchestrator_e2e.py (the only externally-facing
pieces that would otherwise need network/API access).

Each CLI invocation is a fresh Typer command run, and every read-only
command (`runs`/`show`/`report`/`replay`) builds a brand new
FileRunRepository/RunService via build_read_only_components() -- there is
no shared in-memory state between commands, so this also proves that
reads survive a "fresh process" (a new repository/service instance reading
what a previous instance wrote).
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import app.cli.main as cli_main
from app.agent.agent import Agent
from app.cli.composition import RunComponents
from app.evidence.pipeline import EvidencePipeline
from app.persistence.file_repository import FileRunRepository
from app.policies.execution import ExecutionPolicy
from app.providers.mock_llm import MockLLMProvider
from app.providers.search import MockSearchProvider, SearchHit
from app.reports.markdown import MarkdownReportRenderer
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.evidence import Claim
from app.schemas.run import RunRecord
from app.schemas.tool import ToolCall
from app.services.orchestrator import ResearchOrchestrator
from app.services.run_service import RunService
from app.synthesis.mock_provider import MockSynthesisProvider
from app.synthesis.service import SynthesisService
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool

runner = CliRunner()


def _build_offline_run_components(runs_dir: Path) -> RunComponents:
    call_id = "c_search_1"
    evidence_id = f"ev_{call_id}_0"
    query = "vector database for postgres"

    search_provider = MockSearchProvider(
        [
            SearchHit(
                url="https://postgres.example/pgvector",
                title="pgvector: Vector similarity search for Postgres",
                snippet="pgvector adds vector similarity search directly to PostgreSQL.",
            )
        ]
    )
    tool_registry = ToolRegistry([WebSearchTool(search_provider)])  # type: ignore[list-item]

    llm_provider = MockLLMProvider(
        [
            LLMDecision(
                action=DecisionAction.TOOL_CALL,
                rationale="Search for vector database solutions for postgres.",
                tool_calls=[ToolCall(call_id=call_id, tool_name="web_search", arguments={"query": query})],
            ),
            LLMDecision(
                action=DecisionAction.SYNTHESIZE,
                rationale="Sufficient evidence found; synthesizing final answer.",
            ),
        ]
    )

    agent = Agent(
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        policy=ExecutionPolicy(max_steps=5, max_tool_calls=5),
        evidence_pipeline=EvidencePipeline(),
    )

    claim = Claim(
        claim_id="claim_1",
        text="pgvector adds vector similarity search directly to PostgreSQL",
        evidence_ids=[evidence_id],
    )
    answer = FinalAnswer(
        answer_text="pgvector adds vector similarity search directly to PostgreSQL [1].",
        claims=[claim],
        citations=[Citation(claim=claim.text, evidence_ids=[evidence_id])],
        is_complete=True,
    )
    synthesis_service = SynthesisService(MockSynthesisProvider([answer]))

    run_service = RunService(FileRunRepository(runs_dir))
    report_renderer = MarkdownReportRenderer()

    orchestrator = ResearchOrchestrator(
        agent=agent,
        synthesis_service=synthesis_service,
        run_service=run_service,
        report_renderer=report_renderer,
    )
    return RunComponents(
        orchestrator=orchestrator,
        run_service=run_service,
        report_renderer=report_renderer,
        is_mock=True,
    )


class TestCLIOfflineEndToEnd:
    def test_run_runs_show_report_replay_over_the_same_persisted_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runs_dir = tmp_path / "runs"
        monkeypatch.setattr(cli_main, "build_run_components", lambda *a, **k: _build_offline_run_components(runs_dir))

        # 1-3. `run` executes the full offline pipeline and persists a RunRecord.
        run_result = runner.invoke(
            cli_main.app,
            ["run", "Which vector database works directly in Postgres?", "--json"],
        )
        assert run_result.exit_code == 0, run_result.output
        run_payload = json.loads(run_result.output)
        run_id = run_payload["run_id"]
        assert run_payload["success"] is True
        assert run_payload["termination_reason"] == "finished"

        # Confirm the file actually landed on disk (FileRunRepository's contract).
        assert (runs_dir / run_id / "run.json").is_file()

        # 4. `runs` -- a brand new FileRunRepository/RunService reads it back.
        runs_result = runner.invoke(cli_main.app, ["runs", "--runs-dir", str(runs_dir)])
        assert runs_result.exit_code == 0
        assert run_id in runs_result.output

        # 5. `show` -- another fresh RunService instance.
        show_result = runner.invoke(cli_main.app, ["show", run_id, "--runs-dir", str(runs_dir), "--json"])
        assert show_result.exit_code == 0
        record = RunRecord.model_validate_json(show_result.output)
        assert record.run_id == run_id
        assert record.research_state.final_answer is not None
        assert "pgvector" in record.research_state.final_answer.answer_text

        # 6. `report` -- MarkdownReportRenderer over the reloaded record.
        report_result = runner.invoke(cli_main.app, ["report", run_id, "--runs-dir", str(runs_dir)])
        assert report_result.exit_code == 0
        assert "# Research Report" in report_result.output
        assert "pgvector: Vector similarity search for Postgres" in report_result.output
        assert "https://postgres.example/pgvector" in report_result.output

        # 7. `replay` -- deterministic offline narrative, no orchestrator/agent involved.
        def fail_if_run_components_built(*args: object, **kwargs: object) -> RunComponents:
            raise AssertionError("replay must never build run (Agent/LLM/tool) components")

        monkeypatch.setattr(cli_main, "build_run_components", fail_if_run_components_built)
        replay_result = runner.invoke(cli_main.app, ["replay", run_id, "--runs-dir", str(runs_dir)])
        assert replay_result.exit_code == 0
        assert "Question: Which vector database works directly in Postgres?" in replay_result.output
        assert "web_search(call_id=c_search_1" in replay_result.output
        assert "pgvector adds vector similarity search directly to PostgreSQL" in replay_result.output
        assert "Termination: finished" in replay_result.output

        # 9-10. A completely independent RunService/FileRunRepository (a new
        # "process") reading the same directory sees the identical record.
        independent_run_service = RunService(FileRunRepository(runs_dir))
        reloaded = independent_run_service.load_run(run_id)
        assert reloaded == record
