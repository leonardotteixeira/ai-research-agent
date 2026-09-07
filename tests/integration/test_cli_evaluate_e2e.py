"""Offline end-to-end test for `python -m app.cli evaluate`.

Drives the real `run` command (real Agent/ToolRegistry/EvidencePipeline/
SynthesisService/RunService/FileRunRepository -- only the LLM/search/
synthesis providers are mocked, exactly like test_cli_e2e.py and
test_e2e_offline.py) to produce a real persisted RunRecord, then drives
the real `evaluate` command against it and against the real sample
dataset at evals/datasets/research_quality_v1.json. Also exercises the
multi-case --runs-map path with a small ad-hoc two-case dataset.

Confirms `evaluate` never builds an Agent/LLM stack (never calls
build_run_components) and that a completely independent RunService
reads the same persisted data `evaluate` consumed.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import app.cli.main as cli_main
from app.agent.agent import Agent
from app.cli.composition import RunComponents
from app.evaluation.schemas import EvaluationComparison
from app.evidence.pipeline import EvidencePipeline
from app.persistence.file_repository import FileRunRepository
from app.policies.execution import ExecutionPolicy
from app.providers.mock_llm import MockLLMProvider
from app.providers.search import MockSearchProvider, SearchHit
from app.reports.markdown import MarkdownReportRenderer
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.evidence import Claim
from app.schemas.tool import ToolCall
from app.services.orchestrator import ResearchOrchestrator
from app.services.run_service import RunService
from app.synthesis.mock_provider import MockSynthesisProvider
from app.synthesis.service import SynthesisService
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool

runner = CliRunner()

SAMPLE_DATASET_PATH = "evals/datasets/research_quality_v1.json"


def _build_pgvector_run_components(runs_dir: Path) -> RunComponents:
    """Matches evals/datasets/research_quality_v1.json's single case exactly
    (same answer/claim/evidence/source text its `expected` block checks for)."""
    call_id = "c_search_1"
    evidence_id = f"ev_{call_id}_0"

    search_provider = MockSearchProvider(
        [
            SearchHit(
                url="https://fixture.example/pgvector-overview",
                title="pgvector overview",
                snippet="pgvector adds vector columns and similarity search directly to PostgreSQL.",
            )
        ]
    )
    tool_registry = ToolRegistry([WebSearchTool(search_provider)])  # type: ignore[list-item]

    llm_provider = MockLLMProvider(
        [
            LLMDecision(
                action=DecisionAction.TOOL_CALL,
                rationale="Search for the vector database.",
                tool_calls=[
                    ToolCall(
                        call_id=call_id, tool_name="web_search", arguments={"query": "vector database postgres"}
                    )
                ],
            ),
            LLMDecision(action=DecisionAction.SYNTHESIZE, rationale="Sufficient evidence found."),
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
        text="pgvector adds vector columns and similarity search directly to PostgreSQL",
        evidence_ids=[evidence_id],
    )
    answer = FinalAnswer(
        answer_text="pgvector adds vector columns and similarity search directly to PostgreSQL [1].",
        claims=[claim],
        citations=[Citation(claim=claim.text, evidence_ids=[evidence_id])],
        is_complete=True,
    )
    synthesis_service = SynthesisService(MockSynthesisProvider([answer]))

    run_service = RunService(FileRunRepository(runs_dir))
    orchestrator = ResearchOrchestrator(
        agent=agent,
        synthesis_service=synthesis_service,
        run_service=run_service,
        report_renderer=MarkdownReportRenderer(),
    )
    return RunComponents(
        orchestrator=orchestrator, run_service=run_service, report_renderer=MarkdownReportRenderer(), is_mock=True
    )


class TestEvaluateCLIOfflineEndToEnd:
    def test_evaluate_single_case_dataset_via_run_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runs_dir = tmp_path / "runs"
        monkeypatch.setattr(
            cli_main, "build_run_components", lambda *a, **k: _build_pgvector_run_components(runs_dir)
        )

        run_result = runner.invoke(
            cli_main.app, ["run", "Which vector database works directly in Postgres?", "--json"]
        )
        assert run_result.exit_code == 0, run_result.output
        run_id = json.loads(run_result.output)["run_id"]

        # `evaluate` must never build the Agent/LLM stack.
        def fail_if_run_components_built(*args: object, **kwargs: object) -> RunComponents:
            raise AssertionError("evaluate must never build run (Agent/LLM/tool) components")

        monkeypatch.setattr(cli_main, "build_run_components", fail_if_run_components_built)

        evaluate_result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", SAMPLE_DATASET_PATH, "--run", run_id, "--runs-dir", str(runs_dir)],
        )

        assert evaluate_result.exit_code == 0, evaluate_result.output
        assert "# Evaluation Report" in evaluate_result.output
        assert "Dataset: `research_quality`" in evaluate_result.output
        assert "Cases: `1`" in evaluate_result.output
        assert "Passed: `1`" in evaluate_result.output
        assert "Overall: PASS" in evaluate_result.output

        # A completely independent RunService reads the same persisted data.
        independent = RunService(FileRunRepository(runs_dir))
        assert independent.load_run(run_id).run_id == run_id

    def test_evaluate_multi_case_dataset_via_runs_map(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runs_dir = tmp_path / "runs"
        monkeypatch.setattr(
            cli_main, "build_run_components", lambda *a, **k: _build_pgvector_run_components(runs_dir)
        )

        first = runner.invoke(cli_main.app, ["run", "Which vector database works directly in Postgres?", "--json"])
        second = runner.invoke(cli_main.app, ["run", "Which vector database works directly in Postgres?", "--json"])
        assert first.exit_code == 0 and second.exit_code == 0
        run_id_1 = json.loads(first.output)["run_id"]
        run_id_2 = json.loads(second.output)["run_id"]
        assert run_id_1 != run_id_2

        dataset_path = tmp_path / "two_case_dataset.json"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "two_case",
                    "version": "1",
                    "name": "Two Case",
                    "cases": [
                        {"id": "case_a", "question": "Which vector database works directly in Postgres?"},
                        {"id": "case_b", "question": "Which vector database works directly in Postgres?"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        runs_map_path = tmp_path / "runs_map.json"
        runs_map_path.write_text(json.dumps({"case_a": run_id_1, "case_b": run_id_2}), encoding="utf-8")

        evaluate_result = runner.invoke(
            cli_main.app,
            [
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--runs-map",
                str(runs_map_path),
                "--runs-dir",
                str(runs_dir),
            ],
        )

        assert evaluate_result.exit_code == 0, evaluate_result.output
        assert "Cases: `2`" in evaluate_result.output
        assert "Passed: `2`" in evaluate_result.output
        assert "Overall: PASS" in evaluate_result.output

    def test_evaluate_baseline_vs_candidate_comparison(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runs_dir = tmp_path / "runs"
        monkeypatch.setattr(
            cli_main, "build_run_components", lambda *a, **k: _build_pgvector_run_components(runs_dir)
        )

        run_result = runner.invoke(
            cli_main.app, ["run", "Which vector database works directly in Postgres?", "--json"]
        )
        assert run_result.exit_code == 0
        run_id = json.loads(run_result.output)["run_id"]

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        for path, label in [(baseline_path, "build-100"), (candidate_path, "build-101")]:
            save_result = runner.invoke(
                cli_main.app,
                [
                    "evaluate",
                    "--dataset",
                    SAMPLE_DATASET_PATH,
                    "--run",
                    run_id,
                    "--runs-dir",
                    str(runs_dir),
                    "--json",
                    "--out",
                    str(path),
                    "--candidate-id",
                    label,
                ],
            )
            assert save_result.exit_code == 0, save_result.output
            assert path.is_file()

        # Comparison mode must never touch RunService/FileRunRepository or
        # build an Agent/LLM stack -- it only reads the two JSON files above.
        def fail_if_run_components_built(*args: object, **kwargs: object) -> RunComponents:
            raise AssertionError("comparison mode must never build run (Agent/LLM/tool) components")

        monkeypatch.setattr(cli_main, "build_run_components", fail_if_run_components_built)

        compare_result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert compare_result.exit_code == 0, compare_result.output
        assert "# Evaluation Comparison" in compare_result.output
        assert "Baseline: `build-100`" in compare_result.output
        assert "Candidate: `build-101`" in compare_result.output
        assert "Overall: PASS" in compare_result.output

        # JSON comparison output round-trips into EvaluationComparison.
        json_compare_result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path), "--json"],
        )
        assert json_compare_result.exit_code == 0
        comparison = EvaluationComparison.model_validate_json(json_compare_result.output)
        assert comparison.candidate.candidate_id == "build-101"
        assert comparison.overall_passed is True
