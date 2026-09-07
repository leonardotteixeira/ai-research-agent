import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import app.cli.main as cli_main
from app.cli.composition import EvaluationComponents, RunComponents
from app.evaluation.comparison import EvaluationComparator
from app.evaluation.datasets import EvaluationDatasetLoader
from app.evaluation.reports import EvaluationJSONRenderer, EvaluationMarkdownRenderer
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import EvaluationRunResult
from app.persistence.file_repository import FileRunRepository
from app.persistence.repository import RepositoryError
from app.services.run_service import RunService
from tests.unit.test_run_record import build_record

runner = CliRunner()


def _write_dataset(path: Path, case_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "sample",
                "version": "1",
                "name": "Sample",
                "cases": [{"id": case_id, "question": "What was observed?"} for case_id in case_ids],
            }
        ),
        encoding="utf-8",
    )


def _persist_record(runs_dir: Path, run_id: str = "run_1") -> None:
    record = build_record()
    if run_id != "run_1":
        assert record.research_result is not None
        record.run_id = run_id
        record.research_state.research_id = run_id
        record.research_result = record.research_result.model_copy(update={"research_id": run_id})
    RunService(FileRunRepository(runs_dir)).save_run(record)


class TestEvaluateCommandSingleCase:
    def test_success_with_run_flag_exit_code_0(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "# Evaluation Report" in result.output
        assert "Overall: PASS" in result.output

    def test_candidate_id_is_forwarded(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)

        result = runner.invoke(
            cli_main.app,
            [
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--run",
                "run_1",
                "--runs-dir",
                str(tmp_path),
                "--candidate-id",
                "nightly-build-42",
            ],
        )

        assert result.exit_code == 0
        assert "Candidate: `nightly-build-42`" in result.output

    def test_run_flag_rejected_for_multi_case_dataset(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1", "case_2"])

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 2
        assert "single-case dataset" in result.output
        assert "--runs-map" in result.output

    def test_neither_run_nor_runs_map_is_usage_error(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])

        result = runner.invoke(cli_main.app, ["evaluate", "--dataset", str(dataset_path), "--runs-dir", str(tmp_path)])

        assert result.exit_code == 2
        assert "exactly one of --run or --runs-map" in result.output

    def test_both_run_and_runs_map_is_usage_error(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        runs_map_path = tmp_path / "map.json"
        runs_map_path.write_text("{}", encoding="utf-8")

        result = runner.invoke(
            cli_main.app,
            [
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--run",
                "run_1",
                "--runs-map",
                str(runs_map_path),
                "--runs-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 2
        assert "exactly one of --run or --runs-map" in result.output

    def test_dataset_not_found_exit_code_2(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(tmp_path / "missing.json"), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 2
        assert "not found" in result.output

    def test_invalid_dataset_schema_exit_code_2(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "bad.json"
        dataset_path.write_text(json.dumps({"id": "x"}), encoding="utf-8")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 2

    def test_run_not_found_exit_code_2(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "does-not-exist", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 2
        assert "not found" in result.output


class TestEvaluateCommandMultiCase:
    def test_success_with_runs_map(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1", "case_2"])
        _persist_record(tmp_path, "run_1")
        _persist_record(tmp_path, "run_2")
        runs_map_path = tmp_path / "map.json"
        runs_map_path.write_text(json.dumps({"case_1": "run_1", "case_2": "run_2"}), encoding="utf-8")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--runs-map", str(runs_map_path), "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "Cases: `2`" in result.output

    def test_runs_map_not_found_exit_code_2(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1", "case_2"])

        result = runner.invoke(
            cli_main.app,
            [
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--runs-map",
                str(tmp_path / "missing_map.json"),
                "--runs-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 2
        assert "runs-map file not found" in result.output

    def test_runs_map_invalid_json_exit_code_2(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        runs_map_path = tmp_path / "map.json"
        runs_map_path.write_text("{not json", encoding="utf-8")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--runs-map", str(runs_map_path), "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 2
        assert "not valid JSON" in result.output

    def test_runs_map_not_flat_string_object_exit_code_2(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        runs_map_path = tmp_path / "map.json"
        runs_map_path.write_text(json.dumps({"case_1": 123}), encoding="utf-8")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--runs-map", str(runs_map_path), "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 2
        assert "flat JSON object" in result.output

    def test_runs_map_missing_case_exit_code_2(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1", "case_2"])
        _persist_record(tmp_path, "run_1")
        runs_map_path = tmp_path / "map.json"
        runs_map_path.write_text(json.dumps({"case_1": "run_1"}), encoding="utf-8")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--runs-map", str(runs_map_path), "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 2
        assert "case_2" in result.output

    def test_runs_map_extra_case_exit_code_2(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path, "run_1")
        _persist_record(tmp_path, "run_2")
        runs_map_path = tmp_path / "map.json"
        runs_map_path.write_text(json.dumps({"case_1": "run_1", "case_2": "run_2"}), encoding="utf-8")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--runs-map", str(runs_map_path), "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 2
        assert "case_2" in result.output


class TestEvaluateCommandInfrastructureAndUnexpectedErrors:
    def test_repository_error_exit_code_3(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])

        class FailingRunService:
            def load_run(self, run_id: str):
                raise RepositoryError("disk failure")

        components = EvaluationComponents(
            run_service=FailingRunService(),  # type: ignore[arg-type]
            dataset_loader=EvaluationDatasetLoader(),
            runner=EvaluationRunner(),
            markdown_renderer=EvaluationMarkdownRenderer(),
            json_renderer=EvaluationJSONRenderer(),
            comparator=EvaluationComparator(),
        )
        monkeypatch.setattr(cli_main, "build_evaluation_components", lambda *a, **k: components)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 3
        assert "disk failure" in result.output

    def test_unexpected_error_exit_code_4(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])

        class BoomRunService:
            def load_run(self, run_id: str):
                raise RuntimeError("boom")

        components = EvaluationComponents(
            run_service=BoomRunService(),  # type: ignore[arg-type]
            dataset_loader=EvaluationDatasetLoader(),
            runner=EvaluationRunner(),
            markdown_renderer=EvaluationMarkdownRenderer(),
            json_renderer=EvaluationJSONRenderer(),
            comparator=EvaluationComparator(),
        )
        monkeypatch.setattr(cli_main, "build_evaluation_components", lambda *a, **k: components)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 4
        assert "Traceback" in result.output

    def test_unexpected_error_during_dataset_loading_exit_code_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])

        class BoomDatasetLoader:
            def load_json(self, path: Path):
                raise RuntimeError("boom")

        components = EvaluationComponents(
            run_service=RunService(FileRunRepository(tmp_path)),
            dataset_loader=BoomDatasetLoader(),  # type: ignore[arg-type]
            runner=EvaluationRunner(),
            markdown_renderer=EvaluationMarkdownRenderer(),
            json_renderer=EvaluationJSONRenderer(),
            comparator=EvaluationComparator(),
        )
        monkeypatch.setattr(cli_main, "build_evaluation_components", lambda *a, **k: components)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 4
        assert "Traceback" in result.output

    def test_unexpected_error_during_evaluate_records_exit_code_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)

        class BoomRunner:
            def evaluate_records(self, *args: Any, **kwargs: Any):
                raise RuntimeError("boom")

        components = EvaluationComponents(
            run_service=RunService(FileRunRepository(tmp_path)),
            dataset_loader=EvaluationDatasetLoader(),
            runner=BoomRunner(),  # type: ignore[arg-type]
            markdown_renderer=EvaluationMarkdownRenderer(),
            json_renderer=EvaluationJSONRenderer(),
            comparator=EvaluationComparator(),
        )
        monkeypatch.setattr(cli_main, "build_evaluation_components", lambda *a, **k: components)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 4
        assert "Traceback" in result.output

    def test_unexpected_error_during_render_exit_code_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)

        class BoomRenderer:
            def render(self, result: Any) -> str:
                raise RuntimeError("boom")

        components = EvaluationComponents(
            run_service=RunService(FileRunRepository(tmp_path)),
            dataset_loader=EvaluationDatasetLoader(),
            runner=EvaluationRunner(),
            markdown_renderer=BoomRenderer(),  # type: ignore[arg-type]
            json_renderer=EvaluationJSONRenderer(),
            comparator=EvaluationComparator(),
        )
        monkeypatch.setattr(cli_main, "build_evaluation_components", lambda *a, **k: components)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 4
        assert "Traceback" in result.output

    def test_unexpected_error_during_json_render_exit_code_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)

        class BoomJSONRenderer:
            def render(self, result: Any) -> str:
                raise RuntimeError("boom")

        components = EvaluationComponents(
            run_service=RunService(FileRunRepository(tmp_path)),
            dataset_loader=EvaluationDatasetLoader(),
            runner=EvaluationRunner(),
            markdown_renderer=EvaluationMarkdownRenderer(),
            json_renderer=BoomJSONRenderer(),  # type: ignore[arg-type]
            comparator=EvaluationComparator(),
        )
        monkeypatch.setattr(cli_main, "build_evaluation_components", lambda *a, **k: components)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path), "--json"],
        )

        assert result.exit_code == 4
        assert "Traceback" in result.output


class TestEvaluateOutputs:
    def test_json_flag_prints_valid_json_and_no_human_text(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path), "--json"],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["dataset_id"] == "sample"
        assert payload["overall_passed"] is True
        assert "# Evaluation Report" not in result.output

    def test_default_output_is_markdown(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 0
        assert "# Evaluation Report" in result.output

    def test_out_writes_markdown_to_file(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)
        out_file = tmp_path / "report.md"

        result = runner.invoke(
            cli_main.app,
            [
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--run",
                "run_1",
                "--runs-dir",
                str(tmp_path),
                "--out",
                str(out_file),
            ],
        )

        assert result.exit_code == 0
        assert out_file.is_file()
        assert "# Evaluation Report" in out_file.read_text(encoding="utf-8")
        assert f"Evaluation report written to {out_file}" in result.output
        assert "# Evaluation Report" not in result.output

    def test_out_writes_json_to_file_when_json_flag_set(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)
        out_file = tmp_path / "report.json"

        result = runner.invoke(
            cli_main.app,
            [
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--run",
                "run_1",
                "--runs-dir",
                str(tmp_path),
                "--json",
                "--out",
                str(out_file),
            ],
        )

        assert result.exit_code == 0
        assert out_file.is_file()
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert payload["overall_passed"] is True
        assert "{" not in result.output  # stdout only carries the confirmation message

    def test_json_output_round_trips_evaluation_run_result(self, tmp_path: Path) -> None:
        from app.evaluation.schemas import EvaluationRunResult

        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path), "--json"],
        )

        assert result.exit_code == 0
        parsed = EvaluationRunResult.model_validate_json(result.output)
        assert parsed.dataset_id == "sample"


class TestEvaluateNeverExecutesResearch:
    def test_never_builds_run_components_dataset_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """evaluate must never touch Agent/LLM/tools -- assert build_run_components is never called."""
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        _persist_record(tmp_path)

        def fail_if_called(*args: Any, **kwargs: Any) -> RunComponents:
            raise AssertionError("evaluate must never call build_run_components")

        monkeypatch.setattr(cli_main, "build_run_components", fail_if_called)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--dataset", str(dataset_path), "--run", "run_1", "--runs-dir", str(tmp_path)],
        )

        assert result.exit_code == 0


def _save_evaluation_result(path: Path, candidate_id: str, overall_passed: bool = True, pass_rate: float = 1.0) -> None:
    from app.evaluation.schemas import MetricResult

    result = EvaluationRunResult(
        evaluation_id="research_quality_v1",
        dataset_id="research_quality",
        dataset_version="1",
        candidate_id=candidate_id,
        metrics=[MetricResult(name="pass_rate", value=pass_rate, unit="ratio")],
        overall_passed=overall_passed,
    )
    path.write_text(result.model_dump_json(), encoding="utf-8")


class TestEvaluateComparisonMode:
    def test_success_prints_markdown_comparison(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "baseline-build")
        _save_evaluation_result(candidate_path, "candidate-build")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 0, result.output
        assert "# Evaluation Comparison" in result.output
        assert "Baseline: `baseline-build`" in result.output
        assert "Candidate: `candidate-build`" in result.output
        assert "Overall: PASS" in result.output

    def test_json_flag_prints_valid_json_comparison(self, tmp_path: Path) -> None:
        from app.evaluation.schemas import EvaluationComparison

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "baseline-build")
        _save_evaluation_result(candidate_path, "candidate-build")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path), "--json"],
        )

        assert result.exit_code == 0
        parsed = EvaluationComparison.model_validate_json(result.output)
        assert parsed.candidate.candidate_id == "candidate-build"
        assert "# Evaluation Comparison" not in result.output

    def test_out_writes_comparison_markdown_to_file(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "baseline-build")
        _save_evaluation_result(candidate_path, "candidate-build")
        out_file = tmp_path / "comparison.md"

        result = runner.invoke(
            cli_main.app,
            [
                "evaluate",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--out",
                str(out_file),
            ],
        )

        assert result.exit_code == 0
        assert out_file.is_file()
        assert "# Evaluation Comparison" in out_file.read_text(encoding="utf-8")
        assert f"Evaluation comparison written to {out_file}" in result.output

    def test_candidate_overall_passed_false_propagates_exit_code_1(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "baseline-build", overall_passed=True)
        _save_evaluation_result(candidate_path, "candidate-build", overall_passed=False, pass_rate=0.5)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 1
        assert "Overall: FAIL" in result.output

    def test_baseline_and_dataset_together_is_usage_error(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.json"
        _write_dataset(dataset_path, ["case_1"])
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "b")
        _save_evaluation_result(candidate_path, "c")

        result = runner.invoke(
            cli_main.app,
            [
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
            ],
        )

        assert result.exit_code == 2
        assert "cannot be combined" in result.output

    def test_baseline_without_candidate_is_usage_error(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline.json"
        _save_evaluation_result(baseline_path, "b")

        result = runner.invoke(cli_main.app, ["evaluate", "--baseline", str(baseline_path)])

        assert result.exit_code == 2
        assert "must be provided together" in result.output

    def test_candidate_without_baseline_is_usage_error(self, tmp_path: Path) -> None:
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(candidate_path, "c")

        result = runner.invoke(cli_main.app, ["evaluate", "--candidate", str(candidate_path)])

        assert result.exit_code == 2
        assert "must be provided together" in result.output

    def test_no_mode_flags_is_usage_error(self) -> None:
        result = runner.invoke(cli_main.app, ["evaluate"])

        assert result.exit_code == 2
        assert "provide either --dataset" in result.output

    def test_baseline_file_not_found_exit_code_2(self, tmp_path: Path) -> None:
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(candidate_path, "c")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(tmp_path / "missing.json"), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 2
        assert "not found" in result.output

    def test_candidate_invalid_json_exit_code_2(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline.json"
        _save_evaluation_result(baseline_path, "b")
        candidate_path = tmp_path / "candidate.json"
        candidate_path.write_text("{not json", encoding="utf-8")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 2
        assert "not valid JSON" in result.output

    def test_candidate_wrong_schema_exit_code_2(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline.json"
        _save_evaluation_result(baseline_path, "b")
        candidate_path = tmp_path / "candidate.json"
        candidate_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 2
        assert "does not match EvaluationRunResult schema" in result.output

    def test_comparison_mode_never_calls_run_service_load_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "b")
        _save_evaluation_result(candidate_path, "c")

        class FailIfCalledRunService:
            def load_run(self, run_id: str):
                raise AssertionError("comparison mode must never call RunService.load_run")

        components = EvaluationComponents(
            run_service=FailIfCalledRunService(),  # type: ignore[arg-type]
            dataset_loader=EvaluationDatasetLoader(),
            runner=EvaluationRunner(),
            markdown_renderer=EvaluationMarkdownRenderer(),
            json_renderer=EvaluationJSONRenderer(),
            comparator=EvaluationComparator(),
        )
        monkeypatch.setattr(cli_main, "build_evaluation_components", lambda *a, **k: components)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 0, result.output

    def test_comparison_mode_never_builds_run_components(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "b")
        _save_evaluation_result(candidate_path, "c")

        def fail_if_called(*args: Any, **kwargs: Any) -> RunComponents:
            raise AssertionError("comparison mode must never call build_run_components")

        monkeypatch.setattr(cli_main, "build_run_components", fail_if_called)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 0

    def test_unexpected_error_building_components_exit_code_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "b")
        _save_evaluation_result(candidate_path, "c")

        def boom(*args: Any, **kwargs: Any) -> EvaluationComponents:
            raise RuntimeError("boom")

        monkeypatch.setattr(cli_main, "build_evaluation_components", boom)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 4
        assert "Traceback" in result.output

    def test_unexpected_error_during_compare_exit_code_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "b")
        _save_evaluation_result(candidate_path, "c")

        class BoomComparator:
            def compare(self, *args: Any, **kwargs: Any):
                raise RuntimeError("boom")

        components = EvaluationComponents(
            run_service=RunService(FileRunRepository(tmp_path)),
            dataset_loader=EvaluationDatasetLoader(),
            runner=EvaluationRunner(),
            markdown_renderer=EvaluationMarkdownRenderer(),
            json_renderer=EvaluationJSONRenderer(),
            comparator=BoomComparator(),  # type: ignore[arg-type]
        )
        monkeypatch.setattr(cli_main, "build_evaluation_components", lambda *a, **k: components)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 4
        assert "Traceback" in result.output

    def test_unexpected_error_during_comparison_render_exit_code_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        _save_evaluation_result(baseline_path, "b")
        _save_evaluation_result(candidate_path, "c")

        class BoomRenderer:
            def render_comparison(self, comparison: Any) -> str:
                raise RuntimeError("boom")

        components = EvaluationComponents(
            run_service=RunService(FileRunRepository(tmp_path)),
            dataset_loader=EvaluationDatasetLoader(),
            runner=EvaluationRunner(),
            markdown_renderer=BoomRenderer(),  # type: ignore[arg-type]
            json_renderer=EvaluationJSONRenderer(),
            comparator=EvaluationComparator(),
        )
        monkeypatch.setattr(cli_main, "build_evaluation_components", lambda *a, **k: components)

        result = runner.invoke(
            cli_main.app,
            ["evaluate", "--baseline", str(baseline_path), "--candidate", str(candidate_path)],
        )

        assert result.exit_code == 4
        assert "Traceback" in result.output
