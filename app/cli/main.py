"""Typer CLI entry point.

Thin layer: parses arguments into existing domain models (ResearchRequest),
delegates to the composition root (app/cli/composition.py) and existing
services (ResearchOrchestrator, RunService, MarkdownReportRenderer), and
formats the result (app/cli/formatting.py). No Agent, tool, synthesis,
persistence, or Markdown-rendering logic lives here.

Run with:
    python -m app.cli run "question"
    python -m app.cli runs
    python -m app.cli show <run_id>
    python -m app.cli report <run_id>
    python -m app.cli replay <run_id>
    python -m app.cli resume <run_id>
    python -m app.cli academic-report <run_id>

Exit codes:
    0 - complete success
    1 - the run finished but OrchestratorResult.success is False
        (partial result / controlled failure -- e.g. max_steps, tool_error);
        for `replay`, a divergent (non-equivalent) reproduction
    2 - usage error: invalid arguments, unknown run_id, a run in the wrong
        state for the requested operation (e.g. `replay` on an unfinished
        run, `resume` on an already-finished one), or live providers
        requested without the required configuration/API keys
    3 - infrastructure error surfaced from RunService/FileRunRepository or
        MarkdownReportRenderer (never swallowed, per app/services/orchestrator.py)
    4 - unexpected error (full traceback is still printed, never hidden)
"""

import asyncio
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

import typer
from pydantic import ValidationError

from app.academic.builder import build_academic_report
from app.academic.schemas import AcademicMetadata
from app.cli.composition import (
    DEFAULT_RUNS_DIR,
    MissingProviderCredentialsError,
    MockProvidersNotSupportedError,
    build_academic_report_components,
    build_evaluation_components,
    build_read_only_components,
    build_replay_components,
    build_resume_components,
    build_run_components,
)
from app.cli.formatting import (
    format_replay,
    format_replay_comparison,
    format_run_human,
    format_run_json,
    format_runs_table,
    format_show_human,
    format_show_json,
    summarize_runs_json,
)
from app.core.exceptions import ReportRenderingError
from app.evaluation.datasets import EvaluationDatasetError
from app.evaluation.runner import MissingRunRecordError, UnknownCaseRunRecordError
from app.evaluation.schemas import EvaluationDataset, EvaluationRunResult
from app.persistence.repository import (
    ConcurrentResumeError,
    InvalidRunIdError,
    RepositoryError,
    RunAlreadyFinalizedError,
    RunNotFinalizedError,
    RunNotFoundError,
)
from app.schemas.state import ResearchRequest

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Thin CLI over the AI Research Agent's ResearchOrchestrator and RunService.",
)


def _parse_allowed_tools(value: str | None) -> list[str] | None:
    if value is None:
        return None
    tools = [item.strip() for item in value.split(",") if item.strip()]
    return tools or None


def _usage_error(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=2)


def _infra_error(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=3)


def _unexpected_error() -> typer.Exit:
    traceback.print_exc()
    return typer.Exit(code=4)


@app.command("run")
def run_command(
    question: str = typer.Argument(..., help="The research question to investigate."),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Override ExecutionPolicy.max_steps."),
    max_tool_calls: int | None = typer.Option(
        None, "--max-tool-calls", help="Override ExecutionPolicy.max_tool_calls."
    ),
    max_same_tool_calls: int | None = typer.Option(
        None, "--max-same-tool-calls", help="Override ExecutionPolicy.max_same_tool_calls."
    ),
    timeout: float | None = typer.Option(
        None, "--timeout", help="Override ExecutionPolicy.global_timeout_seconds."
    ),
    allowed_tools: str | None = typer.Option(
        None, "--allowed-tools", help="Comma-separated list of allowed tool names."
    ),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", help="Directory runs are persisted under."),
    json_output: bool = typer.Option(False, "--json", help="Print OrchestratorResult as JSON instead of text."),
) -> None:
    """Execute new research end-to-end via ResearchOrchestrator."""
    try:
        request = ResearchRequest(
            question=question,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            max_same_tool_calls=max_same_tool_calls,
            total_timeout_seconds=timeout,
            allowed_tools=_parse_allowed_tools(allowed_tools),
        )
    except ValidationError as exc:
        raise _usage_error(str(exc)) from exc

    try:
        components = build_run_components(runs_dir)
    except (MockProvidersNotSupportedError, MissingProviderCredentialsError) as exc:
        raise _usage_error(str(exc)) from exc

    try:
        result = asyncio.run(components.orchestrator.run(request))
    except ReportRenderingError as exc:
        raise _infra_error(str(exc)) from exc
    except RepositoryError as exc:
        raise _infra_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    if json_output:
        typer.echo(format_run_json(result))
    else:
        typer.echo(format_run_human(result, is_mock=components.is_mock))

    raise typer.Exit(code=0 if result.success else 1)


@app.command("runs")
def runs_command(
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", help="Directory runs are persisted under."),
    json_output: bool = typer.Option(False, "--json", help="Print a summarized JSON list instead of a table."),
) -> None:
    """List persisted research runs."""
    try:
        components = build_read_only_components(runs_dir)
        records = components.run_service.list_runs()
    except RepositoryError as exc:
        raise _infra_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    if json_output:
        typer.echo(json.dumps(summarize_runs_json(records), indent=2))
    else:
        typer.echo(format_runs_table(records))


@app.command("show")
def show_command(
    run_id: str = typer.Argument(..., help="The run ID to display."),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", help="Directory runs are persisted under."),
    json_output: bool = typer.Option(False, "--json", help="Print the full RunRecord as JSON."),
) -> None:
    """Show a summary of one persisted run."""
    try:
        components = build_read_only_components(runs_dir)
        record = components.run_service.load_run(run_id)
    except (RunNotFoundError, InvalidRunIdError) as exc:
        raise _usage_error(str(exc)) from exc
    except RepositoryError as exc:
        raise _infra_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    typer.echo(format_show_json(record) if json_output else format_show_human(record))


@app.command("report")
def report_command(
    run_id: str = typer.Argument(..., help="The run ID to render a Markdown report for."),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", help="Directory runs are persisted under."),
    out: Path | None = typer.Option(None, "--out", help="Write the Markdown report to this file instead of stdout."),
) -> None:
    """Render the Markdown report for one persisted run."""
    try:
        components = build_read_only_components(runs_dir)
        record = components.run_service.load_run(run_id)
        markdown = components.report_renderer.render(record)
    except (RunNotFoundError, InvalidRunIdError) as exc:
        raise _usage_error(str(exc)) from exc
    except ReportRenderingError as exc:
        raise _infra_error(str(exc)) from exc
    except RepositoryError as exc:
        raise _infra_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    if out is not None:
        out.write_text(markdown, encoding="utf-8")
        typer.echo(f"Report written to {out}")
    else:
        typer.echo(markdown)


@app.command("replay")
def replay_command(
    run_id: str = typer.Argument(..., help="The run ID to replay."),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", help="Directory runs are persisted under."),
) -> None:
    """Narrate a persisted run, then reproduce its execution offline and
    verify it against what was recorded.

    Two things happen, both fully offline and read-only:
      1. The persisted RunRecord's decisions/tool calls/evidence/answer are
         narrated exactly as before (no re-execution involved in this part).
      2. The recorded decisions and tool results are fed back through the
         real Agent/SynthesisService (app/replay/) -- never Agent,
         LLMProvider, ToolRegistry, tools, or SynthesisService talking to a
         real network/LLM -- and the freshly-reproduced execution is
         compared to the persisted one. This proves the persisted run is
         internally consistent with its own recorded trace; it does not
         prove that a real LLM would make the same decisions again.
    """
    try:
        components = build_replay_components(runs_dir)
        record = components.run_service.load_run(run_id)
    except (RunNotFoundError, InvalidRunIdError) as exc:
        raise _usage_error(str(exc)) from exc
    except RepositoryError as exc:
        raise _infra_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    try:
        comparison = asyncio.run(components.replay_service.replay(record))
    except RunNotFinalizedError as exc:
        raise _usage_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    typer.echo(format_replay(record))
    typer.echo(format_replay_comparison(comparison))

    raise typer.Exit(code=0 if comparison.equivalent else 1)


@app.command("resume")
def resume_command(
    run_id: str = typer.Argument(..., help="The run ID to resume."),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", help="Directory runs are persisted under."),
    json_output: bool = typer.Option(False, "--json", help="Print OrchestratorResult as JSON instead of text."),
) -> None:
    """Continue an interrupted (checkpointed but unfinished) run using real
    providers/tools.

    Unlike `replay` (100% offline, read-only, never re-executes anything),
    `resume` can call a real LLM and real tools again -- exactly like `run`
    does -- to finish a run whose Agent loop or synthesis never completed
    before a crash. It always uses the run's own persisted run_id,
    question, and ExecutionPolicy; it never restarts or re-runs a run that
    already has a `research_result` (already finished).
    """
    try:
        components = build_resume_components(runs_dir)
    except (MockProvidersNotSupportedError, MissingProviderCredentialsError) as exc:
        raise _usage_error(str(exc)) from exc

    try:
        result = asyncio.run(components.resume_service.resume(run_id))
    except (RunNotFoundError, InvalidRunIdError, RunAlreadyFinalizedError, ConcurrentResumeError) as exc:
        raise _usage_error(str(exc)) from exc
    except ReportRenderingError as exc:
        raise _infra_error(str(exc)) from exc
    except RepositoryError as exc:
        raise _infra_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    if json_output:
        typer.echo(format_run_json(result))
    else:
        typer.echo(format_run_human(result, is_mock=False))

    raise typer.Exit(code=0 if result.success else 1)


@app.command("academic-report")
def academic_report_command(
    run_id: str = typer.Argument(..., help="The run ID to generate an ABNT-oriented academic PDF for."),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", help="Directory runs are persisted under."),
    out: Path | None = typer.Option(
        None, "--out", help="Write the PDF here instead of <runs-dir>/<run_id>/academic_report.pdf."
    ),
    title: str | None = typer.Option(None, "--title", help="Override the document title (defaults to the question)."),
    author: str = typer.Option("Leonardo Teixeira", "--author"),
    registration: str | None = typer.Option("245602", "--registration", help="Student registration number (RA)."),
    institution: str = typer.Option("Universidade Estadual de Campinas – UNICAMP", "--institution"),
    unit: str | None = typer.Option("Faculdade de Engenharia Agrícola – FEAGRI", "--unit"),
    city: str = typer.Option("Campinas – SP", "--city"),
    year: int | None = typer.Option(None, "--year", help="Defaults to the current year."),
    course: str | None = typer.Option(None, "--course"),
    advisor: str | None = typer.Option(None, "--advisor"),
    force: bool = typer.Option(False, "--force", help="Overwrite the output file if it already exists."),
) -> None:
    """Render an ABNT-oriented academic PDF (capa, folha de rosto, resumo,
    sumário, seções, referências) from an already-completed, persisted
    run -- never re-executes research, never calls an LLM/tool/network.
    See README "Academic Report Generation" for exactly which ABNT rules
    this does and doesn't implement.
    """
    try:
        components = build_academic_report_components(runs_dir)
        record = components.run_service.load_run(run_id)
    except (RunNotFoundError, InvalidRunIdError) as exc:
        raise _usage_error(str(exc)) from exc
    except RepositoryError as exc:
        raise _infra_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    if record.research_result is None:
        raise _usage_error(f"run {run_id!r} is not finalized; cannot generate an academic report from it")

    out_path = out or (runs_dir / record.run_id / "academic_report.pdf")
    if out_path.exists() and not force:
        raise _usage_error(f"{out_path} already exists; pass --force to overwrite")

    metadata = AcademicMetadata(
        author=author,
        registration=registration,
        institution=institution,
        unit=unit,
        city=city,
        year=year or datetime.now(UTC).year,
        course=course,
        advisor=advisor,
    )
    try:
        report = build_academic_report(record, metadata, title=title)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        components.renderer.render(report, out_path)
    except Exception:
        raise _unexpected_error() from None

    typer.echo(f"Academic report written to {out_path}")


@app.command("evaluate")
def evaluate_command(
    dataset: Path | None = typer.Option(None, "--dataset", help="Path to an EvaluationDataset JSON file."),
    run_id: str | None = typer.Option(
        None, "--run", help="RunRecord id to evaluate. Only valid when the dataset has exactly one case."
    ),
    runs_map: Path | None = typer.Option(
        None,
        "--runs-map",
        help="JSON file mapping case_id -> run_id. Required for datasets with more than one case.",
    ),
    baseline: Path | None = typer.Option(
        None, "--baseline", help="Path to a previously-saved EvaluationRunResult JSON file (baseline)."
    ),
    candidate: Path | None = typer.Option(
        None, "--candidate", help="Path to a previously-saved EvaluationRunResult JSON file (candidate)."
    ),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", help="Directory runs are persisted under."),
    candidate_id: str = typer.Option(
        "candidate",
        "--candidate-id",
        help="Label identifying this evaluation run in the report (dataset mode only).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the result as JSON instead of Markdown."),
    out: Path | None = typer.Option(
        None, "--out", help="Write the rendered report to this file instead of stdout."
    ),
) -> None:
    """Evaluate persisted RunRecords against an EvaluationDataset, or compare
    two previously-saved EvaluationRunResult JSON files.

    Two mutually exclusive modes:
      - Dataset mode: --dataset with --run (single-case dataset) or --runs-map
        (multi-case). Loads already-persisted RunRecords via RunService and
        runs the deterministic evaluation core from app.evaluation.
      - Comparison mode: --baseline and --candidate, each a path to a
        previously-saved EvaluationRunResult JSON file (e.g. produced by a
        prior `evaluate --json --out`). Delegates to EvaluationComparator.

    Fully offline in both modes -- comparison mode reads two plain JSON
    files from disk and never calls RunService/FileRunRepository (no
    RunRecord is ever loaded or saved). Neither mode ever invokes Agent,
    LLMProvider, ToolRegistry, tools, or SynthesisService, or generates a
    new RunRecord.
    """
    comparison_requested = baseline is not None or candidate is not None
    dataset_requested = dataset is not None or run_id is not None or runs_map is not None

    if comparison_requested and dataset_requested:
        raise _usage_error("--baseline/--candidate cannot be combined with --dataset/--run/--runs-map")
    if not comparison_requested and not dataset_requested:
        raise _usage_error("provide either --dataset (with --run or --runs-map) or both --baseline and --candidate")

    if comparison_requested:
        if baseline is None or candidate is None:
            raise _usage_error("--baseline and --candidate must be provided together")
        # Defensive: _run_comparison_mode always terminates via typer.Exit
        # (success or error), but this explicit return guards against
        # falling through to dataset mode with dataset=None if that
        # invariant is ever broken by a future change.
        _run_comparison_mode(baseline, candidate, runs_dir, json_output, out)
        return

    assert dataset is not None
    _run_dataset_mode(dataset, run_id, runs_map, runs_dir, candidate_id, json_output, out)


def _run_dataset_mode(
    dataset: Path,
    run_id: str | None,
    runs_map: Path | None,
    runs_dir: Path,
    candidate_id: str,
    json_output: bool,
    out: Path | None,
) -> None:
    if (run_id is None) == (runs_map is None):
        raise _usage_error("exactly one of --run or --runs-map must be provided")

    try:
        components = build_evaluation_components(runs_dir)
        evaluation_dataset = components.dataset_loader.load_json(dataset)
    except EvaluationDatasetError as exc:
        raise _usage_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    try:
        case_run_ids = _resolve_case_run_ids(evaluation_dataset, run_id, runs_map)
        records_by_case_id = {
            case_id: components.run_service.load_run(mapped_run_id)
            for case_id, mapped_run_id in case_run_ids.items()
        }
    except ValueError as exc:
        raise _usage_error(str(exc)) from exc
    except (RunNotFoundError, InvalidRunIdError) as exc:
        raise _usage_error(str(exc)) from exc
    except RepositoryError as exc:
        raise _infra_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    try:
        result = components.runner.evaluate_records(
            evaluation_dataset,
            records_by_case_id,
            evaluation_id=f"{evaluation_dataset.id}_v{evaluation_dataset.version}",
            candidate_id=candidate_id,
        )
    except (MissingRunRecordError, UnknownCaseRunRecordError) as exc:
        raise _usage_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    try:
        rendered = (
            components.json_renderer.render(result) if json_output else components.markdown_renderer.render(result)
        )
    except Exception:
        raise _unexpected_error() from None

    _emit(rendered, out, "Evaluation report written to")
    raise typer.Exit(code=0 if result.overall_passed else 1)


def _run_comparison_mode(
    baseline: Path,
    candidate: Path,
    runs_dir: Path,
    json_output: bool,
    out: Path | None,
) -> None:
    try:
        components = build_evaluation_components(runs_dir)
        baseline_result = _load_evaluation_run_result(baseline)
        candidate_result = _load_evaluation_run_result(candidate)
    except ValueError as exc:
        raise _usage_error(str(exc)) from exc
    except Exception:
        raise _unexpected_error() from None

    try:
        comparison = components.comparator.compare(baseline_result, candidate_result)
    except Exception:
        raise _unexpected_error() from None

    try:
        rendered = (
            components.json_renderer.render(comparison)
            if json_output
            else components.markdown_renderer.render_comparison(comparison)
        )
    except Exception:
        raise _unexpected_error() from None

    _emit(rendered, out, "Evaluation comparison written to")
    raise typer.Exit(code=0 if comparison.overall_passed else 1)


def _emit(rendered: str, out: Path | None, written_message_prefix: str) -> None:
    if out is not None:
        out.write_text(rendered, encoding="utf-8")
        typer.echo(f"{written_message_prefix} {out}")
    else:
        typer.echo(rendered)


def _resolve_case_run_ids(
    dataset: EvaluationDataset, run_id: str | None, runs_map_path: Path | None
) -> dict[str, str]:
    if run_id is not None:
        if len(dataset.cases) != 1:
            raise ValueError(
                f"--run requires a single-case dataset; {dataset.id!r} has {len(dataset.cases)} cases. "
                "Use --runs-map instead."
            )
        return {dataset.cases[0].id: run_id}
    assert runs_map_path is not None
    return _load_runs_map(runs_map_path)


def _load_runs_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"runs-map file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"runs-map file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError(f"runs-map must be a flat JSON object of case_id -> run_id strings: {path}")
    return payload


def _load_evaluation_run_result(path: Path) -> EvaluationRunResult:
    if not path.is_file():
        raise ValueError(f"evaluation result file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"evaluation result file is not valid JSON: {path}") from exc
    try:
        return EvaluationRunResult.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"evaluation result file does not match EvaluationRunResult schema: {path}") from exc


if __name__ == "__main__":
    app()
