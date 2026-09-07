import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import app.cli.main as cli_main
from app.agent.agent import Agent
from app.cli.composition import (
    MissingProviderCredentialsError,
    MockProvidersNotSupportedError,
    ReadOnlyComponents,
    ReplayComponents,
    ResumeComponents,
    RunComponents,
    build_read_only_components,
)
from app.core.exceptions import ReportRenderingError
from app.evidence.pipeline import EvidencePipeline
from app.persistence.file_repository import FileRunRepository
from app.persistence.repository import (
    RepositoryError,
    RunAlreadyFinalizedError,
    RunNotFoundError,
)
from app.policies.execution import ExecutionPolicy
from app.providers.mock_llm import MockLLMProvider
from app.providers.search import MockSearchProvider, SearchHit
from app.replay.service import ReplayService
from app.reports.markdown import MarkdownReportRenderer
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.run import RunRecord
from app.schemas.state import ResearchRequest, ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall
from app.services.orchestrator import OrchestratorResult, apply_synthesis_if_requested
from app.services.run_service import RunService
from app.synthesis.mock_provider import MockSynthesisProvider
from app.synthesis.service import SynthesisService
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool

runner = CliRunner()


def _timestamp() -> datetime:
    return datetime.now(UTC)


def _persist_sample_run(runs_dir: Path, run_id: str = "sample_run") -> OrchestratorResult:
    source = Source(
        source_id="src_1", url="https://example.com/a", title="A", tool_call_id="c1", timestamp=_timestamp()
    )
    evidence = Evidence(
        evidence_id="ev_1", content="Evidence text", source_id="src_1", tool_call_id="c1", timestamp=_timestamp()
    )
    claim = Claim(claim_id="cl_1", text="A claim", evidence_ids=["ev_1"])
    answer = FinalAnswer(
        answer_text="A claim [1].",
        claims=[claim],
        citations=[Citation(claim="A claim", evidence_ids=["ev_1"])],
        is_complete=True,
    )
    state = ResearchState(
        research_id=run_id,
        original_question="Sample question?",
        created_at=_timestamp(),
        current_step=1,
        decisions=[LLMDecision(action=DecisionAction.SYNTHESIZE, rationale="synth")],
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        final_answer=answer,
        termination_reason=TerminationReason.FINISHED,
    )
    result = ResearchResult.from_state(state)
    run_service = RunService(FileRunRepository(runs_dir))
    record = run_service.build_run(state, ExecutionPolicy(), result)
    run_service.save_run(record)
    return OrchestratorResult(
        run_id=run_id,
        result=result,
        record=record,
        report_markdown=MarkdownReportRenderer().render(record),
        termination_reason=TerminationReason.FINISHED,
        success=True,
        errors=[],
    )


class FakeOrchestrator:
    def __init__(self, outcome: OrchestratorResult | Exception) -> None:
        self._outcome = outcome
        self.calls: list[ResearchRequest] = []

    async def run(self, request: ResearchRequest) -> OrchestratorResult:
        self.calls.append(request)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _FailingRunService:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def list_runs(self) -> list[RunRecord]:
        raise self._exc

    def load_run(self, run_id: str) -> RunRecord:
        raise self._exc


def _fake_read_only_components(exc: Exception) -> ReadOnlyComponents:
    return ReadOnlyComponents(
        run_service=_FailingRunService(exc),  # type: ignore[arg-type]
        report_renderer=MarkdownReportRenderer(),
    )


def _fake_replay_components(exc: Exception) -> ReplayComponents:
    return ReplayComponents(
        run_service=_FailingRunService(exc),  # type: ignore[arg-type]
        replay_service=ReplayService(),
    )


class FakeResumeService:
    def __init__(self, outcome: OrchestratorResult | Exception) -> None:
        self._outcome = outcome
        self.calls: list[str] = []

    async def resume(self, run_id: str) -> OrchestratorResult:
        self.calls.append(run_id)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _fake_resume_components(outcome: OrchestratorResult | Exception) -> ResumeComponents:
    return ResumeComponents(
        run_service=RunService(FileRunRepository(Path("unused-for-resume-command"))),
        resume_service=FakeResumeService(outcome),  # type: ignore[arg-type]
    )


async def _persist_replayable_sample_run(runs_dir: Path, run_id: str = "sample_run") -> None:
    """Unlike `_persist_sample_run` (hand-built sources/evidence detached
    from any real tool call -- fine for tests that never re-execute the
    run, but not internally consistent with what Agent+EvidencePipeline
    would actually derive), this builds a RunRecord via the real
    Agent/EvidencePipeline/SynthesisService flow, so `replay` reproduces it
    with equivalent=True instead of flagging a spurious divergence.
    """
    call_id = "c1"
    search_provider = MockSearchProvider(
        [SearchHit(url="https://example.com/a", title="A", snippet="Evidence text")]
    )
    tool_registry = ToolRegistry([WebSearchTool(search_provider)])  # type: ignore[list-item]
    llm_provider = MockLLMProvider(
        [
            LLMDecision(
                action=DecisionAction.TOOL_CALL,
                tool_calls=[ToolCall(call_id=call_id, tool_name="web_search", arguments={"query": "a"})],
            ),
            LLMDecision(action=DecisionAction.SYNTHESIZE, rationale="synth"),
        ]
    )
    policy = ExecutionPolicy()
    agent = Agent(
        llm_provider=llm_provider, tool_registry=tool_registry, policy=policy, evidence_pipeline=EvidencePipeline()
    )
    await agent.run(ResearchRequest(question="Sample question?"))
    state = agent.last_state
    assert state is not None
    evidence_id = f"ev_{call_id}_0"
    claim = Claim(claim_id="cl_1", text="A claim", evidence_ids=[evidence_id])
    answer = FinalAnswer(
        answer_text="A claim [1].",
        claims=[claim],
        citations=[Citation(claim="A claim", evidence_ids=[evidence_id])],
        is_complete=True,
    )
    await apply_synthesis_if_requested(state, SynthesisService(MockSynthesisProvider([answer])))

    state.research_id = run_id
    result = ResearchResult.from_state(state)
    run_service = RunService(FileRunRepository(runs_dir))
    record = run_service.build_run(state, policy, result)
    run_service.save_run(record)


def _fake_run_components(outcome: OrchestratorResult | Exception, is_mock: bool = False) -> RunComponents:
    orchestrator = FakeOrchestrator(outcome)
    return RunComponents(
        orchestrator=orchestrator,  # type: ignore[arg-type]
        run_service=RunService(FileRunRepository(Path("unused-for-run-command"))),
        report_renderer=MarkdownReportRenderer(),
        is_mock=is_mock,
    )


class TestRunCommand:
    def test_refuses_mock_mode_with_exit_code_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_mock_error(*args: Any, **kwargs: Any) -> RunComponents:
            raise MockProvidersNotSupportedError("mock mode not supported")

        monkeypatch.setattr(cli_main, "build_run_components", raise_mock_error)
        result = runner.invoke(cli_main.app, ["run", "question"])
        assert result.exit_code == 2
        assert "mock mode not supported" in result.output

    def test_refuses_missing_credentials_with_exit_code_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_missing_creds(*args: Any, **kwargs: Any) -> RunComponents:
            raise MissingProviderCredentialsError("OPENAI_API_KEY missing")

        monkeypatch.setattr(cli_main, "build_run_components", raise_missing_creds)
        result = runner.invoke(cli_main.app, ["run", "question"])
        assert result.exit_code == 2
        assert "OPENAI_API_KEY" in result.output

    def test_successful_run_exit_code_0_human(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        outcome = _persist_sample_run(tmp_path)
        monkeypatch.setattr(
            cli_main, "build_run_components", lambda *a, **k: _fake_run_components(outcome, is_mock=False)
        )
        result = runner.invoke(cli_main.app, ["run", "Sample question?"])
        assert result.exit_code == 0
        assert "Mode: LIVE" in result.output
        assert "Success: True" in result.output
        assert "Run ID: sample_run" in result.output

    def test_successful_run_json_is_valid(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        outcome = _persist_sample_run(tmp_path)
        monkeypatch.setattr(cli_main, "build_run_components", lambda *a, **k: _fake_run_components(outcome))
        result = runner.invoke(cli_main.app, ["run", "Sample question?", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["run_id"] == "sample_run"
        assert "Mode:" not in result.output  # never mix human text into --json output

    def test_partial_result_exit_code_1(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        outcome = _persist_sample_run(tmp_path)
        failed_outcome = outcome.model_copy(update={"success": False, "termination_reason": TerminationReason.MAX_STEPS})
        monkeypatch.setattr(cli_main, "build_run_components", lambda *a, **k: _fake_run_components(failed_outcome))
        result = runner.invoke(cli_main.app, ["run", "Sample question?"])
        assert result.exit_code == 1
        assert "Success: False" in result.output

    def test_repository_error_exit_code_3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli_main,
            "build_run_components",
            lambda *a, **k: _fake_run_components(RepositoryError("disk full")),
        )
        result = runner.invoke(cli_main.app, ["run", "Sample question?"])
        assert result.exit_code == 3
        assert "disk full" in result.output

    def test_report_rendering_error_exit_code_3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli_main,
            "build_run_components",
            lambda *a, **k: _fake_run_components(ReportRenderingError("render crash")),
        )
        result = runner.invoke(cli_main.app, ["run", "Sample question?"])
        assert result.exit_code == 3
        assert "render crash" in result.output

    def test_unexpected_error_exit_code_4_prints_traceback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli_main,
            "build_run_components",
            lambda *a, **k: _fake_run_components(RuntimeError("boom")),
        )
        result = runner.invoke(cli_main.app, ["run", "Sample question?"])
        assert result.exit_code == 4
        assert "Traceback" in result.output
        assert "boom" in result.output

    def test_policy_flags_are_forwarded_to_research_request(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        outcome = _persist_sample_run(tmp_path)
        orchestrator = FakeOrchestrator(outcome)
        components = RunComponents(
            orchestrator=orchestrator,  # type: ignore[arg-type]
            run_service=RunService(FileRunRepository(tmp_path)),
            report_renderer=MarkdownReportRenderer(),
            is_mock=False,
        )
        monkeypatch.setattr(cli_main, "build_run_components", lambda *a, **k: components)
        result = runner.invoke(
            cli_main.app,
            [
                "run",
                "Sample question?",
                "--max-steps",
                "3",
                "--max-tool-calls",
                "4",
                "--max-same-tool-calls",
                "2",
                "--timeout",
                "10.5",
                "--allowed-tools",
                "web_search,calculator",
            ],
        )
        assert result.exit_code == 0
        request = orchestrator.calls[0]
        assert request.max_steps == 3
        assert request.max_tool_calls == 4
        assert request.max_same_tool_calls == 2
        assert request.total_timeout_seconds == 10.5
        assert request.allowed_tools == ["web_search", "calculator"]

    def test_invalid_max_steps_argument_is_usage_error(self) -> None:
        result = runner.invoke(cli_main.app, ["run", "Sample question?", "--max-steps", "not-a-number"])
        assert result.exit_code == 2

    def test_research_request_validation_error_is_usage_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_validation_error(*args: Any, **kwargs: Any) -> ResearchRequest:
            from pydantic import ValidationError

            raise ValidationError.from_exception_data("ResearchRequest", [])

        monkeypatch.setattr(cli_main, "ResearchRequest", raise_validation_error)
        result = runner.invoke(cli_main.app, ["run", "Sample question?"])
        assert result.exit_code == 2


class TestRunsCommand:
    def test_empty_runs_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(cli_main.app, ["runs", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No runs found." in result.output

    def test_lists_multiple_runs(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="run_a")
        _persist_sample_run(tmp_path, run_id="run_b")
        result = runner.invoke(cli_main.app, ["runs", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "run_a" in result.output
        assert "run_b" in result.output

    def test_json_output_is_summarized_list(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="run_a")
        result = runner.invoke(cli_main.app, ["runs", "--runs-dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert isinstance(payload, list)
        assert payload[0]["run_id"] == "run_a"
        assert set(payload[0].keys()) == {"run_id", "created_at", "status", "question", "termination_reason"}

    def test_repository_error_exit_code_3(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            cli_main, "build_read_only_components", lambda *a, **k: _fake_read_only_components(RepositoryError("io error"))
        )
        result = runner.invoke(cli_main.app, ["runs", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 3
        assert "io error" in result.output

    def test_unexpected_error_exit_code_4(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            cli_main, "build_read_only_components", lambda *a, **k: _fake_read_only_components(RuntimeError("boom"))
        )
        result = runner.invoke(cli_main.app, ["runs", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 4
        assert "Traceback" in result.output


class TestShowCommand:
    def test_shows_existing_run(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="run_show")
        result = runner.invoke(cli_main.app, ["show", "run_show", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Run ID: run_show" in result.output
        assert "Sample question?" in result.output

    def test_not_found_exit_code_2(self, tmp_path: Path) -> None:
        result = runner.invoke(cli_main.app, ["show", "does-not-exist", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 2
        assert "not found" in result.output

    def test_json_output_round_trips(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="run_show_json")
        result = runner.invoke(cli_main.app, ["show", "run_show_json", "--runs-dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        record = RunRecord.model_validate_json(result.output)
        assert record.run_id == "run_show_json"

    def test_repository_error_exit_code_3(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            cli_main, "build_read_only_components", lambda *a, **k: _fake_read_only_components(RepositoryError("io error"))
        )
        result = runner.invoke(cli_main.app, ["show", "any-id", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 3
        assert "io error" in result.output

    def test_unexpected_error_exit_code_4(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            cli_main, "build_read_only_components", lambda *a, **k: _fake_read_only_components(RuntimeError("boom"))
        )
        result = runner.invoke(cli_main.app, ["show", "any-id", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 4
        assert "Traceback" in result.output


class TestReportCommand:
    def test_prints_markdown_to_stdout(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="run_report")
        result = runner.invoke(cli_main.app, ["report", "run_report", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "# Research Report" in result.output

    def test_writes_to_out_file(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="run_report_out")
        out_file = tmp_path / "report.md"
        result = runner.invoke(
            cli_main.app,
            ["report", "run_report_out", "--runs-dir", str(tmp_path), "--out", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.is_file()
        assert "# Research Report" in out_file.read_text(encoding="utf-8")

    def test_not_found_exit_code_2(self, tmp_path: Path) -> None:
        result = runner.invoke(cli_main.app, ["report", "does-not-exist", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 2

    def test_repository_error_exit_code_3(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            cli_main, "build_read_only_components", lambda *a, **k: _fake_read_only_components(RepositoryError("io error"))
        )
        result = runner.invoke(cli_main.app, ["report", "any-id", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 3
        assert "io error" in result.output

    def test_report_rendering_error_exit_code_3(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="run_render_fail")

        class FailingRenderer:
            def render(self, run: RunRecord) -> str:
                raise ReportRenderingError("render crash")

        components = ReadOnlyComponents(
            run_service=RunService(FileRunRepository(tmp_path)),
            report_renderer=FailingRenderer(),  # type: ignore[arg-type]
        )
        monkeypatch.setattr(cli_main, "build_read_only_components", lambda *a, **k: components)
        result = runner.invoke(cli_main.app, ["report", "run_render_fail", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 3
        assert "render crash" in result.output

    def test_unexpected_error_exit_code_4(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            cli_main, "build_read_only_components", lambda *a, **k: _fake_read_only_components(RuntimeError("boom"))
        )
        result = runner.invoke(cli_main.app, ["report", "any-id", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 4
        assert "Traceback" in result.output


class TestReplayCommand:
    def test_replays_persisted_run(self, tmp_path: Path) -> None:
        asyncio.run(_persist_replayable_sample_run(tmp_path, run_id="run_replay"))
        result = runner.invoke(cli_main.app, ["replay", "run_replay", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Question: Sample question?" in result.output
        assert "Final Answer" in result.output
        assert "Termination: finished" in result.output
        assert "Replay Verification" in result.output
        assert "Equivalent: YES" in result.output
        assert "Differences: none" in result.output

    def test_replay_without_tool_calls_does_not_crash(self, tmp_path: Path) -> None:
        state = ResearchState(
            research_id="run_no_tools",
            original_question="No tools needed?",
            created_at=_timestamp(),
            current_step=1,
            decisions=[LLMDecision(action=DecisionAction.FINISH, rationale="direct answer")],
            termination_reason=TerminationReason.FINISHED,
        )
        run_service = RunService(FileRunRepository(tmp_path))
        record = run_service.build_run(state, ExecutionPolicy(), ResearchResult.from_state(state))
        run_service.save_run(record)

        result = runner.invoke(cli_main.app, ["replay", "run_no_tools", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Question: No tools needed?" in result.output
        assert "Equivalent: YES" in result.output

    def test_not_found_exit_code_2(self, tmp_path: Path) -> None:
        result = runner.invoke(cli_main.app, ["replay", "does-not-exist", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 2

    def test_repository_error_exit_code_3(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            cli_main, "build_replay_components", lambda *a, **k: _fake_replay_components(RepositoryError("io error"))
        )
        result = runner.invoke(cli_main.app, ["replay", "any-id", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 3
        assert "io error" in result.output

    def test_unexpected_error_exit_code_4(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            cli_main, "build_replay_components", lambda *a, **k: _fake_replay_components(RuntimeError("boom"))
        )
        result = runner.invoke(cli_main.app, ["replay", "any-id", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 4
        assert "Traceback" in result.output

    def test_never_touches_run_components(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Replay must not build/require Agent/LLM/tool components at all."""

        def fail_if_called(*args: Any, **kwargs: Any) -> RunComponents:
            raise AssertionError("replay must never call build_run_components")

        monkeypatch.setattr(cli_main, "build_run_components", fail_if_called)
        asyncio.run(_persist_replayable_sample_run(tmp_path, run_id="run_isolated"))
        result = runner.invoke(cli_main.app, ["replay", "run_isolated", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output

    def test_divergent_replay_detected_and_exit_code_1(self, tmp_path: Path) -> None:
        asyncio.run(_persist_replayable_sample_run(tmp_path, run_id="run_tampered"))
        run_service = RunService(FileRunRepository(tmp_path))
        record = run_service.load_run("run_tampered")
        # Corrupt the persisted record so the tool call has no matching
        # result -- a genuinely inconsistent run.json, independent of any
        # object-aliasing concern.
        record.research_state.tool_results = []
        (tmp_path / "run_tampered" / "run.json").write_text(record.model_dump_json(), encoding="utf-8")

        result = runner.invoke(cli_main.app, ["replay", "run_tampered", "--runs-dir", str(tmp_path)])

        assert result.exit_code == 1
        assert "Equivalent: NO" in result.output
        assert "Differences:" in result.output

    def test_rejects_an_unfinished_run_exit_code_2(self, tmp_path: Path) -> None:
        """Fase 11E: a checkpoint (no research_result yet) is not
        replayable -- replay is for finished runs; resume is for these."""
        state = ResearchState(
            research_id="run_checkpoint",
            original_question="In progress?",
            created_at=_timestamp(),
            current_step=1,
        )
        run_service = RunService(FileRunRepository(tmp_path))
        run_service.save_checkpoint(run_service.build_run(state, ExecutionPolicy(), research_result=None))

        result = runner.invoke(cli_main.app, ["replay", "run_checkpoint", "--runs-dir", str(tmp_path)])

        assert result.exit_code == 2
        assert "not finalized" in result.output


class TestResumeCommand:
    def test_refuses_mock_mode_with_exit_code_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_mock_error(*args: Any, **kwargs: Any) -> ResumeComponents:
            raise MockProvidersNotSupportedError("mock mode not supported")

        monkeypatch.setattr(cli_main, "build_resume_components", raise_mock_error)
        result = runner.invoke(cli_main.app, ["resume", "some-run"])
        assert result.exit_code == 2
        assert "mock mode not supported" in result.output

    def test_refuses_missing_credentials_with_exit_code_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_missing_creds(*args: Any, **kwargs: Any) -> ResumeComponents:
            raise MissingProviderCredentialsError("OPENAI_API_KEY missing")

        monkeypatch.setattr(cli_main, "build_resume_components", raise_missing_creds)
        result = runner.invoke(cli_main.app, ["resume", "some-run"])
        assert result.exit_code == 2
        assert "OPENAI_API_KEY" in result.output

    def test_successful_resume_exit_code_0_human(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        outcome = _persist_sample_run(tmp_path)
        monkeypatch.setattr(cli_main, "build_resume_components", lambda *a, **k: _fake_resume_components(outcome))
        result = runner.invoke(cli_main.app, ["resume", "sample_run"])
        assert result.exit_code == 0
        assert "Success: True" in result.output
        assert "Run ID: sample_run" in result.output

    def test_successful_resume_json_is_valid(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        outcome = _persist_sample_run(tmp_path)
        monkeypatch.setattr(cli_main, "build_resume_components", lambda *a, **k: _fake_resume_components(outcome))
        result = runner.invoke(cli_main.app, ["resume", "sample_run", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["run_id"] == "sample_run"

    def test_partial_resume_exit_code_1(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        outcome = _persist_sample_run(tmp_path)
        failed = outcome.model_copy(update={"success": False, "termination_reason": TerminationReason.MAX_STEPS})
        monkeypatch.setattr(cli_main, "build_resume_components", lambda *a, **k: _fake_resume_components(failed))
        result = runner.invoke(cli_main.app, ["resume", "sample_run"])
        assert result.exit_code == 1
        assert "Success: False" in result.output

    def test_run_not_found_exit_code_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli_main,
            "build_resume_components",
            lambda *a, **k: _fake_resume_components(RunNotFoundError("run not found: 'missing'")),
        )
        result = runner.invoke(cli_main.app, ["resume", "missing"])
        assert result.exit_code == 2

    def test_already_finalized_exit_code_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli_main,
            "build_resume_components",
            lambda *a, **k: _fake_resume_components(RunAlreadyFinalizedError("already finalized")),
        )
        result = runner.invoke(cli_main.app, ["resume", "done-run"])
        assert result.exit_code == 2
        assert "already finalized" in result.output

    def test_repository_error_exit_code_3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli_main,
            "build_resume_components",
            lambda *a, **k: _fake_resume_components(RepositoryError("disk full")),
        )
        result = runner.invoke(cli_main.app, ["resume", "any-id"])
        assert result.exit_code == 3
        assert "disk full" in result.output

    def test_unexpected_error_exit_code_4(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli_main,
            "build_resume_components",
            lambda *a, **k: _fake_resume_components(RuntimeError("boom")),
        )
        result = runner.invoke(cli_main.app, ["resume", "any-id"])
        assert result.exit_code == 4
        assert "Traceback" in result.output

    def test_never_calls_build_replay_components(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Resume and replay are deliberately separate capabilities."""

        def fail_if_called(*args: Any, **kwargs: Any) -> ReplayComponents:
            raise AssertionError("resume must never call build_replay_components")

        monkeypatch.setattr(cli_main, "build_replay_components", fail_if_called)
        outcome = _persist_sample_run(tmp_path)
        monkeypatch.setattr(cli_main, "build_resume_components", lambda *a, **k: _fake_resume_components(outcome))
        result = runner.invoke(cli_main.app, ["resume", "sample_run"])
        assert result.exit_code == 0


class TestAcademicReportCommand:
    def test_generates_a_pdf_at_the_default_path(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="sample_run")

        result = runner.invoke(cli_main.app, ["academic-report", "sample_run", "--runs-dir", str(tmp_path)])

        assert result.exit_code == 0, result.output
        out_path = tmp_path / "sample_run" / "academic_report.pdf"
        assert out_path.is_file()
        assert out_path.read_bytes()[:5] == b"%PDF-"
        assert str(out_path) in result.output

    def test_pdf_contains_the_core_metadata_and_content(self, tmp_path: Path) -> None:
        from pypdf import PdfReader

        _persist_sample_run(tmp_path, run_id="sample_run")

        result = runner.invoke(cli_main.app, ["academic-report", "sample_run", "--runs-dir", str(tmp_path)])

        assert result.exit_code == 0, result.output
        reader = PdfReader(str(tmp_path / "sample_run" / "academic_report.pdf"))
        text = "\n".join(page.extract_text() for page in reader.pages)
        assert "LEONARDO TEIXEIRA" in text
        assert "245602" in text
        assert "UNICAMP" in text
        assert "A claim" in text  # _persist_sample_run's own claim text, reproduced verbatim

    def test_custom_metadata_flags_are_used(self, tmp_path: Path) -> None:
        from pypdf import PdfReader

        _persist_sample_run(tmp_path, run_id="sample_run")

        result = runner.invoke(
            cli_main.app,
            [
                "academic-report",
                "sample_run",
                "--runs-dir",
                str(tmp_path),
                "--author",
                "Someone Else",
                "--registration",
                "999999",
                "--institution",
                "Another University",
            ],
        )

        assert result.exit_code == 0, result.output
        reader = PdfReader(str(tmp_path / "sample_run" / "academic_report.pdf"))
        text = "\n".join(page.extract_text() for page in reader.pages)
        assert "SOMEONE ELSE" in text
        assert "999999" in text
        assert "ANOTHER UNIVERSITY" in text
        assert "LEONARDO TEIXEIRA" not in text

    def test_custom_out_path_is_honored(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="sample_run")
        custom_out = tmp_path / "custom" / "my_report.pdf"

        result = runner.invoke(
            cli_main.app, ["academic-report", "sample_run", "--runs-dir", str(tmp_path), "--out", str(custom_out)]
        )

        assert result.exit_code == 0, result.output
        assert custom_out.is_file()
        assert not (tmp_path / "sample_run" / "academic_report.pdf").exists()

    def test_run_not_found_exit_code_2(self, tmp_path: Path) -> None:
        result = runner.invoke(cli_main.app, ["academic-report", "missing-run", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 2

    def test_refuses_an_unfinished_run_exit_code_2(self, tmp_path: Path) -> None:
        state = ResearchState(
            research_id="run_incomplete",
            original_question="In progress?",
            created_at=_timestamp(),
            current_step=1,
        )
        run_service = RunService(FileRunRepository(tmp_path))
        run_service.save_checkpoint(run_service.build_run(state, ExecutionPolicy(), research_result=None))

        result = runner.invoke(cli_main.app, ["academic-report", "run_incomplete", "--runs-dir", str(tmp_path)])

        assert result.exit_code == 2
        assert "not finalized" in result.output

    def test_refuses_to_overwrite_an_existing_file_without_force(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="sample_run")
        out_path = tmp_path / "sample_run" / "academic_report.pdf"
        out_path.write_bytes(b"already here")

        result = runner.invoke(cli_main.app, ["academic-report", "sample_run", "--runs-dir", str(tmp_path)])

        assert result.exit_code == 2
        assert "already exists" in result.output
        assert out_path.read_bytes() == b"already here"  # never silently clobbered

    def test_force_overwrites_an_existing_file(self, tmp_path: Path) -> None:
        _persist_sample_run(tmp_path, run_id="sample_run")
        out_path = tmp_path / "sample_run" / "academic_report.pdf"
        out_path.write_bytes(b"already here")

        result = runner.invoke(
            cli_main.app, ["academic-report", "sample_run", "--runs-dir", str(tmp_path), "--force"]
        )

        assert result.exit_code == 0, result.output
        assert out_path.read_bytes()[:5] == b"%PDF-"

    def test_never_calls_build_run_components(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """academic-report only ever reads an already-persisted run --
        no LLM/tool/network access, matching replay's isolation."""

        def fail_if_called(*args: Any, **kwargs: Any) -> RunComponents:
            raise AssertionError("academic-report must never call build_run_components")

        monkeypatch.setattr(cli_main, "build_run_components", fail_if_called)
        _persist_sample_run(tmp_path, run_id="sample_run")

        result = runner.invoke(cli_main.app, ["academic-report", "sample_run", "--runs-dir", str(tmp_path)])

        assert result.exit_code == 0, result.output


class TestReadOnlyComponentsIndependenceFromRunComponents:
    def test_read_only_components_do_not_require_run_components(self, tmp_path: Path) -> None:
        components: ReadOnlyComponents = build_read_only_components(tmp_path)
        assert isinstance(components, ReadOnlyComponents)
