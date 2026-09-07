"""Production composition root for the CLI.

Wires Settings -> providers -> tools -> ToolRegistry -> EvidencePipeline ->
Agent -> SynthesisService -> FileRunRepository -> RunService ->
MarkdownReportRenderer -> ResearchOrchestrator.

This module only constructs and connects existing domain objects. It
contains no LLM, tool, synthesis, or persistence *logic* of its own.

Read-only commands (`runs`, `show`, `report`, `replay`) never need an LLM
or search provider, so they get a smaller `ReadOnlyComponents` bundle that
never touches app.providers/app.agent at all.

`run` needs a real Agent, which needs a real LLMProvider. app.providers.
mock_llm.MockLLMProvider and app.synthesis.mock_provider.MockSynthesisProvider
are *scripted* test doubles (they replay a fixed, pre-supplied list of
decisions/answers) -- they cannot answer an arbitrary question typed at the
CLI. So when `Settings.use_mock_providers` is true, `build_run_components`
refuses up front with `MockProvidersNotSupportedError` instead of silently
running a broken or misleading "mock research". Live use requires
`use_mock_providers=false` plus real API keys.
"""

from dataclasses import dataclass
from pathlib import Path

from app.academic.assets import resolve_logo_paths
from app.academic.renderer import AcademicPDFRenderer
from app.agent.agent import Agent
from app.core.config import Settings, get_settings
from app.evaluation.comparison import EvaluationComparator
from app.evaluation.datasets import EvaluationDatasetLoader
from app.evaluation.reports import EvaluationJSONRenderer, EvaluationMarkdownRenderer
from app.evaluation.runner import EvaluationRunner
from app.evidence.pipeline import EvidencePipeline
from app.persistence.file_repository import FileRunRepository
from app.policies.execution import ExecutionPolicy
from app.providers.anthropic_llm import AnthropicProvider
from app.providers.openai_llm import OpenAIProvider
from app.providers.search import RealSearchProvider
from app.replay.service import ReplayService
from app.reports.markdown import MarkdownReportRenderer
from app.resume.service import ResumeService
from app.services.orchestrator import ResearchOrchestrator
from app.services.run_service import RunService
from app.synthesis.service import SynthesisService
from app.tools.base import Tool
from app.tools.calculator import CalculatorTool
from app.tools.fetch_url import FetchURLTool
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool

DEFAULT_RUNS_DIR = Path("runs")


class MockProvidersNotSupportedError(RuntimeError):
    """Raised by `build_run_components` when `use_mock_providers` is true.

    The scripted Mock* providers cannot serve an arbitrary CLI question;
    running research for real requires live providers and API keys.
    """


class MissingProviderCredentialsError(RuntimeError):
    """Raised when live providers are requested but a required API key is unset."""


@dataclass(frozen=True)
class ReadOnlyComponents:
    """Everything the read-only commands (runs/show/report/replay) need."""

    run_service: RunService
    report_renderer: MarkdownReportRenderer


@dataclass(frozen=True)
class RunComponents:
    """Everything the `run` command needs to execute new research."""

    orchestrator: ResearchOrchestrator
    run_service: RunService
    report_renderer: MarkdownReportRenderer
    is_mock: bool


def build_read_only_components(runs_dir: Path = DEFAULT_RUNS_DIR) -> ReadOnlyComponents:
    """Build the components used by `runs`, `show`, `report`, and `replay`.

    These commands only ever read a persisted RunRecord and render it, so
    this never constructs an LLM/search provider, Agent, or ToolRegistry.
    """
    repository = FileRunRepository(runs_dir)
    run_service = RunService(repository)
    report_renderer = MarkdownReportRenderer()
    return ReadOnlyComponents(run_service=run_service, report_renderer=report_renderer)


@dataclass(frozen=True)
class AcademicReportComponents:
    """Everything the `academic-report` command needs. Built on top of
    `build_read_only_components` -- generating the PDF never touches
    app.providers/app.agent/network, only an already-persisted RunRecord
    (see app/academic/).
    """

    run_service: RunService
    renderer: AcademicPDFRenderer


def build_academic_report_components(runs_dir: Path = DEFAULT_RUNS_DIR) -> AcademicReportComponents:
    read_only = build_read_only_components(runs_dir)
    renderer = AcademicPDFRenderer(logo_paths=resolve_logo_paths())
    return AcademicReportComponents(run_service=read_only.run_service, renderer=renderer)


@dataclass(frozen=True)
class ReplayComponents:
    """Everything the `replay` command needs.

    Built on top of `build_read_only_components` -- `ReplayService` itself
    never touches app.providers/app.tools/network; it only reconstructs an
    Agent driven by replay adapters fed from the RunRecord's own recorded
    decisions/tool-results (see app/replay/).
    """

    run_service: RunService
    replay_service: ReplayService


def build_replay_components(runs_dir: Path = DEFAULT_RUNS_DIR) -> ReplayComponents:
    read_only = build_read_only_components(runs_dir)
    return ReplayComponents(run_service=read_only.run_service, replay_service=ReplayService())


@dataclass(frozen=True)
class EvaluationComponents:
    """Everything the `evaluate` command needs.

    Built entirely on top of `build_read_only_components` -- `evaluate`
    only ever reads already-persisted RunRecords and a dataset file, so
    (like the other read-only commands) this never touches app.providers,
    app.agent, or app.tools.
    """

    run_service: RunService
    dataset_loader: EvaluationDatasetLoader
    runner: EvaluationRunner
    markdown_renderer: EvaluationMarkdownRenderer
    json_renderer: EvaluationJSONRenderer
    comparator: EvaluationComparator


def build_evaluation_components(runs_dir: Path = DEFAULT_RUNS_DIR) -> EvaluationComponents:
    read_only = build_read_only_components(runs_dir)
    return EvaluationComponents(
        run_service=read_only.run_service,
        dataset_loader=EvaluationDatasetLoader(),
        runner=EvaluationRunner(),
        markdown_renderer=EvaluationMarkdownRenderer(),
        json_renderer=EvaluationJSONRenderer(),
        comparator=EvaluationComparator(),
    )


def _ensure_live_settings(settings: Settings | None) -> Settings:
    """Shared by `build_run_components` and `build_resume_components` --
    both need real LLM/search credentials, since both can invoke a real
    LLM and real tools (unlike replay, which never does). Raises
    `MockProvidersNotSupportedError` if `settings.use_mock_providers` is
    true, and `MissingProviderCredentialsError` if a required API key (or
    `llm_provider` itself) is missing/invalid.
    """
    settings = settings or get_settings()

    if settings.use_mock_providers:
        raise MockProvidersNotSupportedError(
            "use_mock_providers is true: MockLLMProvider/MockSynthesisProvider are "
            "scripted test doubles and cannot answer an arbitrary question. Set "
            "use_mock_providers=false and configure OPENAI_API_KEY (and SEARCH_API_KEY "
            "for real web search) to run live research."
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise MissingProviderCredentialsError(
                "llm_provider is 'openai' but OPENAI_API_KEY is not set."
            )
    elif settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise MissingProviderCredentialsError(
                "llm_provider is 'anthropic' but ANTHROPIC_API_KEY is not set."
            )
    else:
        raise MissingProviderCredentialsError(
            f"LLM_PROVIDER={settings.llm_provider!r} is not supported; use 'openai' or 'anthropic'."
        )
    if not settings.search_api_key:
        raise MissingProviderCredentialsError(
            "use_mock_providers is false but SEARCH_API_KEY is not set."
        )
    return settings


def _build_llm_provider(settings: Settings) -> OpenAIProvider | AnthropicProvider:
    """Picks the concrete LLM provider per `settings.llm_provider` --
    `_ensure_live_settings` already validated the matching credential is
    present. Both implementations satisfy the exact same LLMProvider/
    SynthesisProvider Protocols (returning a Union of the two concrete
    classes, rather than just `LLMProvider`, so the same object can still
    be handed to `SynthesisService`, which needs `generate_answer` too),
    so nothing downstream (Agent, SynthesisService) needs to know or care
    which one this returns.
    """
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_seconds=settings.provider_timeout_seconds,
        )
    return OpenAIProvider(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.provider_timeout_seconds,
    )


def _build_live_ingredients(
    settings: Settings,
) -> tuple[OpenAIProvider | AnthropicProvider, ToolRegistry, EvidencePipeline, SynthesisService]:
    """The live LLM provider, tool registry, evidence pipeline, and
    synthesis service shared by `run` and `resume` -- both execute real
    research, just against a different (fresh vs. persisted) policy and
    ResearchState starting point. Extracted so this wiring is built in
    exactly one place rather than duplicated between the two.
    """
    # mypy note: `settings.search_api_key`'s `str | None` type doesn't
    # narrow across the call to `_ensure_live_settings` (a separate
    # function) even though it already guarantees non-None here -- see
    # its docstring. Asserted rather than re-implementing that check.
    assert settings.search_api_key is not None
    llm_provider = _build_llm_provider(settings)
    search_provider = RealSearchProvider(
        api_key=settings.search_api_key,
        base_url=settings.search_api_base_url,
        timeout_seconds=settings.provider_timeout_seconds,
    )

    # mypy note: app.tools.base.Tool declares `execute(self, **kwargs: Any)`,
    # which mypy's Protocol matching does not structurally accept from a
    # concrete tool with named parameters (e.g. `execute(self, query: str,
    # ...)`), even though it's exactly how ToolRegistry.execute() calls every
    # tool (`tool.execute(**validated_args.model_dump())`) at runtime. This is
    # a pre-existing gap in the Tool Protocol itself (app/tools/base.py),
    # not something specific to this composition root -- left alone here
    # rather than editing domain typing as a side effect of the CLI.
    tools: list[Tool] = [
        WebSearchTool(search_provider),  # type: ignore[list-item]
        FetchURLTool(  # type: ignore[list-item]
            timeout_seconds=settings.fetch_timeout_seconds,
            max_content_bytes=settings.fetch_max_content_bytes,
        ),
        CalculatorTool(),  # type: ignore[list-item]
    ]
    tool_registry = ToolRegistry(tools)
    evidence_pipeline = EvidencePipeline()
    synthesis_service = SynthesisService(llm_provider)
    return llm_provider, tool_registry, evidence_pipeline, synthesis_service


def build_run_components(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    settings: Settings | None = None,
) -> RunComponents:
    """Build the full stack needed to execute new research via `run`.

    Raises `MockProvidersNotSupportedError` if `settings.use_mock_providers`
    is true, and `MissingProviderCredentialsError` if live mode is active but
    a required API key is missing.
    """
    settings = _ensure_live_settings(settings)
    llm_provider, tool_registry, evidence_pipeline, synthesis_service = _build_live_ingredients(settings)

    policy = ExecutionPolicy()
    agent = Agent(
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        policy=policy,
        evidence_pipeline=evidence_pipeline,
    )

    repository = FileRunRepository(runs_dir)
    run_service = RunService(repository)
    report_renderer = MarkdownReportRenderer()

    orchestrator = ResearchOrchestrator(
        agent=agent,
        synthesis_service=synthesis_service,
        run_service=run_service,
        report_renderer=report_renderer,
        # Fase 11E: real runs checkpoint after every completed step, so a
        # crash mid-run leaves an identifiable, resumable RunRecord instead
        # of losing all intermediate state.
        checkpoints_enabled=True,
    )
    return RunComponents(
        orchestrator=orchestrator,
        run_service=run_service,
        report_renderer=report_renderer,
        is_mock=False,
    )


@dataclass(frozen=True)
class ResumeComponents:
    """Everything the `resume` command needs.

    Like `run`, this requires real LLM/search credentials -- resuming an
    interrupted run can mean calling OpenAI and real tools again (see
    app/resume/service.py). It never reuses `ReplayComponents`: replay and
    resume are deliberately separate capabilities.
    """

    run_service: RunService
    resume_service: ResumeService


def build_resume_components(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    settings: Settings | None = None,
) -> ResumeComponents:
    settings = _ensure_live_settings(settings)
    llm_provider, tool_registry, evidence_pipeline, synthesis_service = _build_live_ingredients(settings)

    repository = FileRunRepository(runs_dir)
    run_service = RunService(repository)
    report_renderer = MarkdownReportRenderer()

    def agent_factory(policy: ExecutionPolicy) -> Agent:
        # A fresh Agent per resume call, built from the *persisted*
        # policy -- never a new default -- so a resumed run keeps the
        # exact limits it was checkpointed under (see ResumeService).
        return Agent(
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            policy=policy,
            evidence_pipeline=evidence_pipeline,
        )

    resume_service = ResumeService(run_service, agent_factory, synthesis_service, report_renderer)
    return ResumeComponents(run_service=run_service, resume_service=resume_service)
