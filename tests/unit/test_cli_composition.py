from pathlib import Path

import pytest

from app.cli.composition import (
    EvaluationComponents,
    MissingProviderCredentialsError,
    MockProvidersNotSupportedError,
    ReadOnlyComponents,
    ResumeComponents,
    RunComponents,
    build_evaluation_components,
    build_read_only_components,
    build_resume_components,
    build_run_components,
)
from app.core.config import Settings
from app.evaluation.datasets import EvaluationDatasetLoader
from app.evaluation.reports import EvaluationJSONRenderer, EvaluationMarkdownRenderer
from app.evaluation.runner import EvaluationRunner
from app.persistence.file_repository import FileRunRepository
from app.policies.execution import ExecutionPolicy
from app.providers.anthropic_llm import AnthropicProvider
from app.providers.openai_llm import OpenAIProvider
from app.reports.markdown import MarkdownReportRenderer
from app.resume.service import ResumeService
from app.services.orchestrator import ResearchOrchestrator
from app.services.run_service import RunService


class TestBuildReadOnlyComponents:
    def test_wires_run_service_and_renderer_to_given_dir(self, tmp_path: Path) -> None:
        components = build_read_only_components(tmp_path)

        assert isinstance(components, ReadOnlyComponents)
        assert isinstance(components.run_service, RunService)
        assert isinstance(components.report_renderer, MarkdownReportRenderer)
        # Reaches the same directory a FileRunRepository built directly would.
        assert components.run_service._repository._root_dir == tmp_path  # type: ignore[attr-defined]

    def test_never_touches_llm_or_search_providers(self, tmp_path: Path) -> None:
        # No API keys configured anywhere; construction must still succeed
        # because read-only components never build an LLM/search provider.
        build_read_only_components(tmp_path)


class TestBuildRunComponents:
    def test_refuses_mock_providers(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=True)
        with pytest.raises(MockProvidersNotSupportedError, match="scripted test doubles"):
            build_run_components(tmp_path, settings=settings)

    def test_refuses_missing_openai_key(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=False, llm_provider="openai", openai_api_key=None, search_api_key="search-key")
        with pytest.raises(MissingProviderCredentialsError, match="OPENAI_API_KEY"):
            build_run_components(tmp_path, settings=settings)

    def test_refuses_missing_search_key(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=False, openai_api_key="sk-test", search_api_key=None)
        with pytest.raises(MissingProviderCredentialsError, match="SEARCH_API_KEY"):
            build_run_components(tmp_path, settings=settings)

    def test_builds_full_live_stack_when_configured(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=False, openai_api_key="sk-test", search_api_key="search-key")

        components = build_run_components(tmp_path, settings=settings)

        assert isinstance(components, RunComponents)
        assert components.is_mock is False
        assert isinstance(components.orchestrator, ResearchOrchestrator)
        assert isinstance(components.run_service, RunService)
        assert isinstance(components.report_renderer, MarkdownReportRenderer)
        assert isinstance(components.run_service._repository, FileRunRepository)  # type: ignore[attr-defined]

    def test_enables_checkpointing_by_default(self, tmp_path: Path) -> None:
        """Fase 11E: real runs via `run` checkpoint after every completed
        step, so a crash mid-run leaves a resumable RunRecord."""
        settings = Settings(use_mock_providers=False, openai_api_key="sk-test", search_api_key="search-key")

        components = build_run_components(tmp_path, settings=settings)

        assert components.orchestrator._checkpoints_enabled is True  # type: ignore[attr-defined]

    def test_selects_anthropic_provider_when_configured(self, tmp_path: Path) -> None:
        settings = Settings(
            use_mock_providers=False,
            llm_provider="anthropic",
            anthropic_api_key="sk-ant-test",
            search_api_key="search-key",
        )

        components = build_run_components(tmp_path, settings=settings)

        agent = components.orchestrator._agent  # type: ignore[attr-defined]
        assert isinstance(agent._llm_provider, AnthropicProvider)  # type: ignore[attr-defined]

    def test_refuses_missing_anthropic_key_when_selected(self, tmp_path: Path) -> None:
        settings = Settings(
            use_mock_providers=False, llm_provider="anthropic", anthropic_api_key=None, search_api_key="search-key"
        )
        with pytest.raises(MissingProviderCredentialsError, match="ANTHROPIC_API_KEY"):
            build_run_components(tmp_path, settings=settings)

    def test_refuses_unsupported_llm_provider(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=False, llm_provider="mistral", search_api_key="search-key")
        with pytest.raises(MissingProviderCredentialsError, match="not supported"):
            build_run_components(tmp_path, settings=settings)

    def test_defaults_to_openai_provider(self, tmp_path: Path) -> None:
        settings = Settings(
            use_mock_providers=False, llm_provider="openai", openai_api_key="sk-test", search_api_key="search-key"
        )

        components = build_run_components(tmp_path, settings=settings)

        agent = components.orchestrator._agent  # type: ignore[attr-defined]
        assert isinstance(agent._llm_provider, OpenAIProvider)  # type: ignore[attr-defined]


class TestBuildResumeComponents:
    def test_refuses_mock_providers(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=True)
        with pytest.raises(MockProvidersNotSupportedError, match="scripted test doubles"):
            build_resume_components(tmp_path, settings=settings)

    def test_refuses_missing_openai_key(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=False, llm_provider="openai", openai_api_key=None, search_api_key="search-key")
        with pytest.raises(MissingProviderCredentialsError, match="OPENAI_API_KEY"):
            build_resume_components(tmp_path, settings=settings)

    def test_refuses_missing_search_key(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=False, openai_api_key="sk-test", search_api_key=None)
        with pytest.raises(MissingProviderCredentialsError, match="SEARCH_API_KEY"):
            build_resume_components(tmp_path, settings=settings)

    def test_builds_full_live_stack_when_configured(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=False, openai_api_key="sk-test", search_api_key="search-key")

        components = build_resume_components(tmp_path, settings=settings)

        assert isinstance(components, ResumeComponents)
        assert isinstance(components.run_service, RunService)
        assert isinstance(components.resume_service, ResumeService)
        assert isinstance(components.run_service._repository, FileRunRepository)  # type: ignore[attr-defined]

    def test_agent_factory_uses_the_persisted_policy(self, tmp_path: Path) -> None:
        """The Agent built by resume must use whatever ExecutionPolicy it
        is handed at resume time, not a fresh default -- see
        ResumeService, which calls agent_factory(record.execution_policy)."""
        settings = Settings(use_mock_providers=False, openai_api_key="sk-test", search_api_key="search-key")
        components = build_resume_components(tmp_path, settings=settings)
        custom_policy = ExecutionPolicy(max_steps=42)

        agent = components.resume_service._agent_factory(custom_policy)  # type: ignore[attr-defined]

        assert agent._policy == custom_policy  # type: ignore[attr-defined]

    def test_reuses_read_only_run_service_wiring(self, tmp_path: Path) -> None:
        settings = Settings(use_mock_providers=False, openai_api_key="sk-test", search_api_key="search-key")
        read_only = build_read_only_components(tmp_path)
        resume = build_resume_components(tmp_path, settings=settings)

        assert (
            resume.run_service._repository._root_dir  # type: ignore[attr-defined]
            == read_only.run_service._repository._root_dir  # type: ignore[attr-defined]
        )


class TestBuildEvaluationComponents:
    def test_wires_run_service_dataset_loader_runner_and_renderer(self, tmp_path: Path) -> None:
        components = build_evaluation_components(tmp_path)

        assert isinstance(components, EvaluationComponents)
        assert isinstance(components.run_service, RunService)
        assert isinstance(components.dataset_loader, EvaluationDatasetLoader)
        assert isinstance(components.runner, EvaluationRunner)
        assert isinstance(components.markdown_renderer, EvaluationMarkdownRenderer)
        assert isinstance(components.json_renderer, EvaluationJSONRenderer)
        assert components.run_service._repository._root_dir == tmp_path  # type: ignore[attr-defined]

    def test_never_touches_llm_or_search_providers(self, tmp_path: Path) -> None:
        # No API keys configured anywhere; construction must still succeed
        # because evaluation components never build an LLM/search provider.
        build_evaluation_components(tmp_path)

    def test_reuses_read_only_run_service_wiring(self, tmp_path: Path) -> None:
        read_only = build_read_only_components(tmp_path)
        evaluation = build_evaluation_components(tmp_path)

        # Same repository root, same RunService shape -- not a second,
        # divergent persistence mechanism.
        assert (
            evaluation.run_service._repository._root_dir  # type: ignore[attr-defined]
            == read_only.run_service._repository._root_dir  # type: ignore[attr-defined]
        )
