"""Central configuration. Mirrors the pattern from ai-gateway/production-rag/
ai-eval-lab: everything defaults to mock/offline so the app, tests, and CI
never require a paid API by default.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # When true (default), both the LLM decisions and web search are served
    # by deterministic in-process mocks — no API key needed, no network
    # calls. Real research (real LLM + real search) requires this false
    # AND the relevant keys set.
    use_mock_providers: bool = True

    # LLM provider -- "openai" (default) or "anthropic". Both implement the
    # same LLMProvider/SynthesisProvider Protocols (app/providers/llm.py,
    # app/synthesis/provider.py), so nothing else in the app needs to know
    # which one is active.
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_input_price_per_1m: float = 0.15
    llm_output_price_per_1m: float = 0.60

    # Anthropic (Claude) -- alternative to OpenAI, selected by llm_provider.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    # Search provider (Brave Search API — see app/providers/search)
    search_api_key: str | None = None
    search_api_base_url: str = "https://api.search.brave.com/res/v1/web/search"

    # Shared provider/tool timeouts
    provider_timeout_seconds: float = 30.0
    fetch_timeout_seconds: float = 15.0
    fetch_max_content_bytes: int = 1_000_000

    # Default execution policy (see app/policies) — overridable per request,
    # e.g. by the CLI's --max-steps.
    default_max_steps: int = 8
    default_max_tool_calls: int = 12
    default_max_same_tool_calls: int = 3
    default_total_timeout_seconds: float = 120.0
    default_per_tool_timeout_seconds: float = 20.0

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
