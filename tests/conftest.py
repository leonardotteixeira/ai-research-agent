"""Session-wide test isolation.

`Settings` (app/core/config.py) reads from a real `.env` file in the repo
root by design, so a developer running the CLI picks up their own
credentials -- but that also means any test constructing `Settings()`
without overriding every single field can silently pick up whatever the
developer's own `.env` currently contains (a real API key, or
LLM_PROVIDER=anthropic instead of the class's built-in "openai" default),
making test outcomes depend on machine state instead of being
self-contained. Confirmed in practice: several `test_cli_composition.py`
tests started failing the moment a real `.env` was created with
LLM_PROVIDER=anthropic for manual CLI testing, because they only
overrode `openai_api_key`, not `llm_provider`.

This autouse fixture neutralizes that for every test in the suite by
disabling env_file loading and clearing `Settings`' own environment
variables (in case the shell running pytest happens to export any of
them), and by dropping `get_settings()`'s process-wide `lru_cache` so a
stale, previously-cached real Settings object can never leak into a
later test either.
"""

import pytest

from app.core.config import Settings, get_settings

_SETTINGS_ENV_VARS = (
    "ENVIRONMENT",
    "USE_MOCK_PROVIDERS",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "LLM_MODEL",
    "LLM_INPUT_PRICE_PER_1M",
    "LLM_OUTPUT_PRICE_PER_1M",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "SEARCH_API_KEY",
    "SEARCH_API_BASE_URL",
    "PROVIDER_TIMEOUT_SECONDS",
    "FETCH_TIMEOUT_SECONDS",
    "FETCH_MAX_CONTENT_BYTES",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_MAX_SAME_TOOL_CALLS",
    "DEFAULT_TOTAL_TIMEOUT_SECONDS",
    "DEFAULT_PER_TOOL_TIMEOUT_SECONDS",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
