"""Deterministic, offline replay of a persisted research run.

Distinct from `app.cli.formatting.format_replay` (pure narrative
rendering of a RunRecord): this package actually re-runs the real Agent/
SynthesisService control flow, driven by replay adapters that feed back
the exact LLMDecision/ToolResult/FinalAnswer already recorded in the
RunRecord instead of calling OpenAI, a search API, or fetching a URL.
It then compares the freshly-reproduced ResearchState against the
persisted one, ignoring fields that are naturally nondeterministic across
any two executions (new UUIDs, wall-clock timestamps, token usage).

This is NOT event sourcing, NOT a general replay/migration framework, and
NOT a second Agent implementation -- it is three small adapters
(ReplayLLMProvider, ReplayToolRegistry, ReplaySynthesisProvider) that
satisfy the existing LLMProvider/ToolRegistry/SynthesisProvider contracts,
plus a semantic comparator. The real Agent and SynthesisService classes
run unmodified.
"""

from app.replay.comparator import compare_states
from app.replay.providers import (
    ReplayDivergenceError,
    ReplayLLMProvider,
    ReplaySynthesisProvider,
    ReplayToolRegistry,
)
from app.replay.schemas import ReplayComparison
from app.replay.service import ReplayService

__all__ = [
    "ReplayComparison",
    "ReplayDivergenceError",
    "ReplayLLMProvider",
    "ReplayService",
    "ReplaySynthesisProvider",
    "ReplayToolRegistry",
    "compare_states",
]
