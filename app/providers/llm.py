"""LLM provider contracts and structured-decision validation helpers."""

from typing import Any, Protocol

from app.schemas.cost import TokenUsage
from app.schemas.decision import LLMDecision


class LLMProvider(Protocol):
    async def generate_decision(
        self,
        question: str,
        context: list[dict[str, Any]] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> tuple[LLMDecision, TokenUsage | None]: ...