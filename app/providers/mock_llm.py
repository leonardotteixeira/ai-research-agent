"""Deterministic LLM provider for tests and offline development."""

from typing import Any

from app.core.exceptions import InvalidStructuredOutputError
from app.schemas.cost import TokenUsage
from app.schemas.decision import LLMDecision


class MockLLMProvider:
    def __init__(
        self,
        decisions: list[LLMDecision | dict[str, Any]],
        usages: list[TokenUsage | None] | None = None,
    ) -> None:
        self._decisions = [self._validate(decision) for decision in decisions]
        if usages is not None and len(usages) != len(self._decisions):
            raise ValueError("usages must have the same length as decisions when provided")
        self._usages = [self._as_mock_usage(usage) for usage in usages] if usages is not None else None
        self.calls: list[dict[str, Any]] = []

    async def generate_decision(
        self,
        question: str,
        context: list[dict[str, Any]] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> tuple[LLMDecision, TokenUsage | None]:
        self.calls.append(
            {
                "question": question,
                "context": context or [],
                "available_tools": available_tools or [],
            }
        )
        if not self._decisions:
            raise InvalidStructuredOutputError("MockLLMProvider has no decisions remaining")
        decision = self._decisions.pop(0)
        usage = self._usages.pop(0) if self._usages is not None else None
        return decision, usage

    @staticmethod
    def _validate(decision: LLMDecision | dict[str, Any]) -> LLMDecision:
        try:
            if isinstance(decision, LLMDecision):
                return decision
            return LLMDecision.model_validate(decision)
        except Exception as exc:
            raise InvalidStructuredOutputError(f"invalid mock LLM decision: {exc}") from exc

    @staticmethod
    def _as_mock_usage(usage: TokenUsage | None) -> TokenUsage | None:
        """A mock never claims to have measured real OpenAI consumption --
        whatever the caller configures is forced to actual=False."""
        if usage is None:
            return None
        return usage.model_copy(update={"actual": False})
