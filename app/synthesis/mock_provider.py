"""Deterministic synthesis provider for tests and offline development."""

from typing import Any

from app.core.exceptions import InvalidStructuredOutputError
from app.schemas.answer import FinalAnswer
from app.schemas.cost import TokenUsage


class MockSynthesisProvider:
    def __init__(
        self,
        answers: list[FinalAnswer | dict[str, Any]],
        usages: list[TokenUsage | None] | None = None,
    ) -> None:
        self._answers = [self._validate(answer) for answer in answers]
        if usages is not None and len(usages) != len(self._answers):
            raise ValueError("usages must have the same length as answers when provided")
        self._usages = [self._as_mock_usage(usage) for usage in usages] if usages is not None else None
        self.calls: list[dict[str, Any]] = []

    async def generate_answer(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        claims: list[dict[str, Any]],
    ) -> tuple[FinalAnswer, TokenUsage | None]:
        self.calls.append(
            {"question": question, "evidence": evidence, "sources": sources, "claims": claims}
        )
        if not self._answers:
            raise InvalidStructuredOutputError("MockSynthesisProvider has no answers remaining")
        answer = self._answers.pop(0)
        usage = self._usages.pop(0) if self._usages is not None else None
        return answer, usage

    @staticmethod
    def _validate(answer: FinalAnswer | dict[str, Any]) -> FinalAnswer:
        try:
            return answer if isinstance(answer, FinalAnswer) else FinalAnswer.model_validate(answer)
        except Exception as exc:
            raise InvalidStructuredOutputError(f"invalid mock synthesis answer: {exc}") from exc

    @staticmethod
    def _as_mock_usage(usage: TokenUsage | None) -> TokenUsage | None:
        """A mock never claims to have measured real OpenAI consumption --
        whatever the caller configures is forced to actual=False."""
        if usage is None:
            return None
        return usage.model_copy(update={"actual": False})
