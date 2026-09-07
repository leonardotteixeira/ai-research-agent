"""Provider contract for synthesis; it never executes tools."""

from typing import Any, Protocol

from app.schemas.answer import FinalAnswer
from app.schemas.cost import TokenUsage


class SynthesisProvider(Protocol):
    async def generate_answer(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        claims: list[dict[str, Any]],
    ) -> tuple[FinalAnswer, TokenUsage | None]: ...