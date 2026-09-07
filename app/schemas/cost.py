"""Token/cost accounting — same real/estimated/unavailable discipline used
across the portfolio (ai-gateway, production-rag, ai-eval-lab): a field
left as `None` means the information is genuinely unavailable, never
guessed or defaulted to 0 (which would misleadingly read as "measured, and
it was free").
"""

from pydantic import BaseModel


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    # False by default: an empty TokenUsage() has measured nothing, so it
    # must not assert "this is real usage". Only a value with real numbers
    # attached should ever claim actual=True (see providers/openai_llm.py);
    # explicitly-configured mock/test usage must set actual=False.
    actual: bool = False

    def has_data(self) -> bool:
        return self.input_tokens is not None or self.output_tokens is not None or self.total_tokens is not None

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Accumulate another call's usage into this one. Cost/token
        fields are summed only where both sides have a value — otherwise
        the total would silently understate usage by treating a missing
        value as 0.

        `actual` folds like an AND *once both sides have contributed real
        numbers* — an empty accumulator (nothing measured yet) is neutral
        and simply inherits whichever side does have data, so starting
        from `TokenUsage()` and adding one real (`actual=True`) usage
        yields `actual=True`, not `False`.
        """

        def _sum(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        def _sum_cost(a: float | None, b: float | None) -> float | None:
            if a is None and b is None:
                return None
            return (a or 0.0) + (b or 0.0)

        if not self.has_data():
            combined_actual = other.actual
        elif not other.has_data():
            combined_actual = self.actual
        else:
            combined_actual = self.actual and other.actual

        return TokenUsage(
            input_tokens=_sum(self.input_tokens, other.input_tokens),
            output_tokens=_sum(self.output_tokens, other.output_tokens),
            total_tokens=_sum(self.total_tokens, other.total_tokens),
            estimated_cost_usd=_sum_cost(self.estimated_cost_usd, other.estimated_cost_usd),
            actual=combined_actual,
        )
