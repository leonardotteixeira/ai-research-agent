"""Replay adapters: satisfy the existing LLMProvider/ToolRegistry-shaped/
SynthesisProvider contracts by feeding back data already recorded in a
persisted RunRecord, instead of calling OpenAI, a search API, or fetching
a URL. None of these classes execute a real tool, make an HTTP request, or
import/eval/exec anything from the replayed content -- everything they
return is data that was already validated as a Pydantic model when it was
first persisted.

Token usage: replay never fabricates consumption. Every adapter here
always returns `usage=None` -- reproducing a run must never look like a
new, billable OpenAI call.
"""

from typing import Any

from app.schemas.answer import FinalAnswer
from app.schemas.cost import TokenUsage
from app.schemas.decision import LLMDecision
from app.schemas.tool import ToolCall, ToolResult


class ReplayDivergenceError(Exception):
    """Raised when the (deterministically replayed) Agent/SynthesisService
    control flow asks for something the persisted run does not contain --
    e.g. more decisions than were recorded, a tool call with no matching
    persisted call/result, a duplicate call, or a call whose arguments
    don't match the record.

    Agent and SynthesisService already convert any provider/tool exception
    into a terminated state with a diagnostic message recorded in
    `state.errors` (see app/agent/agent.py, app/synthesis/service.py) --
    this exception is deliberately just a plain Exception so it is caught
    by that same, already-tested handling instead of requiring a second,
    replay-specific error path.
    """


class ReplayLLMProvider:
    """Feeds back the exact LLMDecision sequence recorded in a persisted
    run, in order. Never calls a network or LLM."""

    def __init__(self, decisions: list[LLMDecision]) -> None:
        self._decisions = list(decisions)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def consumed_count(self) -> int:
        return self._index

    @property
    def recorded_count(self) -> int:
        return len(self._decisions)

    async def generate_decision(
        self,
        question: str,
        context: list[dict[str, Any]] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> tuple[LLMDecision, TokenUsage | None]:
        self.calls.append({"question": question, "context": context or [], "available_tools": available_tools or []})
        if self._index >= len(self._decisions):
            raise ReplayDivergenceError(
                f"replay requested decision #{self._index + 1}, but the persisted run only recorded "
                f"{len(self._decisions)} decision(s)"
            )
        decision = self._decisions[self._index]
        self._index += 1
        return decision, None


class ReplayToolRegistry:
    """Returns exactly the ToolResult persisted for each ToolCall's
    call_id. Never executes a real tool -- describe() returns an empty
    catalog because replay never asks an LLM to choose from a tool menu;
    the decisions (and the tool calls they carry) are already fixed."""

    def __init__(self, tool_calls: list[ToolCall], tool_results: list[ToolResult]) -> None:
        self._expected_calls = {call.call_id: call for call in tool_calls}
        self._results = {result.call_id: result for result in tool_results}
        self._consumed: set[str] = set()

    def describe(self) -> list[dict[str, Any]]:
        return []

    async def execute(self, call: ToolCall, timeout_seconds: float | None = None) -> ToolResult:
        if call.call_id in self._consumed:
            raise ReplayDivergenceError(f"duplicate tool call during replay: call_id={call.call_id!r}")
        expected = self._expected_calls.get(call.call_id)
        if expected is None:
            raise ReplayDivergenceError(
                f"unexpected tool call during replay: call_id={call.call_id!r} tool={call.tool_name!r} "
                "was not present in the persisted run"
            )
        if expected.tool_name != call.tool_name or expected.arguments != call.arguments:
            raise ReplayDivergenceError(
                f"tool call during replay does not match the persisted record for call_id={call.call_id!r}: "
                f"expected {expected.tool_name}({expected.arguments!r}), got {call.tool_name}({call.arguments!r})"
            )
        result = self._results.get(call.call_id)
        if result is None:
            raise ReplayDivergenceError(f"no persisted tool result for call_id={call.call_id!r}")
        self._consumed.add(call.call_id)
        return result


class ReplaySynthesisProvider:
    """Returns the FinalAnswer already persisted in the original run,
    without contacting OpenAI or any synthesis backend."""

    def __init__(self, answer: FinalAnswer) -> None:
        self._answer = answer
        self.calls: list[dict[str, Any]] = []

    async def generate_answer(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        claims: list[dict[str, Any]],
    ) -> tuple[FinalAnswer, TokenUsage | None]:
        self.calls.append({"question": question, "evidence": evidence, "sources": sources, "claims": claims})
        return self._answer, None
