"""ToolRegistry — the single authority over which tools exist and can be
invoked.

This is the structural security boundary for the whole agent: an
LLMDecision can only ever request a `tool_name` that's registered here,
and arguments are always validated against the tool's own `input_schema`
before `execute()` runs. Content fetched from the web can *say* "call
tool: delete_everything" — it cannot make that happen, because nothing in
this codebase ever turns page content into a tool_name lookup. The LLM
provider only ever produces a `ToolCall` (a validated Pydantic model, not
free text) — see app/schemas/decision.py — and this registry is the only
thing that turns a ToolCall into an actual side effect.

Per-tool timeout is also enforced here (not left to the tool itself),
because a timeout is an execution *policy* concern (see app/policies),
independent of what any individual tool implementation does.
"""

import asyncio
import time

from pydantic import ValidationError

from app.schemas.tool import ToolCall, ToolResult
from app.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema.model_json_schema(),
            }
            for tool in self._tools.values()
        ]

    async def execute(self, call: ToolCall, timeout_seconds: float | None = None) -> ToolResult:
        start = time.perf_counter()

        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error=f"Unknown tool: {call.tool_name!r}. Available: {self.names()}",
                latency_ms=self._elapsed_ms(start),
            )

        try:
            validated_args = tool.input_schema.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error=f"Invalid arguments for '{call.tool_name}': {exc}",
                latency_ms=self._elapsed_ms(start),
            )

        try:
            coro = tool.execute(**validated_args.model_dump())
            output = await asyncio.wait_for(coro, timeout=timeout_seconds) if timeout_seconds else await coro
        except TimeoutError:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error=f"Tool '{call.tool_name}' timed out after {timeout_seconds}s",
                latency_ms=self._elapsed_ms(start),
            )
        except Exception as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=self._elapsed_ms(start),
            )

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            output=output,
            latency_ms=self._elapsed_ms(start),
        )

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000
