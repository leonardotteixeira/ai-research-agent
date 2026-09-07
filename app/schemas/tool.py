"""Structured tool-calling schemas. The agent never parses free text like
"CALL_TOOL: calculator(...)" — every tool invocation and its result is a
validated Pydantic model, decoupled from however the LLM provider phrases
its decision (see app/schemas/decision.py for LLMDecision).
"""

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """`success=False` with `error` set covers both "the tool ran and
    failed" and "the tool couldn't be found/validated" — the caller
    doesn't need to distinguish those to build an Observation, but the
    `error` message does (see app/core/exceptions.py for the underlying
    exception types that get stringified into it).
    """

    call_id: str
    tool_name: str
    success: bool
    output: Any | None = None
    error: str | None = None
    latency_ms: float | None = None
