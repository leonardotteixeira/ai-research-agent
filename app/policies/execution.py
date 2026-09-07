"""Explicit, serializable limits for one Agent execution."""

from pydantic import BaseModel, Field

from app.core.config import get_settings


class ExecutionPolicy(BaseModel):
    max_steps: int = Field(default_factory=lambda: get_settings().default_max_steps, ge=1)
    max_tool_calls: int = Field(default_factory=lambda: get_settings().default_max_tool_calls, ge=0)
    max_same_tool_calls: int = Field(
        default_factory=lambda: get_settings().default_max_same_tool_calls,
        ge=1,
    )
    allowed_tools: set[str] | None = None
    global_timeout_seconds: float = Field(
        default_factory=lambda: get_settings().default_total_timeout_seconds,
        gt=0,
    )
    # Fase 11F: Settings.default_per_tool_timeout_seconds already existed
    # but was never wired into ExecutionPolicy or consumed by the Agent --
    # a single hanging tool call was only ever bounded by the (much
    # coarser) global_timeout_seconds. See Agent._run_loop, which now
    # passes this to ToolRegistry.execute(timeout_seconds=...).
    per_tool_timeout_seconds: float = Field(
        default_factory=lambda: get_settings().default_per_tool_timeout_seconds,
        gt=0,
    )