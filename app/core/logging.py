"""Structured, correlatable logging for observing a research run.

Minimal by design: emits lifecycle events (run_started, decision,
tool_started, tool_completed, tool_failed, synthesis_started/completed/
failed, run_finished) from Agent/ResearchOrchestrator onto one named
stdlib logger, each record carrying structured context (run_id, call_id,
step, ...) via `extra` instead of interpolated free text.

Deliberately does NOT attach a handler or call logging.basicConfig() --
standard practice for a library logger: it only emits, a consumer attaches
its own handler to see output. No handler means no output (Python's
default logging behavior), so this has zero effect anywhere until
something explicitly configures it -- existing tests and CLI behavior are
unaffected. Wiring a handler (e.g. a CLI flag streaming to stderr) is a
deliberately separate decision, not part of this module.
"""

import logging
from typing import Any

LOGGER_NAME = "ai_research_agent"

_MAX_TEXT_FIELD_LENGTH = 300


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured lifecycle event.

    `event` is both the log message and the record's `event` field.
    Every other keyword becomes a `LogRecord` attribute (readable via a
    handler/formatter, or in tests via `caplog.records[i].<field>`) --
    never interpolated into free text, so a handler can render or filter
    on them as data, not by parsing a message string.
    """
    safe_fields = {key: _truncate(value) for key, value in fields.items()}
    logger.log(level, event, extra={"event": event, **safe_fields})


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_TEXT_FIELD_LENGTH:
        return value[: _MAX_TEXT_FIELD_LENGTH - 3] + "..."
    return value
