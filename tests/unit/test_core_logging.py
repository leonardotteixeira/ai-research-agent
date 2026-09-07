import logging
from typing import Any

import pytest

from app.core.logging import LOGGER_NAME, get_logger, log_event


def _field(record: logging.LogRecord, name: str) -> Any:
    """LogRecord attributes attached via `extra=` are dynamic -- mypy's
    stub for LogRecord has no knowledge of them, so tests read them
    through this instead of `record.<name>` to stay type-clean."""
    return getattr(record, name)


class TestGetLogger:
    def test_returns_named_logger(self) -> None:
        assert get_logger().name == LOGGER_NAME == "ai_research_agent"

    def test_returns_same_logger_instance_across_calls(self) -> None:
        assert get_logger() is get_logger()

    def test_no_handler_attached_by_default(self) -> None:
        # Library-logger discipline: this module never calls
        # logging.basicConfig() or attaches a handler itself -- a consumer
        # must opt in. Confirms adding this module has zero side effects
        # on existing behavior until something explicitly configures it.
        assert get_logger().handlers == []


class TestLogEvent:
    def test_sets_message_and_event_field(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        log_event(get_logger(), "run_started", run_id="run_1", question="What?")

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.getMessage() == "run_started"
        assert _field(record, "event") == "run_started"
        assert _field(record, "run_id") == "run_1"
        assert _field(record, "question") == "What?"

    def test_default_level_is_info(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        log_event(get_logger(), "decision", run_id="run_1")

        assert caplog.records[0].levelno == logging.INFO

    def test_custom_level_is_honored(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        log_event(get_logger(), "tool_failed", level=logging.WARNING, run_id="run_1", error="boom")

        assert caplog.records[0].levelno == logging.WARNING
        assert _field(caplog.records[0], "error") == "boom"

    def test_below_configured_level_is_not_captured(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
        log_event(get_logger(), "decision", run_id="run_1")

        assert caplog.records == []

    def test_long_text_field_is_truncated(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        long_rationale = "x" * 1000
        log_event(get_logger(), "decision", run_id="run_1", rationale=long_rationale)

        truncated = _field(caplog.records[0], "rationale")
        assert len(truncated) == 300
        assert truncated.endswith("...")

    def test_short_text_field_is_not_truncated(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        log_event(get_logger(), "decision", run_id="run_1", rationale="short")

        assert _field(caplog.records[0], "rationale") == "short"

    def test_non_string_fields_pass_through_untouched(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        log_event(get_logger(), "tool_completed", run_id="run_1", latency_ms=12.5, success=True)

        record = caplog.records[0]
        assert _field(record, "latency_ms") == 12.5
        assert _field(record, "success") is True
