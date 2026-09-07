import json
import logging
from pathlib import Path
from typing import Any

import pytest

from app.persistence import (
    DuplicateRunError,
    FileRunRepository,
    InvalidRunIdError,
    RepositoryError,
    RunAlreadyFinalizedError,
    RunNotFoundError,
    UnsupportedSchemaVersionError,
)
from app.persistence import file_repository as file_repository_module
from app.schemas.run import CURRENT_SCHEMA_VERSION
from tests.unit.test_run_record import build_record


def _field(record: logging.LogRecord, name: str) -> Any:
    """LogRecord attributes attached via `extra=` are dynamic -- mypy's
    stub for LogRecord doesn't know them, and ruff (B009) objects to a
    literal `getattr(x, "constant")` at the call site, so this wrapper is
    used instead of either `record.<name>` or an inline `getattr` call."""
    return getattr(record, name)


class TestFileRunRepository:
    def test_save_creates_run_directory_and_json(self, tmp_path: Path):
        record = build_record()
        repository = FileRunRepository(tmp_path / "custom-runs")

        repository.save(record)

        run_file = tmp_path / "custom-runs" / "run_1" / "run.json"
        assert run_file.is_file()
        assert repository.exists("run_1") is True

    def test_get_after_save_preserves_complete_record(self, tmp_path: Path):
        record = build_record()
        repository = FileRunRepository(tmp_path)

        repository.save(record)

        assert repository.get("run_1") == record

    def test_save_rejects_duplicate_run_id_without_overwriting(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path)
        record = build_record()
        repository.save(record)

        with pytest.raises(DuplicateRunError, match="already exists"):
            repository.save(record)
        assert repository.get("run_1") == record

    def test_get_missing_run_has_controlled_error(self, tmp_path: Path):
        with pytest.raises(RunNotFoundError, match="run not found"):
            FileRunRepository(tmp_path).get("missing")

    def test_exists_is_false_for_missing_run_and_missing_root(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path / "does-not-exist")

        assert repository.exists("missing") is False
        assert repository.list_runs() == []

    def test_list_runs_is_sorted_deterministically(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path)
        for run_id in ("run_c", "run_a", "run_b"):
            record = build_record()
            record.research_state.research_id = run_id
            record.run_id = run_id
            record.research_result = None
            repository.save(record)

        assert [record.run_id for record in repository.list_runs()] == ["run_a", "run_b", "run_c"]

    @pytest.mark.parametrize("run_id", ["../evil", "..\\evil", "C:\\absolute", "/absolute", "run/child", ""])
    def test_rejects_unsafe_run_id(self, tmp_path: Path, run_id: str):
        repository = FileRunRepository(tmp_path)
        record = build_record()
        record.run_id = run_id

        with pytest.raises(InvalidRunIdError):
            repository.save(record)
        assert list(tmp_path.iterdir()) == []

    def test_get_rejects_unsafe_run_id(self, tmp_path: Path):
        with pytest.raises(InvalidRunIdError):
            FileRunRepository(tmp_path).get("../evil")

    def test_invalid_json_has_controlled_error(self, tmp_path: Path):
        run_dir = tmp_path / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("not-json", encoding="utf-8")
        repository = FileRunRepository(tmp_path)

        with pytest.raises(RepositoryError, match="invalid or unreadable"):
            repository.get("run_1")
        # "found but corrupted" must never be reported the same as "not
        # found" (Fase 11C) -- exists() propagates the same RepositoryError.
        with pytest.raises(RepositoryError, match="invalid or unreadable"):
            repository.exists("run_1")

    def test_invalid_record_has_controlled_error(self, tmp_path: Path):
        run_dir = tmp_path / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text('{"run_id":"run_1","schema_version":"1.0"}', encoding="utf-8")

        with pytest.raises(RepositoryError, match="invalid or unreadable"):
            FileRunRepository(tmp_path).get("run_1")

    def test_list_ignores_non_run_entries(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path)
        (tmp_path / "README.txt").write_text("ignored", encoding="utf-8")
        (tmp_path / "empty-run").mkdir()

        assert repository.list_runs() == []

    def test_save_filesystem_failure_has_context(self, tmp_path: Path, monkeypatch):
        repository = FileRunRepository(tmp_path)

        def fail_replace(source: str, destination: Path) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(file_repository_module.os, "replace", fail_replace)

        with pytest.raises(RepositoryError, match="failed to save run"):
            repository.save(build_record())
        assert not list((tmp_path / "run_1").glob(".run-*.tmp"))

    def test_get_filesystem_failure_has_context(self, tmp_path: Path, monkeypatch):
        repository = FileRunRepository(tmp_path)
        repository.save(build_record())

        def fail_read_text(path: Path, encoding: str = "utf-8") -> str:
            raise OSError("read failure")

        monkeypatch.setattr(Path, "read_text", fail_read_text)

        with pytest.raises(RepositoryError, match="failed to load run"):
            repository.get("run_1")

    def test_corrupt_run_does_not_escape_root(self, tmp_path: Path):
        outside = tmp_path.parent / "evil.json"
        repository = FileRunRepository(tmp_path / "runs")
        repository.save(build_record())

        assert not outside.exists()
        assert (tmp_path / "runs" / "run_1" / "run.json").is_file()


def _checkpoint_record():
    """A build_record() with research_result stripped -- Fase 11E's
    in-progress checkpoint shape (see app/schemas/run.py: research_result
    is already Optional, no schema change needed for checkpointing)."""
    record = build_record()
    record.research_result = None
    return record


class TestSaveCheckpoint:
    def test_creates_run_directory_and_json_when_absent(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path)

        repository.save_checkpoint(_checkpoint_record())

        run_file = tmp_path / "run_1" / "run.json"
        assert run_file.is_file()
        loaded = repository.get("run_1")
        assert loaded.research_result is None

    def test_overwrites_an_existing_checkpoint(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path)
        first = _checkpoint_record()
        first.research_state.current_step = 1
        repository.save_checkpoint(first)

        second = _checkpoint_record()
        second.research_state.current_step = 2
        repository.save_checkpoint(second)

        assert repository.get("run_1").research_state.current_step == 2

    def test_never_raises_duplicate_run_error(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path)
        repository.save_checkpoint(_checkpoint_record())

        # Unlike save(), repeated checkpointing of the same run_id is the
        # whole point -- this must never raise DuplicateRunError.
        repository.save_checkpoint(_checkpoint_record())

    def test_checkpoint_can_transition_into_a_finalized_run(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path)
        repository.save_checkpoint(_checkpoint_record())

        finalized = build_record()  # has research_result set
        repository.save_checkpoint(finalized)

        assert repository.get("run_1").research_result is not None

    def test_refuses_to_overwrite_an_already_finalized_run(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path)
        repository.save_checkpoint(build_record())  # already finalized

        with pytest.raises(RunAlreadyFinalizedError, match="already finalized"):
            repository.save_checkpoint(_checkpoint_record())
        # The finalized record on disk must be untouched by the refusal.
        assert repository.get("run_1").research_result is not None

    def test_filesystem_failure_has_context_and_leaves_no_partial_file(self, tmp_path: Path, monkeypatch):
        repository = FileRunRepository(tmp_path)

        def fail_replace(source: str, destination: Path) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(file_repository_module.os, "replace", fail_replace)

        with pytest.raises(RepositoryError, match="failed to save run"):
            repository.save_checkpoint(_checkpoint_record())
        assert not list((tmp_path / "run_1").glob(".run-*.tmp"))

    def test_overwrites_a_corrupt_existing_file_rather_than_erroring(self, tmp_path: Path):
        """A corrupt run.json isn't save_checkpoint()'s problem to diagnose
        (get() already raises a clear RepositoryError for that) -- it's
        simply treated as not-yet-finalized and overwritten."""
        run_dir = tmp_path / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("not-json", encoding="utf-8")
        repository = FileRunRepository(tmp_path)

        repository.save_checkpoint(_checkpoint_record())

        assert repository.get("run_1").research_result is None

    def test_crash_mid_checkpoint_leaves_prior_checkpoint_intact(self, tmp_path: Path, monkeypatch):
        """Simulates crash-window F (Fase 11E): a checkpoint write that
        fails atomically must never leave run.json partially written --
        the previous, fully-valid checkpoint must still load correctly."""
        repository = FileRunRepository(tmp_path)
        good = _checkpoint_record()
        good.research_state.current_step = 1
        repository.save_checkpoint(good)

        def fail_replace(source: str, destination: Path) -> None:
            raise OSError("simulated crash during checkpoint write")

        monkeypatch.setattr(file_repository_module.os, "replace", fail_replace)
        failing = _checkpoint_record()
        failing.research_state.current_step = 2
        with pytest.raises(RepositoryError):
            repository.save_checkpoint(failing)

        monkeypatch.undo()
        reloaded = repository.get("run_1")
        assert reloaded.research_state.current_step == 1
        assert reloaded.research_result is None


def _write_run_with_version(tmp_path: Path, run_id: str, schema_version: object) -> None:
    """Writes a run.json for `run_id` with an arbitrary (or absent, via
    `schema_version=...MISSING...`) schema_version -- bypassing save(),
    which always stamps the real current version, so tests can simulate
    a future/unknown/missing version without touching production code."""
    record = build_record()
    record.run_id = run_id
    record.research_state.research_id = run_id
    record.research_result = None
    payload = json.loads(record.model_dump_json())
    if schema_version is _MISSING:
        del payload["schema_version"]
    else:
        payload["schema_version"] = schema_version
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")


class _Missing:
    """Sentinel meaning 'omit schema_version entirely from the payload'."""


_MISSING = _Missing()


class TestSchemaVersioning:
    def test_current_version_loads_normally(self, tmp_path: Path):
        _write_run_with_version(tmp_path, "run_current", CURRENT_SCHEMA_VERSION)
        repository = FileRunRepository(tmp_path)

        record = repository.get("run_current")

        assert record.run_id == "run_current"
        assert record.schema_version == CURRENT_SCHEMA_VERSION

    def test_future_version_is_rejected(self, tmp_path: Path):
        _write_run_with_version(tmp_path, "run_future", "2.0")
        repository = FileRunRepository(tmp_path)

        with pytest.raises(UnsupportedSchemaVersionError):
            repository.get("run_future")

    def test_unknown_version_is_rejected(self, tmp_path: Path):
        _write_run_with_version(tmp_path, "run_unknown", "999.999")
        repository = FileRunRepository(tmp_path)

        with pytest.raises(UnsupportedSchemaVersionError):
            repository.get("run_unknown")

    def test_missing_version_is_rejected_not_assumed_current(self, tmp_path: Path):
        """No historical fixture proves what an absent schema_version would
        have meant, so it must be treated as unsupported/indeterminate --
        never silently assumed to be the current version."""
        _write_run_with_version(tmp_path, "run_no_version", _MISSING)
        repository = FileRunRepository(tmp_path)

        with pytest.raises(UnsupportedSchemaVersionError) as exc_info:
            repository.get("run_no_version")
        assert exc_info.value.found_version is None

    def test_error_message_contains_found_and_supported_versions(self, tmp_path: Path):
        _write_run_with_version(tmp_path, "run_future", "2.0")
        repository = FileRunRepository(tmp_path)

        with pytest.raises(UnsupportedSchemaVersionError) as exc_info:
            repository.get("run_future")

        message = str(exc_info.value)
        assert "2.0" in message
        assert CURRENT_SCHEMA_VERSION in message
        assert exc_info.value.found_version == "2.0"
        assert exc_info.value.supported_versions == frozenset({CURRENT_SCHEMA_VERSION})

    def test_list_runs_skips_incompatible_run_without_raising(self, tmp_path: Path):
        _write_run_with_version(tmp_path, "run_001", CURRENT_SCHEMA_VERSION)
        _write_run_with_version(tmp_path, "run_002", "2.0")
        _write_run_with_version(tmp_path, "run_003", CURRENT_SCHEMA_VERSION)
        repository = FileRunRepository(tmp_path)

        records = repository.list_runs()

        assert [record.run_id for record in records] == ["run_001", "run_003"]

    def test_list_runs_logs_run_load_failed_for_incompatible_run(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.WARNING, logger="ai_research_agent")
        _write_run_with_version(tmp_path, "run_good", CURRENT_SCHEMA_VERSION)
        _write_run_with_version(tmp_path, "run_bad", "2.0")
        repository = FileRunRepository(tmp_path)

        repository.list_runs()

        failed_events = [r for r in caplog.records if _field(r, "event") == "run_load_failed"]
        assert len(failed_events) == 1
        assert _field(failed_events[0], "run_id") == "run_bad"
        assert _field(failed_events[0], "error_type") == "UnsupportedSchemaVersionError"
        # Only diagnostic metadata -- never prompt/completion/API key/payload content.
        error_message = _field(failed_events[0], "error")
        assert "sk-" not in error_message
        assert "Authorization" not in error_message

    def test_exists_true_for_valid_run(self, tmp_path: Path):
        _write_run_with_version(tmp_path, "run_ok", CURRENT_SCHEMA_VERSION)
        repository = FileRunRepository(tmp_path)

        assert repository.exists("run_ok") is True

    def test_exists_false_for_missing_run(self, tmp_path: Path):
        repository = FileRunRepository(tmp_path)

        assert repository.exists("does-not-exist") is False

    def test_exists_raises_for_incompatible_version_not_false(self, tmp_path: Path):
        _write_run_with_version(tmp_path, "run_future", "2.0")
        repository = FileRunRepository(tmp_path)

        with pytest.raises(UnsupportedSchemaVersionError):
            repository.exists("run_future")
