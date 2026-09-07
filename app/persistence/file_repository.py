"""Local JSON-file implementation of the run repository.

Schema versioning (Fase 11C): `get()` checks the persisted `schema_version`
against `app.schemas.run.SUPPORTED_SCHEMA_VERSIONS` *before* full model
validation, so a run written under a version this codebase doesn't know how
to load is rejected with a distinct `UnsupportedSchemaVersionError` instead
of either silently loading (if it happens to still validate structurally)
or being lumped into the same generic "invalid or unreadable run.json"
message as a genuinely corrupted file. No migration is attempted -- an
unsupported version is simply refused. A missing `schema_version` key is
treated the same as an unsupported version, not silently assumed to be the
current one: there is no historical fixture proving what an absent version
would have meant, so guessing "1.0" would be exactly the kind of silent
assumption this check exists to avoid.
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.core.logging import get_logger, log_event
from app.persistence.repository import (
    ConcurrentResumeError,
    DuplicateRunError,
    InvalidRunIdError,
    RepositoryError,
    RunAlreadyFinalizedError,
    RunNotFoundError,
    UnsupportedSchemaVersionError,
)
from app.schemas.run import SUPPORTED_SCHEMA_VERSIONS, RunRecord

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class FileRunRepository:
    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)
        self._logger = get_logger()

    def save(self, run: RunRecord) -> None:
        """Create-only: the single first (and only) write for a run_id.

        Raises `DuplicateRunError` if anything is already on disk for this
        run_id. Used when the caller knows this must be the run's one and
        only persistence -- e.g. test fixtures and any hand-built,
        already-complete RunRecord. Real executions that checkpoint during
        their run use `save_checkpoint` instead (see its docstring), since
        that first checkpoint write already claims the run_id.
        """
        run_dir = self._run_dir(run.run_id)
        run_file = run_dir / "run.json"
        if run_file.exists() or run_dir.exists():
            raise DuplicateRunError(f"run already exists: {run.run_id!r}")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise DuplicateRunError(f"run already exists: {run.run_id!r}") from exc
        self._atomic_write(run_dir, run_file, run)

    def save_checkpoint(self, run: RunRecord) -> None:
        """Create-or-overwrite: the persistence primitive behind both
        in-progress checkpoints and a run's own finalization (Fase 11E).

        Unlike `save()`, this never raises `DuplicateRunError` -- the whole
        point of checkpointing is writing to the same run_id repeatedly as
        an execution progresses. The one thing it refuses is overwriting a
        run that already reached `research_result is not None` (a
        finished run), which raises `RunAlreadyFinalizedError` instead of
        silently clobbering completed data -- this is what stops a stray
        checkpoint or a stale `resume` from corrupting a finished run.
        Every write is atomic (temp file + os.replace), the same mechanism
        `save()` uses: a crash mid-write leaves either the previous
        checkpoint intact or the new one complete, never a partial file.
        """
        run_dir = self._run_dir(run.run_id)
        run_file = run_dir / "run.json"
        if run_file.is_file() and self._is_already_finalized(run_file):
            raise RunAlreadyFinalizedError(
                f"run {run.run_id!r} is already finalized and cannot be checkpointed over"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(run_dir, run_file, run)

    @staticmethod
    def _atomic_write(run_dir: Path, run_file: Path, run: RunRecord) -> None:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=run_dir,
                prefix=".run-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(run.model_dump_json())
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, run_file)
        except OSError as exc:
            FileRunRepository._remove_temporary_file(locals().get("temporary_path"))
            raise RepositoryError(f"failed to save run {run.run_id!r}: {exc}") from exc

    def acquire_resume_lock(self, run_id: str) -> None:
        """Local-filesystem-only exclusion (Fase 11F) so two `resume` calls
        for the same run_id -- from the same machine/`runs_dir`, whether
        in-process or two separate CLI invocations -- don't both continue
        the Agent loop from the same checkpoint and race to persist. Not a
        distributed lock: a shared network filesystem with weaker atomicity
        guarantees than POSIX/NTFS `O_CREAT|O_EXCL` is out of scope.
        Raises `ConcurrentResumeError` if the lock is already held; the
        caller must call `release_resume_lock` (typically via try/finally)
        once resume finishes, succeeds or fails.
        """
        lock_file = self._run_dir(run_id) / ".resume.lock"
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ConcurrentResumeError(
                f"run {run_id!r} is already being resumed (lock file present: {lock_file})"
            ) from exc
        except OSError as exc:
            raise RepositoryError(f"failed to acquire resume lock for {run_id!r}: {exc}") from exc
        os.close(fd)

    def release_resume_lock(self, run_id: str) -> None:
        lock_file = self._run_dir(run_id) / ".resume.lock"
        lock_file.unlink(missing_ok=True)

    @staticmethod
    def _is_already_finalized(run_file: Path) -> bool:
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # An unreadable/corrupt existing file is not our problem to
            # diagnose here -- get() already raises a clear RepositoryError
            # for that; a checkpoint write is allowed to proceed and
            # overwrite it (this is a genuinely new record for this run_id).
            return False
        return isinstance(payload, dict) and payload.get("research_result") is not None

    def get(self, run_id: str) -> RunRecord:
        run_file = self._run_file(run_id)
        if not run_file.is_file():
            raise RunNotFoundError(f"run not found: {run_id!r}")
        try:
            raw = run_file.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise RepositoryError(f"failed to load run {run_id!r}: invalid or unreadable run.json") from exc

        self._check_schema_version(payload)

        try:
            return RunRecord.model_validate(payload)
        except ValidationError as exc:
            raise RepositoryError(f"failed to load run {run_id!r}: invalid or unreadable run.json") from exc

    def exists(self, run_id: str) -> bool:
        try:
            self.get(run_id)
        except RunNotFoundError:
            return False
        return True

    def list_runs(self) -> list[RunRecord]:
        if not self._root_dir.is_dir():
            return []
        records: list[RunRecord] = []
        for run_dir in sorted(self._root_dir.iterdir(), key=lambda path: path.name):
            if run_dir.is_dir() and (run_dir / "run.json").is_file():
                try:
                    records.append(self.get(run_dir.name))
                except RepositoryError as exc:
                    # One bad run must never take down the listing of every
                    # other, healthy run -- it's skipped and reported via
                    # the structured logger (Fase 11A) instead, never via
                    # stdout/print, and never with prompt/response content.
                    log_event(
                        self._logger,
                        "run_load_failed",
                        level=logging.WARNING,
                        run_id=run_dir.name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
        return records

    def _run_file(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run.json"

    def _run_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
            raise InvalidRunIdError(f"unsafe run ID: {run_id!r}")
        return self._root_dir / run_id

    @staticmethod
    def _check_schema_version(payload: Any) -> None:
        found = payload.get("schema_version") if isinstance(payload, dict) else None
        if found not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedSchemaVersionError(found, SUPPORTED_SCHEMA_VERSIONS)

    @staticmethod
    def _remove_temporary_file(path: object) -> None:
        if isinstance(path, Path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
