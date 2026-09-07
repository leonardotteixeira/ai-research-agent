"""Repository contracts and errors for persisted research runs."""

from typing import Protocol

from app.schemas.run import RunRecord


class RepositoryError(Exception):
    """Base error for run repository failures."""


class RunNotFoundError(RepositoryError):
    """The requested run does not exist."""


class DuplicateRunError(RepositoryError):
    """A run with the same ID already exists."""


class InvalidRunIdError(RepositoryError):
    """The run ID cannot safely be used as a directory name."""


class UnsupportedSchemaVersionError(RepositoryError):
    """The persisted run's `schema_version` is missing or not one this
    codebase knows how to load -- surfaced distinctly from generic
    corruption (see FileRunRepository.get()) so callers can tell "this
    file is unreadable" apart from "this file is a version we don't
    understand yet". No migration is attempted; the run is simply rejected.
    """

    def __init__(self, found_version: str | None, supported_versions: frozenset[str]) -> None:
        self.found_version = found_version
        self.supported_versions = supported_versions
        found = found_version if found_version is not None else "missing"
        supported = ", ".join(sorted(supported_versions))
        super().__init__(f"Unsupported run schema version {found!r}; supported versions: {supported}")


class RunAlreadyFinalizedError(RepositoryError):
    """Raised when something tries to checkpoint over, or resume, a run
    whose `research_result` is already set (Fase 11E) -- a finished run's
    persisted file must never be silently overwritten by an in-progress
    checkpoint or a stale resume attempt.
    """


class RunNotFinalizedError(RepositoryError):
    """Raised when an operation that requires a completed run (currently:
    `replay`, Fase 11D) is given a run whose `research_result` is still
    `None` -- i.e. a checkpoint from an interrupted execution (Fase 11E).
    Replay reproduces a finished run's own recorded trace; it is not the
    tool for continuing an unfinished one (see app/resume/ for that).
    """


class ConcurrentResumeError(RepositoryError):
    """Raised when `resume` is attempted for a run_id that another resume
    is already in progress for (Fase 11F). Two concurrent resumes of the
    same run would each independently continue the Agent loop from the
    same checkpoint and race to persist -- not a corruption risk (every
    write is still atomic) but a silent lost-update risk, plus duplicate
    real LLM/tool calls billed for no reason. No distributed lock is
    implemented; this is a plain local-filesystem exclusion (an
    atomically-created lock file) that only protects concurrent resumes
    of the same run_id from the same machine/`runs_dir` -- see
    FileRunRepository.acquire_resume_lock().
    """


class RunRepository(Protocol):
    def save(self, run: RunRecord) -> None: ...

    def save_checkpoint(self, run: RunRecord) -> None: ...

    def get(self, run_id: str) -> RunRecord: ...

    def exists(self, run_id: str) -> bool: ...

    def list_runs(self) -> list[RunRecord]: ...

    def acquire_resume_lock(self, run_id: str) -> None: ...

    def release_resume_lock(self, run_id: str) -> None: ...