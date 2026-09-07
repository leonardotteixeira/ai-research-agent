"""Domain service for constructing and accessing persisted research runs."""

from datetime import UTC, datetime

from app.persistence.repository import RunRepository
from app.policies.execution import ExecutionPolicy
from app.schemas.run import RunRecord
from app.schemas.state import ResearchResult, ResearchState


class RunService:
    def __init__(self, repository: RunRepository) -> None:
        self._repository = repository

    @staticmethod
    def build_run(
        state: ResearchState,
        execution_policy: ExecutionPolicy,
        research_result: ResearchResult | None = None,
        run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> RunRecord:
        return RunRecord(
            run_id=run_id or state.research_id,
            created_at=created_at or state.created_at or datetime.now(UTC),
            question=state.original_question,
            execution_policy=execution_policy,
            research_state=state,
            research_result=research_result,
        )

    def save_run(self, run: RunRecord) -> None:
        self._repository.save(run)

    def save_checkpoint(self, run: RunRecord) -> None:
        self._repository.save_checkpoint(run)

    def load_run(self, run_id: str) -> RunRecord:
        return self._repository.get(run_id)

    def exists(self, run_id: str) -> bool:
        return self._repository.exists(run_id)

    def list_runs(self) -> list[RunRecord]:
        return self._repository.list_runs()

    def export_json(self, run_id: str) -> str:
        return self.load_run(run_id).model_dump_json()

    def acquire_resume_lock(self, run_id: str) -> None:
        self._repository.acquire_resume_lock(run_id)

    def release_resume_lock(self, run_id: str) -> None:
        self._repository.release_resume_lock(run_id)