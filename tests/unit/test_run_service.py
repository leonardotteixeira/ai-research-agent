from datetime import UTC, datetime

import pytest

from app.persistence.repository import RepositoryError
from app.policies.execution import ExecutionPolicy
from app.schemas.run import RunRecord
from app.schemas.state import ResearchResult, ResearchState
from app.services.run_service import RunService
from tests.unit.test_run_record import build_state


class InMemoryRepository:
    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}

    def save(self, run: RunRecord) -> None:
        self.runs[run.run_id] = run

    def get(self, run_id: str) -> RunRecord:
        return self.runs[run_id]

    def exists(self, run_id: str) -> bool:
        return run_id in self.runs

    def list_runs(self) -> list[RunRecord]:
        return [self.runs[key] for key in sorted(self.runs)]


class FailingRepository:
    def save(self, run: RunRecord) -> None:
        raise RepositoryError("repository unavailable")

    def get(self, run_id: str) -> RunRecord:
        raise RepositoryError("repository unavailable")

    def exists(self, run_id: str) -> bool:
        raise RepositoryError("repository unavailable")

    def list_runs(self) -> list[RunRecord]:
        raise RepositoryError("repository unavailable")


class TestRunService:
    def test_builds_run_from_state_without_result(self):
        state = ResearchState(research_id="run_no_result", original_question="Question", created_at=datetime.now(UTC))
        policy = ExecutionPolicy(max_steps=3, max_tool_calls=4)

        record = RunService.build_run(state, policy)

        assert record.run_id == "run_no_result"
        assert record.question == "Question"
        assert record.research_result is None
        assert record.execution_policy == policy

    def test_builds_run_with_result_and_preserves_identity(self):
        state = build_state()
        result = ResearchResult.from_state(state)

        record = RunService.build_run(state, ExecutionPolicy(), result)

        assert record.research_result == result
        assert record.run_id == state.research_id
        assert record.question == state.original_question

    def test_delegates_save_load_exists_list_and_export(self):
        repository = InMemoryRepository()
        service = RunService(repository)
        record = RunService.build_run(build_state(), ExecutionPolicy())

        service.save_run(record)

        assert service.exists(record.run_id) is True
        assert service.load_run(record.run_id) == record
        assert service.list_runs() == [record]
        assert RunRecord.model_validate_json(service.export_json(record.run_id)) == record

    def test_propagates_repository_errors(self):
        service = RunService(FailingRepository())
        record = RunService.build_run(build_state(), ExecutionPolicy())

        with pytest.raises(RepositoryError, match="repository unavailable"):
            service.save_run(record)
        with pytest.raises(RepositoryError, match="repository unavailable"):
            service.load_run(record.run_id)
        with pytest.raises(RepositoryError, match="repository unavailable"):
            service.exists(record.run_id)
        with pytest.raises(RepositoryError, match="repository unavailable"):
            service.list_runs()

    def test_explicit_run_id_and_created_at_are_supported(self):
        state = build_state()
        created_at = datetime(2020, 1, 1, tzinfo=UTC)

        record = RunService.build_run(
            state,
            ExecutionPolicy(),
            run_id="run_1",
            created_at=created_at,
        )

        assert record.run_id == "run_1"
        assert record.created_at == created_at