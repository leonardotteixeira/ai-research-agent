import json

import pytest

from app.evaluation.datasets import (
    EvaluationDatasetJSONError,
    EvaluationDatasetLoader,
    EvaluationDatasetNotFoundError,
    EvaluationDatasetSchemaError,
)
from app.evaluation.schemas import EvaluationDataset


def test_loads_valid_dataset_and_preserves_metadata(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "id": "research_quality",
                "version": "1",
                "name": "Research Quality",
                "metadata": {"owner": "tests"},
                "cases": [
                    {
                        "id": "case_1",
                        "question": "What works in Postgres?",
                        "expected": {
                            "answer_contains": ["pgvector"],
                            "source_urls": ["https://fixture.example/pgvector-overview"],
                        },
                        "policy": {"max_steps": 3, "allowed_tools": ["web_search"]},
                        "metadata": {"topic": "vector-db"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDatasetLoader().load_json(path)

    assert isinstance(dataset, EvaluationDataset)
    assert dataset.version == "1"
    assert dataset.metadata == {"owner": "tests"}
    assert dataset.cases[0].metadata == {"topic": "vector-db"}
    assert dataset.cases[0].policy is not None
    assert dataset.cases[0].policy.allowed_tools == ["web_search"]


def test_missing_file_raises_structured_error(tmp_path):
    with pytest.raises(EvaluationDatasetNotFoundError):
        EvaluationDatasetLoader().load_json(tmp_path / "missing.json")


def test_invalid_json_raises_structured_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(EvaluationDatasetJSONError):
        EvaluationDatasetLoader().load_json(path)


def test_invalid_schema_raises_structured_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"id": "missing cases"}), encoding="utf-8")

    with pytest.raises(EvaluationDatasetSchemaError):
        EvaluationDatasetLoader().load_json(path)


def test_empty_dataset_is_rejected(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps({"id": "empty", "version": "1", "name": "Empty", "cases": []}),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDatasetSchemaError):
        EvaluationDatasetLoader().load_json(path)


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(
            {
                "id": "dupes",
                "version": "1",
                "name": "Dupes",
                "cases": [
                    {"id": "case_1", "question": "Q1"},
                    {"id": "case_1", "question": "Q2"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDatasetSchemaError):
        EvaluationDatasetLoader().load_json(path)
