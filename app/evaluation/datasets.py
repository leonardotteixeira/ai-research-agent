"""Filesystem-contained JSON dataset loading for evaluation."""

import json
from pathlib import Path

from pydantic import ValidationError

from app.evaluation.schemas import EvaluationDataset


class EvaluationDatasetError(Exception):
    """Base error for evaluation dataset loading."""


class EvaluationDatasetNotFoundError(EvaluationDatasetError):
    """The requested dataset file does not exist."""


class EvaluationDatasetJSONError(EvaluationDatasetError):
    """The dataset file is not valid JSON."""


class EvaluationDatasetSchemaError(EvaluationDatasetError):
    """The dataset JSON does not match EvaluationDataset."""


class EvaluationDatasetLoader:
    def load_json(self, path: str | Path) -> EvaluationDataset:
        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise EvaluationDatasetNotFoundError(f"evaluation dataset not found: {dataset_path}")
        try:
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvaluationDatasetJSONError(f"invalid evaluation dataset JSON: {dataset_path}") from exc
        except OSError as exc:
            raise EvaluationDatasetNotFoundError(f"evaluation dataset not readable: {dataset_path}") from exc
        try:
            return EvaluationDataset.model_validate(payload)
        except ValidationError as exc:
            raise EvaluationDatasetSchemaError(f"invalid evaluation dataset schema: {dataset_path}") from exc
