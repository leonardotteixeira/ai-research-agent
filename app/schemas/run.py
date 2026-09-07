"""Serializable aggregate representing one complete research execution."""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.policies.execution import ExecutionPolicy
from app.schemas.state import ResearchResult, ResearchState

# Single source of truth for the run schema version. FileRunRepository
# validates loaded runs against SUPPORTED_SCHEMA_VERSIONS -- see its
# docstring for why an unrecognized/missing version is rejected rather
# than silently accepted or defaulted.
CURRENT_SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({CURRENT_SCHEMA_VERSION})


class RunRecord(BaseModel):
    run_id: str = Field(..., min_length=1)
    created_at: datetime
    schema_version: str = CURRENT_SCHEMA_VERSION
    question: str = Field(..., min_length=1)
    execution_policy: ExecutionPolicy
    research_state: ResearchState
    research_result: ResearchResult | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> "RunRecord":
        if self.run_id != self.research_state.research_id:
            raise ValueError("run_id must match research_state.research_id")
        if self.question != self.research_state.original_question:
            raise ValueError("question must match research_state.original_question")
        if self.research_result is not None:
            if self.research_result.research_id != self.run_id:
                raise ValueError("research_result.research_id must match run_id")
            if self.research_result.question != self.question:
                raise ValueError("research_result.question must match question")
        return self