"""The minimal result of a replay: whether the reproduced execution is
semantically equivalent to the persisted one, and if not, why. Kept small
and separate from `app.evaluation` on purpose -- replay verifies
reproduction of an execution, evaluation measures quality against a
dataset. They are different concerns and are not merged here.
"""

from pydantic import BaseModel, Field


class ReplayComparison(BaseModel):
    run_id: str
    equivalent: bool
    differences: list[str] = Field(default_factory=list)
