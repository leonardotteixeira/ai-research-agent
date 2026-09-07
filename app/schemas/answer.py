"""The structured final answer — synthesis output. `is_complete=False`
means the agent is explicitly saying "I don't have enough evidence for a
confident answer" rather than papering over the gap with confident-sounding
prose (see app/synthesis).
"""

from pydantic import BaseModel, Field

from app.schemas.evidence import Claim


class Citation(BaseModel):
    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    answer_text: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    is_complete: bool
    caveats: str | None = None
