"""ResearchRequest (input), ResearchState (the agent's working memory for
one run), and ResearchResult (the public snapshot returned to callers).

ResearchState is deliberately fully JSON-serializable (plain Pydantic
model, no runtime-only objects) — that's what makes save/replay
(app/reports) possible: a finished or in-progress state is just
`model_dump_json()` / `model_validate_json()`, nothing more.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.answer import FinalAnswer
from app.schemas.cost import TokenUsage
from app.schemas.decision import LLMDecision, Observation, ResearchPlan
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.tool import ToolCall, ToolResult


class TerminationReason(str, Enum):
    """Why a research run stopped. Distinguishing these is the whole
    point of "don't turn an error into a seemingly-correct answer" — a
    caller can tell COMPLETED apart from every flavor of "stopped early".
    """

    FINISHED = "finished"
    COMPLETED = "finished"
    SYNTHESIS_REQUESTED = "synthesis_requested"
    MAX_STEPS = "max_steps"
    MAX_STEPS_REACHED = "max_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_TOOL_CALLS_REACHED = "max_tool_calls"
    TIMEOUT = "timeout"
    TOTAL_TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"
    POLICY_BLOCKED = "policy_blocked"
    NO_EVIDENCE = "no_evidence"
    LOOP_DETECTED = "loop_detected"
    PROVIDER_ERROR = "provider_error"
    INVALID_OUTPUT = "invalid_output"


class ResearchRequest(BaseModel):
    """The input to one research run. Limits are optional overrides of the
    configured defaults (see app/core/config.py); `allowed_tools=None`
    means every registered tool is available.
    """

    question: str
    max_steps: int | None = None
    max_tool_calls: int | None = None
    max_same_tool_calls: int | None = None
    total_timeout_seconds: float | None = None
    allowed_tools: list[str] | None = None


class ResearchState(BaseModel):
    research_id: str
    original_question: str
    created_at: datetime
    current_step: int = 0
    current_plan: ResearchPlan | None = None
    decisions: list[LLMDecision] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    elapsed_seconds: float | None = None
    final_answer: FinalAnswer | None = None
    termination_reason: TerminationReason | None = None

    @model_validator(mode="after")
    def validate_evidence_graph(self) -> "ResearchState":
        source_ids = {source.source_id for source in self.sources}
        evidence_ids = {evidence.evidence_id for evidence in self.evidence}
        missing_sources = {evidence.source_id for evidence in self.evidence} - source_ids
        missing_evidence = {
            evidence_id
            for claim in self.claims
            for evidence_id in claim.evidence_ids
            if evidence_id not in evidence_ids
        }
        if missing_sources:
            raise ValueError(f"evidence references unknown source(s): {sorted(missing_sources)}")
        if missing_evidence:
            raise ValueError(f"claim references unknown evidence(s): {sorted(missing_evidence)}")
        return self


class ResearchResult(BaseModel):
    """The public result of a completed (or terminated) run. A separate
    schema from ResearchState on purpose: internal bookkeeping on
    ResearchState (e.g. `current_plan`, mid-run scratch fields) can change
    without changing this public contract.
    """

    research_id: str
    question: str
    final_answer: FinalAnswer | None
    termination_reason: TerminationReason
    steps: int
    tool_call_count: int
    sources: list[Source]
    evidence: list[Evidence]
    claims: list[Claim]
    errors: list[str]
    token_usage: TokenUsage
    elapsed_seconds: float

    @classmethod
    def from_state(cls, state: ResearchState) -> "ResearchResult":
        if state.termination_reason is None:
            raise ValueError("cannot build a ResearchResult from a state with no termination_reason")
        return cls(
            research_id=state.research_id,
            question=state.original_question,
            final_answer=state.final_answer,
            termination_reason=state.termination_reason,
            steps=state.current_step,
            tool_call_count=len(state.tool_calls),
            sources=state.sources,
            evidence=state.evidence,
            claims=state.claims,
            errors=state.errors,
            token_usage=state.token_usage,
            elapsed_seconds=state.elapsed_seconds or 0.0,
        )
