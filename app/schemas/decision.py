"""What the LLM provider returns each PLAN step, and what the agent records
after executing the resulting tool calls (Observation). `rationale`/
`summary` fields are free text kept purely for observability/logs — they
never drive control flow. Only `tool_calls` and `should_finish` do, and
even those are subordinate to the runtime's ExecutionPolicy (see
app/policies): the model requests, the code decides.
"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.tool import ToolCall, ToolResult


class DecisionAction(str, Enum):
    TOOL_CALL = "tool_call"
    FINISH = "finish"
    SYNTHESIZE = "synthesize"


class LLMDecision(BaseModel):
    """One PLAN step's output. If `should_finish` is true, `tool_calls` is
    ignored by the agent loop even if non-empty — finishing always takes
    priority. If neither `should_finish` nor any `tool_calls` are present,
    the agent loop treats that as an invalid decision (the model must
    either ask for tools or signal it's done), not as "do nothing".
    """

    action: DecisionAction | None = None
    rationale: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    should_finish: bool = False

    @model_validator(mode="after")
    def validate_action(self) -> "LLMDecision":
        # Keep the original fields usable while making new provider output explicit.
        if self.action is None:
            return self
        if self.action == DecisionAction.TOOL_CALL and not self.tool_calls:
            raise ValueError("tool_call decisions require at least one tool call")
        if self.action in {DecisionAction.FINISH, DecisionAction.SYNTHESIZE} and self.tool_calls:
            raise ValueError(f"{self.action.value} decisions cannot include tool calls")
        return self


class ResearchPlan(BaseModel):
    """A short, human-readable snapshot of the agent's current plan —
    updated each step from the LLM's rationale. Stored on ResearchState
    for observability/replay, not consulted by the control flow itself.
    """

    summary: str
    generated_at_step: int


class Observation(BaseModel):
    """What the agent recorded after executing one step's tool calls —
    the bridge between raw ToolResults and the evidence pipeline.
    """

    step: int
    tool_results: list[ToolResult] = Field(default_factory=list)
    summary: str
    new_evidence_ids: list[str] = Field(default_factory=list)
