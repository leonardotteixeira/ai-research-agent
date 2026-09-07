"""Human-readable and JSON formatting for CLI output.

Kept separate from main.py so it's testable without invoking Typer, and so
main.py stays a thin argument-parsing/dispatch layer. No domain logic lives
here either -- these functions only read already-computed fields off
existing Pydantic models (OrchestratorResult, RunRecord, ResearchState) and
arrange them for display. They never recompute a success/failure verdict
that the Orchestrator itself owns.
"""

from app.replay.schemas import ReplayComparison
from app.schemas.decision import LLMDecision
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.run import RunRecord
from app.schemas.state import ResearchState
from app.services.orchestrator import OrchestratorResult


def format_run_human(result: OrchestratorResult, is_mock: bool) -> str:
    mode = "MOCK" if is_mock else "LIVE"
    answer = result.result.final_answer
    lines = [
        f"Mode: {mode}",
        f"Run ID: {result.run_id}",
        f"Termination: {result.termination_reason.value}",
        f"Success: {result.success}",
        f"Sources: {len(result.result.sources)}",
        f"Evidence: {len(result.result.evidence)}",
        f"Claims: {len(result.result.claims)}",
    ]
    if answer is not None:
        lines.append(f"Answer: {answer.answer_text}")
        if not answer.is_complete:
            lines.append(f"Caveats: {answer.caveats or 'insufficient evidence'}")
    else:
        lines.append("Answer: (none)")
    if result.errors:
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in result.errors)
    return "\n".join(lines)


def format_run_json(result: OrchestratorResult) -> str:
    return result.model_dump_json()


def format_runs_table(records: list[RunRecord]) -> str:
    if not records:
        return "No runs found."
    header = f"{'run_id':<38} {'created_at':<26} {'status':<10} {'termination':<20} question"
    lines = [header, "-" * len(header)]
    for record in records:
        termination = _termination_label(record.research_state)
        status = _completion_status(record)
        question = record.question if len(record.question) <= 60 else record.question[:57] + "..."
        lines.append(
            f"{record.run_id:<38} {record.created_at.isoformat():<26} {status:<10} {termination:<20} {question}"
        )
    return "\n".join(lines)


def summarize_runs_json(records: list[RunRecord]) -> list[dict]:
    return [
        {
            "run_id": record.run_id,
            "created_at": record.created_at.isoformat(),
            "status": _completion_status(record),
            "question": record.question,
            "termination_reason": _termination_label(record.research_state),
        }
        for record in records
    ]


def format_show_human(record: RunRecord) -> str:
    state = record.research_state
    answer = state.final_answer
    lines = [
        f"Run ID: {record.run_id}",
        f"Question: {record.question}",
        f"Created at: {record.created_at.isoformat()}",
        f"Status: {_completion_status(record)}",
        "Execution Policy:",
        f"  max_steps: {record.execution_policy.max_steps}",
        f"  max_tool_calls: {record.execution_policy.max_tool_calls}",
        f"  max_same_tool_calls: {record.execution_policy.max_same_tool_calls}",
        f"  global_timeout_seconds: {record.execution_policy.global_timeout_seconds}",
        f"Termination: {_termination_label(state)}",
        f"Steps: {state.current_step}",
        f"Tool calls: {len(state.tool_calls)}",
        f"Sources: {len(state.sources)}",
        f"Evidence: {len(state.evidence)}",
        f"Claims: {len(state.claims)}",
    ]
    if answer is not None:
        lines.append(f"Answer complete: {answer.is_complete}")
        lines.append(f"Answer: {answer.answer_text}")
    else:
        lines.append("Answer: (none)")
    if state.errors:
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in state.errors)
    return "\n".join(lines)


def format_show_json(record: RunRecord) -> str:
    return record.model_dump_json()


def format_replay(record: RunRecord) -> str:
    """Deterministic, offline narrative of a persisted run.

    Reads only `record` (already loaded from disk by the caller) -- no
    Agent, LLMProvider, ToolRegistry, tool, or SynthesisService is ever
    invoked here. Evidence is correlated to tool results via
    `tool_call_id`, since `Observation.new_evidence_ids` is not populated
    by the Agent.
    """
    state = record.research_state
    lines = [f"Question: {state.original_question}", ""]

    results_by_call_id = {result.call_id: result for result in state.tool_results}
    evidence_by_call_id: dict[str, list[Evidence]] = {}
    for item in state.evidence:
        evidence_by_call_id.setdefault(item.tool_call_id, []).append(item)
    observations_by_step = {observation.step: observation for observation in state.observations}

    for index, decision in enumerate(state.decisions):
        step = index + 1
        lines.append(f"Step {step}")
        lines.append(f"  Action: {_decision_action_label(decision)}")
        if decision.rationale:
            lines.append(f"  Rationale: {decision.rationale}")
        if decision.tool_calls:
            lines.append("  Tool calls:")
            for call in decision.tool_calls:
                result = results_by_call_id.get(call.call_id)
                lines.append(f"    - {call.tool_name}(call_id={call.call_id}, arguments={call.arguments})")
                if result is None:
                    lines.append("      -> result: (not executed)")
                elif result.success:
                    lines.append(f"      -> success: output={result.output!r}")
                else:
                    lines.append(f"      -> failed: {result.error}")
                for item in evidence_by_call_id.get(call.call_id, []):
                    lines.append(f"      -> evidence {item.evidence_id}: {item.content}")
        observation = observations_by_step.get(step)
        if observation is not None:
            lines.append(f"  Observation: {observation.summary}")
        lines.append("")

    lines.append(f"Sources ({len(state.sources)})")
    lines.extend(_format_sources(state.sources))
    lines.append("")

    lines.append(f"Evidence ({len(state.evidence)})")
    lines.extend(_format_evidence(state.evidence))
    lines.append("")

    lines.append(f"Claims ({len(state.claims)})")
    lines.extend(_format_claims(state.claims))
    lines.append("")

    lines.append("Final Answer")
    if state.final_answer is not None:
        lines.append(f"  {state.final_answer.answer_text}")
        lines.append(f"  is_complete: {state.final_answer.is_complete}")
        if state.final_answer.caveats:
            lines.append(f"  caveats: {state.final_answer.caveats}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append(f"Termination: {_termination_label(state)}")
    if state.errors:
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in state.errors)
    return "\n".join(lines)


def format_replay_comparison(comparison: ReplayComparison) -> str:
    """Renders the result of actually re-running a persisted RunRecord
    offline (app/replay/) -- distinct from format_replay() above, which
    only narrates the recorded data without re-executing anything.
    """
    lines = [
        "",
        "Replay Verification",
        f"  Run ID: {comparison.run_id}",
        f"  Equivalent: {'YES' if comparison.equivalent else 'NO'}",
    ]
    if comparison.differences:
        lines.append("  Differences:")
        lines.extend(f"    - {difference}" for difference in comparison.differences)
    else:
        lines.append("  Differences: none")
    return "\n".join(lines)


def _decision_action_label(decision: LLMDecision) -> str:
    if decision.action is not None:
        return decision.action.value
    if decision.should_finish:
        return "finish"
    if decision.tool_calls:
        return "tool_call"
    return "unknown"


def _termination_label(state: ResearchState) -> str:
    return state.termination_reason.value if state.termination_reason is not None else "(none)"


def _completion_status(record: RunRecord) -> str:
    """Fase 11E: a run is complete iff it has a `research_result` -- a
    checkpoint from an interrupted execution never has one (see
    app/schemas/run.py). No separate RunStatus field is needed for this.
    """
    return "complete" if record.research_result is not None else "incomplete"


def _format_sources(sources: list[Source]) -> list[str]:
    if not sources:
        return ["  (none)"]
    return [f"  - {source.source_id}: {source.title or source.url} ({source.url})" for source in sources]


def _format_evidence(evidence: list[Evidence]) -> list[str]:
    if not evidence:
        return ["  (none)"]
    return [f"  - {item.evidence_id} (source={item.source_id}): {item.content}" for item in evidence]


def _format_claims(claims: list[Claim]) -> list[str]:
    if not claims:
        return ["  (none)"]
    return [f"  - {claim.claim_id}: {claim.text} [evidence: {', '.join(claim.evidence_ids)}]" for claim in claims]
