"""Semantic comparison between an original persisted ResearchState and one
reproduced by replaying it.

Deliberately ignores fields that are naturally nondeterministic across any
two executions of the exact same recorded decisions/tool-results:
- `research_id` (a fresh UUID every Agent.run() call);
- `created_at`/`elapsed_seconds` (wall-clock);
- `token_usage` (replay never fabricates consumption -- see
  app/replay/providers.py);
- `Source.retrieved_at`/`timestamp` and `Evidence.timestamp` (stamped
  fresh by EvidencePipeline at merge time, from `datetime.now(UTC)`).

Everything else that reflects what the run actually *did* -- decisions,
tool calls, tool results, observations, sources/evidence content, claims,
the final answer, termination reason, and errors -- is compared directly.
"""

from app.schemas.state import ResearchState

_IGNORED_SOURCE_FIELDS = {"retrieved_at", "timestamp"}
_IGNORED_EVIDENCE_FIELDS = {"timestamp"}


def _dump_all(items: list) -> list[dict]:
    return [item.model_dump(mode="json") for item in items]


def _normalize_sources(state: ResearchState) -> list[dict]:
    return [
        {key: value for key, value in source.model_dump(mode="json").items() if key not in _IGNORED_SOURCE_FIELDS}
        for source in state.sources
    ]


def _normalize_evidence(state: ResearchState) -> list[dict]:
    return [
        {key: value for key, value in item.model_dump(mode="json").items() if key not in _IGNORED_EVIDENCE_FIELDS}
        for item in state.evidence
    ]


def compare_states(original: ResearchState, replayed: ResearchState) -> list[str]:
    """Returns a list of human-readable differences; an empty list means
    the two states are semantically equivalent for replay purposes."""
    differences: list[str] = []

    if original.original_question != replayed.original_question:
        differences.append(
            f"original_question: expected {original.original_question!r}, got {replayed.original_question!r}"
        )

    original_decisions = _dump_all(original.decisions)
    replayed_decisions = _dump_all(replayed.decisions)
    if original_decisions != replayed_decisions:
        if len(original_decisions) != len(replayed_decisions):
            differences.append(
                f"decisions: expected {len(original_decisions)} decision(s), replay produced "
                f"{len(replayed_decisions)}"
            )
        else:
            differences.append("decisions: sequence differs from the persisted run")

    if _dump_all(original.tool_calls) != _dump_all(replayed.tool_calls):
        differences.append("tool_calls: sequence or arguments differ from the persisted run")

    if _dump_all(original.tool_results) != _dump_all(replayed.tool_results):
        differences.append("tool_results: differ from the persisted run")

    if _dump_all(original.observations) != _dump_all(replayed.observations):
        differences.append("observations: differ from the persisted run")

    if _normalize_sources(original) != _normalize_sources(replayed):
        differences.append("sources: differ from the persisted run (ignoring retrieved_at/timestamp)")

    if _normalize_evidence(original) != _normalize_evidence(replayed):
        differences.append("evidence: differ from the persisted run (ignoring timestamp)")

    if _dump_all(original.claims) != _dump_all(replayed.claims):
        differences.append("claims: differ from the persisted run")

    original_answer = original.final_answer.model_dump(mode="json") if original.final_answer is not None else None
    replayed_answer = replayed.final_answer.model_dump(mode="json") if replayed.final_answer is not None else None
    if original_answer != replayed_answer:
        differences.append("final_answer: differs from the persisted run")

    if original.termination_reason != replayed.termination_reason:
        differences.append(
            f"termination_reason: expected {original.termination_reason}, got {replayed.termination_reason}"
        )

    if original.errors != replayed.errors:
        differences.append(f"errors: expected {original.errors}, replay produced {replayed.errors}")

    return differences
