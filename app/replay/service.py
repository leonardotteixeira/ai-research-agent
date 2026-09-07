"""Entry point: replay a persisted RunRecord and compare the reproduced
execution to the one that was persisted. Read-only -- never mutates or
re-saves the original RunRecord, never touches RunService/FileRunRepository
itself (the caller already loaded the record).
"""

from app.agent.agent import Agent
from app.evidence.pipeline import EvidencePipeline
from app.persistence.repository import RunNotFinalizedError
from app.replay.comparator import compare_states
from app.replay.providers import ReplayLLMProvider, ReplaySynthesisProvider, ReplayToolRegistry
from app.replay.schemas import ReplayComparison
from app.schemas.run import RunRecord
from app.schemas.state import ResearchRequest
from app.services.orchestrator import apply_synthesis_if_requested
from app.synthesis.service import SynthesisService


class ReplayService:
    def __init__(self, evidence_pipeline: EvidencePipeline | None = None) -> None:
        self._evidence_pipeline = evidence_pipeline or EvidencePipeline()

    async def replay(self, record: RunRecord) -> ReplayComparison:
        # Fase 11E: a checkpoint from an interrupted execution has no
        # research_result yet. Replay reproduces a *finished* run's own
        # recorded trace -- it is not the tool for continuing an
        # unfinished one (that's app/resume/), so it fails clearly here
        # rather than attempting to replay a partial decision sequence.
        if record.research_result is None:
            raise RunNotFinalizedError(
                f"run {record.run_id!r} is not finalized and cannot be replayed"
            )

        # Deep-copied so the comparison baseline is fully decoupled in
        # memory from whatever the replay adapters are handed -- `record`
        # (the caller's object) is never read from again after this line,
        # and can never be mutated by anything replay does.
        original_state = record.research_state.model_copy(deep=True)

        llm_provider = ReplayLLMProvider(original_state.decisions)
        tool_registry = ReplayToolRegistry(original_state.tool_calls, original_state.tool_results)
        agent = Agent(
            llm_provider=llm_provider,
            # mypy note: Agent.__init__ types tool_registry as the concrete
            # app.tools.registry.ToolRegistry rather than a Protocol (a
            # pre-existing gap, same class of issue already noted in
            # app/cli/composition.py for app.tools.base.Tool). ReplayToolRegistry
            # implements the same execute()/describe() surface Agent actually
            # calls; left as an ignore here rather than widening Agent's typed
            # contract as a side effect of adding replay.
            tool_registry=tool_registry,  # type: ignore[arg-type]
            policy=record.execution_policy,
            evidence_pipeline=self._evidence_pipeline,
        )

        await agent.run(ResearchRequest(question=original_state.original_question))
        replayed_state = agent.last_state
        if replayed_state is None:
            raise RuntimeError("replay agent finished without recording last_state")

        if original_state.final_answer is not None:
            synthesis_service = SynthesisService(ReplaySynthesisProvider(original_state.final_answer))
            await apply_synthesis_if_requested(replayed_state, synthesis_service)

        differences = compare_states(original_state, replayed_state)
        return ReplayComparison(
            run_id=record.run_id,
            equivalent=not differences,
            differences=differences,
        )
