"""ResumeService: the only component allowed to continue a checkpointed,
unfinished run's real execution (see app/resume/__init__.py for why this
must stay separate from app/replay/).

Two cases, both driven purely by the two already-existing Optional fields
on a persisted RunRecord (no new status enum -- see the Fase 11E audit):

- `research_state.termination_reason is None`: the Agent loop itself was
  interrupted mid-run. A fresh `Agent` (built from the *persisted*
  ExecutionPolicy, via `agent_factory`) continues it from its own
  `current_step`, checkpointing again as it goes.
- `research_state.termination_reason is not None` (but `research_result`
  is still `None`): the Agent loop had already reached a terminal
  decision before the crash -- most commonly SYNTHESIS_REQUESTED, meaning
  only synthesis (and the final persistence) never completed. No Agent
  loop is re-entered at all in this case; `apply_synthesis_if_requested`
  is a no-op for any other terminal reason, so this degrades to simply
  finishing the write that the crash interrupted, with zero provider
  calls.

`record.research_result is not None` (a genuinely finished run) is
rejected outright with `RunAlreadyFinalizedError` -- resume never
restarts, never re-runs providers for, and never touches a completed run.
"""

import logging
from collections.abc import Callable

from app.agent.agent import Agent
from app.core.logging import get_logger, log_event
from app.persistence.repository import RunAlreadyFinalizedError
from app.policies.execution import ExecutionPolicy
from app.reports.markdown import MarkdownReportRenderer
from app.schemas.run import RunRecord
from app.schemas.state import ResearchRequest
from app.services.orchestrator import (
    OrchestratorResult,
    apply_synthesis_if_requested,
    finalize_run,
    make_checkpoint_callback,
)
from app.services.run_service import RunService
from app.synthesis.service import SynthesisService

AgentFactory = Callable[[ExecutionPolicy], Agent]


class ResumeService:
    def __init__(
        self,
        run_service: RunService,
        agent_factory: AgentFactory,
        synthesis_service: SynthesisService,
        report_renderer: MarkdownReportRenderer,
    ) -> None:
        self._run_service = run_service
        self._agent_factory = agent_factory
        self._synthesis_service = synthesis_service
        self._report_renderer = report_renderer
        self._logger = get_logger()

    async def resume(self, run_id: str) -> OrchestratorResult:
        try:
            return await self._resume(run_id)
        except Exception as exc:
            log_event(
                self._logger,
                "resume_failed",
                level=logging.WARNING,
                run_id=run_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    async def _resume(self, run_id: str) -> OrchestratorResult:
        record = self._run_service.load_run(run_id)
        if record.research_result is not None:
            raise RunAlreadyFinalizedError(f"run {run_id!r} is already finalized and cannot be resumed")

        # Fase 11F: local-filesystem exclusion so a second, concurrent
        # `resume` of the same run_id fails fast instead of racing this
        # one to continue the same checkpoint and persist over it (see
        # ConcurrentResumeError's docstring for exactly what this does and
        # does not protect against). Acquired only after confirming the
        # run exists and isn't already finished, so those errors don't
        # need lock cleanup; released unconditionally below.
        self._run_service.acquire_resume_lock(run_id)
        try:
            return await self._resume_locked(run_id, record)
        finally:
            self._run_service.release_resume_lock(run_id)

    async def _resume_locked(self, run_id: str, record: RunRecord) -> OrchestratorResult:
        state = record.research_state
        policy = record.execution_policy
        log_event(
            self._logger,
            "resume_started",
            run_id=run_id,
            step=state.current_step,
            termination_reason=state.termination_reason.value if state.termination_reason is not None else None,
        )

        if state.termination_reason is None:
            # Mid-loop crash: reconstruct an Agent using the *persisted*
            # policy (never a fresh default) and continue from
            # state.current_step. A bare ResearchRequest carrying only the
            # original question has every override field at None, so
            # Agent._effective_policy() folds in no changes -- the loop
            # runs under exactly the policy that was checkpointed.
            agent = self._agent_factory(policy)
            checkpoint_callback = make_checkpoint_callback(self._run_service, policy, self._logger)
            request = ResearchRequest(question=state.original_question)
            await agent.run(request, initial_state=state, checkpoint_callback=checkpoint_callback)
            resumed_state = agent.last_state
            if resumed_state is None:
                raise RuntimeError("resumed Agent finished without recording last_state")
            state = resumed_state

        # Terminal-but-unfinalized (including the state produced just
        # above): apply_synthesis_if_requested() is a no-op unless
        # termination_reason is SYNTHESIS_REQUESTED, so a run that had
        # already reached e.g. MAX_STEPS before the crash costs zero
        # provider calls here -- it only finishes the persistence that was
        # interrupted.
        await apply_synthesis_if_requested(state, self._synthesis_service, self._logger)

        result = await finalize_run(state, policy, self._run_service, self._report_renderer, self._logger)

        log_event(
            self._logger,
            "resume_finished",
            run_id=run_id,
            termination_reason=result.termination_reason.value,
            success=result.success,
            elapsed_seconds=state.elapsed_seconds,
        )
        return result
