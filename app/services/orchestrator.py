"""High-level E2E research orchestrator.

Coordinates:
    ResearchRequest
        ↓
    Agent
        ↓
    SynthesisService
        ↓
    RunService
        ↓
    MarkdownReportRenderer
        ↓
    OrchestratorResult

Orchestrates the workflow without implementing LLM logic, tools,
persistence internals, or Markdown formatting itself.
"""

import logging

from pydantic import BaseModel, Field

from app.agent.agent import Agent, CheckpointCallback
from app.core.exceptions import ReportRenderingError
from app.core.logging import get_logger, log_event
from app.policies.execution import ExecutionPolicy
from app.reports.markdown import MarkdownReportRenderer
from app.schemas.run import RunRecord
from app.schemas.state import ResearchRequest, ResearchResult, ResearchState, TerminationReason
from app.services.run_service import RunService
from app.synthesis.service import SynthesisService


def make_checkpoint_callback(
    run_service: RunService, policy: ExecutionPolicy, logger: logging.Logger | None = None
) -> CheckpointCallback:
    """Builds the Agent-facing checkpoint callback (Fase 11E): wraps the
    in-progress ResearchState into a RunRecord with no `research_result`
    (a checkpoint is never a finished run -- see app/schemas/run.py) and
    persists it via `RunService.save_checkpoint`, which atomically
    overwrites the same `run.json` a prior checkpoint (or none at all) may
    already be at. Shared by `ResearchOrchestrator.run()` and
    `app/resume/service.py.ResumeService.resume()` so this exact wiring
    is never duplicated between a fresh run and a resumed one.
    """

    async def checkpoint(state: ResearchState) -> None:
        record = run_service.build_run(state, policy, research_result=None)
        run_service.save_checkpoint(record)

    return checkpoint


async def finalize_run(
    state: ResearchState,
    policy: ExecutionPolicy,
    run_service: RunService,
    report_renderer: MarkdownReportRenderer,
    logger: logging.Logger | None = None,
) -> "OrchestratorResult":
    """Builds the final ResearchResult/RunRecord from a terminated state,
    persists it (overwriting any earlier checkpoint for the same run_id
    via `save_checkpoint` -- see its docstring for why this never touches
    an already-finalized run), renders the report, and computes success.
    Shared by `ResearchOrchestrator.run()` and `ResumeService.resume()` so
    this exact finalization logic is never duplicated between a fresh run
    and a resumed one.
    """
    log = logger or get_logger()
    result = ResearchResult.from_state(state)
    record = run_service.build_run(state, policy, result)
    run_service.save_checkpoint(record)

    try:
        report_markdown = report_renderer.render(record)
    except Exception as exc:
        state.errors.append(f"Report rendering failed: {type(exc).__name__}: {exc}")
        raise ReportRenderingError(
            f"research {record.run_id} completed but report rendering failed: {exc}"
        ) from exc

    is_answer_complete = state.final_answer.is_complete if state.final_answer is not None else True
    has_critical_failure = any(
        err.startswith(("ProviderError", "RuntimeError", "Synthesis failed", "Evidence pipeline error"))
        for err in state.errors
    )
    success = (
        state.termination_reason == TerminationReason.FINISHED
        and is_answer_complete
        and not has_critical_failure
    )

    log_event(
        log,
        "run_finished",
        run_id=record.run_id,
        termination_reason=result.termination_reason.value,
        success=success,
        elapsed_seconds=state.elapsed_seconds,
    )

    return OrchestratorResult(
        run_id=record.run_id,
        result=result,
        record=record,
        report_markdown=report_markdown,
        termination_reason=result.termination_reason,
        success=success,
        errors=list(state.errors),
    )


async def apply_synthesis_if_requested(
    state: ResearchState,
    synthesis_service: SynthesisService,
    logger: logging.Logger | None = None,
) -> None:
    """If the agent requested synthesis, run it and fold the result into
    `state` -- shared by `ResearchOrchestrator.run()` and the replay service
    (app/replay/service.py) so this exact mutation logic is never
    duplicated between a real run and a replayed one.
    """
    if state.termination_reason != TerminationReason.SYNTHESIS_REQUESTED:
        return
    log = logger or get_logger()
    log_event(log, "synthesis_started", run_id=state.research_id)
    try:
        answer = await synthesis_service.synthesize(state)
        state.final_answer = answer
        state.claims = list(answer.claims)
        state.termination_reason = TerminationReason.FINISHED
        log_event(log, "synthesis_completed", run_id=state.research_id)
    except Exception as exc:
        log_event(
            log,
            "synthesis_failed",
            level=logging.WARNING,
            run_id=state.research_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        state.errors.append(f"Synthesis failed: {type(exc).__name__}: {exc}")
        state.termination_reason = TerminationReason.INVALID_OUTPUT


class OrchestratorResult(BaseModel):
    run_id: str
    result: ResearchResult
    record: RunRecord
    report_markdown: str
    termination_reason: TerminationReason
    success: bool
    errors: list[str] = Field(default_factory=list)


class ResearchOrchestrator:
    def __init__(
        self,
        agent: Agent,
        synthesis_service: SynthesisService,
        run_service: RunService,
        report_renderer: MarkdownReportRenderer,
        checkpoints_enabled: bool = False,
    ) -> None:
        self._agent = agent
        self._synthesis_service = synthesis_service
        self._run_service = run_service
        self._report_renderer = report_renderer
        self._checkpoints_enabled = checkpoints_enabled
        self._logger = get_logger()

    async def run(self, request: ResearchRequest) -> OrchestratorResult:
        # 1. Resolve policy up front -- both the checkpoint callback and
        # the final RunRecord need the same effective policy.
        policy = self._resolve_policy(request)

        # 2. Execute agent loop, checkpointing after each completed step
        # when enabled (Fase 11E) -- disabled by default so callers that
        # never opted in (e.g. existing tests constructing this class
        # directly) see no behavior change.
        if self._checkpoints_enabled:
            checkpoint_callback = make_checkpoint_callback(self._run_service, policy, self._logger)
            await self._agent.run(request, checkpoint_callback=checkpoint_callback)
        else:
            await self._agent.run(request)
        state = self._agent.last_state
        if state is None:
            raise RuntimeError("Agent finished without recording last_state")

        # 3. Evaluate termination reason & perform synthesis when requested
        await apply_synthesis_if_requested(state, self._synthesis_service, self._logger)

        # 4. Produce the final ResearchResult/RunRecord, persist, render,
        # and compute success -- shared with ResumeService.resume().
        return await finalize_run(state, policy, self._run_service, self._report_renderer, self._logger)

    def _resolve_policy(self, request: ResearchRequest) -> ExecutionPolicy:
        if hasattr(self._agent, "effective_policy"):
            return self._agent.effective_policy(request)
        if hasattr(self._agent, "_policy"):
            policy = self._agent._policy
            if isinstance(policy, ExecutionPolicy):
                return policy
        return ExecutionPolicy()
