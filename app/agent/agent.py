"""The minimal PLAN/EXECUTE/OBSERVE loop for one research request."""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from app.core.exceptions import InvalidStructuredOutputError
from app.core.logging import get_logger, log_event
from app.evidence.pipeline import EvidencePipeline
from app.policies.execution import ExecutionPolicy
from app.providers.llm import LLMProvider
from app.schemas.decision import DecisionAction, LLMDecision, Observation, ResearchPlan
from app.schemas.state import ResearchRequest, ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall, ToolResult
from app.tools.registry import ToolRegistry

# Fase 11E: invoked with the (still JSON-serializable) in-progress
# ResearchState after each completed step and at every termination point.
# The Agent has no idea what the callback does with it -- persisting a
# checkpoint is an application/persistence concern (app/services/orchestrator.py
# builds and saves a RunRecord from it), never something the Agent touches
# the filesystem for directly.
CheckpointCallback = Callable[[ResearchState], Awaitable[None]]


class Agent:
    """Coordinates an injected LLM provider and tool registry.

    This class deliberately owns no provider or tool construction. It only
    records serializable state and delegates tool validation/execution to the
    registry.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        max_steps: int | None = None,
        policy: ExecutionPolicy | None = None,
        evidence_pipeline: EvidencePipeline | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._tool_registry = tool_registry
        self._policy = policy or ExecutionPolicy(
            max_steps=max_steps if max_steps is not None else ExecutionPolicy().max_steps
        )
        self._evidence_pipeline = evidence_pipeline or EvidencePipeline()
        self.last_state: ResearchState | None = None
        self._logger = get_logger()

    async def run(
        self,
        request: ResearchRequest,
        initial_state: ResearchState | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> ResearchResult:
        """Run the PLAN/EXECUTE/OBSERVE loop for one request.

        `initial_state` is Fase 11E's resume seam: pass a previously
        checkpointed (i.e. not yet terminated) ResearchState to continue it
        from its own `current_step` instead of starting a fresh run with a
        new `research_id`. Raises `ValueError` if that state already has a
        `termination_reason` -- resuming an already-terminated state would
        silently re-run a decision that was already made; callers with a
        terminal-but-unfinalized state (e.g. SYNTHESIS_REQUESTED) should go
        straight to synthesis instead of calling this at all (see
        app/resume/service.py).
        """
        prior_elapsed = 0.0
        if initial_state is not None:
            if initial_state.termination_reason is not None:
                raise ValueError(
                    "cannot resume a ResearchState that already has a termination_reason "
                    f"({initial_state.termination_reason!r}); it has nothing left for the Agent to do"
                )
            state = initial_state
            prior_elapsed = state.elapsed_seconds or 0.0
        else:
            state = ResearchState(
                research_id=str(uuid.uuid4()),
                original_question=request.question,
                created_at=datetime.now(UTC),
            )
        # Offsetting the clock (rather than adding prior_elapsed at every
        # `elapsed_seconds = time.perf_counter() - started` computation)
        # means resumed and fresh runs share the exact same downstream
        # arithmetic -- the accumulated pre-crash time is simply folded in
        # once, here, since it was never itself wall-clock-observable.
        started = time.perf_counter() - prior_elapsed
        policy = self._effective_policy(request)
        log_event(self._logger, "run_started", run_id=state.research_id, question=request.question)

        try:
            return await asyncio.wait_for(
                self._run_loop(state, request, policy, started, checkpoint_callback),
                timeout=policy.global_timeout_seconds,
            )
        except TimeoutError:
            state.errors.append(f"Agent run timed out after {policy.global_timeout_seconds}s")
            return await self._finish(state, TerminationReason.TIMEOUT, started, checkpoint_callback)

    async def _run_loop(
        self,
        state: ResearchState,
        request: ResearchRequest,
        policy: ExecutionPolicy,
        started: float,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> ResearchResult:
        # Seeded from any tool_calls already on `state` (empty for a fresh
        # run) so that resuming a checkpointed run keeps enforcing
        # max_same_tool_calls across the crash, instead of resetting the
        # count to zero and allowing more repeats than the policy intends.
        signatures: dict[str, int] = {}
        for call in state.tool_calls:
            signature = self._call_signature(call)
            signatures[signature] = signatures.get(signature, 0) + 1
        while state.current_step < policy.max_steps:
            state.current_step += 1
            try:
                decision, usage = await self._llm_provider.generate_decision(
                    question=state.original_question,
                    context=self._context_for(state),
                    available_tools=self._available_tools(policy),
                )
            except InvalidStructuredOutputError as exc:
                state.errors.append(str(exc))
                return await self._finish(state, TerminationReason.INVALID_OUTPUT, started, checkpoint_callback)
            except Exception as exc:
                state.errors.append(f"{type(exc).__name__}: {exc}")
                return await self._finish(state, TerminationReason.PROVIDER_ERROR, started, checkpoint_callback)

            if usage is not None:
                state.token_usage = state.token_usage.add(usage)

            state.decisions.append(decision)
            state.current_plan = ResearchPlan(
                summary=decision.rationale,
                generated_at_step=state.current_step,
            )

            action = self._resolve_action(decision)
            log_event(
                self._logger,
                "decision",
                run_id=state.research_id,
                step=state.current_step,
                action=action.value if action is not None else None,
                rationale=decision.rationale,
                tool_call_count=len(decision.tool_calls),
            )
            if action is None:
                state.errors.append("LLM decision has no actionable operation")
                return await self._finish(state, TerminationReason.INVALID_OUTPUT, started, checkpoint_callback)
            if action == DecisionAction.FINISH:
                return await self._finish(state, TerminationReason.FINISHED, started, checkpoint_callback)
            if action == DecisionAction.SYNTHESIZE:
                return await self._finish(
                    state, TerminationReason.SYNTHESIS_REQUESTED, started, checkpoint_callback
                )

            results: list[ToolResult] = []
            for call in decision.tool_calls:
                if len(state.tool_calls) >= policy.max_tool_calls:
                    self._append_observation(state, results)
                    return await self._finish(
                        state, TerminationReason.MAX_TOOL_CALLS, started, checkpoint_callback
                    )
                if policy.allowed_tools is not None and call.tool_name not in policy.allowed_tools:
                    state.tool_calls.append(call)
                    blocked = ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        success=False,
                        error=f"Tool {call.tool_name!r} blocked by execution policy",
                    )
                    state.tool_results.append(blocked)
                    state.errors.append(blocked.error or "tool blocked by execution policy")
                    results.append(blocked)
                    self._append_observation(state, results)
                    return await self._finish(
                        state, TerminationReason.POLICY_BLOCKED, started, checkpoint_callback
                    )
                signature = self._call_signature(call)
                if signatures.get(signature, 0) >= policy.max_same_tool_calls:
                    self._append_observation(state, results)
                    state.errors.append(f"Repeated tool call detected: {call.tool_name}")
                    return await self._finish(
                        state, TerminationReason.LOOP_DETECTED, started, checkpoint_callback
                    )
                signatures[signature] = signatures.get(signature, 0) + 1
                state.tool_calls.append(call)
                log_event(
                    self._logger,
                    "tool_started",
                    run_id=state.research_id,
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    step=state.current_step,
                )
                try:
                    result = await self._tool_registry.execute(call, timeout_seconds=policy.per_tool_timeout_seconds)
                except Exception as exc:
                    log_event(
                        self._logger,
                        "tool_failed",
                        level=logging.WARNING,
                        run_id=state.research_id,
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    state.errors.append(f"{type(exc).__name__}: {exc}")
                    self._append_observation(state, results)
                    return await self._finish(
                        state, TerminationReason.TOOL_ERROR, started, checkpoint_callback
                    )
                state.tool_results.append(result)
                results.append(result)
                if result.success:
                    log_event(
                        self._logger,
                        "tool_completed",
                        run_id=state.research_id,
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        latency_ms=result.latency_ms,
                    )
                else:
                    log_event(
                        self._logger,
                        "tool_failed",
                        level=logging.WARNING,
                        run_id=state.research_id,
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        error=result.error,
                    )
                try:
                    self._evidence_pipeline.merge(state.sources, state.evidence, result)
                except Exception as exc:
                    state.errors.append(f"Evidence pipeline error: {type(exc).__name__}: {exc}")
                if not result.success and result.error:
                    state.errors.append(result.error)

            self._append_observation(state, results)
            await self._checkpoint(state, started, checkpoint_callback)

        return await self._finish(state, TerminationReason.MAX_STEPS, started, checkpoint_callback)

    def effective_policy(self, request: ResearchRequest) -> ExecutionPolicy:
        return self._effective_policy(request)

    def _effective_policy(self, request: ResearchRequest) -> ExecutionPolicy:
        updates: dict[str, Any] = {}
        for field in ("max_steps", "max_tool_calls", "max_same_tool_calls"):
            value = getattr(request, field)
            if value is not None:
                updates[field] = value
        if request.total_timeout_seconds is not None:
            updates["global_timeout_seconds"] = request.total_timeout_seconds
        if request.allowed_tools is not None:
            requested = set(request.allowed_tools)
            updates["allowed_tools"] = (
                requested
                if self._policy.allowed_tools is None
                else self._policy.allowed_tools & requested
            )
        return ExecutionPolicy.model_validate({**self._policy.model_dump(), **updates})

    def _available_tools(self, policy: ExecutionPolicy) -> list[dict[str, Any]]:
        descriptions = self._tool_registry.describe()
        if policy.allowed_tools is None:
            return descriptions
        return [description for description in descriptions if description["name"] in policy.allowed_tools]

    @staticmethod
    def _context_for(state: ResearchState) -> list[dict[str, Any]]:
        return [
            {
                "step": observation.step,
                "summary": observation.summary,
                "tool_results": [result.model_dump(mode="json") for result in observation.tool_results],
            }
            for observation in state.observations
        ]

    @staticmethod
    def _resolve_action(decision: LLMDecision) -> DecisionAction | None:
        if decision.action is not None:
            return decision.action
        if decision.should_finish:
            return DecisionAction.FINISH
        if decision.tool_calls:
            return DecisionAction.TOOL_CALL
        return None

    @staticmethod
    def _observation_summary(results: list[ToolResult]) -> str:
        failures = sum(not result.success for result in results)
        if failures:
            return f"Executed {len(results)} tool call(s); {failures} failed."
        return f"Executed {len(results)} tool call(s) successfully."

    @staticmethod
    def _append_observation(state: ResearchState, results: list[ToolResult]) -> None:
        if results:
            state.observations.append(
                Observation(
                    step=state.current_step,
                    tool_results=results,
                    summary=Agent._observation_summary(results),
                )
            )

    @staticmethod
    def _call_signature(call: ToolCall) -> str:
        return json.dumps(
            {"tool_name": call.tool_name, "arguments": call.arguments},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    async def _checkpoint(
        self, state: ResearchState, started: float, checkpoint_callback: CheckpointCallback | None
    ) -> None:
        """Fase 11E: fires once per fully-completed step (never mid-step),
        so a checkpoint always reflects a decision plus all of its tool
        executions, evidence merges, and observation -- never a partial
        one. `checkpoint_callback` is application-injected (see
        app/services/orchestrator.py); the Agent only ever calls it with
        the current state, never touching a filesystem itself.
        """
        if checkpoint_callback is None:
            return
        state.elapsed_seconds = time.perf_counter() - started
        log_event(
            self._logger,
            "checkpoint_saved",
            run_id=state.research_id,
            step=state.current_step,
            termination_reason=state.termination_reason.value if state.termination_reason is not None else None,
        )
        await checkpoint_callback(state)

    async def _finish(
        self,
        state: ResearchState,
        reason: TerminationReason,
        started: float,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> ResearchResult:
        state.termination_reason = reason
        state.elapsed_seconds = time.perf_counter() - started
        self.last_state = state
        await self._checkpoint(state, started, checkpoint_callback)
        return ResearchResult.from_state(state)