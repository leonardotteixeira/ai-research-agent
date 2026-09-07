"""Deterministic evaluators over persisted research RunRecords."""

import json
from typing import Protocol

from app.evaluation.schemas import EvaluationCase, EvaluationCaseResult
from app.schemas.answer import FinalAnswer
from app.schemas.evidence import normalize_source_url
from app.schemas.run import RunRecord
from app.schemas.state import TerminationReason
from app.schemas.tool import ToolCall
from app.synthesis.service import SynthesisService


class Evaluator(Protocol):
    name: str

    def evaluate(self, case: EvaluationCase, run_record: RunRecord) -> EvaluationCaseResult: ...


def _answer_for(run_record: RunRecord) -> FinalAnswer | None:
    if run_record.research_result is not None and run_record.research_result.final_answer is not None:
        return run_record.research_result.final_answer
    return run_record.research_state.final_answer


def _contains_all(haystack: str, needles: list[str]) -> tuple[bool, list[str]]:
    lowered = haystack.lower()
    missing = [needle for needle in needles if needle.lower() not in lowered]
    return not missing, missing


def _case_result(
    case: EvaluationCase,
    run_record: RunRecord,
    evaluator: str,
    violations: list[str],
    score: float | None = None,
    **details: object,
) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        case_id=case.id,
        run_id=run_record.run_id,
        evaluator=evaluator,
        passed=not violations,
        score=score,
        details=details,
        violations=violations,
    )


class AnswerEvaluator:
    name = "answer"

    def evaluate(self, case: EvaluationCase, run_record: RunRecord) -> EvaluationCaseResult:
        answer = _answer_for(run_record)
        violations: list[str] = []
        expected = case.expected
        checks = 3 + (len(expected.answer_contains) if expected is not None else 0)
        passed_checks = 0

        if answer is None:
            violations.append("final answer is missing")
            return _case_result(case, run_record, self.name, violations, score=0.0)

        if answer.answer_text.strip():
            passed_checks += 1
        else:
            violations.append("final answer text is empty")
        if answer.is_complete:
            passed_checks += 1
        else:
            violations.append("final answer is marked incomplete")
        if run_record.research_state.termination_reason == TerminationReason.FINISHED:
            passed_checks += 1
        else:
            violations.append("run did not finish successfully")

        missing_answer_terms: list[str] = []
        if expected is not None and expected.answer_contains:
            matched, missing_answer_terms = _contains_all(answer.answer_text, expected.answer_contains)
            passed_checks += len(expected.answer_contains) - len(missing_answer_terms)
            if not matched:
                violations.append(f"answer is missing expected text: {missing_answer_terms}")

        return _case_result(
            case,
            run_record,
            self.name,
            violations,
            score=passed_checks / checks if checks else None,
            is_complete=answer.is_complete,
            missing_answer_terms=missing_answer_terms,
        )


class CitationEvaluator:
    name = "citation"

    def evaluate(self, case: EvaluationCase, run_record: RunRecord) -> EvaluationCaseResult:
        answer = _answer_for(run_record)
        violations: list[str] = []
        if answer is None:
            violations.append("final answer is missing")
            return _case_result(case, run_record, self.name, violations, score=0.0)
        if answer.is_complete and not answer.citations:
            violations.append("complete answer has no citations")
        try:
            SynthesisService.validate_answer(answer, run_record.research_state)
        except Exception as exc:
            violations.append(f"citation graph is invalid: {exc}")

        citation_count = len(answer.citations)
        valid_count = 0 if violations and citation_count else citation_count
        return _case_result(
            case,
            run_record,
            self.name,
            violations,
            score=(valid_count / citation_count) if citation_count else None,
            citation_count=citation_count,
            valid_citation_count=valid_count,
        )


class EvidenceEvaluator:
    name = "evidence"

    def evaluate(self, case: EvaluationCase, run_record: RunRecord) -> EvaluationCaseResult:
        state = run_record.research_state
        expected = case.expected
        source_ids = {source.source_id for source in state.sources}
        evidence_ids = {item.evidence_id for item in state.evidence}
        violations: list[str] = []

        if not state.evidence:
            violations.append("no evidence was recorded")
        missing_sources = sorted({item.source_id for item in state.evidence} - source_ids)
        if missing_sources:
            violations.append(f"evidence references unknown sources: {missing_sources}")
        missing_evidence = sorted(
            {
                evidence_id
                for claim in state.claims
                for evidence_id in claim.evidence_ids
                if evidence_id not in evidence_ids
            }
        )
        if missing_evidence:
            violations.append(f"claims reference unknown evidence: {missing_evidence}")

        missing_expected_sources: list[str] = []
        missing_expected_evidence: list[str] = []
        missing_expected_claims: list[str] = []
        if expected is not None:
            actual_urls = {normalize_source_url(source.url) for source in state.sources}
            missing_expected_sources = [
                url for url in expected.source_urls if normalize_source_url(url) not in actual_urls
            ]
            if missing_expected_sources:
                violations.append(f"expected source URLs were not found: {missing_expected_sources}")
            evidence_text = "\n".join(item.content for item in state.evidence)
            _, missing_expected_evidence = _contains_all(evidence_text, expected.evidence_contains)
            if missing_expected_evidence:
                violations.append(f"expected evidence text was not found: {missing_expected_evidence}")
            claim_text = "\n".join(claim.text for claim in state.claims)
            _, missing_expected_claims = _contains_all(claim_text, expected.claims)
            if missing_expected_claims:
                violations.append(f"expected claims were not found: {missing_expected_claims}")

        claims_with_valid_evidence = sum(
            1 for claim in state.claims if claim.evidence_ids and set(claim.evidence_ids) <= evidence_ids
        )
        return _case_result(
            case,
            run_record,
            self.name,
            violations,
            score=(claims_with_valid_evidence / len(state.claims)) if state.claims else None,
            evidence_count=len(state.evidence),
            source_count=len(state.sources),
            claim_count=len(state.claims),
            claims_with_valid_evidence=claims_with_valid_evidence,
            missing_expected_sources=missing_expected_sources,
            missing_expected_evidence=missing_expected_evidence,
            missing_expected_claims=missing_expected_claims,
        )


class ExecutionEvaluator:
    name = "execution"

    def evaluate(self, case: EvaluationCase, run_record: RunRecord) -> EvaluationCaseResult:
        state = run_record.research_state
        policy = run_record.execution_policy
        reason = state.termination_reason
        violations: list[str] = []
        if reason != TerminationReason.FINISHED:
            violations.append(f"run termination reason is {reason.value if reason else 'missing'}")
        if state.current_step > policy.max_steps:
            violations.append("run exceeded max_steps")
        if len(state.tool_calls) > policy.max_tool_calls:
            violations.append("run exceeded max_tool_calls")
        if reason == TerminationReason.TIMEOUT:
            violations.append("run timed out")
        if reason == TerminationReason.LOOP_DETECTED:
            violations.append("loop detected")
        if reason == TerminationReason.POLICY_BLOCKED:
            violations.append("execution policy blocked a tool call")

        return _case_result(
            case,
            run_record,
            self.name,
            violations,
            score=1.0 if not violations else 0.0,
            termination_reason=reason.value if reason else None,
            steps=state.current_step,
            tool_calls=len(state.tool_calls),
            elapsed_seconds=state.elapsed_seconds,
        )


class ErrorEvaluator:
    name = "error"

    def evaluate(self, case: EvaluationCase, run_record: RunRecord) -> EvaluationCaseResult:
        state = run_record.research_state
        violations: list[str] = []
        if state.errors:
            violations.append("run recorded errors")
        if state.termination_reason in {
            TerminationReason.PROVIDER_ERROR,
            TerminationReason.TOOL_ERROR,
            TerminationReason.INVALID_OUTPUT,
        }:
            violations.append(f"run ended with failure reason: {state.termination_reason.value}")
        return _case_result(
            case,
            run_record,
            self.name,
            violations,
            score=1.0 if not violations else 0.0,
            error_count=len(state.errors),
            errors=list(state.errors),
        )


class PolicyEvaluator:
    name = "policy"

    def evaluate(self, case: EvaluationCase, run_record: RunRecord) -> EvaluationCaseResult:
        state = run_record.research_state
        policy = run_record.execution_policy
        violations: list[str] = []
        if state.termination_reason == TerminationReason.POLICY_BLOCKED:
            violations.append("run was blocked by execution policy")
        if policy.allowed_tools is not None:
            forbidden = sorted(
                {call.tool_name for call in state.tool_calls if call.tool_name not in policy.allowed_tools}
            )
            if forbidden:
                violations.append(f"tool calls outside allowed_tools: {forbidden}")
        repeated = self._repeated_calls(state.tool_calls, policy.max_same_tool_calls)
        if repeated:
            violations.append(f"repeated tool calls exceeded max_same_tool_calls: {repeated}")
        return _case_result(
            case,
            run_record,
            self.name,
            violations,
            score=1.0 if not violations else 0.0,
            repeated_calls=repeated,
        )

    @staticmethod
    def _repeated_calls(calls: list[ToolCall], max_same_tool_calls: int) -> list[str]:
        counts: dict[str, int] = {}
        for call in calls:
            signature = json.dumps(
                {"tool_name": call.tool_name, "arguments": call.arguments},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            counts[signature] = counts.get(signature, 0) + 1
        return sorted(signature for signature, count in counts.items() if count > max_same_tool_calls)


class GroundingEvaluator:
    """Checks only structural grounding, not semantic truth."""

    name = "grounding"

    def evaluate(self, case: EvaluationCase, run_record: RunRecord) -> EvaluationCaseResult:
        answer = _answer_for(run_record)
        state = run_record.research_state
        evidence_by_id = {item.evidence_id: item for item in state.evidence}
        source_ids = {source.source_id for source in state.sources}
        violations: list[str] = []
        claims = answer.claims if answer is not None else state.claims
        if answer is None:
            violations.append("final answer is missing")
        if not claims:
            violations.append("no claims available for structural grounding")
        for claim in claims:
            if not claim.evidence_ids:
                violations.append(f"claim has no evidence ids: {claim.claim_id}")
            missing_evidence = [item for item in claim.evidence_ids if item not in evidence_by_id]
            if missing_evidence:
                violations.append(f"claim references unknown evidence: {claim.claim_id}")
            missing_sources = [
                evidence_by_id[item].source_id
                for item in claim.evidence_ids
                if item in evidence_by_id and evidence_by_id[item].source_id not in source_ids
            ]
            if missing_sources:
                violations.append(f"claim evidence references unknown sources: {claim.claim_id}")
        structurally_grounded = len(claims) - len(
            [
                claim
                for claim in claims
                if not claim.evidence_ids or any(item not in evidence_by_id for item in claim.evidence_ids)
            ]
        )
        return _case_result(
            case,
            run_record,
            self.name,
            violations,
            score=(structurally_grounded / len(claims)) if claims else None,
            grounded_claims=structurally_grounded,
            total_claims=len(claims),
            limitation="structural grounding only; this does not prove semantic factuality",
        )


def default_evaluators() -> list[Evaluator]:
    return [
        AnswerEvaluator(),
        CitationEvaluator(),
        EvidenceEvaluator(),
        ExecutionEvaluator(),
        ErrorEvaluator(),
        PolicyEvaluator(),
        GroundingEvaluator(),
    ]
