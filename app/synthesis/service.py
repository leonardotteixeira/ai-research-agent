"""Evidence-only synthesis orchestration and reference validation."""

from app.core.exceptions import InvalidStructuredOutputError
from app.schemas.answer import FinalAnswer
from app.schemas.state import ResearchState
from app.synthesis.provider import SynthesisProvider


class SynthesisService:
    def __init__(self, provider: SynthesisProvider) -> None:
        self._provider = provider

    async def synthesize(self, state: ResearchState) -> FinalAnswer:
        answer, usage = await self._provider.generate_answer(
            question=state.original_question,
            evidence=[item.model_dump(mode="json") for item in state.evidence],
            sources=[item.model_dump(mode="json") for item in state.sources],
            claims=[item.model_dump(mode="json") for item in state.claims],
        )
        if usage is not None:
            state.token_usage = state.token_usage.add(usage)
        self.validate_answer(answer, state)
        return answer

    @staticmethod
    def validate_answer(answer: FinalAnswer, state: ResearchState) -> None:
        evidence_by_id = {item.evidence_id: item for item in state.evidence}
        source_by_id = {item.source_id: item for item in state.sources}
        claim_by_text = {claim.text: claim for claim in answer.claims}
        for claim in answer.claims:
            missing = set(claim.evidence_ids) - evidence_by_id.keys()
            if missing:
                raise InvalidStructuredOutputError(
                    f"claim references unknown evidence: {sorted(missing)}"
                )
        for citation in answer.citations:
            linked_claim = claim_by_text.get(citation.claim)
            evidence_ids = set(citation.evidence_ids)
            if linked_claim is not None:
                evidence_ids.update(linked_claim.evidence_ids)
            if not evidence_ids and not citation.source_ids:
                raise InvalidStructuredOutputError("citation must reference evidence or sources")
            missing_evidence = evidence_ids - evidence_by_id.keys()
            if missing_evidence:
                raise InvalidStructuredOutputError(
                    f"citation references unknown evidence: {sorted(missing_evidence)}"
                )
            source_ids = set(citation.source_ids)
            source_ids.update(evidence_by_id[item].source_id for item in evidence_ids)
            missing_sources = source_ids - source_by_id.keys()
            if missing_sources:
                raise InvalidStructuredOutputError(
                    f"citation references unknown source: {sorted(missing_sources)}"
                )
