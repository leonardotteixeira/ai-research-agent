"""Deterministic rendering of validated claims and source references."""

from app.schemas.answer import FinalAnswer
from app.schemas.state import ResearchState
from app.synthesis.service import SynthesisService


class CitationRenderer:
    def render(self, answer: FinalAnswer, state: ResearchState) -> str:
        SynthesisService.validate_answer(answer, state)
        source_by_id = {source.source_id: source for source in state.sources}
        source_numbers: dict[str, int] = {}
        ordered_source_ids: list[str] = []
        for citation in answer.citations:
            evidence_ids = set(citation.evidence_ids)
            claim = next((item for item in answer.claims if item.text == citation.claim), None)
            if claim is not None:
                evidence_ids.update(claim.evidence_ids)
            ids = list(citation.source_ids)
            ids.extend(
                evidence.source_id
                for evidence in state.evidence
                if evidence.evidence_id in evidence_ids and evidence.source_id not in ids
            )
            for source_id in ids:
                if source_id not in source_numbers:
                    source_numbers[source_id] = len(source_numbers) + 1
                    ordered_source_ids.append(source_id)

        rendered_answer = answer.answer_text
        for citation in answer.citations:
            claim = next((item for item in answer.claims if item.text == citation.claim), None)
            if claim is None or claim.text not in rendered_answer:
                continue
            evidence_ids = set(citation.evidence_ids) | set(claim.evidence_ids)
            source_ids = list(citation.source_ids)
            source_ids.extend(
                evidence.source_id
                for evidence in state.evidence
                if evidence.evidence_id in evidence_ids and evidence.source_id not in source_ids
            )
            markers = "".join(f"[{source_numbers[source_id]}]" for source_id in source_ids)
            if markers and markers not in rendered_answer:
                rendered_answer = rendered_answer.replace(claim.text, f"{claim.text} {markers}", 1)

        lines: list[str] = [rendered_answer]
        if ordered_source_ids:
            lines.extend(["", "Sources:"])
            lines.extend(
                f"[{source_numbers[source_id]}] {source_by_id[source_id].title or source_by_id[source_id].url} "
                f"- {source_by_id[source_id].url}"
                for source_id in ordered_source_ids
            )
        return "\n".join(lines)