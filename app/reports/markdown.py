"""Deterministic, filesystem-free Markdown rendering for research runs."""

from app.schemas.answer import FinalAnswer
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.run import RunRecord


def _inline(value: str | None) -> str:
    if value is None:
        return "Not available"
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", "").replace("\n", " ")


class MarkdownReportRenderer:
    def render(self, run: RunRecord) -> str:
        state = run.research_state
        result = run.research_result
        answer = result.final_answer if result is not None and result.final_answer is not None else state.final_answer
        termination = (
            result.termination_reason.value
            if result is not None
            else state.termination_reason.value if state.termination_reason is not None else "Not available"
        )
        sections = [
            "# Research Report",
            "",
            "## Question",
            "",
            _inline(run.question),
            "",
            "## Execution",
            "",
            f"- Run ID: `{_inline(run.run_id)}`",
            f"- Created at: `{run.created_at.isoformat()}`",
            f"- Schema version: `{_inline(run.schema_version)}`",
            f"- Termination reason: `{_inline(termination)}`",
            "- Execution Policy:",
            f"  - max_steps: `{run.execution_policy.max_steps}`",
            f"  - max_tool_calls: `{run.execution_policy.max_tool_calls}`",
            f"  - max_same_tool_calls: `{run.execution_policy.max_same_tool_calls}`",
            f"  - global_timeout_seconds: `{run.execution_policy.global_timeout_seconds}`",
            f"  - allowed_tools: `{_inline(', '.join(sorted(run.execution_policy.allowed_tools)) if run.execution_policy.allowed_tools is not None else 'all')}`",
            "",
            "## Sources",
            "",
            *self._sources(state.sources),
            "",
            "## Evidence",
            "",
            *self._evidence(state.evidence),
            "",
            "## Claims",
            "",
            *self._claims(state.claims, state.evidence),
            "",
            "## Answer",
            "",
            *self._answer(answer),
            "",
            "## Citations",
            "",
            *self._citations(answer),
        ]
        return "\n".join(sections)

    @staticmethod
    def _sources(sources: list[Source]) -> list[str]:
        if not sources:
            return ["No data."]
        lines: list[str] = []
        for source in sources:
            lines.extend(
                [
                    f"- **{_inline(source.title)}**",
                    f"  - source_id: `{_inline(source.source_id)}`",
                    f"  - URL: `{_inline(source.url)}`",
                    f"  - type: `{_inline(source.source_type)}`",
                    f"  - retrieved_at: `{source.retrieved_at.isoformat() if source.retrieved_at else 'Not available'}`",
                ]
            )
        return lines

    @staticmethod
    def _evidence(evidence: list[Evidence]) -> list[str]:
        if not evidence:
            return ["No data."]
        lines: list[str] = []
        for item in evidence:
            lines.extend(
                [
                    f"- evidence_id: `{_inline(item.evidence_id)}`",
                    f"  - source_id: `{_inline(item.source_id)}`",
                    f"  - content: {_inline(item.content)}",
                ]
            )
        return lines

    @staticmethod
    def _claims(claims: list[Claim], evidence: list[Evidence]) -> list[str]:
        if not claims:
            return ["No data."]
        evidence_by_id = {item.evidence_id: item for item in evidence}
        lines: list[str] = []
        for claim in claims:
            lines.append(f"- claim_id: `{_inline(claim.claim_id)}`")
            lines.append(f"  - claim: {_inline(claim.text)}")
            lines.append(f"  - evidence_ids: `{_inline(', '.join(claim.evidence_ids))}`")
            for evidence_id in claim.evidence_ids:
                if evidence_id in evidence_by_id:
                    lines.append(f"    - evidence `{_inline(evidence_id)}`: {_inline(evidence_by_id[evidence_id].content)}")
        return lines

    @staticmethod
    def _answer(answer: FinalAnswer | None) -> list[str]:
        if answer is None:
            return ["No data."]
        lines = [answer.answer_text]
        if answer.caveats:
            lines.append(f"\nCaveats: {_inline(answer.caveats)}")
        return lines

    @staticmethod
    def _citations(answer: FinalAnswer | None) -> list[str]:
        if answer is None or not answer.citations:
            return ["No data."]
        return [
            f"- claim: {_inline(citation.claim)}; evidence_ids: `{_inline(', '.join(citation.evidence_ids))}`; source_ids: `{_inline(', '.join(citation.source_ids))}`"
            for citation in answer.citations
        ]