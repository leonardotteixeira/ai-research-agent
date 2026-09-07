from datetime import UTC, datetime

import pytest

from app.academic.builder import build_academic_report
from app.academic.schemas import AcademicMetadata
from app.policies.execution import ExecutionPolicy
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.run import RunRecord
from app.schemas.state import ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall

TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


def _metadata() -> AcademicMetadata:
    return AcademicMetadata(year=2026)


def _finalized_record(state: ResearchState) -> RunRecord:
    return RunRecord(
        run_id=state.research_id,
        created_at=state.created_at,
        question=state.original_question,
        execution_policy=ExecutionPolicy(),
        research_state=state,
        research_result=ResearchResult.from_state(state),
    )


def _complete_state_with_answer() -> ResearchState:
    source = Source(source_id="src_1", url="https://example.com/a", title="A Page", tool_call_id="c1", timestamp=TIMESTAMP)
    evidence = Evidence(evidence_id="ev_1", content="Paris is the capital.", source_id="src_1", tool_call_id="c1", timestamp=TIMESTAMP)
    claim = Claim(claim_id="cl_1", text="Paris is the capital of France.", evidence_ids=["ev_1"])
    answer = FinalAnswer(
        answer_text="Paris is the capital of France [1].",
        claims=[claim],
        citations=[Citation(claim=claim.text, evidence_ids=["ev_1"])],
        is_complete=True,
    )
    return ResearchState(
        research_id="run_1",
        original_question="What is the capital of France?",
        created_at=TIMESTAMP,
        current_step=2,
        decisions=[
            LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=[ToolCall(call_id="c1", tool_name="web_search", arguments={"query": "x"})]),
            LLMDecision(action=DecisionAction.SYNTHESIZE),
        ],
        tool_calls=[ToolCall(call_id="c1", tool_name="web_search", arguments={"query": "x"})],
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        final_answer=answer,
        termination_reason=TerminationReason.FINISHED,
    )


class TestBuildAcademicReportHappyPath:
    def test_produces_all_core_sections_with_traceable_citation(self):
        record = _finalized_record(_complete_state_with_answer())

        report = build_academic_report(record, _metadata())

        section_titles = [s.title for s in report.sections]
        assert "INTRODUÇÃO" in section_titles
        assert "METODOLOGIA" in section_titles
        assert "DESENVOLVIMENTO" in section_titles
        assert "RESULTADOS E ANÁLISE" in section_titles
        assert "CONCLUSÃO" in section_titles
        assert len(report.references) == 1
        assert report.references[0].citation_key == "1"
        results_section = next(s for s in report.sections if s.title == "RESULTADOS E ANÁLISE")
        assert results_section.cited_claims[0].citation_keys == ["1"]

    def test_title_defaults_to_the_question_but_is_overridable(self):
        record = _finalized_record(_complete_state_with_answer())

        default_report = build_academic_report(record, _metadata())
        overridden_report = build_academic_report(record, _metadata(), title="Custom Title")

        assert default_report.title == "What is the capital of France?"
        assert overridden_report.title == "Custom Title"

    def test_source_run_id_matches_the_run(self):
        record = _finalized_record(_complete_state_with_answer())
        report = build_academic_report(record, _metadata())
        assert report.source_run_id == "run_1"

    def test_is_deterministic_across_repeated_builds(self):
        record = _finalized_record(_complete_state_with_answer())

        first = build_academic_report(record, _metadata())
        second = build_academic_report(record, _metadata())

        assert first.model_dump(exclude={"generated_at"}) == second.model_dump(exclude={"generated_at"})


class TestBuildAcademicReportMissingData:
    def test_refuses_an_unfinished_run(self):
        state = _complete_state_with_answer()
        state.termination_reason = None
        state.final_answer = None
        record = RunRecord(
            run_id=state.research_id,
            created_at=state.created_at,
            question=state.original_question,
            execution_policy=ExecutionPolicy(),
            research_state=state,
            research_result=None,
        )

        with pytest.raises(ValueError, match="not finalized"):
            build_academic_report(record, _metadata())

    def test_no_claims_omits_the_results_section_without_crashing(self):
        state = ResearchState(
            research_id="run_2",
            original_question="A question with no claims.",
            created_at=TIMESTAMP,
            current_step=1,
            termination_reason=TerminationReason.FINISHED,
        )
        record = _finalized_record(state)

        report = build_academic_report(record, _metadata())

        assert "RESULTADOS E ANÁLISE" not in [s.title for s in report.sections]
        assert report.references == []

    def test_no_final_answer_uses_neutral_text_not_fabricated_content(self):
        state = ResearchState(
            research_id="run_3",
            original_question="A question without synthesis.",
            created_at=TIMESTAMP,
            current_step=1,
            termination_reason=TerminationReason.MAX_STEPS,
        )
        record = _finalized_record(state)

        report = build_academic_report(record, _metadata())

        development = next(s for s in report.sections if s.title == "DESENVOLVIMENTO")
        assert "Nenhuma resposta final foi produzida" in development.paragraphs[0]

    def test_incomplete_answer_adds_a_discussion_section_with_its_own_caveats(self):
        state = _complete_state_with_answer()
        state.final_answer = FinalAnswer(
            answer_text="Partial answer.",
            claims=[],
            citations=[],
            is_complete=False,
            caveats="Insufficient evidence was found.",
        )
        record = _finalized_record(state)

        report = build_academic_report(record, _metadata())

        discussion = next(s for s in report.sections if s.title == "DISCUSSÃO")
        assert "Insufficient evidence was found." in discussion.paragraphs

    def test_recorded_errors_are_summarized_by_type_not_dumped_verbatim(self):
        state = _complete_state_with_answer()
        state.errors = ["ValueError: https://x returned HTTP 403", "ValueError: https://y timed out"]
        record = _finalized_record(state)

        report = build_academic_report(record, _metadata())

        discussion = next(s for s in report.sections if s.title == "DISCUSSÃO")
        assert any("2x ValueError" in paragraph for paragraph in discussion.paragraphs)
        # The raw, possibly-noisy error text itself is never embedded verbatim.
        assert not any("403" in paragraph for paragraph in discussion.paragraphs)


class TestKeywordExtraction:
    def test_keywords_are_derived_only_from_the_runs_own_text(self):
        record = _finalized_record(_complete_state_with_answer())
        report = build_academic_report(record, _metadata())

        assert report.keywords
        assert all(isinstance(word, str) for word in report.keywords)
        # Nothing outside the question/claims text could have been used.
        combined = (record.question + " " + record.research_state.claims[0].text).lower()
        assert all(word.lower() in combined for word in report.keywords)
