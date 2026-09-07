import json
from datetime import UTC, datetime

from app.cli.formatting import (
    format_replay,
    format_run_human,
    format_run_json,
    format_runs_table,
    format_show_human,
    format_show_json,
    summarize_runs_json,
)
from app.policies.execution import ExecutionPolicy
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.decision import DecisionAction, LLMDecision, Observation
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.run import RunRecord
from app.schemas.state import ResearchResult, ResearchState, TerminationReason
from app.schemas.tool import ToolCall, ToolResult
from app.services.orchestrator import OrchestratorResult


def _timestamp() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _record_with_tool_call() -> RunRecord:
    source = Source(
        source_id="src_1",
        url="https://example.com/a",
        title="A page",
        tool_call_id="call_1",
        timestamp=_timestamp(),
    )
    evidence = Evidence(
        evidence_id="ev_1",
        content="Some evidence text",
        source_id="src_1",
        tool_call_id="call_1",
        timestamp=_timestamp(),
    )
    claim = Claim(claim_id="cl_1", text="Some claim", evidence_ids=["ev_1"])
    answer = FinalAnswer(
        answer_text="Some claim [1].",
        claims=[claim],
        citations=[Citation(claim="Some claim", evidence_ids=["ev_1"])],
        is_complete=True,
    )
    call = ToolCall(call_id="call_1", tool_name="web_search", arguments={"query": "a"})
    result = ToolResult(call_id="call_1", tool_name="web_search", success=True, output=[{"url": source.url}])
    decision_1 = LLMDecision(action=DecisionAction.TOOL_CALL, rationale="search first", tool_calls=[call])
    decision_2 = LLMDecision(action=DecisionAction.SYNTHESIZE, rationale="enough evidence")
    state = ResearchState(
        research_id="run_1",
        original_question="What is the answer?",
        created_at=_timestamp(),
        current_step=2,
        decisions=[decision_1, decision_2],
        tool_calls=[call],
        tool_results=[result],
        observations=[Observation(step=1, tool_results=[result], summary="Executed 1 tool call(s) successfully.")],
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        final_answer=answer,
        termination_reason=TerminationReason.FINISHED,
    )
    research_result = ResearchResult.from_state(state)
    return RunRecord(
        run_id="run_1",
        created_at=_timestamp(),
        question="What is the answer?",
        execution_policy=ExecutionPolicy(),
        research_state=state,
        research_result=research_result,
    )


def _result_of(record: RunRecord) -> ResearchResult:
    assert record.research_result is not None
    return record.research_result


def _record_without_tool_calls() -> RunRecord:
    state = ResearchState(
        research_id="run_2",
        original_question="Quick question?",
        created_at=_timestamp(),
        current_step=1,
        decisions=[LLMDecision(action=DecisionAction.FINISH, rationale="no research needed")],
        termination_reason=TerminationReason.FINISHED,
    )
    research_result = ResearchResult.from_state(state)
    return RunRecord(
        run_id="run_2",
        created_at=_timestamp(),
        question="Quick question?",
        execution_policy=ExecutionPolicy(),
        research_state=state,
        research_result=research_result,
    )


class TestFormatRun:
    def test_human_output_shows_mode_and_key_fields(self) -> None:
        record = _record_with_tool_call()
        result = OrchestratorResult(
            run_id=record.run_id,
            result=_result_of(record),
            record=record,
            report_markdown="# Research Report",
            termination_reason=TerminationReason.FINISHED,
            success=True,
            errors=[],
        )
        output = format_run_human(result, is_mock=False)
        assert "Mode: LIVE" in output
        assert "Run ID: run_1" in output
        assert "Termination: finished" in output
        assert "Success: True" in output
        assert "Sources: 1" in output
        assert "Evidence: 1" in output
        assert "Claims: 1" in output
        assert "Answer: Some claim [1]." in output

    def test_human_output_shows_caveats_and_errors(self) -> None:
        record = _record_without_tool_calls()
        incomplete_answer = FinalAnswer(
            answer_text="Not enough evidence.", claims=[], citations=[], is_complete=False, caveats="No sources found."
        )
        research_result = _result_of(record).model_copy(
            update={"final_answer": incomplete_answer, "errors": ["ProviderError: boom"]}
        )
        result = OrchestratorResult(
            run_id=record.run_id,
            result=research_result,
            record=record,
            report_markdown="# Research Report",
            termination_reason=TerminationReason.FINISHED,
            success=False,
            errors=["ProviderError: boom"],
        )
        output = format_run_human(result, is_mock=False)
        assert "Caveats: No sources found." in output
        assert "Errors:" in output
        assert "  - ProviderError: boom" in output

    def test_human_output_marks_mock_mode(self) -> None:
        record = _record_without_tool_calls()
        result = OrchestratorResult(
            run_id=record.run_id,
            result=_result_of(record),
            record=record,
            report_markdown="# Research Report",
            termination_reason=TerminationReason.FINISHED,
            success=True,
            errors=[],
        )
        output = format_run_human(result, is_mock=True)
        assert "Mode: MOCK" in output

    def test_json_output_is_valid_and_matches_model(self) -> None:
        record = _record_with_tool_call()
        result = OrchestratorResult(
            run_id=record.run_id,
            result=_result_of(record),
            record=record,
            report_markdown="# Research Report",
            termination_reason=TerminationReason.FINISHED,
            success=True,
            errors=[],
        )
        payload = json.loads(format_run_json(result))
        assert payload["run_id"] == "run_1"
        assert payload["success"] is True
        assert OrchestratorResult.model_validate(payload) == result


class TestFormatRuns:
    def test_table_lists_columns(self) -> None:
        record = _record_with_tool_call()
        output = format_runs_table([record])
        assert "run_id" in output
        assert "run_1" in output
        assert "What is the answer?" in output
        assert "finished" in output

    def test_table_empty(self) -> None:
        assert format_runs_table([]) == "No runs found."

    def test_summarize_runs_json_is_minimal(self) -> None:
        record = _record_with_tool_call()
        summary = summarize_runs_json([record])
        assert summary == [
            {
                "run_id": "run_1",
                "created_at": record.created_at.isoformat(),
                "status": "complete",
                "question": "What is the answer?",
                "termination_reason": "finished",
            }
        ]


class TestFormatShow:
    def test_human_output_contains_expected_fields(self) -> None:
        record = _record_with_tool_call()
        output = format_show_human(record)
        assert "Run ID: run_1" in output
        assert "Question: What is the answer?" in output
        assert "Termination: finished" in output
        assert "Steps: 2" in output
        assert "Tool calls: 1" in output
        assert "Answer complete: True" in output

    def test_json_output_round_trips_run_record(self) -> None:
        record = _record_with_tool_call()
        payload = format_show_json(record)
        assert RunRecord.model_validate_json(payload) == record

    def test_human_output_no_answer_and_errors(self) -> None:
        state = ResearchState(
            research_id="run_3",
            original_question="Failed question?",
            created_at=_timestamp(),
            current_step=1,
            errors=["ToolError: registry failure"],
            termination_reason=TerminationReason.TOOL_ERROR,
        )
        record = RunRecord(
            run_id="run_3",
            created_at=_timestamp(),
            question="Failed question?",
            execution_policy=ExecutionPolicy(),
            research_state=state,
            research_result=ResearchResult.from_state(state),
        )
        output = format_show_human(record)
        assert "Answer: (none)" in output
        assert "Errors:" in output
        assert "  - ToolError: registry failure" in output


class TestFormatReplay:
    def test_narrates_steps_tool_calls_and_evidence(self) -> None:
        record = _record_with_tool_call()
        narrative = format_replay(record)
        assert "Question: What is the answer?" in narrative
        assert "Step 1" in narrative
        assert "web_search(call_id=call_1" in narrative
        assert "evidence ev_1: Some evidence text" in narrative
        assert "Step 2" in narrative
        assert "Action: synthesize" in narrative
        assert "Sources (1)" in narrative
        assert "Evidence (1)" in narrative
        assert "Claims (1)" in narrative
        assert "Final Answer" in narrative
        assert "Some claim [1]." in narrative
        assert "Termination: finished" in narrative

    def test_narrates_failed_and_unexecuted_tool_calls_plus_caveats_and_errors(self) -> None:
        failed_call = ToolCall(call_id="call_failed", tool_name="fetch_url", arguments={"url": "https://x"})
        failed_result = ToolResult(
            call_id="call_failed", tool_name="fetch_url", success=False, error="BlockedURLError: refused"
        )
        unexecuted_call = ToolCall(call_id="call_missing", tool_name="calculator", arguments={"expression": "1+1"})
        answer = FinalAnswer(
            answer_text="Partial answer.", claims=[], citations=[], is_complete=False, caveats="Limited evidence."
        )
        state = ResearchState(
            research_id="run_replay_edge",
            original_question="Edge case question?",
            created_at=_timestamp(),
            current_step=2,
            decisions=[
                LLMDecision(should_finish=False, rationale="fetch", tool_calls=[failed_call]),
                LLMDecision(rationale="no explicit action, no tool calls"),
            ],
            tool_calls=[failed_call, unexecuted_call],
            tool_results=[failed_result],
            final_answer=answer,
            errors=["BlockedURLError: refused"],
            termination_reason=TerminationReason.TOOL_ERROR,
        )
        record = RunRecord(
            run_id="run_replay_edge",
            created_at=_timestamp(),
            question="Edge case question?",
            execution_policy=ExecutionPolicy(),
            research_state=state,
            research_result=ResearchResult.from_state(state),
        )
        narrative = format_replay(record)
        assert "Action: tool_call" in narrative
        assert "-> failed: BlockedURLError: refused" in narrative
        assert "Action: unknown" in narrative
        assert "caveats: Limited evidence." in narrative
        assert "Errors:" in narrative
        assert "  - BlockedURLError: refused" in narrative

    def test_narrates_tool_call_missing_result_as_not_executed(self) -> None:
        call = ToolCall(call_id="call_x", tool_name="web_search", arguments={"query": "x"})
        state = ResearchState(
            research_id="run_replay_missing_result",
            original_question="Missing result question?",
            created_at=_timestamp(),
            current_step=1,
            decisions=[LLMDecision(action=DecisionAction.TOOL_CALL, rationale="search", tool_calls=[call])],
            tool_calls=[call],
            termination_reason=TerminationReason.TOOL_ERROR,
        )
        record = RunRecord(
            run_id="run_replay_missing_result",
            created_at=_timestamp(),
            question="Missing result question?",
            execution_policy=ExecutionPolicy(),
            research_state=state,
            research_result=ResearchResult.from_state(state),
        )
        narrative = format_replay(record)
        assert "-> result: (not executed)" in narrative

    def test_narrates_legacy_should_finish_decision_as_finish(self) -> None:
        state = ResearchState(
            research_id="run_replay_should_finish",
            original_question="Legacy finish question?",
            created_at=_timestamp(),
            current_step=1,
            decisions=[LLMDecision(should_finish=True, rationale="done")],
            termination_reason=TerminationReason.FINISHED,
        )
        record = RunRecord(
            run_id="run_replay_should_finish",
            created_at=_timestamp(),
            question="Legacy finish question?",
            execution_policy=ExecutionPolicy(),
            research_state=state,
            research_result=ResearchResult.from_state(state),
        )
        narrative = format_replay(record)
        assert "Action: finish" in narrative

    def test_works_for_runs_without_tool_calls(self) -> None:
        record = _record_without_tool_calls()
        narrative = format_replay(record)
        assert "Question: Quick question?" in narrative
        assert "Step 1" in narrative
        assert "Action: finish" in narrative
        assert "Sources (0)" in narrative
        assert "(none)" in narrative
        assert "Termination: finished" in narrative
