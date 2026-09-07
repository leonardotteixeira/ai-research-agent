from pathlib import Path

from app.agent.agent import Agent
from app.evidence.pipeline import EvidencePipeline
from app.persistence.file_repository import FileRunRepository
from app.policies.execution import ExecutionPolicy
from app.providers.mock_llm import MockLLMProvider
from app.providers.search import MockSearchProvider, SearchHit
from app.reports.markdown import MarkdownReportRenderer
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.cost import TokenUsage
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.evidence import Claim
from app.schemas.state import ResearchRequest, TerminationReason
from app.schemas.tool import ToolCall
from app.services.orchestrator import OrchestratorResult, ResearchOrchestrator
from app.services.run_service import RunService
from app.synthesis.mock_provider import MockSynthesisProvider
from app.synthesis.service import SynthesisService
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


class TestOrchestratorE2EOffline:
    """Verifies the complete offline E2E workflow:
    ResearchRequest
    → real Agent
    → MockLLMProvider
    → real ToolRegistry (with deterministic WebSearchTool & MockSearchProvider)
    → real EvidencePipeline
    → real SynthesisService (with MockSynthesisProvider)
    → real RunService
    → real FileRunRepository (writing to tmp_path)
    → real MarkdownReportRenderer
    → OrchestratorResult
    """

    async def test_full_offline_research_pipeline(self, tmp_path: Path) -> None:
        call_id = "c_search_1"
        evidence_id = f"ev_{call_id}_0"
        query = "vector database for postgres"

        search_corpus = [
            SearchHit(
                url="https://postgres.example/pgvector",
                title="pgvector: Vector similarity search for Postgres",
                snippet="pgvector adds vector similarity search directly to PostgreSQL.",
            )
        ]
        search_provider = MockSearchProvider(search_corpus)
        web_search_tool = WebSearchTool(search_provider)
        tool_registry = ToolRegistry([web_search_tool])

        llm_decisions = [
            LLMDecision(
                action=DecisionAction.TOOL_CALL,
                rationale="Search for vector database solutions for postgres.",
                tool_calls=[ToolCall(call_id=call_id, tool_name="web_search", arguments={"query": query})],
            ),
            LLMDecision(
                action=DecisionAction.SYNTHESIZE,
                rationale="Sufficient evidence found; synthesizing final answer.",
            ),
        ]
        llm_provider = MockLLMProvider(llm_decisions)

        evidence_pipeline = EvidencePipeline()
        policy = ExecutionPolicy(max_steps=5, max_tool_calls=5)

        agent = Agent(
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            policy=policy,
            evidence_pipeline=evidence_pipeline,
        )

        expected_claim = Claim(
            claim_id="claim_1",
            text="pgvector adds vector similarity search directly to PostgreSQL",
            evidence_ids=[evidence_id],
        )
        expected_answer = FinalAnswer(
            answer_text="pgvector adds vector similarity search directly to PostgreSQL [1].",
            claims=[expected_claim],
            citations=[Citation(claim=expected_claim.text, evidence_ids=[evidence_id])],
            is_complete=True,
        )
        synthesis_provider = MockSynthesisProvider([expected_answer])
        synthesis_service = SynthesisService(synthesis_provider)

        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)
        report_renderer = MarkdownReportRenderer()

        orchestrator = ResearchOrchestrator(
            agent=agent,
            synthesis_service=synthesis_service,
            run_service=run_service,
            report_renderer=report_renderer,
        )

        request = ResearchRequest(question="Which vector database works directly in Postgres?")
        result = await orchestrator.run(request)

        # 1. Output structure assertions
        assert isinstance(result, OrchestratorResult)
        assert result.success is True
        assert result.termination_reason == TerminationReason.FINISHED
        assert result.result.final_answer == expected_answer
        assert len(result.result.sources) == 1
        assert len(result.result.evidence) == 1
        assert result.result.evidence[0].evidence_id == evidence_id
        assert len(result.result.claims) == 1

        # 2. File persistence assertions (atomic file on disk)
        run_dir = tmp_path / result.run_id
        run_file = run_dir / "run.json"
        assert run_dir.is_dir()
        assert run_file.is_file()

        loaded_record = run_service.load_run(result.run_id)
        assert loaded_record == result.record
        assert loaded_record.run_id == result.run_id
        assert loaded_record.question == request.question
        assert loaded_record.research_result == result.result
        assert loaded_record.research_state.final_answer == expected_answer

        # 3. Markdown report assertions
        assert "# Research Report" in result.report_markdown
        assert "## Question" in result.report_markdown
        assert request.question in result.report_markdown
        assert "## Sources" in result.report_markdown
        assert "pgvector: Vector similarity search for Postgres" in result.report_markdown
        assert "https://postgres.example/pgvector" in result.report_markdown
        assert "## Evidence" in result.report_markdown
        assert "pgvector adds vector similarity search" in result.report_markdown
        assert "## Answer" in result.report_markdown
        assert "pgvector adds vector similarity search directly to PostgreSQL [1]." in result.report_markdown

    async def test_token_usage_survives_persistence_round_trip(self, tmp_path: Path) -> None:
        """Agent decisions (280 + 120 = 400 tokens) plus synthesis (300
        tokens) must accumulate to exactly 500 input / 200 output / 700
        total, and that total must be byte-identical after being persisted
        by FileRunRepository and reloaded by an independent RunService --
        Agent -> ResearchState -> SynthesisService -> ResearchResult ->
        RunRecord -> FileRunRepository -> load.
        """
        call_id = "c_search_1"
        search_provider = MockSearchProvider(
            [SearchHit(url="https://fixture.example/page", title="Page", snippet="Some fact.")]
        )
        tool_registry = ToolRegistry([WebSearchTool(search_provider)])  # type: ignore[list-item]

        llm_provider = MockLLMProvider(
            [
                LLMDecision(
                    action=DecisionAction.TOOL_CALL,
                    tool_calls=[ToolCall(call_id=call_id, tool_name="web_search", arguments={"query": "fact"})],
                ),
                LLMDecision(action=DecisionAction.SYNTHESIZE),
            ],
            usages=[
                TokenUsage(input_tokens=200, output_tokens=80, total_tokens=280),
                TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            ],
        )
        agent = Agent(
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            policy=ExecutionPolicy(max_steps=5, max_tool_calls=5),
            evidence_pipeline=EvidencePipeline(),
        )

        evidence_id = f"ev_{call_id}_0"
        claim = Claim(claim_id="claim_1", text="Some fact.", evidence_ids=[evidence_id])
        answer = FinalAnswer(
            answer_text="Some fact. [1].",
            claims=[claim],
            citations=[Citation(claim=claim.text, evidence_ids=[evidence_id])],
            is_complete=True,
        )
        synthesis_service = SynthesisService(
            MockSynthesisProvider(
                [answer],
                usages=[TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300)],
            )
        )

        repository = FileRunRepository(tmp_path)
        run_service = RunService(repository)
        orchestrator = ResearchOrchestrator(
            agent=agent,
            synthesis_service=synthesis_service,
            run_service=run_service,
            report_renderer=MarkdownReportRenderer(),
        )

        result = await orchestrator.run(ResearchRequest(question="What is the fact?"))

        expected = {"input_tokens": 500, "output_tokens": 200, "total_tokens": 700}
        assert result.record.research_state.token_usage.input_tokens == expected["input_tokens"]
        assert result.record.research_state.token_usage.output_tokens == expected["output_tokens"]
        assert result.record.research_state.token_usage.total_tokens == expected["total_tokens"]
        assert result.result.token_usage.total_tokens == expected["total_tokens"]

        # Independent RunService/FileRunRepository instance -- a fresh "process".
        reloaded = RunService(FileRunRepository(tmp_path)).load_run(result.run_id)
        assert reloaded.research_state.token_usage.input_tokens == expected["input_tokens"]
        assert reloaded.research_state.token_usage.output_tokens == expected["output_tokens"]
        assert reloaded.research_state.token_usage.total_tokens == expected["total_tokens"]
        assert reloaded.research_result is not None
        assert reloaded.research_result.token_usage.total_tokens == expected["total_tokens"]
        assert reloaded == result.record
