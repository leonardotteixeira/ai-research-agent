import json
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from app.agent.agent import Agent
from app.core.exceptions import InvalidStructuredOutputError, ProviderError
from app.providers.mock_llm import MockLLMProvider
from app.providers.openai_llm import OpenAIProvider
from app.providers.search import MockSearchProvider, SearchHit
from app.schemas.answer import Citation, FinalAnswer
from app.schemas.cost import TokenUsage
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.state import ResearchRequest, ResearchState
from app.schemas.tool import ToolCall
from app.synthesis.mock_provider import MockSynthesisProvider
from app.synthesis.renderer import CitationRenderer
from app.synthesis.service import SynthesisService
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool

TIMESTAMP = datetime.now(UTC)
SOURCE_A = Source(
    source_id="src_a",
    url="https://a.example/article",
    title="Source A",
    tool_call_id="c1",
    timestamp=TIMESTAMP,
)
SOURCE_B = Source(
    source_id="src_b",
    url="https://b.example/report",
    title="Source B",
    tool_call_id="c2",
    timestamp=TIMESTAMP,
)
EVIDENCE_A = Evidence(
    evidence_id="ev_a",
    content="Fact A",
    source_id="src_a",
    tool_call_id="c1",
    timestamp=TIMESTAMP,
)
EVIDENCE_B = Evidence(
    evidence_id="ev_b",
    content="Fact B",
    source_id="src_b",
    tool_call_id="c2",
    timestamp=TIMESTAMP,
)


def state_with_evidence() -> ResearchState:
    return ResearchState(
        research_id="r1",
        original_question="What happened?",
        created_at=TIMESTAMP,
        sources=[SOURCE_A, SOURCE_B],
        evidence=[EVIDENCE_A, EVIDENCE_B],
    )


def answer_with_claims() -> FinalAnswer:
    return FinalAnswer(
        answer_text="Fact A and Fact B.",
        claims=[
            Claim(claim_id="cl_a", text="Fact A", evidence_ids=["ev_a"]),
            Claim(claim_id="cl_b", text="Fact B", evidence_ids=["ev_b"]),
        ],
        citations=[
            Citation(claim="Fact A", evidence_ids=["ev_a"]),
            Citation(claim="Fact B", evidence_ids=["ev_b"]),
        ],
        is_complete=True,
    )


class TestStructuredAnswer:
    def test_claim_without_evidence_is_rejected(self):
        with pytest.raises(ValidationError):
            Claim(claim_id="cl1", text="unsupported", evidence_ids=[])

    def test_answer_serializes_and_restores(self):
        answer = answer_with_claims()

        assert FinalAnswer.model_validate_json(answer.model_dump_json()) == answer


class TestMockSynthesisProvider:
    async def test_returns_deterministic_structured_answer_and_records_context(self):
        provider = MockSynthesisProvider([answer_with_claims()])

        answer, usage = await provider.generate_answer(
            "What happened?",
            [EVIDENCE_A.model_dump(mode="json")],
            [SOURCE_A.model_dump(mode="json")],
            [],
        )

        assert answer.claims[0].evidence_ids == ["ev_a"]
        assert provider.calls[0]["question"] == "What happened?"
        assert provider.calls[0]["evidence"][0]["content"] == "Fact A"
        assert usage is None

    async def test_invalid_answer_is_rejected(self):
        with pytest.raises(InvalidStructuredOutputError):
            MockSynthesisProvider([{"answer_text": "bad", "is_complete": True, "claims": [{"claim_id": "c", "text": "x", "evidence_ids": []}]}])

    async def test_no_answer_remaining_is_controlled_error(self):
        provider = MockSynthesisProvider([])

        with pytest.raises(InvalidStructuredOutputError, match="no answers remaining"):
            await provider.generate_answer("q", [], [], [])

    async def test_configured_usage_is_returned_and_forced_to_actual_false(self):
        provider = MockSynthesisProvider(
            [answer_with_claims()],
            usages=[TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300, actual=True)],
        )

        _, usage = await provider.generate_answer("q", [], [], [])

        assert usage == TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300, actual=False)

    async def test_usages_list_can_contain_none(self):
        provider = MockSynthesisProvider(
            [answer_with_claims(), answer_with_claims()],
            usages=[TokenUsage(total_tokens=10), None],
        )

        _, first_usage = await provider.generate_answer("q", [], [], [])
        _, second_usage = await provider.generate_answer("q", [], [], [])

        assert first_usage is not None and first_usage.total_tokens == 10
        assert second_usage is None

    def test_usages_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="usages must have the same length as answers"):
            MockSynthesisProvider([answer_with_claims()], usages=[TokenUsage(total_tokens=1), TokenUsage(total_tokens=2)])


class TestSynthesisService:
    async def test_agent_evidence_flows_into_synthesis_and_renderer(self):
        search = WebSearchTool(
            MockSearchProvider([SearchHit("https://a.example", "Source A", "Fact A")])
        )
        agent = Agent(
            MockLLMProvider(
                [
                    {"action": "tool_call", "tool_calls": [{"call_id": "c1", "tool_name": "web_search", "arguments": {"query": "fact"}}]},
                    {"action": "synthesize"},
                ]
            ),
            ToolRegistry([search]),
        )

        result = await agent.run(ResearchRequest(question="fact"))
        answer = await SynthesisService(
            MockSynthesisProvider(
                [
                    {
                        "answer_text": "Fact A.",
                        "claims": [{"claim_id": "cl1", "text": "Fact A", "evidence_ids": [result.evidence[0].evidence_id]}],
                        "citations": [{"claim": "Fact A", "evidence_ids": [result.evidence[0].evidence_id]}],
                        "is_complete": True,
                    }
                ]
            )
        ).synthesize(agent.last_state)

        rendered = CitationRenderer().render(answer, agent.last_state)

        assert result.termination_reason.value == "synthesis_requested"
        assert "Fact A [1]." in rendered

    async def test_synthesis_usage_adds_to_agent_usage_exactly_once(self):
        search = WebSearchTool(MockSearchProvider([SearchHit("https://a.example", "Source A", "Fact A")]))
        agent = Agent(
            MockLLMProvider(
                [
                    LLMDecision(
                        action=DecisionAction.TOOL_CALL,
                        tool_calls=[ToolCall(call_id="c1", tool_name="web_search", arguments={"query": "fact"})],
                    ),
                    LLMDecision(action=DecisionAction.SYNTHESIZE),
                ],
                usages=[
                    TokenUsage(input_tokens=250, output_tokens=100, total_tokens=350),
                    TokenUsage(input_tokens=150, output_tokens=100, total_tokens=250),
                ],
            ),
            ToolRegistry([search]),
        )

        result = await agent.run(ResearchRequest(question="fact"))
        assert agent.last_state.token_usage.total_tokens == 600  # 350 + 250 from the two decisions

        await SynthesisService(
            MockSynthesisProvider(
                [
                    FinalAnswer(
                        answer_text="Fact A.",
                        claims=[Claim(claim_id="cl1", text="Fact A", evidence_ids=[result.evidence[0].evidence_id])],
                        citations=[Citation(claim="Fact A", evidence_ids=[result.evidence[0].evidence_id])],
                        is_complete=True,
                    )
                ],
                usages=[TokenUsage(input_tokens=200, output_tokens=50, total_tokens=250)],
            )
        ).synthesize(agent.last_state)

        # 600 (agent: 350 + 250) + 250 (synthesis) = 850, counted exactly once.
        assert agent.last_state.token_usage.total_tokens == 850

    async def test_synthesizes_only_from_state_data(self):
        provider = MockSynthesisProvider([answer_with_claims()])
        service = SynthesisService(provider)

        answer = await service.synthesize(state_with_evidence())

        assert answer.is_complete is True
        assert provider.calls[0]["sources"][0]["url"] == SOURCE_A.url

    async def test_no_evidence_can_return_explicitly_incomplete_answer(self):
        incomplete = FinalAnswer(
            answer_text="There is not enough evidence to answer safely.",
            is_complete=False,
            caveats="No sources were available.",
        )

        answer = await SynthesisService(MockSynthesisProvider([incomplete])).synthesize(
            ResearchState(research_id="r1", original_question="q", created_at=TIMESTAMP)
        )

        assert answer.is_complete is False
        assert answer.citations == []

    async def test_rejects_claim_with_unknown_evidence(self):
        invalid = FinalAnswer(
            answer_text="Unsupported.",
            claims=[Claim(claim_id="cl1", text="Unsupported", evidence_ids=["missing"])],
            is_complete=True,
        )

        with pytest.raises(InvalidStructuredOutputError, match="unknown evidence"):
            await SynthesisService(MockSynthesisProvider([invalid])).synthesize(state_with_evidence())

    async def test_rejects_citation_with_unknown_source(self):
        invalid = FinalAnswer(
            answer_text="Fact.",
            citations=[Citation(claim="Fact", source_ids=["missing"])],
            is_complete=True,
        )

        with pytest.raises(InvalidStructuredOutputError, match="unknown source"):
            await SynthesisService(MockSynthesisProvider([invalid])).synthesize(state_with_evidence())

    async def test_provider_error_is_propagated(self):
        class FailingProvider:
            async def generate_answer(self, question, evidence, sources, claims):
                raise ProviderError("synthesis unavailable")

        with pytest.raises(ProviderError, match="synthesis unavailable"):
            await SynthesisService(FailingProvider()).synthesize(state_with_evidence())


class TestCitationRenderer:
    def test_renders_one_citation_and_source(self):
        answer = FinalAnswer(
            answer_text="Fact A [1].",
            claims=[Claim(claim_id="cl_a", text="Fact A", evidence_ids=["ev_a"])],
            citations=[Citation(claim="Fact A", evidence_ids=["ev_a"])],
            is_complete=True,
        )

        rendered = CitationRenderer().render(answer, state_with_evidence())

        assert rendered == "Fact A [1].\n\nSources:\n[1] Source A - https://a.example/article"

    def test_same_source_is_numbered_once_for_multiple_claims(self):
        answer = FinalAnswer(
            answer_text="Two facts.",
            claims=[
                Claim(claim_id="cl_a", text="Fact A", evidence_ids=["ev_a"]),
                Claim(claim_id="cl_a2", text="Fact A again", evidence_ids=["ev_a"]),
            ],
            citations=[
                Citation(claim="Fact A", evidence_ids=["ev_a"]),
                Citation(claim="Fact A again", evidence_ids=["ev_a"]),
            ],
            is_complete=True,
        )

        rendered = CitationRenderer().render(answer, state_with_evidence())

        assert rendered.count("[1] Source A") == 1
        assert "Sources:" in rendered

    def test_multiple_sources_have_deterministic_order(self):
        first = CitationRenderer().render(answer_with_claims(), state_with_evidence())
        second = CitationRenderer().render(answer_with_claims(), state_with_evidence())

        assert first == second
        assert first.index("[1] Source A") < first.index("[2] Source B")

    def test_renderer_rejects_broken_reference(self):
        answer = FinalAnswer(
            answer_text="Broken.",
            citations=[Citation(claim="Broken", evidence_ids=["missing"])],
            is_complete=True,
        )

        with pytest.raises(InvalidStructuredOutputError):
            CitationRenderer().render(answer, state_with_evidence())


class TestOpenAISynthesis:
    async def test_openai_provider_parses_structured_answer_without_real_network(self):
        body = {
            "choices": [{"message": {"content": json.dumps(answer_with_claims().model_dump(mode="json"))}}]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert payload["response_format"] == {"type": "json_object"}
            assert "untrusted data" in payload["messages"][0]["content"]
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider("test-key", "test-model", 5, client=client)
        answer, usage = await provider.generate_answer("q", [], [], [])

        assert answer.claims[0].evidence_ids == ["ev_a"]
        assert usage is None
        await client.aclose()

    async def test_openai_synthesis_rejects_malformed_json(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider("test-key", "test-model", 5, client=client)

        with pytest.raises(InvalidStructuredOutputError):
            await provider.generate_answer("q", [], [], [])
        await client.aclose()


class TestOpenAISynthesisTokenUsage:
    async def test_extracts_usage_when_present(self):
        body = {
            "choices": [{"message": {"content": json.dumps(answer_with_claims().model_dump(mode="json"))}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider("test-key", "test-model", 5, client=client)

        answer, usage = await provider.generate_answer("q", [], [], [])

        assert answer.claims[0].evidence_ids == ["ev_a"]
        assert usage == TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300, actual=True)
        await client.aclose()

    async def test_usage_is_none_when_absent(self):
        body = {"choices": [{"message": {"content": json.dumps(answer_with_claims().model_dump(mode="json"))}}]}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider("test-key", "test-model", 5, client=client)

        _, usage = await provider.generate_answer("q", [], [], [])

        assert usage is None
        await client.aclose()
