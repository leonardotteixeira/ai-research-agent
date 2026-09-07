from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.evidence.pipeline import EvidencePipeline
from app.schemas.evidence import Claim, Evidence, Source, normalize_source_url
from app.schemas.state import ResearchState
from app.schemas.tool import ToolResult


def result(tool_name: str, output, call_id: str = "c1") -> ToolResult:
    return ToolResult(call_id=call_id, tool_name=tool_name, success=True, output=output)


class TestEvidenceModels:
    def test_source_serializes_with_metadata(self):
        source = Source(
            source_id="src1",
            url="https://example.com/article",
            title="Article",
            source_type="search_result",
            retrieved_at=datetime.now(UTC),
            tool_call_id="c1",
            timestamp=datetime.now(UTC),
        )

        restored = Source.model_validate_json(source.model_dump_json())

        assert restored == source
        assert restored.source_type == "search_result"

    @pytest.mark.parametrize("url", ["", "file:///tmp/x", "https://", "https://example.com:bad"])
    def test_source_rejects_invalid_url(self, url):
        with pytest.raises(ValidationError, match="absolute HTTP"):
            Source(source_id="src1", url=url, tool_call_id="c1", timestamp=datetime.now(UTC))

    def test_normalization_removes_fragment_and_trailing_slash_only(self):
        assert normalize_source_url("HTTPS://Example.com/article/#section") == "https://example.com/article"
        assert normalize_source_url("https://example.com/article?x=1/") == "https://example.com/article?x=1/"

    def test_evidence_serializes_and_links_to_source(self):
        evidence = Evidence(
            evidence_id="ev1",
            content="The source says this.",
            source_id="src1",
            tool_call_id="c1",
            timestamp=datetime.now(UTC),
        )

        assert Evidence.model_validate_json(evidence.model_dump_json()) == evidence

    def test_claim_links_to_evidence(self):
        claim = Claim(claim_id="cl1", text="A claim", evidence_ids=["ev1"])
        assert Claim.model_validate_json(claim.model_dump_json()) == claim


class TestEvidencePipeline:
    def test_search_results_become_sources_and_evidence(self):
        pipeline = EvidencePipeline()
        sources, evidence = pipeline.process(
            result(
                "web_search",
                [
                    {"url": "https://a.example", "title": "A", "snippet": "Snippet A"},
                    {"url": "https://b.example", "title": "B", "snippet": "Snippet B"},
                ],
            )
        )

        assert [source.url for source in sources] == ["https://a.example", "https://b.example"]
        assert [item.content for item in evidence] == ["Snippet A", "Snippet B"]
        assert all(item.source_id == source.source_id for item, source in zip(evidence, sources, strict=True))

    def test_fetch_result_becomes_source_and_full_text_evidence(self):
        sources, evidence = EvidencePipeline().process(
            result("fetch_url", {"url": "https://a.example/page", "text": "Full page text", "truncated": False})
        )

        assert sources[0].source_type == "fetched_page"
        assert evidence[0].content == "Full page text"
        assert evidence[0].source_id == sources[0].source_id

    def test_same_url_is_reused_across_search_and_fetch(self):
        pipeline = EvidencePipeline()
        sources: list[Source] = []
        evidence: list[Evidence] = []
        pipeline.merge(
            sources,
            evidence,
            result("web_search", [{"url": "https://a.example/page/#top", "title": "A", "snippet": "A snippet"}]),
        )
        pipeline.merge(
            sources,
            evidence,
            result("fetch_url", {"url": "https://a.example/page/", "text": "Page text", "truncated": False}, "c2"),
        )

        assert len(sources) == 1
        assert len(evidence) == 2
        assert {item.source_id for item in evidence} == {sources[0].source_id}

    def test_duplicate_search_urls_are_deduplicated(self):
        pipeline = EvidencePipeline()
        sources: list[Source] = []
        evidence: list[Evidence] = []
        pipeline.merge(
            sources,
            evidence,
            result(
                "web_search",
                [
                    {"url": "https://a.example/page", "title": "A", "snippet": "First"},
                    {"url": "https://a.example/page/#section", "title": "A", "snippet": "Second"},
                ],
            ),
        )

        assert len(sources) == 1
        assert len(evidence) == 2

    def test_non_source_tool_output_is_ignored(self):
        assert EvidencePipeline().process(result("calculator", 5)) == ([], [])


class TestEvidenceGraphValidation:
    def test_state_accepts_valid_claim_evidence_source_graph(self):
        timestamp = datetime.now(UTC)
        source = Source(source_id="src1", url="https://example.com", tool_call_id="c1", timestamp=timestamp)
        evidence = Evidence(evidence_id="ev1", content="text", source_id="src1", tool_call_id="c1", timestamp=timestamp)
        state = ResearchState(
            research_id="r1",
            original_question="Q",
            created_at=timestamp,
            sources=[source],
            evidence=[evidence],
            claims=[Claim(claim_id="cl1", text="claim", evidence_ids=["ev1"])],
        )

        assert state.claims[0].evidence_ids == ["ev1"]

    def test_state_rejects_evidence_with_unknown_source(self):
        timestamp = datetime.now(UTC)
        with pytest.raises(ValidationError, match="unknown source"):
            ResearchState(
                research_id="r1",
                original_question="Q",
                created_at=timestamp,
                evidence=[Evidence(evidence_id="ev1", content="text", source_id="missing", tool_call_id="c1", timestamp=timestamp)],
            )

    def test_state_rejects_claim_with_unknown_evidence(self):
        timestamp = datetime.now(UTC)
        with pytest.raises(ValidationError, match="unknown evidence"):
            ResearchState(
                research_id="r1",
                original_question="Q",
                created_at=timestamp,
                claims=[Claim(claim_id="cl1", text="claim", evidence_ids=["missing"])],
            )