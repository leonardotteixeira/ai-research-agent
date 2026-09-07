from datetime import UTC, datetime

from app.academic.citations import build_citation_keys, build_cited_claims, build_references
from app.schemas.evidence import Claim, Evidence, Source
from app.schemas.state import ResearchState

TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


def _state(sources, evidence, claims) -> ResearchState:
    return ResearchState(
        research_id="run_1",
        original_question="Q?",
        created_at=TIMESTAMP,
        sources=sources,
        evidence=evidence,
        claims=claims,
    )


def _source(source_id: str, url: str, title: str | None = None) -> Source:
    return Source(source_id=source_id, url=url, title=title, tool_call_id="c1", timestamp=TIMESTAMP)


def _evidence(evidence_id: str, source_id: str, content: str = "x") -> Evidence:
    return Evidence(evidence_id=evidence_id, content=content, source_id=source_id, tool_call_id="c1", timestamp=TIMESTAMP)


class TestBuildCitationKeys:
    def test_numbers_sources_in_first_use_order(self):
        state = _state(
            sources=[_source("src_a", "https://a"), _source("src_b", "https://b")],
            evidence=[_evidence("ev_a", "src_a"), _evidence("ev_b", "src_b")],
            claims=[
                Claim(claim_id="c1", text="Claim B first.", evidence_ids=["ev_b"]),
                Claim(claim_id="c2", text="Claim A second.", evidence_ids=["ev_a"]),
            ],
        )

        keys = build_citation_keys(state)

        assert keys == {"src_b": "1", "src_a": "2"}

    def test_a_source_never_reached_by_a_claim_gets_no_key(self):
        state = _state(
            sources=[_source("src_used", "https://a"), _source("src_unused", "https://b")],
            evidence=[_evidence("ev_used", "src_used")],
            claims=[Claim(claim_id="c1", text="Only cites the used source.", evidence_ids=["ev_used"])],
        )

        keys = build_citation_keys(state)

        assert keys == {"src_used": "1"}
        assert "src_unused" not in keys

    def test_no_claims_means_no_citations(self):
        state = _state(
            sources=[_source("src_a", "https://a")], evidence=[_evidence("ev_a", "src_a")], claims=[]
        )
        assert build_citation_keys(state) == {}

    def test_same_source_reused_by_multiple_claims_gets_one_key(self):
        state = _state(
            sources=[_source("src_a", "https://a")],
            evidence=[_evidence("ev_1", "src_a"), _evidence("ev_2", "src_a")],
            claims=[
                Claim(claim_id="c1", text="First claim.", evidence_ids=["ev_1"]),
                Claim(claim_id="c2", text="Second claim.", evidence_ids=["ev_2"]),
            ],
        )

        keys = build_citation_keys(state)

        assert keys == {"src_a": "1"}


class TestBuildReferences:
    def test_never_fabricates_author_or_year(self):
        state = _state(
            sources=[_source("src_a", "https://a/page", title="A Page")],
            evidence=[_evidence("ev_a", "src_a")],
            claims=[Claim(claim_id="c1", text="Claim.", evidence_ids=["ev_a"])],
        )
        keys = build_citation_keys(state)

        references = build_references(state, keys)

        assert len(references) == 1
        assert references[0].authors is None
        assert references[0].year is None
        assert references[0].title == "A Page"
        assert references[0].url == "https://a/page"

    def test_falls_back_to_url_when_title_is_missing(self):
        state = _state(
            sources=[_source("src_a", "https://a/page", title=None)],
            evidence=[_evidence("ev_a", "src_a")],
            claims=[Claim(claim_id="c1", text="Claim.", evidence_ids=["ev_a"])],
        )
        keys = build_citation_keys(state)

        references = build_references(state, keys)

        assert references[0].title == "https://a/page"

    def test_references_are_sorted_by_citation_number(self):
        state = _state(
            sources=[_source("src_a", "https://a"), _source("src_b", "https://b")],
            evidence=[_evidence("ev_a", "src_a"), _evidence("ev_b", "src_b")],
            claims=[
                Claim(claim_id="c1", text="B first.", evidence_ids=["ev_b"]),
                Claim(claim_id="c2", text="A second.", evidence_ids=["ev_a"]),
            ],
        )
        keys = build_citation_keys(state)

        references = build_references(state, keys)

        assert [r.citation_key for r in references] == ["1", "2"]
        assert [r.source_id for r in references] == ["src_b", "src_a"]


class TestBuildCitedClaims:
    def test_each_claim_carries_its_own_sources_citation_keys(self):
        state = _state(
            sources=[_source("src_a", "https://a"), _source("src_b", "https://b")],
            evidence=[_evidence("ev_a", "src_a"), _evidence("ev_b", "src_b")],
            claims=[Claim(claim_id="c1", text="Multi-source claim.", evidence_ids=["ev_a", "ev_b"])],
        )
        keys = build_citation_keys(state)

        cited = build_cited_claims(state.claims, state, keys)

        assert len(cited) == 1
        assert cited[0].text == "Multi-source claim."
        assert sorted(cited[0].citation_keys) == sorted(keys.values())

    def test_claim_with_no_resolvable_evidence_has_no_citation_keys(self):
        # A claim can be built directly (outside ResearchState's own
        # cross-field validation) referencing an evidence_id the state
        # itself doesn't have -- build_cited_claims must degrade to "no
        # citation" rather than raise or fabricate one.
        state = _state(sources=[], evidence=[], claims=[])
        orphan_claim = Claim(claim_id="c1", text="Dangling claim.", evidence_ids=["ev_missing"])

        cited = build_cited_claims([orphan_claim], state, {})

        assert cited[0].citation_keys == []

    def test_duplicate_source_citations_are_not_repeated(self):
        state = _state(
            sources=[_source("src_a", "https://a")],
            evidence=[_evidence("ev_1", "src_a"), _evidence("ev_2", "src_a")],
            claims=[Claim(claim_id="c1", text="Cites the same source twice.", evidence_ids=["ev_1", "ev_2"])],
        )
        keys = build_citation_keys(state)

        cited = build_cited_claims(state.claims, state, keys)

        assert cited[0].citation_keys == ["1"]
