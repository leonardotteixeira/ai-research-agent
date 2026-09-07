"""Derives traceable, numbered citations strictly from a ResearchState's
own Claim -> Evidence -> Source graph -- the exact same graph
`SynthesisService.validate_answer` already validates (app/synthesis/
service.py) and `ResearchState.validate_evidence_graph` already enforces
can't reference anything nonexistent (app/schemas/state.py). This module
adds nothing new to that graph; it only assigns citation numbers and
formats a Reference list from it. A Source that was fetched but never
actually backs a claim's evidence gets no citation number -- citing it
would misrepresent it as having supported a conclusion it never did.
"""

from app.academic.schemas import CitedClaim, Reference
from app.schemas.evidence import Claim
from app.schemas.state import ResearchState


def build_citation_keys(state: ResearchState) -> dict[str, str]:
    """Maps `source_id -> citation_key` ("1", "2", ...), numbered in the
    order each source is first reached by a claim's evidence -- so
    citation [1] in the body always corresponds to REFERÊNCIAS entry 1.
    """
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    source_ids = {source.source_id for source in state.sources}

    ordered_source_ids: list[str] = []
    seen: set[str] = set()
    for claim in state.claims:
        for evidence_id in claim.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None or evidence.source_id not in source_ids:
                continue
            if evidence.source_id not in seen:
                seen.add(evidence.source_id)
                ordered_source_ids.append(evidence.source_id)

    return {source_id: str(index + 1) for index, source_id in enumerate(ordered_source_ids)}


def build_references(state: ResearchState, citation_keys: dict[str, str]) -> list[Reference]:
    source_by_id = {source.source_id: source for source in state.sources}
    references = []
    for source_id, key in citation_keys.items():
        source = source_by_id[source_id]
        references.append(
            Reference(
                citation_key=key,
                source_id=source_id,
                title=source.title or source.url,
                url=source.url,
                accessed_at=source.retrieved_at,
            )
        )
    references.sort(key=lambda reference: int(reference.citation_key))
    return references


def build_cited_claims(claims: list[Claim], state: ResearchState, citation_keys: dict[str, str]) -> list[CitedClaim]:
    """One CitedClaim per persisted Claim, each carrying the citation
    keys of every source its own evidence traces back to (deduplicated,
    in citation-number order) -- never a citation key the claim's
    evidence doesn't actually reach.
    """
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    cited_claims = []
    for claim in claims:
        keys: list[str] = []
        for evidence_id in claim.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            key = citation_keys.get(evidence.source_id)
            if key is not None and key not in keys:
                keys.append(key)
        keys.sort(key=int)
        cited_claims.append(CitedClaim(text=claim.text, citation_keys=keys))
    return cited_claims
