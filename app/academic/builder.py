"""Transforms an already-persisted, already-complete RunRecord into a
structured `AcademicReport` -- the only place in this package that makes
any decision about *what* goes into the document. Never calls an LLM,
a tool, or the network; never re-derives anything the research core
didn't already compute and persist.

Determinism: the same RunRecord always produces the same AcademicReport
(field order, keyword extraction, and citation numbering are all pure
functions of already-persisted data) -- the only thing that legitimately
varies between two builds of the same run is `generated_at`, which is
carried purely as metadata and never influences the document's content
or section ordering.
"""

import re
from collections import Counter
from datetime import UTC, datetime

from app.academic.citations import build_citation_keys, build_cited_claims, build_references
from app.academic.schemas import AcademicMetadata, AcademicReport, Section
from app.schemas.run import RunRecord
from app.schemas.state import TerminationReason

_STOPWORDS = {
    # Portuguese
    "a", "as", "o", "os", "de", "da", "do", "das", "dos", "e", "em", "um", "uma",
    "uns", "umas", "para", "por", "com", "sem", "sobre", "entre", "que", "qual",
    "quais", "quando", "como", "onde", "mais", "menos", "muito", "pouco", "ser",
    "estar", "ter", "haver", "sua", "seu", "suas", "seus", "no", "na", "nos",
    "nas", "ao", "aos", "à", "às", "é", "são", "foi", "foram", "ou", "não", "se",
    "pela", "pelo", "pelas", "pelos", "isso", "esta", "este", "estes", "estas",
    # English (question/content may be in English)
    "the", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "be", "been", "with", "without", "about", "between", "what",
    "which", "when", "how", "where", "more", "less", "most", "least", "it",
    "its", "this", "that", "these", "those", "at", "by", "from", "into",
}

_WORD_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ]{4,}")


def _extract_keywords(texts: list[str], limit: int = 6) -> list[str]:
    """Deterministic, frequency-based extraction over the research's own
    text (question + claim texts) -- no LLM call, no external wordlist
    beyond a small built-in stopword set. Ties are broken by first
    appearance, so the result is stable across repeated builds.
    """
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    position = 0
    for text in texts:
        for match in _WORD_PATTERN.finditer(text.lower()):
            word = match.group(0)
            if word in _STOPWORDS:
                continue
            counts[word] += 1
            first_seen.setdefault(word, position)
            position += 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], first_seen[pair[0]]))
    return [word.capitalize() for word, _ in ranked[:limit]]


def _methodology_paragraph(run: RunRecord) -> str:
    state = run.research_state
    tool_names = sorted({call.tool_name for call in state.tool_calls})
    tools_text = ", ".join(tool_names) if tool_names else "nenhuma ferramenta externa"
    return (
        f"Esta pesquisa foi conduzida por um agente autônomo que executou {state.current_step} "
        f"etapa(s) de decisão, utilizando as seguintes ferramentas: {tools_text}. "
        f"Foram consultadas {len(state.sources)} fonte(s) distinta(s), gerando "
        f"{len(state.evidence)} unidade(s) de evidência. A execução foi encerrada com o "
        f"motivo '{_termination_label(state.termination_reason)}'."
    )


def _termination_label(reason: TerminationReason | None) -> str:
    return reason.value if reason is not None else "indisponível"


def _abstract(run: RunRecord, claims_count: int) -> str:
    state = run.research_state
    answer = state.final_answer
    completion = (
        "A resposta foi considerada completa com base nas evidências reunidas."
        if answer is not None and answer.is_complete
        else "A resposta foi sinalizada como incompleta em relação à evidência disponível."
        if answer is not None
        else "Nenhuma resposta final foi sintetizada para esta execução."
    )
    return (
        f"Este trabalho investiga a questão: \"{run.question}\". A pesquisa foi conduzida por um "
        f"agente autônomo que selecionou e executou ferramentas de busca e análise, reunindo "
        f"{len(state.sources)} fonte(s) e {claims_count} afirmação(ões) sustentada(s) por evidência. "
        f"{completion}"
    )


def build_academic_report(run: RunRecord, metadata: AcademicMetadata, title: str | None = None) -> AcademicReport:
    if run.research_result is None:
        raise ValueError(f"run {run.run_id!r} is not finalized; cannot build an academic report from it")

    state = run.research_state
    answer = state.final_answer
    claims = list(answer.claims) if answer is not None else list(state.claims)

    citation_keys = build_citation_keys(state)
    references = build_references(state, citation_keys)
    cited_claims = build_cited_claims(claims, state, citation_keys)

    sections: list[Section] = [
        Section(number="1", title="INTRODUÇÃO", paragraphs=[run.question]),
        Section(number="2", title="METODOLOGIA", paragraphs=[_methodology_paragraph(run)]),
    ]

    if answer is not None and answer.answer_text.strip():
        paragraphs = [p.strip() for p in answer.answer_text.split("\n") if p.strip()]
    else:
        paragraphs = ["Nenhuma resposta final foi produzida para esta execução."]
    sections.append(Section(number="3", title="DESENVOLVIMENTO", paragraphs=paragraphs))

    if cited_claims:
        sections.append(Section(number="4", title="RESULTADOS E ANÁLISE", cited_claims=cited_claims))

    discussion_paragraphs: list[str] = []
    if answer is not None and not answer.is_complete:
        discussion_paragraphs.append(
            answer.caveats or "A evidência disponível não foi suficiente para uma resposta completa."
        )
    if state.errors:
        error_types = Counter(error.split(":", 1)[0] for error in state.errors)
        summary = ", ".join(f"{count}x {kind}" for kind, count in sorted(error_types.items()))
        discussion_paragraphs.append(
            f"Durante a execução, foram registrados os seguintes eventos não críticos: {summary}. "
            "Nenhum deles impediu a conclusão da pesquisa."
        )
    if discussion_paragraphs:
        sections.append(
            Section(number=str(len(sections) + 1), title="DISCUSSÃO", paragraphs=discussion_paragraphs)
        )

    conclusion_number = str(len(sections) + 1)
    sections.append(
        Section(
            number=conclusion_number,
            title="CONCLUSÃO",
            paragraphs=[_abstract(run, len(claims))],
        )
    )

    keywords = _extract_keywords([run.question, *(claim.text for claim in claims)])

    return AcademicReport(
        metadata=metadata,
        title=title or run.question,
        abstract=_abstract(run, len(claims)),
        keywords=keywords,
        sections=sections,
        references=references,
        source_run_id=run.run_id,
        generated_at=datetime.now(UTC),
    )
