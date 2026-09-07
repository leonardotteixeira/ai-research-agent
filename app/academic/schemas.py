"""Structured intermediate models between a persisted RunRecord and the
rendered PDF. `AcademicReportBuilder` (builder.py) is the only thing that
constructs an `AcademicReport`; `AcademicPDFRenderer` (renderer.py) only
ever reads one -- neither ever touches a RunRecord directly, keeping the
"what data" and "how it looks on paper" concerns separate.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class AcademicMetadata(BaseModel):
    """Everything about the *submission* that a RunRecord can never know
    on its own -- who is submitting the work, and to which institution.
    Defaults reflect this project's own author/institution, but every
    field is a plain, overridable value -- nothing here is hardcoded into
    the builder or renderer, so a future user of this project can supply
    entirely different values (e.g. via CLI flags) without touching code.
    """

    author: str = "Leonardo Teixeira"
    registration: str | None = "245602"
    institution: str = "Universidade Estadual de Campinas – UNICAMP"
    unit: str | None = "Faculdade de Engenharia Agrícola – FEAGRI"
    city: str = "Campinas – SP"
    year: int
    course: str | None = None
    advisor: str | None = None
    note: str | None = (
        "Trabalho acadêmico apresentado como parte das atividades acadêmicas."
    )


class Reference(BaseModel):
    """One REFERÊNCIAS entry, derived strictly from a persisted `Source`
    that was actually cited by at least one claim (see citations.py) --
    never fabricated. `authors`/`year` are left `None` (never guessed)
    when the Source itself doesn't carry that information, which is the
    normal case for a fetched web page: title + URL + retrieval date are
    the only fields ever actually available.
    """

    citation_key: str
    source_id: str
    title: str
    url: str
    accessed_at: datetime | None = None
    authors: str | None = None
    year: int | None = None


class CitedClaim(BaseModel):
    """One claim from the research, with the citation keys of every
    source its evidence traces back to -- the rendered form of the
    Claim -> Evidence -> Source -> Citation chain the research core
    already builds (see app/schemas/evidence.py, app/evidence/pipeline.py).
    """

    text: str
    citation_keys: list[str] = Field(default_factory=list)


class Section(BaseModel):
    number: str
    title: str
    paragraphs: list[str] = Field(default_factory=list)
    cited_claims: list[CitedClaim] = Field(default_factory=list)
    subsections: list["Section"] = Field(default_factory=list)


Section.model_rebuild()


class AcademicReport(BaseModel):
    """The fully-resolved content of the document -- nothing the renderer
    reads from here required any decision-making of its own; all of that
    already happened in `AcademicReportBuilder`.
    """

    metadata: AcademicMetadata
    title: str
    abstract: str
    keywords: list[str] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    source_run_id: str
    generated_at: datetime
