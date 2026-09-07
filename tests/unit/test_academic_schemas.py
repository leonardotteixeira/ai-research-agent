from datetime import UTC, datetime

from app.academic.schemas import AcademicMetadata, AcademicReport, CitedClaim, Reference, Section


class TestAcademicMetadata:
    def test_defaults_match_the_projects_own_author_and_institution(self):
        metadata = AcademicMetadata(year=2026)

        assert metadata.author == "Leonardo Teixeira"
        assert metadata.registration == "245602"
        assert "UNICAMP" in metadata.institution
        assert metadata.unit is not None and "FEAGRI" in metadata.unit
        assert metadata.city == "Campinas – SP"

    def test_every_field_is_overridable(self):
        metadata = AcademicMetadata(
            author="Other Author",
            registration=None,
            institution="Other University",
            unit=None,
            city="Other City",
            year=2030,
            course="Other Course",
            advisor="Other Advisor",
            note=None,
        )

        assert metadata.author == "Other Author"
        assert metadata.registration is None
        assert metadata.institution == "Other University"
        assert metadata.unit is None
        assert metadata.course == "Other Course"
        assert metadata.advisor == "Other Advisor"
        assert metadata.note is None


class TestSection:
    def test_supports_nested_subsections(self):
        section = Section(
            number="1",
            title="INTRODUÇÃO",
            paragraphs=["Top-level paragraph."],
            subsections=[Section(number="1.1", title="Contexto", paragraphs=["Nested paragraph."])],
        )

        assert section.subsections[0].number == "1.1"
        assert section.subsections[0].paragraphs == ["Nested paragraph."]

    def test_cited_claims_default_to_empty(self):
        section = Section(number="4", title="RESULTADOS")
        assert section.cited_claims == []
        assert section.paragraphs == []
        assert section.subsections == []


class TestReferenceAndCitedClaim:
    def test_reference_never_requires_authors_or_year(self):
        reference = Reference(
            citation_key="1", source_id="src_1", title="A page", url="https://example.com/a"
        )
        assert reference.authors is None
        assert reference.year is None
        assert reference.accessed_at is None

    def test_cited_claim_defaults_to_no_citations(self):
        claim = CitedClaim(text="A claim with no traceable evidence.")
        assert claim.citation_keys == []


class TestAcademicReport:
    def test_round_trips_through_json(self):
        report = AcademicReport(
            metadata=AcademicMetadata(year=2026),
            title="A title",
            abstract="An abstract.",
            keywords=["Keyword"],
            sections=[Section(number="1", title="INTRODUÇÃO", paragraphs=["Text."])],
            references=[Reference(citation_key="1", source_id="src_1", title="A page", url="https://example.com/a")],
            source_run_id="run_1",
            generated_at=datetime.now(UTC),
        )

        restored = AcademicReport.model_validate_json(report.model_dump_json())
        assert restored == report
