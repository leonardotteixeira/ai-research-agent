"""ABNT-oriented PDF rendering of an already-built `AcademicReport`.

Reads only the `AcademicReport` object it's given (plus, optionally, a
list of local logo image paths) -- never a RunRecord, never the
filesystem beyond the logo files and the one output path it writes to.
See app/academic/abnt.py for exactly which layout rules are centralized
there, and the module docstring in app/academic/abnt.py for what NBR
14724/6023 coverage this claims (and doesn't).
"""

import html
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from app.academic.abnt import DEFAULT_ABNT_STYLE, AbntStyle
from app.academic.schemas import AcademicReport, Section

_HEADING_STYLE_NAME = "AcademicHeading"

_MONTHS_PT = {
    1: "jan.", 2: "fev.", 3: "mar.", 4: "abr.", 5: "maio", 6: "jun.",
    7: "jul.", 8: "ago.", 9: "set.", 10: "out.", 11: "nov.", 12: "dez.",
}

# Pre-textual pages (capa, folha de rosto, resumo, sumário) are not
# numbered, matching common ABNT practice of counting but not printing
# page numbers before the first textual section -- everything from
# INTRODUÇÃO onward prints its real, physical page number.
_UNNUMBERED_LEADING_PAGES = 4


def _escape(text: str) -> str:
    """reportlab's Paragraph markup treats `<`/`&` specially -- escape
    them so persisted content (a question, an answer, a page/claim text)
    is always rendered as literal text, never interpreted as markup."""
    return html.escape(text, quote=False)


class _AcademicDocTemplate(BaseDocTemplate):
    """Registers every heading paragraph with reportlab's TOC machinery
    as it's drawn -- `multiBuild()` then runs enough passes for the
    table of contents to settle on the real page numbers."""

    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == _HEADING_STYLE_NAME:
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


class AcademicPDFRenderer:
    def __init__(self, style: AbntStyle = DEFAULT_ABNT_STYLE, logo_paths: list[Path] | None = None) -> None:
        self._style = style
        self._logo_paths = list(logo_paths or [])
        style_kwargs = dict(fontName=style.body_font)
        self._cover_title_style = ParagraphStyle(
            "CoverTitle", fontName=style.body_font_bold, fontSize=style.body_font_size + 2,
            alignment=TA_CENTER, leading=18,
        )
        self._cover_style = ParagraphStyle(
            "Cover", fontSize=style.body_font_size, alignment=TA_CENTER, leading=16, **style_kwargs
        )
        self._heading_style = ParagraphStyle(
            _HEADING_STYLE_NAME, fontName=style.body_font_bold, fontSize=style.body_font_size,
            alignment=TA_LEFT, spaceBefore=14, spaceAfter=8,
        )
        # Visually identical to _heading_style but a different style
        # *name* -- afterFlowable() (below) tracks TOC entries by style
        # name, and "SUMÁRIO" heading itself must never appear as an
        # entry inside its own table of contents.
        self._untracked_heading_style = ParagraphStyle(
            "AcademicHeadingUntracked", fontName=style.body_font_bold, fontSize=style.body_font_size,
            alignment=TA_LEFT, spaceBefore=14, spaceAfter=8,
        )
        self._body_style = ParagraphStyle(
            "Body", fontSize=style.body_font_size, alignment=TA_JUSTIFY, leading=style.body_leading,
            spaceAfter=6, firstLineIndent=style.paragraph_indent, **style_kwargs
        )
        self._reference_style = ParagraphStyle(
            "Reference", fontSize=style.reference_font_size, alignment=TA_JUSTIFY, leading=14,
            spaceAfter=10, **style_kwargs
        )
        self._toc_entry_style = ParagraphStyle("TOCEntry", fontSize=style.body_font_size, **style_kwargs)

    def render(self, report: AcademicReport, out_path: Path) -> None:
        doc = _AcademicDocTemplate(
            str(out_path),
            pagesize=self._style.page_size,
            topMargin=self._style.margin_top,
            leftMargin=self._style.margin_left,
            rightMargin=self._style.margin_right,
            bottomMargin=self._style.margin_bottom,
            title=report.title,
            author=report.metadata.author,
        )
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
        doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._draw_page_number)])

        story: list[Flowable] = []
        story.extend(self._cover_page(report))
        story.append(PageBreak())
        story.extend(self._title_page(report))
        story.append(PageBreak())
        story.extend(self._abstract_page(report))
        story.append(PageBreak())
        story.extend(self._table_of_contents())
        story.append(PageBreak())
        story.extend(self._sections(report))
        story.append(PageBreak())
        story.extend(self._references(report))

        doc.multiBuild(story)

    def _draw_page_number(self, canvas: Canvas, doc: BaseDocTemplate) -> None:
        page_number = canvas.getPageNumber()
        if page_number <= _UNNUMBERED_LEADING_PAGES:
            return
        canvas.saveState()
        canvas.setFont(self._style.body_font, self._style.small_font_size)
        canvas.drawRightString(
            self._style.page_size[0] - self._style.margin_right, self._style.margin_bottom / 2, str(page_number)
        )
        canvas.restoreState()

    def _content_width(self) -> float:
        return self._style.page_size[0] - self._style.margin_left - self._style.margin_right

    def _logo_header(self) -> Flowable | None:
        """UNICAMP (first known logo) top-left, FEAGRI (second) top-right
        -- a graphic/institutional composition choice, not something NBR
        14724 mandates (it only requires that the institution be
        identified on the capa, not where a logo sits). A single logo is
        placed at the left rather than centered, for consistency with the
        two-logo layout; this project currently only ever has 0, 1, or 2
        logos (see app/academic/assets.py), so anything beyond that just
        spreads evenly rather than raising.
        """
        if not self._logo_paths:
            return None
        logo_box = 70
        content_width = self._content_width()
        if len(self._logo_paths) == 1:
            image = Image(str(self._logo_paths[0]), width=logo_box, height=logo_box, kind="proportional")
            table = Table([[image]], colWidths=[content_width])
            table.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "LEFT"), ("VALIGN", (0, 0), (0, 0), "TOP")]))
            return table
        images = [
            Image(str(path), width=logo_box, height=logo_box, kind="proportional") for path in self._logo_paths
        ]
        column_width = content_width / len(images)
        table = Table([images], colWidths=[column_width] * len(images))
        style_commands = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("ALIGN", (0, 0), (0, 0), "LEFT"),
            ("ALIGN", (-1, 0), (-1, 0), "RIGHT"),
        ]
        for middle_index in range(1, len(images) - 1):
            style_commands.append(("ALIGN", (middle_index, 0), (middle_index, 0), "CENTER"))
        table.setStyle(TableStyle(style_commands))
        return table

    def _cover_page(self, report: AcademicReport) -> list[Flowable]:
        # Vertical budget is deliberately conservative: A4 minus ABNT
        # margins leaves ~700pt of usable height, and this must always
        # fit on ONE page (a capa spilling onto a second, near-empty page
        # is a real layout bug, not a stylistic choice) even with two
        # logos plus institution/title/author/footer text.
        metadata = report.metadata
        elements: list[Flowable] = []
        logo_header = self._logo_header()
        if logo_header is not None:
            elements.append(logo_header)
            elements.append(Spacer(1, 30))

        elements.append(Paragraph(_escape(metadata.institution.upper()), self._cover_style))
        if metadata.unit:
            elements.append(Paragraph(_escape(metadata.unit.upper()), self._cover_style))

        elements.append(Spacer(1, 90))
        elements.append(Paragraph(_escape(report.title.upper()), self._cover_title_style))

        elements.append(Spacer(1, 110))
        elements.append(Paragraph(_escape(metadata.author.upper()), self._cover_style))
        if metadata.registration:
            elements.append(Paragraph(f"RA {_escape(metadata.registration)}", self._cover_style))

        elements.append(Spacer(1, 90))
        footer = " – ".join(part for part in (metadata.city, str(metadata.year)) if part)
        elements.append(Paragraph(_escape(footer), self._cover_style))
        return elements

    def _title_page(self, report: AcademicReport) -> list[Flowable]:
        metadata = report.metadata
        elements: list[Flowable] = [
            Spacer(1, 40),
            Paragraph(_escape(metadata.author.upper()), self._cover_style),
        ]
        if metadata.registration:
            elements.append(Paragraph(f"RA {_escape(metadata.registration)}", self._cover_style))
        elements.append(Spacer(1, 100))
        elements.append(Paragraph(_escape(report.title.upper()), self._cover_title_style))
        elements.append(Spacer(1, 60))
        if metadata.note:
            elements.append(Paragraph(_escape(metadata.note), self._body_style))
        if metadata.course:
            elements.append(Paragraph(f"Curso: {_escape(metadata.course)}", self._body_style))
        if metadata.advisor:
            elements.append(Paragraph(f"Orientador(a): {_escape(metadata.advisor)}", self._body_style))
        elements.append(Spacer(1, 140))
        footer = " – ".join(part for part in (metadata.city, str(metadata.year)) if part)
        elements.append(Paragraph(_escape(footer), self._cover_style))
        return elements

    def _abstract_page(self, report: AcademicReport) -> list[Flowable]:
        elements: list[Flowable] = [
            Paragraph("RESUMO", self._heading_style),
            Paragraph(_escape(report.abstract), self._body_style),
        ]
        if report.keywords:
            elements.append(Spacer(1, 12))
            elements.append(
                Paragraph(f"Palavras-chave: {_escape('; '.join(report.keywords))}.", self._body_style)
            )
        return elements

    def _table_of_contents(self) -> list[Flowable]:
        toc = TableOfContents()
        toc.levelStyles = [self._toc_entry_style]
        return [Paragraph("SUMÁRIO", self._untracked_heading_style), toc]

    def _sections(self, report: AcademicReport) -> list[Flowable]:
        elements: list[Flowable] = []
        for section in report.sections:
            elements.extend(self._render_section(section))
        return elements

    def _render_section(self, section: Section) -> list[Flowable]:
        elements: list[Flowable] = [Paragraph(f"{section.number} {_escape(section.title)}", self._heading_style)]
        for paragraph in section.paragraphs:
            elements.append(Paragraph(_escape(paragraph), self._body_style))
        for cited_claim in section.cited_claims:
            marker = f" [{', '.join(cited_claim.citation_keys)}]" if cited_claim.citation_keys else ""
            elements.append(Paragraph(_escape(cited_claim.text) + marker, self._body_style))
        for subsection in section.subsections:
            elements.extend(self._render_section(subsection))
        return elements

    def _references(self, report: AcademicReport) -> list[Flowable]:
        elements: list[Flowable] = [Paragraph("REFERÊNCIAS", self._heading_style)]
        if not report.references:
            elements.append(Paragraph("Nenhuma fonte externa foi citada nesta pesquisa.", self._body_style))
            return elements
        for reference in report.references:
            accessed = (
                f" Acesso em: {reference.accessed_at.day:02d} {_MONTHS_PT[reference.accessed_at.month]} "
                f"{reference.accessed_at.year}."
                if reference.accessed_at is not None
                else ""
            )
            text = f"[{reference.citation_key}] {_escape(reference.title.upper())}. Disponível em: {_escape(reference.url)}.{accessed}"
            elements.append(Paragraph(text, self._reference_style))
        return elements
