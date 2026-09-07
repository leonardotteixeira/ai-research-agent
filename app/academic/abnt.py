"""Centralized ABNT-oriented layout constants -- every margin/font/
spacing value the renderer uses lives here, never as a magic number
inline in app/academic/renderer.py.

Honesty about scope: this implements the *visual layout* rules from
NBR 14724 (trabalhos acadêmicos) that a single-column PDF renderer can
reasonably express -- margins, font, spacing, first-line indent,
alignment, heading hierarchy, and pagination. It does NOT implement
NBR 6023's full reference-type-specific citation rules (author-year
in-text citation styles, DOI formatting, etc.) or structural elements
like ficha catalográfica. See README "Academic Report Generation" for
the exact, itemized list of what is and isn't covered -- never described
here or anywhere else as "100% ABNT compliant".
"""

from dataclasses import dataclass

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


@dataclass(frozen=True)
class AbntStyle:
    page_size: tuple[float, float] = A4
    margin_left_cm: float = 3.0
    margin_top_cm: float = 3.0
    margin_right_cm: float = 2.0
    margin_bottom_cm: float = 2.0
    body_font: str = "Times-Roman"
    body_font_bold: str = "Times-Bold"
    body_font_size: int = 12
    line_spacing: float = 1.5
    paragraph_indent_cm: float = 1.25
    reference_font_size: int = 12
    small_font_size: int = 10

    @property
    def margin_left(self) -> float:
        return self.margin_left_cm * cm

    @property
    def margin_top(self) -> float:
        return self.margin_top_cm * cm

    @property
    def margin_right(self) -> float:
        return self.margin_right_cm * cm

    @property
    def margin_bottom(self) -> float:
        return self.margin_bottom_cm * cm

    @property
    def paragraph_indent(self) -> float:
        return self.paragraph_indent_cm * cm

    @property
    def body_leading(self) -> float:
        return self.body_font_size * self.line_spacing


DEFAULT_ABNT_STYLE = AbntStyle()
