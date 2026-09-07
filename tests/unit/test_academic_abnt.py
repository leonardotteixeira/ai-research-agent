from reportlab.lib.units import cm

from app.academic.abnt import DEFAULT_ABNT_STYLE, AbntStyle


class TestAbntStyle:
    def test_default_margins_match_nbr_14724(self):
        style = AbntStyle()
        assert style.margin_left_cm == 3.0
        assert style.margin_top_cm == 3.0
        assert style.margin_right_cm == 2.0
        assert style.margin_bottom_cm == 2.0

    def test_cm_properties_convert_to_points(self):
        style = AbntStyle()
        assert style.margin_left == 3.0 * cm
        assert style.margin_top == 3.0 * cm
        assert style.margin_right == 2.0 * cm
        assert style.margin_bottom == 2.0 * cm
        assert style.paragraph_indent == 1.25 * cm

    def test_body_leading_reflects_line_spacing(self):
        style = AbntStyle(body_font_size=12, line_spacing=1.5)
        assert style.body_leading == 18.0

    def test_is_a_frozen_dataclass_not_mutated_by_the_renderer(self):
        style = AbntStyle()
        try:
            style.body_font_size = 99  # type: ignore[misc]
            mutated = True
        except Exception:
            mutated = False
        assert mutated is False

    def test_default_instance_is_reused_not_rebuilt(self):
        assert DEFAULT_ABNT_STYLE.body_font == "Times-Roman"
