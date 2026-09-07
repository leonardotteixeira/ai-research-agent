from app.tools.html_text import extract_text


def test_strips_tags():
    assert extract_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_strips_script_and_style_content():
    html = "<html><head><style>body{color:red}</style></head><body><script>alert(1)</script>Real text</body></html>"
    assert extract_text(html) == "Real text"


def test_collapses_whitespace():
    assert extract_text("<p>Hello\n\n   world</p>") == "Hello world"


def test_plain_text_passthrough():
    assert extract_text("just plain text, no tags") == "just plain text, no tags"


def test_empty_input_returns_empty_string():
    assert extract_text("") == ""


def test_only_tags_no_text_returns_empty_string():
    assert extract_text("<div><span></span></div>") == ""
