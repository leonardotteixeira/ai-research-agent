"""Minimal HTML-to-text extraction using only the standard library
(html.parser) — deliberately avoids adding a dependency like
BeautifulSoup for what only needs to strip tags/scripts/styles and
collapse whitespace into readable plain text for the LLM prompt.
"""

from html.parser import HTMLParser

_SKIPPED_TAGS = {"script", "style"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.chunks.append(data.strip())


def extract_text(html: str) -> str:
    """Strips tags/scripts/styles and collapses whitespace. Works fine on
    plain (non-HTML) text too — HTMLParser treats untagged text as a
    single data chunk, so this is safe to call unconditionally.
    """
    parser = _TextExtractor()
    parser.feed(html)
    return " ".join(" ".join(parser.chunks).split())
