from app.providers.search import MockSearchProvider, SearchHit
from app.tools.web_search import WebSearchTool

CORPUS = [SearchHit(url="https://a.example", title="pgvector", snippet="vector search in postgres")]


async def test_execute_returns_plain_dicts():
    tool = WebSearchTool(provider=MockSearchProvider(corpus=CORPUS))
    results = await tool.execute(query="vector postgres")
    assert results == [{"url": "https://a.example", "title": "pgvector", "snippet": "vector search in postgres"}]


async def test_no_results_returns_empty_list():
    tool = WebSearchTool(provider=MockSearchProvider(corpus=CORPUS))
    results = await tool.execute(query="nonexistent topic entirely")
    assert results == []


def test_input_schema_requires_nonempty_query():
    import pytest
    from pydantic import ValidationError

    from app.tools.web_search import WebSearchInput

    with pytest.raises(ValidationError):
        WebSearchInput(query="")


def test_input_schema_max_results_bounds():
    import pytest
    from pydantic import ValidationError

    from app.tools.web_search import WebSearchInput

    with pytest.raises(ValidationError):
        WebSearchInput(query="x", max_results=0)
    with pytest.raises(ValidationError):
        WebSearchInput(query="x", max_results=11)
