import httpx
import pytest

from app.providers.search import MockSearchProvider, RealSearchProvider, SearchHit

CORPUS = [
    SearchHit(url="https://a.example/pgvector", title="pgvector overview", snippet="pgvector adds vector search to PostgreSQL"),
    SearchHit(url="https://b.example/pinecone", title="Pinecone docs", snippet="Pinecone is a managed vector database"),
    SearchHit(url="https://c.example/unrelated", title="Recipe for bread", snippet="flour, water, yeast, salt"),
]


class TestMockSearchProvider:
    async def test_returns_hits_matching_keywords(self):
        provider = MockSearchProvider(corpus=CORPUS)
        results = await provider.search("vector database")
        urls = [r.url for r in results]
        assert "https://b.example/pinecone" in urls
        assert "https://c.example/unrelated" not in urls

    async def test_is_deterministic(self):
        provider = MockSearchProvider(corpus=CORPUS)
        first = await provider.search("vector search postgres")
        second = await provider.search("vector search postgres")
        assert first == second

    async def test_respects_max_results(self):
        provider = MockSearchProvider(corpus=CORPUS)
        results = await provider.search("vector", max_results=1)
        assert len(results) <= 1

    async def test_no_matching_keywords_returns_empty(self):
        provider = MockSearchProvider(corpus=CORPUS)
        results = await provider.search("xylophone zeppelin quasar")
        assert results == []

    async def test_empty_corpus_returns_empty(self):
        provider = MockSearchProvider(corpus=[])
        results = await provider.search("anything")
        assert results == []

    async def test_default_corpus_is_empty_not_none(self):
        provider = MockSearchProvider()
        results = await provider.search("anything")
        assert results == []


class TestRealSearchProvider:
    async def test_parses_successful_response(self):
        body = {
            "web": {
                "results": [
                    {"url": "https://x.example", "title": "X", "description": "about x"},
                    {"url": "https://y.example", "title": "Y", "description": "about y"},
                ]
            }
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["X-Subscription-Token"] == "test-key"
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = RealSearchProvider(
            api_key="test-key", base_url="https://api.search.brave.com/res/v1/web/search",
            timeout_seconds=10, client=client,
        )
        results = await provider.search("x")
        assert len(results) == 2
        assert results[0].url == "https://x.example"
        await client.aclose()

    async def test_error_status_raises_without_leaking_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "SENTINEL_DETAIL"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = RealSearchProvider(
            api_key="bad-key", base_url="https://api.search.brave.com/res/v1/web/search",
            timeout_seconds=10, client=client,
        )
        with pytest.raises(RuntimeError) as exc_info:
            await provider.search("x")
        assert "SENTINEL_DETAIL" not in str(exc_info.value)
        await client.aclose()

    async def test_missing_results_key_returns_empty_list(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = RealSearchProvider(
            api_key="test-key", base_url="https://api.search.brave.com/res/v1/web/search",
            timeout_seconds=10, client=client,
        )
        results = await provider.search("x")
        assert results == []
        await client.aclose()
