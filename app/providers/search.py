"""SearchProvider abstraction — decouples WebSearchTool (app/tools/
web_search.py) from any specific search API, same Provider-plus-Mock
pattern used across the portfolio (ai-gateway, production-rag,
ai-eval-lab).

`MockSearchProvider` is deterministic and fixture-based — never touches
the network, used by default and in every test. `RealSearchProvider` calls
the Brave Search API over httpx (raw HTTP, no vendor SDK). It is never
exercised against the real API in this environment — see README
"Providers" for why, and how to enable it.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str
    snippet: str


class SearchProvider(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]: ...


def load_corpus_fixture(path: str | Path) -> list[SearchHit]:
    """Loads a JSON fixture corpus for MockSearchProvider. Fixtures are
    plain {url, title, snippet} objects, clearly stored under evals/
    fixtures/ and never presented as real search results — see
    evals/fixtures/vector_db_corpus.json for the one used across this
    project's tests and demos.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SearchHit(url=item["url"], title=item["title"], snippet=item["snippet"]) for item in payload]


class MockSearchProvider:
    """Deterministic keyword-overlap search over a small in-memory corpus
    (see evals/fixtures for the canned corpus shared across tests). The
    same query always returns the same results in the same order, and
    different queries can plausibly disagree — useful for testing the
    agent's "sources disagree, I need another search" decision path.
    """

    def __init__(self, corpus: list[SearchHit] | None = None) -> None:
        self._corpus = corpus if corpus is not None else []

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        query_words = set(query.lower().split())
        scored: list[tuple[int, SearchHit]] = []
        for hit in self._corpus:
            haystack = f"{hit.title} {hit.snippet}".lower()
            score = sum(1 for word in query_words if word in haystack)
            if score > 0:
                scored.append((score, hit))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [hit for _, hit in scored[:max_results]]


class RealSearchProvider:
    """Brave Search API (https://api.search.brave.com) — a real, low-
    friction REST search API needing only a single header API key. Chosen
    as a concrete implementation so this abstraction has a real shape, not
    a stub; swapping to another provider (Serper, SerpAPI, ...) means
    implementing this same Protocol, nothing else in the app changes.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout_seconds
        self._client = client or httpx.AsyncClient()

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        response = await self._client.get(
            self._base_url,
            headers={"X-Subscription-Token": self._api_key, "Accept": "application/json"},
            params={"q": query, "count": max_results},
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            # Never leak the response body — same discipline as the other
            # providers in this portfolio.
            raise RuntimeError(f"Search API call failed with status {response.status_code}")

        payload = response.json()
        results = payload.get("web", {}).get("results", [])
        return [
            SearchHit(url=r.get("url", ""), title=r.get("title", ""), snippet=r.get("description", ""))
            for r in results[:max_results]
        ]
