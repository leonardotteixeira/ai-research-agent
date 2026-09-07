"""WebSearchTool — the agent's primary way to find sources. Wraps a
SearchProvider (mock or real, see app/providers/search.py); the tool
itself only turns a validated query into plain dicts the registry can
attach to a ToolResult.
"""

from pydantic import BaseModel, Field

from app.providers.search import SearchProvider


class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="The search query.")
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchTool:
    name = "web_search"
    description = "Searches the web and returns a list of {url, title, snippet} results."
    input_schema = WebSearchInput

    def __init__(self, provider: SearchProvider) -> None:
        self._provider = provider

    async def execute(self, query: str, max_results: int = 5) -> list[dict]:
        hits = await self._provider.search(query, max_results=max_results)
        return [{"url": hit.url, "title": hit.title, "snippet": hit.snippet} for hit in hits]
