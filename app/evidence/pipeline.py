"""Deterministic conversion of tool output into traceable source data."""

import hashlib
from datetime import UTC, datetime
from typing import Any

from app.schemas.evidence import Evidence, Source, normalize_source_url
from app.schemas.tool import ToolResult


class EvidencePipeline:
    def process(self, result: ToolResult) -> tuple[list[Source], list[Evidence]]:
        output = result.output
        if not result.success or not isinstance(output, list | dict):
            return [], []
        if result.tool_name == "web_search" and isinstance(output, list):
            return self._from_search_results(result, output)
        if result.tool_name == "fetch_url" and isinstance(output, dict):
            return self._from_fetched_page(result, output)
        return [], []

    def merge(
        self,
        existing_sources: list[Source],
        existing_evidence: list[Evidence],
        result: ToolResult,
    ) -> tuple[list[Source], list[Evidence]]:
        new_sources, new_evidence = self.process(result)
        sources_by_url = {normalize_source_url(source.url): source for source in existing_sources}
        source_ids: dict[str, str] = {}
        for source in new_sources:
            key = normalize_source_url(source.url)
            if key in sources_by_url:
                source_ids[source.source_id] = sources_by_url[key].source_id
            else:
                sources_by_url[key] = source
                existing_sources.append(source)
        for evidence in new_evidence:
            evidence.source_id = source_ids.get(evidence.source_id, evidence.source_id)
            if evidence.source_id in {source.source_id for source in existing_sources}:
                existing_evidence.append(evidence)
        return existing_sources, existing_evidence

    def _from_search_results(self, result: ToolResult, output: list[Any]) -> tuple[list[Source], list[Evidence]]:
        sources: list[Source] = []
        evidence: list[Evidence] = []
        for index, item in enumerate(output):
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            url = item["url"]
            source = self._source(url, result, "search_result", item.get("title"), item.get("snippet"))
            sources.append(source)
            snippet = item.get("snippet")
            if isinstance(snippet, str) and snippet.strip():
                evidence.append(self._evidence(snippet, source, result, index))
        return sources, evidence

    def _from_fetched_page(self, result: ToolResult, output: dict[Any, Any]) -> tuple[list[Source], list[Evidence]]:
        if not isinstance(output.get("url"), str) or not isinstance(output.get("text"), str):
            return [], []
        source = self._source(output["url"], result, "fetched_page", output.get("title"))
        if not output["text"].strip():
            return [source], []
        return [source], [self._evidence(output["text"], source, result, 0)]

    @staticmethod
    def _source(
        url: str,
        result: ToolResult,
        source_type: str,
        title: Any = None,
        snippet: Any = None,
    ) -> Source:
        normalized = normalize_source_url(url)
        source_id = "src_" + hashlib.sha256(normalized.encode()).hexdigest()[:16]
        retrieved_at = datetime.now(UTC)
        return Source(
            source_id=source_id,
            url=url,
            title=title if isinstance(title, str) else None,
            snippet=snippet if isinstance(snippet, str) else None,
            source_type=source_type,
            retrieved_at=retrieved_at,
            tool_call_id=result.call_id,
            timestamp=retrieved_at,
        )

    @staticmethod
    def _evidence(text: str, source: Source, result: ToolResult, index: int) -> Evidence:
        evidence_id = f"ev_{result.call_id}_{index}"
        return Evidence(
            evidence_id=evidence_id,
            content=text,
            source_id=source.source_id,
            tool_call_id=result.call_id,
            timestamp=source.timestamp,
        )