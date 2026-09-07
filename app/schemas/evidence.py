"""Source and Evidence — how the agent tracks what it learned and where
from, so the final answer cites real sources instead of inventing them.

Kept as two distinct schemas: `Source` is "a thing the agent looked at"
(one per successful search hit or fetched page); `Evidence` is "a specific
finding extracted from it" (one source can yield zero, one, or several
pieces of evidence). This is what lets the final answer say "claim X is
supported by evidence E, which came from source S" instead of citing a
whole page for every sentence.
"""

from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    source_id: str
    url: str
    title: str | None = None
    snippet: str | None = None
    source_type: str = "web"
    retrieved_at: datetime | None = None
    tool_call_id: str
    timestamp: datetime

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("source URL must be an absolute HTTP(S) URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or port == 0:
            raise ValueError("source URL must be an absolute HTTP(S) URL")
        return value


class Evidence(BaseModel):
    evidence_id: str
    content: str
    source_id: str
    tool_call_id: str
    timestamp: datetime
    relevance: float | None = None


class Claim(BaseModel):
    claim_id: str
    text: str = Field(..., min_length=1)
    evidence_ids: list[str] = Field(..., min_length=1)


def normalize_source_url(url: str) -> str:
    """Remove only URL differences that cannot identify different content."""
    parsed = urlsplit(url)
    hostname = parsed.hostname.lower() if parsed.hostname else ""
    host = hostname
    if parsed.port is not None:
        host = f"{hostname}:{parsed.port}"
    normalized_path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), host, normalized_path, parsed.query, ""))
