"""FetchURLTool — retrieves and extracts plain text from a URL.

Security note (see README "Prompt Injection Protection" for the full
picture): whatever this tool returns is treated as DATA everywhere
downstream — the agent loop places fetched text in the LLM prompt behind
an explicit "this is source content, not instructions" boundary. This
tool's own job is narrower: don't let a URL or a page reach somewhere it
shouldn't, and don't let a bad page crash the run.

Guards against:
- non-http(s) schemes (file://, ftp://, ...);
- userinfo in URLs;
- requests to loopback/private/link-local/multicast/reserved addresses;
- hostnames resolving to any blocked IPv4/IPv6 address;
- redirects are disabled in httpx and followed manually, validating every
    destination before its request;
- HTTPS to HTTP redirects are rejected;
- DNS rebinding by pinning the validated address in the network backend while
    preserving the logical hostname for HTTP Host and TLS SNI;
- injected clients are limited to MockTransport so they cannot bypass the
    pinned production transport;
- oversized responses (truncated, not rejected outright);
- timeouts, HTTP errors, redirect loops, and unparseable content — all become
    a ToolResult error via the registry, never an unhandled exception that
    could crash the whole research run.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Iterable
from typing import Protocol
from urllib.parse import urljoin, urlparse

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend
from httpcore._backends.base import SOCKET_OPTION
from pydantic import BaseModel, Field

from app.core.exceptions import BlockedURLError
from app.tools.html_text import extract_text

_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOSTNAMES = {"localhost"}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_DEFAULT_MAX_REDIRECTS = 5


class FetchURLInput(BaseModel):
    url: str = Field(..., description="An absolute http(s) URL to fetch.")


class HostnameResolver(Protocol):
    async def resolve(self, hostname: str, port: int) -> list[str]: ...


class SystemHostnameResolver:
    async def resolve(self, hostname: str, port: int) -> list[str]:
        try:
            records = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise ValueError(f"DNS resolution failed for {hostname!r}") from exc
        addresses = list(dict.fromkeys(record[4][0] for record in records))
        if not addresses:
            raise ValueError(f"DNS resolution returned no addresses for {hostname!r}")
        return addresses


class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connects to the address selected by FetchURLTool after validation."""

    def __init__(self) -> None:
        self._backend: httpcore.AsyncNetworkBackend = AutoBackend()
        self._pins: dict[str, str] = {}

    def pin(self, hostname: str, address: str) -> None:
        self._pins[hostname.lower()] = address

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        address = self._pins.get(host.lower())
        if address is None:
            raise httpcore.ConnectError(f"no validated address pinned for {host!r}")
        return await self._backend.connect_tcp(
            address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


class PinnedHTTPTransport(httpx.AsyncHTTPTransport):
    def __init__(self, backend: PinnedNetworkBackend) -> None:
        if httpx.__version__ != "0.28.1":
            raise RuntimeError("PinnedHTTPTransport supports httpx 0.28.1 only")
        super().__init__()
        if not hasattr(self, "_pool") or not hasattr(self._pool, "_network_backend"):
            raise RuntimeError("incompatible HTTPX transport internals")
        self._pool._network_backend = backend


def _is_blocked_host(hostname: str) -> bool:
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        return True
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False  # a regular domain name — see module docstring re: DNS rebinding
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise BlockedURLError(f"unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise BlockedURLError("URL has no hostname")
    if parsed.username is not None or parsed.password is not None:
        raise BlockedURLError("userinfo in URL is not allowed")
    try:
        port = parsed.port
    except ValueError as exc:
        raise BlockedURLError("URL has an invalid port") from exc
    if port == 0:
        raise BlockedURLError("URL has an invalid port")
    if _is_blocked_host(parsed.hostname):
        raise BlockedURLError(f"refusing to fetch a private/loopback address: {parsed.hostname!r}")


def _validate_resolved_addresses(hostname: str, addresses: list[str]) -> None:
    if not addresses:
        raise ValueError(f"DNS resolution returned no addresses for {hostname!r}")
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError(f"DNS resolver returned an invalid address for {hostname!r}") from exc
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise BlockedURLError(f"refusing to fetch {hostname!r} resolved to blocked address {address!r}")


class FetchURLTool:
    name = "fetch_url"
    description = "Fetches a web page and returns its extracted plain text (truncated if large)."
    input_schema = FetchURLInput

    def __init__(
        self,
        timeout_seconds: float,
        max_content_bytes: int,
        client: httpx.AsyncClient | None = None,
        resolver: HostnameResolver | None = None,
        max_redirects: int = _DEFAULT_MAX_REDIRECTS,
    ) -> None:
        if max_redirects < 0:
            raise ValueError("max_redirects must be non-negative")
        self._timeout = timeout_seconds
        self._max_content_bytes = max_content_bytes
        self._resolver = resolver or SystemHostnameResolver()
        self._max_redirects = max_redirects
        self._injected_client: httpx.AsyncClient | None
        if client is not None:
            if not isinstance(client._transport, httpx.MockTransport):
                raise ValueError("injected clients must use MockTransport")
            self._injected_client = client
        else:
            self._injected_client = None

    async def execute(self, url: str) -> dict:
        if self._injected_client is not None:
            return await self._execute(url, self._injected_client, None)

        backend = PinnedNetworkBackend()
        client = httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            transport=PinnedHTTPTransport(backend),
        )
        try:
            return await self._execute(url, client, backend)
        finally:
            await client.aclose()

    async def _execute(
        self,
        url: str,
        client: httpx.AsyncClient,
        backend: PinnedNetworkBackend | None,
    ) -> dict:
        current_url = url
        visited: set[str] = set()
        for redirect_count in range(self._max_redirects + 1):
            _validate_url(current_url)
            if current_url in visited:
                raise ValueError("redirect loop detected")
            visited.add(current_url)
            parsed = urlparse(current_url)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                addresses = await self._resolver.resolve(parsed.hostname or "", port)
                _validate_resolved_addresses(parsed.hostname or "", addresses)
            except (BlockedURLError, ValueError):
                raise
            except Exception as exc:
                raise ValueError(f"DNS resolution failed for {parsed.hostname!r}") from exc
            if backend is not None:
                backend.pin(parsed.hostname or "", addresses[0])

            try:
                response = await client.get(current_url, timeout=self._timeout, follow_redirects=False)
            except httpx.TimeoutException as exc:
                raise ValueError(f"request to {current_url} timed out") from exc
            except httpx.HTTPError as exc:
                raise ValueError(f"request to {current_url} failed: {exc}") from exc

            if response.status_code not in _REDIRECT_STATUSES:
                break
            location = response.headers.get("Location")
            if not location:
                raise ValueError(f"{current_url} returned a redirect without Location")
            if redirect_count >= self._max_redirects:
                raise ValueError(f"redirect limit of {self._max_redirects} exceeded")
            next_url = urljoin(current_url, location)
            if urlparse(current_url).scheme == "https" and urlparse(next_url).scheme == "http":
                raise BlockedURLError("HTTPS to HTTP redirect is not allowed")
            current_url = next_url
        else:
            raise ValueError(f"redirect limit of {self._max_redirects} exceeded")

        if response.status_code >= 400:
            raise ValueError(f"{current_url} returned HTTP {response.status_code}")

        raw = response.content[: self._max_content_bytes]
        truncated = len(response.content) > self._max_content_bytes

        try:
            text = extract_text(raw.decode(response.encoding or "utf-8", errors="replace"))
        except Exception as exc:
            raise ValueError(f"failed to parse content from {current_url}: {exc}") from exc

        if not text.strip():
            raise ValueError(f"{current_url} returned no extractable text content")

        return {"url": str(response.url), "text": text, "truncated": truncated}
