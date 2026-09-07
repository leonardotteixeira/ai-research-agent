import asyncio
import socket
from collections.abc import Callable
from typing import Any, cast

import httpcore
import httpx
import pytest

from app.core.exceptions import BlockedURLError
from app.tools import fetch_url as fetch_url_module
from app.tools.fetch_url import (
    FetchURLTool,
    PinnedHTTPTransport,
    PinnedNetworkBackend,
    SystemHostnameResolver,
    _validate_resolved_addresses,
)


class FakeResolver:
    def __init__(self, addresses: dict[str, list[str]], sequences: dict[str, list[list[str]]] | None = None) -> None:
        self._addresses = addresses
        self._sequences = sequences or {}
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, hostname: str, port: int) -> list[str]:
        self.calls.append((hostname, port))
        sequence = self._sequences.get(hostname)
        if sequence:
            return sequence.pop(0)
        return self._addresses[hostname]


class RecordingTransport(httpx.MockTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        super().__init__(handler)
        self.handler = handler
        self.requests: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        return cast(httpx.Response, self.handler(request))


async def fetch_with(
    url: str,
    addresses: dict[str, list[str]],
    transport: RecordingTransport,
    max_redirects: int = 5,
    sequences: dict[str, list[list[str]]] | None = None,
):
    resolver = FakeResolver(addresses, sequences)
    client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    tool = FetchURLTool(
        timeout_seconds=5,
        max_content_bytes=10_000,
        client=client,
        resolver=resolver,
        max_redirects=max_redirects,
    )
    try:
        return await tool.execute(url)
    finally:
        await client.aclose()


class TestResolvedAddressValidation:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "127.0.0.2",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "::1",
            "fc00::1",
            "fe80::1",
            "224.0.0.1",
            "255.255.255.255",
            "0.0.0.0",
            "::",
        ],
    )
    def test_blocks_non_public_ipv4_and_ipv6_addresses(self, address: str):
        with pytest.raises(BlockedURLError):
            _validate_resolved_addresses("public.example", [address])

    def test_blocks_hostname_if_any_resolved_address_is_private(self):
        with pytest.raises(BlockedURLError):
            _validate_resolved_addresses("public.example", ["93.184.216.34", "10.0.0.1"])

    def test_allows_public_hostname_address(self):
        _validate_resolved_addresses("public.example", ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])

    def test_rejects_empty_or_invalid_resolver_output(self):
        with pytest.raises(ValueError, match="no addresses"):
            _validate_resolved_addresses("public.example", [])
        with pytest.raises(ValueError, match="invalid address"):
            _validate_resolved_addresses("public.example", ["not-an-ip"])


class TestSecureFetch:
    def test_arbitrary_injected_transport_is_rejected(self):
        client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport())
        try:
            with pytest.raises(ValueError, match="MockTransport"):
                FetchURLTool(5, 10_000, client=client)
        finally:
            asyncio.run(client.aclose())

    async def test_public_hostname_is_resolved_before_transport(self):
        transport = RecordingTransport(lambda request: httpx.Response(200, text="ok"))
        result = await fetch_with("https://public.example/page", {"public.example": ["93.184.216.34"]}, transport)

        assert result["text"] == "ok"
        assert transport.requests == ["https://public.example/page"]

    @pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "::1", "fc00::1", "fe80::1"])
    async def test_hostname_resolving_to_private_address_never_reaches_transport(self, address: str):
        transport = RecordingTransport(lambda request: (_ for _ in ()).throw(AssertionError("must not connect")))

        with pytest.raises(BlockedURLError):
            await fetch_with("https://public.example/secret", {"public.example": [address]}, transport)

        assert transport.requests == []

    async def test_dns_resolution_failure_is_controlled(self):
        class FailingResolver:
            async def resolve(self, hostname: str, port: int) -> list[str]:
                raise OSError("resolver offline")

        transport = RecordingTransport(lambda request: (_ for _ in ()).throw(AssertionError("must not connect")))
        client = httpx.AsyncClient(transport=transport)
        tool = FetchURLTool(5, 10_000, client=client, resolver=FailingResolver())
        try:
            with pytest.raises(ValueError, match="DNS resolution failed"):
                await tool.execute("https://public.example")
        finally:
            await client.aclose()
        assert transport.requests == []

    async def test_dns_resolution_with_no_addresses_is_controlled(self):
        class EmptyResolver:
            async def resolve(self, hostname: str, port: int) -> list[str]:
                return []

        transport = RecordingTransport(lambda request: (_ for _ in ()).throw(AssertionError("must not connect")))
        client = httpx.AsyncClient(transport=transport)
        tool = FetchURLTool(5, 10_000, client=client, resolver=EmptyResolver())
        try:
            with pytest.raises(ValueError, match="no addresses"):
                await tool.execute("https://public.example")
        finally:
            await client.aclose()
        assert transport.requests == []

    async def test_system_resolver_with_no_addresses_is_controlled(self, monkeypatch):
        monkeypatch.setattr("app.tools.fetch_url.socket.getaddrinfo", lambda *args, **kwargs: [])

        with pytest.raises(ValueError, match="no addresses"):
            await SystemHostnameResolver().resolve("public.example", 443)

    @pytest.mark.parametrize("url", ["https://user@example.com/path", "https://user:password@example.com/path"])
    async def test_userinfo_is_rejected_before_transport(self, url: str):
        transport = RecordingTransport(lambda request: (_ for _ in ()).throw(AssertionError("must not connect")))

        with pytest.raises(BlockedURLError, match="userinfo"):
            await fetch_with(url, {"example.com": ["93.184.216.34"]}, transport)

        assert transport.requests == []

    async def test_invalid_port_is_rejected_before_transport(self):
        transport = RecordingTransport(lambda request: (_ for _ in ()).throw(AssertionError("must not connect")))

        with pytest.raises(BlockedURLError, match="invalid port"):
            await fetch_with("https://public.example:not-a-port/path", {"public.example": ["93.184.216.34"]}, transport)

        assert transport.requests == []

    async def test_zero_port_is_rejected_before_transport(self):
        transport = RecordingTransport(lambda request: (_ for _ in ()).throw(AssertionError("must not connect")))

        with pytest.raises(BlockedURLError, match="invalid port"):
            await fetch_with("https://public.example:0/path", {"public.example": ["93.184.216.34"]}, transport)

        assert transport.requests == []

    async def test_public_redirect_to_public_is_checked_hop_by_hop(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "https://second.example/final"})
            return httpx.Response(200, text="final")

        transport = RecordingTransport(handler)
        result = await fetch_with(
            "https://first.example/start",
            {"first.example": ["93.184.216.34"], "second.example": ["93.184.216.35"]},
            transport,
        )

        assert result["text"] == "final"
        assert transport.requests == ["https://first.example/start", "https://second.example/final"]

    async def test_redirect_without_location_is_controlled(self):
        transport = RecordingTransport(lambda request: httpx.Response(302))

        with pytest.raises(ValueError, match="without Location"):
            await fetch_with("https://public.example/start", {"public.example": ["93.184.216.34"]}, transport)

        assert transport.requests == ["https://public.example/start"]

    async def test_https_to_http_redirect_is_rejected_before_next_request(self):
        transport = RecordingTransport(
            lambda request: httpx.Response(302, headers={"Location": "http://public.example/insecure"})
        )

        with pytest.raises(BlockedURLError, match="HTTPS to HTTP"):
            await fetch_with("https://public.example/start", {"public.example": ["93.184.216.34"]}, transport)

        assert transport.requests == ["https://public.example/start"]

    @pytest.mark.parametrize("private_url", ["http://localhost/internal", "http://127.0.0.1/internal", "http://10.0.0.1/internal"])
    async def test_public_redirect_to_private_never_reaches_private_transport(self, private_url: str):
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            return httpx.Response(302, headers={"Location": private_url})

        transport = RecordingTransport(handler)
        with pytest.raises(BlockedURLError):
            await fetch_with(
                "https://public.example/start",
                {"public.example": ["93.184.216.34"]},
                transport,
            )

        assert requests == ["https://public.example/start"]
        assert all(private_url not in item for item in transport.requests)

    async def test_redirect_dns_rebinding_is_blocked_before_second_request(self):
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            return httpx.Response(302, headers={"Location": "https://same.example/next"})

        transport = RecordingTransport(handler)
        with pytest.raises(BlockedURLError):
            await fetch_with(
                "https://same.example/start",
                {"same.example": ["93.184.216.34"]},
                transport,
                sequences={"same.example": [["93.184.216.34"], ["127.0.0.1"]]},
            )

        assert requests == ["https://same.example/start"]

    async def test_redirect_loop_and_limit_are_controlled(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"Location": "https://public.example/loop"})

        transport = RecordingTransport(handler)
        with pytest.raises(ValueError, match="redirect loop detected"):
            await fetch_with(
                "https://public.example/loop",
                {"public.example": ["93.184.216.34"]},
                transport,
            )

        assert len(transport.requests) == 1

    async def test_redirect_limit_is_enforced(self):
        counter = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal counter
            counter += 1
            return httpx.Response(302, headers={"Location": f"https://public.example/hop{counter + 1}"})

        transport = RecordingTransport(handler)
        with pytest.raises(ValueError, match="redirect limit"):
            await fetch_with(
                "https://public.example/hop1",
                {"public.example": ["93.184.216.34"]},
                transport,
                max_redirects=2,
            )

        assert len(transport.requests) == 3

    async def test_http_transport_error_is_controlled(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection failed", request=request)

        transport = RecordingTransport(handler)
        with pytest.raises(ValueError, match="request to"):
            await fetch_with("https://public.example/start", {"public.example": ["93.184.216.34"]}, transport)

    async def test_concurrent_fetches_use_isolated_pinned_backends(self, monkeypatch):
        class RecordingSocketBackend:
            def __init__(self) -> None:
                self.addresses: list[str] = []

            async def connect_tcp(self, host: str, port: int, **kwargs: Any) -> httpcore.AsyncNetworkStream:
                self.addresses.append(host)
                return cast(httpcore.AsyncNetworkStream, object())

        class FakeClient:
            def __init__(self, backend: PinnedNetworkBackend, sockets: list[RecordingSocketBackend]) -> None:
                self.backend = backend
                self.socket = RecordingSocketBackend()
                sockets.append(self.socket)
                self.backend._backend = cast(httpcore.AsyncNetworkBackend, self.socket)

            async def get(self, url: str, **kwargs: Any) -> httpx.Response:
                await self.backend.connect_tcp("same.example", 443)
                return httpx.Response(200, text="ok", request=httpx.Request("GET", url))

            async def aclose(self) -> None:
                return None

        sockets: list[RecordingSocketBackend] = []

        def client_factory(*args: Any, transport: PinnedHTTPTransport, **kwargs: Any) -> FakeClient:
            backend = cast(PinnedNetworkBackend, transport._pool._network_backend)
            return FakeClient(backend, sockets)

        monkeypatch.setattr(fetch_url_module.httpx, "AsyncClient", client_factory)

        class ConcurrentResolver:
            def __init__(self) -> None:
                self.ready = asyncio.Barrier(2)
                self.addresses = ["93.184.216.34", "93.184.216.35"]

            async def resolve(self, hostname: str, port: int) -> list[str]:
                await self.ready.wait()
                return [self.addresses.pop(0)]

        tool = FetchURLTool(5, 10_000, resolver=ConcurrentResolver())
        results = await asyncio.gather(
            tool.execute("https://same.example/a"),
            tool.execute("https://same.example/b"),
        )

        assert [result["text"] for result in results] == ["ok", "ok"]
        assert sorted(socket.addresses[0] for socket in sockets) == ["93.184.216.34", "93.184.216.35"]

    async def test_separate_executes_have_separate_keep_alive_pools(self, monkeypatch):
        transports: list[PinnedHTTPTransport] = []

        class FakeClient:
            async def get(self, url: str, **kwargs: Any) -> httpx.Response:
                return httpx.Response(200, text="ok", request=httpx.Request("GET", url))

            async def aclose(self) -> None:
                return None

        def client_factory(*args: Any, transport: PinnedHTTPTransport, **kwargs: Any) -> FakeClient:
            transports.append(transport)
            return FakeClient()

        monkeypatch.setattr(fetch_url_module.httpx, "AsyncClient", client_factory)
        tool = FetchURLTool(5, 10_000, resolver=FakeResolver({"same.example": ["93.184.216.34"]}))

        await tool.execute("https://same.example/one")
        await tool.execute("https://same.example/two")

        assert len(transports) == 2
        assert transports[0] is not transports[1]
        assert transports[0]._pool is not transports[1]._pool

    async def test_parser_error_is_wrapped(self, monkeypatch):
        transport = RecordingTransport(lambda request: httpx.Response(200, text="ok"))
        monkeypatch.setattr(fetch_url_module, "extract_text", lambda value: (_ for _ in ()).throw(RuntimeError("bad parser")))

        with pytest.raises(ValueError, match="failed to parse content"):
            await fetch_with("https://public.example/page", {"public.example": ["93.184.216.34"]}, transport)

    def test_negative_redirect_limit_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            FetchURLTool(5, 10_000, max_redirects=-1)


class TestPinnedNetworkBackend:
    async def test_connects_to_validated_ip_not_hostname(self):
        class FakeBackend:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            async def connect_tcp(self, host: str, port: int, **kwargs: Any) -> httpcore.AsyncNetworkStream:
                self.calls.append((host, port))
                return cast(httpcore.AsyncNetworkStream, object())

        backend = PinnedNetworkBackend()
        fake_backend = FakeBackend()
        backend._backend = cast(httpcore.AsyncNetworkBackend, fake_backend)
        backend.pin("public.example", "93.184.216.34")

        await backend.connect_tcp("public.example", 443)

        assert fake_backend.calls == [("93.184.216.34", 443)]

    async def test_missing_pin_is_rejected_before_backend_connection(self):
        backend = PinnedNetworkBackend()

        with pytest.raises(httpcore.ConnectError, match="no validated address"):
            await backend.connect_tcp("unvalidated.example", 443)


class TestSystemResolver:
    async def test_system_resolver_returns_socket_addresses(self, monkeypatch):
        monkeypatch.setattr(
            "app.tools.fetch_url.socket.getaddrinfo",
            lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        )
        addresses = await SystemHostnameResolver().resolve("public.example", 443)

        assert addresses == ["93.184.216.34"]

    async def test_system_resolver_wraps_resolution_failure(self, monkeypatch):
        def fail_getaddrinfo(*args, **kwargs):
            raise OSError("offline")

        monkeypatch.setattr("app.tools.fetch_url.socket.getaddrinfo", fail_getaddrinfo)

        with pytest.raises(ValueError, match="DNS resolution failed"):
            await SystemHostnameResolver().resolve("public.example", 443)
