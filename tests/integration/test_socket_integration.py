import asyncio
from typing import Any

import httpx
import pytest

from app.tools.fetch_url import PinnedHTTPTransport, PinnedNetworkBackend


class TestRealSocketIntegration:
    """Verifies that PinnedHTTPTransport and PinnedNetworkBackend interact
    correctly with real OS TCP sockets, proving that the pinned IP is the
    actual connected destination, the original hostname is preserved in the
    Host header, concurrency is isolated, and keep-alive functions as expected.
    """

    async def test_real_socket_connects_to_pinned_ip_and_preserves_host(self) -> None:
        received_hosts: list[str] = []
        peer_addresses: list[Any] = []

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            peer = writer.get_extra_info("peername")
            peer_addresses.append(peer)
            line = await reader.readline()
            assert line.startswith(b"GET /test-path HTTP/1.1")
            while True:
                header_line = await reader.readline()
                if not header_line or header_line == b"\r\n":
                    break
                if header_line.lower().startswith(b"host:"):
                    received_hosts.append(header_line.decode().strip())
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\nHello Socket")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        backend = PinnedNetworkBackend()
        backend.pin("unresolvable-domain.test", "127.0.0.1")
        transport = PinnedHTTPTransport(backend)
        client = httpx.AsyncClient(transport=transport)

        try:
            response = await client.get(f"http://unresolvable-domain.test:{port}/test-path")
            assert response.status_code == 200
            assert response.text == "Hello Socket"
            assert len(peer_addresses) == 1
            assert peer_addresses[0][0] == "127.0.0.1"
            assert received_hosts == [f"Host: unresolvable-domain.test:{port}"]
        finally:
            await client.aclose()
            server.close()
            await server.wait_closed()

    async def test_real_socket_missing_pin_fails_before_connecting(self) -> None:
        backend = PinnedNetworkBackend()
        transport = PinnedHTTPTransport(backend)
        client = httpx.AsyncClient(transport=transport)

        try:
            with pytest.raises(httpx.ConnectError, match="no validated address pinned"):
                await client.get("http://unpinned-domain.test:80/path")
        finally:
            await client.aclose()

    async def test_real_socket_concurrency_isolated(self) -> None:
        async def create_server(reply_text: str) -> tuple[asyncio.Server, int]:
            async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                while True:
                    line = await reader.readline()
                    if not line or line == b"\r\n":
                        break
                payload = reply_text.encode()
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                    + payload
                )
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            srv = await asyncio.start_server(handler, "127.0.0.1", 0)
            p = srv.sockets[0].getsockname()[1]
            return srv, p

        server1, port1 = await create_server("SERVER_1_OK")
        server2, port2 = await create_server("SERVER_2_OK")

        async def fetch(domain: str, port: int) -> str:
            backend = PinnedNetworkBackend()
            backend.pin(domain, "127.0.0.1")
            transport = PinnedHTTPTransport(backend)
            async with httpx.AsyncClient(transport=transport) as client:
                res = await client.get(f"http://{domain}:{port}/")
                return res.text

        try:
            res1, res2 = await asyncio.gather(
                fetch("site-alpha.test", port1),
                fetch("site-beta.test", port2),
            )
            assert res1 == "SERVER_1_OK"
            assert res2 == "SERVER_2_OK"
        finally:
            server1.close()
            server2.close()
            await server1.wait_closed()
            await server2.wait_closed()

    async def test_real_socket_keep_alive_reuses_connection(self) -> None:
        connections_accepted = 0

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal connections_accepted
            connections_accepted += 1
            while True:
                line = await reader.readline()
                if not line:
                    break
                while True:
                    header_line = await reader.readline()
                    if not header_line or header_line == b"\r\n":
                        break
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
                await writer.drain()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        backend = PinnedNetworkBackend()
        backend.pin("keep-alive-domain.test", "127.0.0.1")
        transport = PinnedHTTPTransport(backend)
        client = httpx.AsyncClient(transport=transport)

        try:
            r1 = await client.get(f"http://keep-alive-domain.test:{port}/first")
            r2 = await client.get(f"http://keep-alive-domain.test:{port}/second")
            assert r1.status_code == 200
            assert r2.status_code == 200
            assert connections_accepted == 1
        finally:
            await client.aclose()
            server.close()
            await server.wait_closed()
