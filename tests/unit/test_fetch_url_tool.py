import httpx
import pytest

from app.core.exceptions import BlockedURLError
from app.tools.fetch_url import FetchURLTool, _is_blocked_host, _validate_url


class TestBlockedHostDetection:
    @pytest.mark.parametrize(
        "hostname",
        ["127.0.0.1", "localhost", "169.254.169.254", "10.0.0.5", "192.168.1.1", "::1", "0.0.0.0"],
    )
    def test_blocks_private_and_loopback_hosts(self, hostname):
        assert _is_blocked_host(hostname) is True

    @pytest.mark.parametrize("hostname", ["example.com", "8.8.8.8", "api.github.com"])
    def test_allows_public_hosts(self, hostname):
        assert _is_blocked_host(hostname) is False


class TestValidateUrl:
    def test_rejects_non_http_scheme(self):
        with pytest.raises(BlockedURLError, match="unsupported URL scheme"):
            _validate_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(BlockedURLError, match="unsupported URL scheme"):
            _validate_url("ftp://example.com/file")

    def test_rejects_url_with_no_hostname(self):
        with pytest.raises(BlockedURLError, match="no hostname"):
            _validate_url("http://")

    def test_rejects_loopback_ip(self):
        with pytest.raises(BlockedURLError, match="private/loopback"):
            _validate_url("http://127.0.0.1/admin")

    def test_accepts_ordinary_https_url(self):
        _validate_url("https://example.com/page")  # should not raise


class TestFetchURLToolExecute:
    async def test_fetches_and_extracts_text(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, html="<html><body><p>Hello world</p></body></html>")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        tool = FetchURLTool(timeout_seconds=5, max_content_bytes=10_000, client=client)
        result = await tool.execute(url="https://example.com")
        assert result["text"] == "Hello world"
        assert result["truncated"] is False
        await client.aclose()

    async def test_blocks_loopback_url_before_any_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("should never reach the network for a blocked host")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        tool = FetchURLTool(timeout_seconds=5, max_content_bytes=10_000, client=client)
        with pytest.raises(BlockedURLError):
            await tool.execute(url="http://127.0.0.1/secret")
        await client.aclose()

    async def test_http_error_status_raises_value_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        tool = FetchURLTool(timeout_seconds=5, max_content_bytes=10_000, client=client)
        with pytest.raises(ValueError, match="HTTP 404"):
            await tool.execute(url="https://example.com/missing")
        await client.aclose()

    async def test_timeout_raises_value_error_not_httpx_exception(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        tool = FetchURLTool(timeout_seconds=5, max_content_bytes=10_000, client=client)
        with pytest.raises(ValueError, match="timed out"):
            await tool.execute(url="https://example.com/slow")
        await client.aclose()

    async def test_empty_content_raises_value_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, html="<html><head></head><body></body></html>")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        tool = FetchURLTool(timeout_seconds=5, max_content_bytes=10_000, client=client)
        with pytest.raises(ValueError, match="no extractable text"):
            await tool.execute(url="https://example.com/blank")
        await client.aclose()

    async def test_oversized_content_is_truncated_not_rejected(self):
        big_html = "<p>" + ("word " * 100_000) + "</p>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, html=big_html)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        tool = FetchURLTool(timeout_seconds=5, max_content_bytes=100, client=client)
        result = await tool.execute(url="https://example.com/huge")
        assert result["truncated"] is True
        assert len(result["text"]) < len(big_html)
        await client.aclose()

    async def test_redirect_to_blocked_host_is_caught(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == "https://example.com/redirect":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1/internal"})
            return httpx.Response(200, html="<p>should not be reached</p>")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
        tool = FetchURLTool(timeout_seconds=5, max_content_bytes=10_000, client=client)
        with pytest.raises(BlockedURLError):
            await tool.execute(url="https://example.com/redirect")
        await client.aclose()

    async def test_ordinary_redirect_to_public_host_succeeds(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == "https://example.com/redirect":
                return httpx.Response(302, headers={"Location": "https://example.com/final"})
            return httpx.Response(200, html="<p>final page</p>")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
        tool = FetchURLTool(timeout_seconds=5, max_content_bytes=10_000, client=client)
        result = await tool.execute(url="https://example.com/redirect")
        assert result["text"] == "final page"
        assert result["url"] == "https://example.com/final"
        await client.aclose()
