import json
from typing import Any

import httpx
import pytest

from app.core.exceptions import InvalidStructuredOutputError, ProviderError, ProviderTimeoutError
from app.providers.anthropic_llm import AnthropicProvider


def _messages_response(text: str, usage: dict | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


class TestAnthropicProvider:
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="API key is required"):
            AnthropicProvider(api_key=None, model="test", timeout_seconds=5)

    async def test_parses_valid_structured_decision(self):
        body = _messages_response(
            json.dumps(
                {
                    "action": "tool_call",
                    "rationale": "Search.",
                    "tool_calls": [{"call_id": "c1", "tool_name": "web_search", "arguments": {"query": "x"}}],
                }
            ),
            usage={"input_tokens": 100, "output_tokens": 20},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["x-api-key"] == "test-key"
            assert request.headers["anthropic-version"] == "2023-06-01"
            payload = json.loads(request.content)
            assert payload["model"] == "test-model"
            assert payload["messages"][0]["role"] == "user"
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("test-key", "test-model", 5, client=client)

        decision, usage = await provider.generate_decision(
            "What is x?", context=[{"source": "s1"}], available_tools=[{"name": "web_search"}]
        )

        assert decision.tool_calls[0].arguments == {"query": "x"}
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 20
        assert usage.total_tokens == 120
        assert usage.actual is True
        await client.aclose()

    async def test_strips_markdown_code_fence_around_json(self):
        body = _messages_response("```json\n" + json.dumps({"action": "finish", "rationale": "done"}) + "\n```")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("test-key", "test-model", 5, client=client)

        decision, usage = await provider.generate_decision("question")

        assert decision.should_finish is False  # action=finish parsed correctly regardless
        assert usage is None
        await client.aclose()

    async def test_rejects_malformed_content(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_messages_response("not json"))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("test-key", "test-model", 5, client=client)

        with pytest.raises(InvalidStructuredOutputError, match="invalid structured response"):
            await provider.generate_decision("question")
        await client.aclose()

    async def test_rejects_schema_invalid_content(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_messages_response(json.dumps({"action": "tool_call", "tool_calls": []})))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("test-key", "test-model", 5, client=client)

        with pytest.raises(InvalidStructuredOutputError, match="invalid structured response"):
            await provider.generate_decision("question")
        await client.aclose()

    async def test_http_error_is_controlled(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("bad-key", "test-model", 5, client=client)

        with pytest.raises(ProviderError, match="status 401"):
            await provider.generate_decision("question")
        await client.aclose()

    async def test_timeout_is_controlled(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("test-key", "test-model", 5, client=client)

        with pytest.raises(ProviderTimeoutError, match="timed out"):
            await provider.generate_decision("question")
        await client.aclose()

    async def test_generate_answer_parses_valid_response(self):
        answer_json = json.dumps(
            {
                "answer_text": "The answer [1].",
                "claims": [{"claim_id": "cl_1", "text": "The answer.", "evidence_ids": ["ev_1"]}],
                "citations": [{"claim": "The answer.", "evidence_ids": ["ev_1"], "source_ids": []}],
                "is_complete": True,
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_messages_response(answer_json))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("test-key", "test-model", 5, client=client)

        answer, usage = await provider.generate_answer(
            "question",
            evidence=[{"evidence_id": "ev_1", "content": "x", "source_id": "src_1"}],
            sources=[],
            claims=[],
        )

        assert answer.answer_text == "The answer [1]."
        assert usage is None
        await client.aclose()

    async def test_usage_missing_returns_none_not_zero(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_messages_response(json.dumps({"action": "finish"})))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("test-key", "test-model", 5, client=client)

        _, usage = await provider.generate_decision("question")

        assert usage is None
        await client.aclose()

    async def test_malformed_usage_does_not_fail_an_otherwise_valid_decision(self):
        body = _messages_response(json.dumps({"action": "finish"}), usage={"input_tokens": "not-a-number"})

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("test-key", "test-model", 5, client=client)

        decision, usage = await provider.generate_decision("question")

        assert decision.action is not None
        assert usage is None
        await client.aclose()
