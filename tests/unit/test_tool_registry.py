import asyncio

from pydantic import BaseModel

from app.schemas.tool import ToolCall
from app.tools.registry import ToolRegistry


class EchoInput(BaseModel):
    text: str


class EchoTool:
    name = "echo"
    description = "Echoes the given text back."
    input_schema = EchoInput

    async def execute(self, text: str) -> str:
        return text


class FailingTool:
    name = "failing"
    description = "Always raises."
    input_schema = EchoInput

    async def execute(self, text: str) -> str:
        raise RuntimeError("boom")


class SlowTool:
    name = "slow"
    description = "Sleeps longer than any reasonable timeout."
    input_schema = EchoInput

    async def execute(self, text: str) -> str:
        await asyncio.sleep(10)
        return text


class TestRegistryBasics:
    def test_names_lists_registered_tools_sorted(self):
        registry = ToolRegistry([EchoTool(), FailingTool()])
        assert registry.names() == ["echo", "failing"]

    def test_get_returns_none_for_unknown_tool(self):
        registry = ToolRegistry([EchoTool()])
        assert registry.get("nonexistent") is None

    def test_describe_includes_schema(self):
        registry = ToolRegistry([EchoTool()])
        [description] = registry.describe()
        assert description["name"] == "echo"
        assert "properties" in description["input_schema"]


class TestExecuteSuccess:
    async def test_successful_call_returns_output(self):
        registry = ToolRegistry([EchoTool()])
        call = ToolCall(call_id="c1", tool_name="echo", arguments={"text": "hello"})
        result = await registry.execute(call)
        assert result.success is True
        assert result.output == "hello"
        assert result.latency_ms is not None


class TestExecuteErrors:
    async def test_unknown_tool_is_a_structured_error_not_a_crash(self):
        registry = ToolRegistry([EchoTool()])
        call = ToolCall(call_id="c1", tool_name="nonexistent", arguments={})
        result = await registry.execute(call)
        assert result.success is False
        assert "Unknown tool" in result.error
        assert "nonexistent" in result.error

    async def test_invalid_arguments_are_rejected_before_execution(self):
        registry = ToolRegistry([EchoTool()])
        call = ToolCall(call_id="c1", tool_name="echo", arguments={"wrong_field": "x"})
        result = await registry.execute(call)
        assert result.success is False
        assert "Invalid arguments" in result.error

    async def test_tool_exception_becomes_structured_error(self):
        registry = ToolRegistry([FailingTool()])
        call = ToolCall(call_id="c1", tool_name="failing", arguments={"text": "x"})
        result = await registry.execute(call)
        assert result.success is False
        assert "RuntimeError" in result.error
        assert "boom" in result.error


class TestExecuteTimeout:
    async def test_slow_tool_is_stopped_by_timeout(self):
        registry = ToolRegistry([SlowTool()])
        call = ToolCall(call_id="c1", tool_name="slow", arguments={"text": "x"})
        result = await registry.execute(call, timeout_seconds=0.05)
        assert result.success is False
        assert "timed out" in result.error

    async def test_no_timeout_configured_lets_fast_tool_finish(self):
        registry = ToolRegistry([EchoTool()])
        call = ToolCall(call_id="c1", tool_name="echo", arguments={"text": "x"})
        result = await registry.execute(call, timeout_seconds=None)
        assert result.success is True
