import asyncio
from typing import Any

from pydantic import BaseModel

from app.agent.agent import Agent
from app.policies.execution import ExecutionPolicy
from app.providers.mock_llm import MockLLMProvider
from app.schemas.decision import DecisionAction, LLMDecision
from app.schemas.state import ResearchRequest, TerminationReason
from app.schemas.tool import ToolCall
from app.tools.registry import ToolRegistry


class InputModel(BaseModel):
    value: str


class CountingTool:
    name = "count"
    description = "Counts executions."
    input_schema = InputModel
    executions: list[str] = []

    async def execute(self, value: str) -> str:
        self.executions.append(value)
        return value


class SlowTool:
    name = "slow"
    description = "Sleeps."
    input_schema = InputModel

    async def execute(self, value: str) -> str:
        await asyncio.sleep(1)
        return value


class RaisingRegistry:
    def describe(self) -> list[dict[str, Any]]:
        return []

    async def execute(self, call: ToolCall, timeout_seconds: float | None = None):
        raise RuntimeError("registry failure")


def tool_call(call_id: str, name: str = "count", value: str = "x") -> ToolCall:
    return ToolCall(call_id=call_id, tool_name=name, arguments={"value": value})


def decision(*calls: ToolCall) -> LLMDecision:
    return LLMDecision(action=DecisionAction.TOOL_CALL, tool_calls=list(calls))


class TestExecutionPolicy:
    async def test_defaults_are_safe_and_configurable(self):
        policy = ExecutionPolicy(max_steps=2, max_tool_calls=3, max_same_tool_calls=1, global_timeout_seconds=4)

        assert policy.max_steps == 2
        assert policy.max_tool_calls == 3
        assert policy.max_same_tool_calls == 1
        assert policy.global_timeout_seconds == 4

    async def test_max_steps_stops_before_an_additional_step(self):
        provider = MockLLMProvider([decision(tool_call("c1")), decision(tool_call("c2"))])
        tool = CountingTool()
        agent = Agent(provider, ToolRegistry([tool]), policy=ExecutionPolicy(max_steps=2, max_tool_calls=10))

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.MAX_STEPS
        assert result.steps == 2
        assert tool.executions == ["x", "x"]

    async def test_max_tool_calls_stops_before_next_call(self):
        CountingTool.executions = []
        provider = MockLLMProvider([decision(tool_call("c1", value="one"), tool_call("c2", value="two"))])
        agent = Agent(
            provider,
            ToolRegistry([CountingTool()]),
            policy=ExecutionPolicy(max_steps=5, max_tool_calls=1),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.MAX_TOOL_CALLS
        assert CountingTool.executions == ["one"]
        assert [call.call_id for call in agent.last_state.tool_calls] == ["c1"]

    async def test_allowed_tool_executes(self):
        CountingTool.executions = []
        provider = MockLLMProvider([decision(tool_call("c1")), {"action": "finish"}])
        agent = Agent(
            provider,
            ToolRegistry([CountingTool()]),
            policy=ExecutionPolicy(allowed_tools={"count"}),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.FINISHED
        assert CountingTool.executions == ["x"]

    async def test_blocked_tool_is_not_executed(self):
        CountingTool.executions = []
        provider = MockLLMProvider([decision(tool_call("c1"))])
        agent = Agent(
            provider,
            ToolRegistry([CountingTool()]),
            policy=ExecutionPolicy(allowed_tools={"other"}),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.POLICY_BLOCKED
        assert CountingTool.executions == []
        assert agent.last_state.tool_results[0].success is False
        assert "blocked" in agent.last_state.errors[0]

    async def test_empty_allowed_tools_blocks_every_tool(self):
        provider = MockLLMProvider([decision(tool_call("c1"))])
        agent = Agent(
            provider,
            ToolRegistry([CountingTool()]),
            policy=ExecutionPolicy(allowed_tools=set()),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.POLICY_BLOCKED
        assert agent.last_state.tool_results[0].success is False

    async def test_unknown_tool_reaches_registry_when_allowed(self):
        provider = MockLLMProvider([decision(tool_call("c1", name="missing")), {"action": "finish"}])
        agent = Agent(
            provider,
            ToolRegistry([]),
            policy=ExecutionPolicy(allowed_tools={"missing"}),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.FINISHED
        assert "Unknown tool" in agent.last_state.errors[0]

    async def test_different_calls_do_not_trigger_loop_detection(self):
        CountingTool.executions = []
        provider = MockLLMProvider(
            [decision(tool_call("c1", value="one")), decision(tool_call("c2", value="two")), {"action": "finish"}]
        )
        agent = Agent(
            provider,
            ToolRegistry([CountingTool()]),
            policy=ExecutionPolicy(max_same_tool_calls=1),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.FINISHED
        assert CountingTool.executions == ["one", "two"]

    async def test_repeated_call_triggers_loop_detection_before_execution(self):
        CountingTool.executions = []
        provider = MockLLMProvider([decision(tool_call("c1")), decision(tool_call("c2"))])
        agent = Agent(
            provider,
            ToolRegistry([CountingTool()]),
            policy=ExecutionPolicy(max_same_tool_calls=1),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.LOOP_DETECTED
        assert CountingTool.executions == ["x"]
        assert len(agent.last_state.tool_results) == 1

    async def test_argument_order_is_normalized_for_loop_detection(self):
        class AnyInput(BaseModel):
            first: str
            second: str

        class ObjectTool:
            name = "object"
            description = "Accepts an object."
            input_schema = AnyInput

            async def execute(self, **kwargs: Any) -> str:
                return "ok"

        calls = [
            ToolCall(call_id="c1", tool_name="object", arguments={"first": "a", "second": "b"}),
            ToolCall(call_id="c2", tool_name="object", arguments={"second": "b", "first": "a"}),
        ]
        provider = MockLLMProvider([decision(*calls)])
        agent = Agent(
            provider,
            ToolRegistry([ObjectTool()]),
            policy=ExecutionPolicy(max_same_tool_calls=1),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.LOOP_DETECTED
        assert len(agent.last_state.tool_results) == 1

    async def test_global_timeout_cancels_slow_tool(self):
        provider = MockLLMProvider([decision(tool_call("c1", name="slow"))])
        agent = Agent(
            provider,
            ToolRegistry([SlowTool()]),
            policy=ExecutionPolicy(global_timeout_seconds=0.01),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.TIMEOUT
        assert result.errors == ["Agent run timed out after 0.01s"]

    async def test_per_tool_timeout_fails_the_call_without_killing_the_run(self):
        """Fase 11F: Settings.default_per_tool_timeout_seconds existed but
        was never wired into ExecutionPolicy/Agent -- a single hanging
        tool was only ever bounded by the much coarser global_timeout.
        Now a per-tool timeout is enforced, and (like any other tool
        failure) it's recorded and the loop continues -- it doesn't need
        the coarse global timeout to recover."""
        provider = MockLLMProvider([decision(tool_call("c1", name="slow")), {"action": "finish"}])
        agent = Agent(
            provider,
            ToolRegistry([SlowTool()]),
            policy=ExecutionPolicy(global_timeout_seconds=5, per_tool_timeout_seconds=0.05),
        )

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.COMPLETED
        assert agent.last_state.tool_results[0].success is False
        assert "timed out after 0.05s" in agent.last_state.tool_results[0].error

    async def test_registry_exception_has_tool_error_reason(self):
        provider = MockLLMProvider([decision(tool_call("c1"))])
        agent = Agent(provider, RaisingRegistry())

        result = await agent.run(ResearchRequest(question="q"))

        assert result.termination_reason == TerminationReason.TOOL_ERROR
        assert result.errors == ["RuntimeError: registry failure"]

    async def test_request_overrides_are_applied_and_allowed_tools_intersect(self):
        CountingTool.executions = []
        provider = MockLLMProvider([decision(tool_call("c1"))])
        agent = Agent(
            provider,
            ToolRegistry([CountingTool()]),
            policy=ExecutionPolicy(max_steps=5, allowed_tools={"count"}),
        )

        result = await agent.run(
            ResearchRequest(question="q", max_steps=1, allowed_tools=[])
        )

        assert result.termination_reason == TerminationReason.POLICY_BLOCKED
        assert CountingTool.executions == []
