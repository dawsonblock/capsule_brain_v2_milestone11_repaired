import asyncio
import json
import pytest

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.llm.tools import ToolCall, ToolRegistry, ToolSpec


class ToolCallingProvider(LLMProvider):
    """A fake provider that simulates a two-step tool-calling conversation.

    Step 1: returns a tool_call requesting ``get_weather``.
    Step 2 (after tool result is fed back): returns a natural-language answer
    that incorporates the tool result.
    """

    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        if self.calls == 1 and request.tools:
            return LLMResult(
                text="",
                model=model_cfg["model_name"],
                provider=self.name,
                latency_ms=1.0,
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", arguments={"city": "Tokyo"})
                ],
                finish_reason="tool_calls",
                raw={},
            )
        # Second call: produce a final answer using the tool result.
        tool_text = ""
        for tr in request.tool_results:
            tool_text = tr.get("content", "")
        return LLMResult(
            text=f"The weather is {tool_text}",
            model=model_cfg["model_name"],
            provider=self.name,
            latency_ms=1.0,
            finish_reason="stop",
            raw={},
        )


def _gateway():
    return LLMGateway(
        {
            "default_model": "test",
            "timeout_s": 1,
            "max_attempts": 1,
            "backoff_s": 0,
            "models": {
                "test": {
                    "provider": "fake",
                    "model_name": "fake-model",
                    "capabilities": ["text", "tools"],
                }
            },
        },
        providers={"fake": ToolCallingProvider()},
    )


@pytest.mark.asyncio
async def test_tool_registry_validates_required_args():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="echo",
        description="echo a message",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )

    async def handler(args):
        return {"echo": args["message"]}

    registry.register(spec, handler)
    await registry.start()

    # Missing required arg -> error result, handler not called.
    result = await registry.execute(ToolCall(id="1", name="echo", arguments={}))
    assert result.is_error
    assert "Missing required arguments" in result.content

    # Valid call -> handler output stringified.
    result = await registry.execute(
        ToolCall(id="2", name="echo", arguments={"message": "hi"})
    )
    assert not result.is_error
    assert json.loads(result.content) == {"echo": "hi"}

    # Unknown tool.
    result = await registry.execute(ToolCall(id="3", name="nope", arguments={}))
    assert result.is_error
    assert "Unknown tool" in result.content

    await registry.stop()


@pytest.mark.asyncio
async def test_generate_with_tools_runs_loop():
    gateway = _gateway()
    await gateway.start()
    registry = ToolRegistry()
    await registry.start()

    async def get_weather(args):
        return f"sunny in {args['city']}"

    registry.register(
        ToolSpec(
            name="get_weather",
            description="Get the weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
        get_weather,
    )

    result, tool_results = await gateway.generate_with_tools(
        LLMRequest(prompt="What is the weather in Tokyo?"),
        registry,
        max_iterations=4,
    )

    assert "sunny in Tokyo" in result.text
    assert len(tool_results) == 1
    assert tool_results[0].name == "get_weather"
    assert "Tokyo" in tool_results[0].content

    await registry.stop()
    await gateway.stop()


@pytest.mark.asyncio
async def test_generate_with_tools_requires_registered_tools():
    gateway = _gateway()
    await gateway.start()
    registry = ToolRegistry()
    await registry.start()

    from capsule_brain.llm.errors import LLMConfigurationError

    with pytest.raises(LLMConfigurationError):
        await gateway.generate_with_tools(
            LLMRequest(prompt="hi"),
            registry,
        )

    await registry.stop()
    await gateway.stop()


@pytest.mark.asyncio
async def test_tool_capability_enforced():
    """A model without the 'tools' capability must reject tool requests."""
    cfg = {
        "default_model": "test",
        "timeout_s": 1,
        "max_attempts": 1,
        "backoff_s": 0,
        "models": {
            "test": {
                "provider": "fake",
                "model_name": "fake-model",
                "capabilities": ["text"],  # no tools
            }
        },
    }
    from capsule_brain.llm.errors import LLMCapabilityError

    class StubProvider(LLMProvider):
        name = "fake"

        async def generate(self, request, model_cfg):
            return LLMResult(text="ok", model="m", provider="fake", latency_ms=1)

    gateway = LLMGateway(cfg, providers={"fake": StubProvider()})
    await gateway.start()

    registry = ToolRegistry()
    await registry.start()
    registry.register(
        ToolSpec(name="t", description="d", parameters={"type": "object"}),
        lambda args: asyncio.sleep(0, result=None),
    )

    with pytest.raises(LLMCapabilityError):
        await gateway.generate_with_tools(LLMRequest(prompt="x"), registry)

    await registry.stop()
    await gateway.stop()


def test_tool_spec_openai_schema():
    spec = ToolSpec(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    schema = spec.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["parameters"]["type"] == "object"
