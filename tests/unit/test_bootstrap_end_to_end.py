"""End-to-end bootstrap integration test.

Exercises ``build_application`` wiring through real SQLite repositories and a
fake LLM provider: conversation -> response provenance, verification failure
triggering reflection, and goal creation. This covers the dependency edges and
event flow that unit tests with isolated services do not.
"""
import pytest

from capsule_brain.events.models import EventEnvelope
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.runtime.bootstrap import build_application


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        # Structured (JSON) requests get a valid JSON object.
        if request.response_format == "json":
            return LLMResult(
                text='{"tasks": ["first", "second"]}',
                model=model_cfg["model_name"],
                provider=self.name,
                latency_ms=1.0,
            )
        # Reflection sends prompts containing "THOUGHT:"; return something
        # stable so the duplicate-revision stop condition triggers quickly.
        if "THOUGHT:" in request.prompt:
            return LLMResult(
                text="revised",
                model=model_cfg["model_name"],
                provider=self.name,
                latency_ms=1.0,
            )
        return LLMResult(
            text="Hello from Capsule Brain.",
            model=model_cfg["model_name"],
            provider=self.name,
            latency_ms=12.5,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


def _config(tmp_path):
    return {
        "memory": {"db_path": str(tmp_path / "memory.sqlite")},
        "memory_consolidator": {"enable": False},
        "llm_gateway": {
            "enable": True,
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake-model",
                    "capabilities": ["text", "json"],
                }
            },
        },
        "conversation": {
            "enable": True,
            "db_path": str(tmp_path / "conversation.sqlite"),
        },
        "feedback": {"enable": True},
        "reflection": {
            "enable": True,
            "db_path": str(tmp_path / "reflection.sqlite"),
            "max_iterations": 2,
        },
        "verification": {
            "enable": True,
            "db_path": str(tmp_path / "verification.sqlite"),
        },
        "execution": {"enable": False},
        "goal_planner": {"db_path": str(tmp_path / "goals.sqlite")},
        "redis_bridge": {"enable": False},
    }


@pytest.mark.asyncio
async def test_bootstrap_end_to_end_event_flow(tmp_path):
    app = build_application(
        _config(tmp_path),
        llm_providers={"fake": FakeProvider()},
    )
    bus = app.services.require("event_bus")

    collected: dict[str, list] = {
        "conversation.response.created": [],
        "gui.response": [],
        "verification.failed": [],
        "reflection.completed": [],
        "goal.created": [],
    }

    for event_type in collected:
        bus.subscribe(
            event_type,
            lambda event, et=event_type: collected[et].append(event.payload),
        )

    await app.start()
    try:
        # 1. Conversation path: operator message -> assistant response.
        await bus.publish(
            EventEnvelope(
                event_type="conversation.message.create",
                source="test",
                payload={"text": "hello", "conversation_id": "conv-1"},
            )
        )
        assert collected["conversation.response.created"]
        assert collected["gui.response"]
        response = collected["conversation.response.created"][0]
        # Provenance must come from the trace travelling with the LLMResult,
        # not from shared gateway state.
        assert response["model_alias"] == "fake"
        assert response["provider"] == "fake"
        assert response["text"] == "Hello from Capsule Brain."

        # 2. Verification failure -> reflection trigger.
        await bus.publish(
            EventEnvelope(
                event_type="verification.request",
                source="test",
                payload={
                    "subject": "",
                    "source": "test",
                    "metadata": {},
                },
            )
        )
        assert collected["verification.failed"]
        # Reflection is triggered synchronously by verification.failed.
        assert collected["reflection.completed"]
        assert collected["reflection.completed"][0]["source"] == "verification"

        # 3. Goal creation path.
        await bus.publish(
            EventEnvelope(
                event_type="goal.request",
                source="test",
                payload={"text": "Build the system"},
            )
        )
        assert collected["goal.created"]
        assert collected["goal.created"][0]["text"] == "Build the system"

        # 4. Every service is healthy.
        health = await app.services.health()
        unhealthy = {
            name: status
            for name, status in health.items()
            if not status.healthy
        }
        assert not unhealthy, f"Unhealthy services: {unhealthy}"
    finally:
        await app.stop()
