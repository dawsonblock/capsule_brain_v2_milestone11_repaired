import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
from capsule_brain.reflection.repository import ReflectionRepository
from capsule_brain.reflection.service import ReflectionService


class ResolveProvider(LLMProvider):
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        return LLMResult(
            text=[
                "The answer ignored the correction.",
                "Use the correction explicitly.",
                "RESOLVED: corrected.",
            ][(self.calls - 1) % 3],
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_negative_feedback_triggers_reflection(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    memory = MemoryService(
        bus,
        repository=SQLiteMemoryRepository(str(tmp_path / "memory.sqlite")),
    )
    await memory.start()
    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
        },
        providers={"fake": ResolveProvider()},
    )
    await gateway.start()

    service = ReflectionService(
        bus,
        gateway,
        memory,
        repository=ReflectionRepository(str(tmp_path / "reflections.sqlite")),
    )
    await service.start()

    completed = []
    bus.subscribe("reflection.completed", lambda event: completed.append(event.payload))

    await bus.publish(
        EventEnvelope(
            event_type="feedback.recorded",
            source="feedback",
            payload={
                "response_id": "r1",
                "classification": "negative",
                "text": "That was wrong.",
            },
        )
    )

    assert len(completed) == 1
    assert completed[0]["source"] == "feedback"

    await service.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()
