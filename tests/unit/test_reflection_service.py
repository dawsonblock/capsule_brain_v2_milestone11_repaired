import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
from capsule_brain.reflection.repository import ReflectionRepository
from capsule_brain.reflection.service import ReflectionService


class ScriptedProvider(LLMProvider):
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        responses = [
            "The thought lacks a measurable next step.",
            "Add a measurable next step and verify the outcome.",
            "RESOLVED: the revision addresses the missing next step.",
        ]
        return LLMResult(
            text=responses[self.calls - 1],
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_reflection_is_iterative_and_bounded(tmp_path):
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
            "routing": {
                "routes": {
                    "reflection": {"models": ["fake"]}
                }
            },
        },
        providers={"fake": ScriptedProvider()},
    )
    await gateway.start()

    service = ReflectionService(
        bus,
        gateway,
        memory,
        cfg={"max_iterations": 4},
        repository=ReflectionRepository(str(tmp_path / "reflections.sqlite")),
    )
    await service.start()

    session = await service.reflect(
        seed="Improve this plan.",
        source="test",
    )

    assert session.stop_reason == "resolved"
    assert len(session.iterations) == 1
    assert session.final_text == "Add a measurable next step and verify the outcome."

    memories = await memory.recent(limit=10)
    assert any(m.type.value == "reflection" for m in memories)

    await service.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()
