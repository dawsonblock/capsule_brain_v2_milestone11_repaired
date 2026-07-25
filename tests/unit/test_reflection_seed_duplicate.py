import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import (
    SQLiteMemoryRepository,
)
from capsule_brain.reflection.repository import ReflectionRepository
from capsule_brain.reflection.service import ReflectionService


class DuplicateProvider(LLMProvider):
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        sequence = [
            "No material issue.",
            "same seed",
            "CONTINUE: no improvement.",
        ]
        return LLMResult(
            text=sequence[(self.calls - 1) % 3],
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_first_revision_equal_to_seed_stops_immediately(
    tmp_path,
):
    bus = LocalEventBus()
    await bus.start()

    memory = MemoryService(
        bus,
        repository=SQLiteMemoryRepository(
            str(tmp_path / "memory.sqlite")
        ),
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
        providers={"fake": DuplicateProvider()},
    )
    await gateway.start()

    reflection = ReflectionService(
        bus,
        gateway,
        memory,
        repository=ReflectionRepository(
            str(tmp_path / "reflection.sqlite")
        ),
    )
    await reflection.start()

    session = await reflection.reflect(
        seed="same seed",
        source="test",
    )

    assert session.stop_reason == "duplicate_revision"
    assert len(session.iterations) == 1

    await reflection.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()
