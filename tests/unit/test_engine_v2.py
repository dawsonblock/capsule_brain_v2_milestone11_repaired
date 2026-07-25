import pytest

from capsule_brain.core.engine_v2 import CapsuleEngineV2
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
from capsule_brain.runtime.registry import ServiceRegistry
from capsule_brain.runtime.tasks import TaskRegistry


@pytest.mark.asyncio
async def test_engine_exposes_memory_service_not_mutable_list(tmp_path):
    bus = LocalEventBus()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    memory = MemoryService(bus, repository=repo)

    services = ServiceRegistry()
    services.register(bus)
    services.register(memory, requires=["event_bus"])
    await services.start_all()

    engine = CapsuleEngineV2(
        services=services,
        tasks=TaskRegistry(),
        event_bus=bus,
        memory=memory,
    )

    assert not hasattr(engine, "_memory")
    created = await engine.memory_write("hello", source="test")
    recent = await engine.memory_recent(limit=10)
    assert recent[0].id == created.id

    await services.stop_all()
