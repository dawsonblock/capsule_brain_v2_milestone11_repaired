import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.memory.models import MemoryType
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository


@pytest.mark.asyncio
async def test_protected_memory_cannot_be_archived(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    memory = MemoryService(bus, repository=repo)
    await memory.start()

    record = await memory.write(
        "operator instruction",
        type=MemoryType.OPERATOR,
        source="operator",
    )
    assert record.protected is True
    assert await memory.archive(record.id) is False

    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_memory_created_event(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    memory = MemoryService(bus, repository=repo)
    await memory.start()

    seen = []
    bus.subscribe("memory.created", lambda event: seen.append(event.payload))

    record = await memory.write("x", source="test")
    assert seen[0]["id"] == record.id

    await memory.stop()
    await bus.stop()
