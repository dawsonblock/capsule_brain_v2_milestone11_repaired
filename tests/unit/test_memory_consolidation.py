from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.memory.consolidation import MemoryConsolidator
from capsule_brain.memory.models import MemoryRecord, MemoryType
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository


def old_timestamp(days=30):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


@pytest.mark.asyncio
async def test_consolidator_archives_old_unprotected_memory(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    memory = MemoryService(bus, repository=repo)
    await memory.start()

    record = MemoryRecord(
        text="old",
        source="test",
        created_at=old_timestamp(),
        updated_at=old_timestamp(),
        protected=False,
    )
    await repo.create(record)

    consolidator = MemoryConsolidator(
        memory,
        {"archive_after_days": 7, "interval_sec": 9999},
    )

    archived = await consolidator.run_once()
    assert archived == 1
    assert await repo.count() == 0
    assert await repo.count(include_archived=True) == 1

    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_consolidator_never_archives_protected_memory(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    memory = MemoryService(bus, repository=repo)
    await memory.start()

    record = MemoryRecord(
        text="protected",
        source="operator",
        type=MemoryType.OPERATOR,
        created_at=old_timestamp(),
        updated_at=old_timestamp(),
        protected=True,
    )
    await repo.create(record)

    consolidator = MemoryConsolidator(
        memory,
        {"archive_after_days": 7},
    )

    archived = await consolidator.run_once()
    assert archived == 0
    assert await repo.count() == 1

    await memory.stop()
    await bus.stop()
