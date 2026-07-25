import asyncio
from dataclasses import replace

import pytest

from capsule_brain.memory.models import MemoryRecord, MemoryType
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository


@pytest.mark.asyncio
async def test_create_get_update_archive(tmp_path):
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    await repo.start()

    record = MemoryRecord(
        text="hello",
        type=MemoryType.OBSERVATION,
        source="test",
    )
    await repo.create(record)

    loaded = await repo.get(record.id)
    assert loaded is not None
    assert loaded.text == "hello"

    updated = await repo.update(replace(loaded, text="changed"))
    assert updated.text == "changed"

    assert await repo.archive(record.id) is True
    assert await repo.count() == 0
    assert await repo.count(include_archived=True) == 1

    await repo.stop()


@pytest.mark.asyncio
async def test_concurrent_writes_are_serialized(tmp_path):
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    await repo.start()

    async def write_one(i):
        await repo.create(
            MemoryRecord(
                text=f"memory-{i}",
                source="test",
            )
        )

    await asyncio.gather(*(write_one(i) for i in range(200)))

    assert await repo.count() == 200
    await repo.stop()
