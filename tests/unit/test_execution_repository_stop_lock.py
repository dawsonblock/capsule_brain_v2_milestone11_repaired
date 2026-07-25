import asyncio

import pytest

from capsule_brain.execution.repository import ExecutionRepository


@pytest.mark.asyncio
async def test_stop_waits_for_repository_lock(tmp_path):
    repo = ExecutionRepository(
        str(tmp_path / "execution.sqlite")
    )
    await repo.start()

    await repo._lock.acquire()
    stop_task = asyncio.create_task(repo.stop())
    await asyncio.sleep(0)

    assert not stop_task.done()

    repo._lock.release()
    await stop_task

    assert repo._conn is None
