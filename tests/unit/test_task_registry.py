import asyncio

import pytest

from capsule_brain.runtime.tasks import TaskRegistry


@pytest.mark.asyncio
async def test_task_registry_cancels_background_tasks():
    registry = TaskRegistry()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def worker():
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            cancelled.set()

    registry.spawn(worker(), name="worker")
    assert registry.active_count == 1

    # Let the worker actually begin so cancellation exercises its cleanup path.
    await started.wait()
    await registry.shutdown()

    assert cancelled.is_set()
    assert registry.active_count == 0
