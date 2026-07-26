import asyncio
import pytest
import sys
from pathlib import Path

from capsule_brain.execution.models import ExecutionPolicy, ExecutionRequest
from capsule_brain.execution.runner import ExecutionRunner
from capsule_brain.execution.worker_pool import ExecutionWorkerPool


def _policy(root):
    return ExecutionPolicy(
        allow=True,
        cwd_root=str(root),
        allowed_commands=(Path(sys.executable).name,),
        timeout_s=5,
    )


@pytest.mark.asyncio
async def test_worker_pool_executes_jobs(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "ok.py").write_text("print('ok')")
    runner = ExecutionRunner(_policy(root))
    pool = ExecutionWorkerPool(runner, cfg={"max_workers": 2})
    await pool.start()

    result = await pool.submit(
        ExecutionRequest(command=[sys.executable, "ok.py"], cwd=str(root))
    )
    assert result.passed
    assert "ok" in result.stdout

    health = await pool.health()
    assert health.details["submitted"] == 1
    assert health.details["completed"] == 1
    assert health.details["failed"] == 0

    await pool.stop()


@pytest.mark.asyncio
async def test_worker_pool_bounds_concurrency(tmp_path):
    """With max_workers=1, jobs run serially even when submitted concurrently."""
    root = tmp_path / "sandbox"
    root.mkdir()
    # A script that sleeps briefly so we can observe serialization.
    (root / "slow.py").write_text(
        "import time; print('start'); time.sleep(0.2); print('done')"
    )
    runner = ExecutionRunner(_policy(root))
    pool = ExecutionWorkerPool(runner, cfg={"max_workers": 1})
    await pool.start()

    # Submit 3 jobs concurrently.
    tasks = [
        asyncio.create_task(
            pool.submit(
                ExecutionRequest(command=[sys.executable, "slow.py"], cwd=str(root))
            )
        )
        for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)
    assert all(r.passed for r in results)

    health = await pool.health()
    # peak_active must never exceed max_workers=1.
    assert health.details["peak_active"] <= 1
    assert health.details["submitted"] == 3
    assert health.details["completed"] == 3

    await pool.stop()


@pytest.mark.asyncio
async def test_worker_pool_parallelism_with_multiple_workers(tmp_path):
    """With max_workers=3, three slow jobs should overlap."""
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "slow.py").write_text(
        "import time; time.sleep(0.2)"
    )
    runner = ExecutionRunner(_policy(root))
    pool = ExecutionWorkerPool(runner, cfg={"max_workers": 3})
    await pool.start()

    import time
    started = time.perf_counter()
    tasks = [
        asyncio.create_task(
            pool.submit(
                ExecutionRequest(command=[sys.executable, "slow.py"], cwd=str(root))
            )
        )
        for _ in range(3)
    ]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started

    # 3 jobs * 0.2s serially = 0.6s. With 3 workers, ~0.2s + overhead.
    # Allow generous slack for process startup.
    assert elapsed < 0.55, f"expected parallel execution, took {elapsed:.2f}s"

    health = await pool.health()
    assert health.details["peak_active"] >= 2
    await pool.stop()


@pytest.mark.asyncio
async def test_worker_pool_propagates_runner_errors(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "bad.py").write_text("raise SystemExit(1)")
    runner = ExecutionRunner(_policy(root))
    pool = ExecutionWorkerPool(runner, cfg={"max_workers": 2})
    await pool.start()

    result = await pool.submit(
        ExecutionRequest(command=[sys.executable, "bad.py"], cwd=str(root))
    )
    assert not result.passed
    assert result.exit_code == 1

    await pool.stop()


@pytest.mark.asyncio
async def test_worker_pool_rejects_when_queue_full(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "slow.py").write_text("import time; time.sleep(0.5)")
    runner = ExecutionRunner(_policy(root))
    pool = ExecutionWorkerPool(
        runner,
        cfg={"max_workers": 1, "queue_max": 1},
    )
    await pool.start()

    # First job occupies the single worker (sleeps 0.5s).
    job1 = asyncio.create_task(
        pool.submit(
            ExecutionRequest(command=[sys.executable, "slow.py"], cwd=str(root))
        )
    )
    # Give the worker time to pick up job1 and enter the subprocess.
    await asyncio.sleep(0.1)
    # Second job fills the queue (queue_max=1).
    job2 = asyncio.create_task(
        pool.submit(
            ExecutionRequest(command=[sys.executable, "slow.py"], cwd=str(root))
        )
    )
    # Give job2 time to be enqueued.
    await asyncio.sleep(0.05)
    assert pool._queue.qsize() >= 1, f"queue should have job2, qsize={pool._queue.qsize()}"
    # Third job should be rejected.
    with pytest.raises(RuntimeError, match="queue is full"):
        await pool.submit(
            ExecutionRequest(command=[sys.executable, "slow.py"], cwd=str(root))
        )

    await asyncio.gather(job1, job2)
    await pool.stop()


@pytest.mark.asyncio
async def test_execution_service_uses_worker_pool(tmp_path):
    """ExecutionService should delegate to the pool when one is provided."""
    from capsule_brain.events.local_bus import LocalEventBus
    from capsule_brain.execution.service import ExecutionService

    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "ok.py").write_text("print('ok')")
    bus = LocalEventBus()
    await bus.start()

    runner = ExecutionRunner(_policy(root))
    pool = ExecutionWorkerPool(runner, cfg={"max_workers": 2})
    service = ExecutionService(
        bus,
        cfg={"allow": True, "cwd_root": str(root),
             "allowed_commands": [Path(sys.executable).name]},
        runner=runner,
        worker_pool=pool,
    )
    await service.start()

    result = await service.execute(
        ExecutionRequest(command=[sys.executable, "ok.py"], cwd=str(root))
    )
    assert result.passed

    health = await service.health()
    assert health.details["worker_pool"] is True
    assert health.details["pool"]["completed"] == 1

    await service.stop()
    await bus.stop()
