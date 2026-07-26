"""Regression tests for REP-01 through REP-07 hardening fixes."""
import asyncio
import sqlite3
import sys
import time
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.memory.models import MemoryRecord
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository


# ---------------------------------------------------------------------------
# REP-01: Batch transaction archival
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rep01_archive_batch_single_transaction(tmp_path: Path):
    """archive_batch() should archive all records in one transaction."""
    repo = SQLiteMemoryRepository(str(tmp_path / "mem.sqlite"))
    await repo.start()

    ids = []
    for i in range(500):
        record = await repo.create(MemoryRecord(
            text=f"record-{i}",
            source="test",
        ))
        ids.append(record.id)

    count = await repo.archive_batch(ids)
    assert count == 500

    # All records should be archived
    active = await repo.count(include_archived=False)
    total = await repo.count(include_archived=True)
    assert active == 0
    assert total == 500

    await repo.stop()


@pytest.mark.asyncio
async def test_rep01_archive_batch_skips_protected(tmp_path: Path):
    """archive_batch() should skip protected records."""
    repo = SQLiteMemoryRepository(str(tmp_path / "mem.sqlite"))
    await repo.start()

    protected_record = await repo.create(MemoryRecord(
        text="protected",
        source="test",
        protected=True,
    ))
    normal_record = await repo.create(MemoryRecord(
        text="normal",
        source="test",
        protected=False,
    ))

    count = await repo.archive_batch([protected_record.id, normal_record.id])
    assert count == 1  # Only the non-protected record

    await repo.stop()


@pytest.mark.asyncio
async def test_rep01_archive_batch_empty_list(tmp_path: Path):
    """archive_batch() with empty list should return 0 without error."""
    repo = SQLiteMemoryRepository(str(tmp_path / "mem.sqlite"))
    await repo.start()
    count = await repo.archive_batch([])
    assert count == 0
    await repo.stop()


@pytest.mark.asyncio
async def test_rep01_consolidator_uses_batch_archival(tmp_path: Path):
    """MemoryConsolidator should use batch archival, not per-record calls."""
    from capsule_brain.memory.consolidation import MemoryConsolidator

    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "mem.sqlite"))
    memory = MemoryService(bus, repository=repo)
    await memory.start()

    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    for i in range(50):
        await repo.create(MemoryRecord(
            text=f"old-{i}",
            source="test",
            created_at=old_ts,
            updated_at=old_ts,
            protected=False,
        ))

    consolidator = MemoryConsolidator(memory, {
        "archive_after_days": 7,
        "max_scan": 1000,
        "interval_sec": 9999,
    })

    archived = await consolidator.run_once()
    assert archived == 50
    assert await repo.count() == 0

    await memory.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# REP-02: SQLite busy_timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rep02_busy_timeout_set_on_all_repositories(tmp_path: Path):
    """All repositories should set PRAGMA busy_timeout = 5000."""
    from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
    from capsule_brain.conversation.repository import ConversationRepository
    from capsule_brain.reflection.repository import ReflectionRepository
    from capsule_brain.verification.repository import VerificationRepository
    from capsule_brain.execution.repository import ExecutionRepository
    from capsule_brain.learning.experience_store import ExperienceStore
    from capsule_brain.core.goal_planner_v2 import GoalPlannerV2

    repos = [
        ("memory", SQLiteMemoryRepository(str(tmp_path / "m.sqlite"))),
        ("conversation", ConversationRepository(str(tmp_path / "c.sqlite"))),
        ("reflection", ReflectionRepository(str(tmp_path / "r.sqlite"))),
        ("verification", VerificationRepository(str(tmp_path / "v.sqlite"))),
        ("execution", ExecutionRepository(str(tmp_path / "e.sqlite"))),
        ("experience", ExperienceStore(db_path=str(tmp_path / "exp.sqlite"))),
    ]

    for name, repo in repos:
        await repo.start()
        conn = repo._conn
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        # Handle both tuple and sqlite3.Row factories
        timeout = row[0] if not isinstance(row, sqlite3.Row) else row[0]
        assert timeout == 5000, f"{name} repository busy_timeout is {timeout}, expected 5000"
        await repo.stop()

    # GoalPlannerV2 stores connection as _db
    planner = GoalPlannerV2(
        event_bus=LocalEventBus(),
        decomposer=None,
        cfg={"db_path": str(tmp_path / "g.sqlite")},
    )
    await planner.start()
    row = planner._db.execute("PRAGMA busy_timeout").fetchone()
    assert row[0] == 5000
    await planner.stop()


# ---------------------------------------------------------------------------
# REP-03: Bytecode suppression in container runner
# ---------------------------------------------------------------------------

def test_rep03_container_runner_includes_bytecode_env_vars():
    """ContainerExecutionRunner should inject PYTHONDONTWRITEBYTECODE and
    PYTHONPYCACHEPREFIX env vars."""
    from capsule_brain.execution.container_runner import ContainerExecutionRunner
    from capsule_brain.execution.models import ExecutionPolicy

    runner = ContainerExecutionRunner(
        ExecutionPolicy(allow=True, allowed_commands=("python",)),
    )

    # We can't easily run a container in tests, but we can verify the command
    # construction by inspecting the runner's configuration and the env vars
    # are part of the command template. Check the source for the env flags.
    import inspect
    source = inspect.getsource(ContainerExecutionRunner.run)
    assert "PYTHONDONTWRITEBYTECODE=1" in source, \
        "ContainerExecutionRunner must inject PYTHONDONTWRITEBYTECODE=1"
    assert "PYTHONPYCACHEPREFIX=/tmp/pycache" in source, \
        "ContainerExecutionRunner must inject PYTHONPYCACHEPREFIX=/tmp/pycache"


def test_rep03_python_compile_verifier_uses_B_flag():
    """PythonCompileVerifier should use -B flag to suppress .pyc writes."""
    from capsule_brain.verification.execution_verifiers import PythonCompileVerifier

    verifier = PythonCompileVerifier.__new__(PythonCompileVerifier)
    assert "-B" in verifier.command_template, \
        "PythonCompileVerifier must use -B flag"


# ---------------------------------------------------------------------------
# REP-04: Redis bridge auto-reconnect
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rep04_redis_bridge_reconnect_loop_exists():
    """RedisBridge._listen() should implement a reconnection loop with
    exponential backoff. We verify this by source inspection since testing
    the actual reconnect requires a real Redis server."""
    import inspect
    from capsule_brain.events.redis_bridge import RedisBridge

    source = inspect.getsource(RedisBridge._listen)
    # Must have a while loop that continues while running/degraded
    assert "while self.state in" in source, \
        "_listen() must have a reconnection while loop"
    # Must have backoff logic
    assert "backoff" in source, \
        "_listen() must implement exponential backoff"
    assert "min(backoff" in source, \
        "_listen() must cap backoff with min()"
    # Must reset backoff on successful connection
    assert "backoff = 1.0" in source, \
        "_listen() must reset backoff on successful reconnection"
    # Must have cleanup on failure
    assert "_cleanup_redis_clients" in source, \
        "_listen() must clean up Redis clients on failure"
    # Must handle CancelledError
    assert "CancelledError" in source, \
        "_listen() must handle asyncio.CancelledError"


# ---------------------------------------------------------------------------
# REP-05: Reflection session timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rep05_reflection_session_timeout(tmp_path: Path):
    """ReflectionService should enforce session_timeout_s and set
    stop_reason='session_timeout' when exceeded."""
    from capsule_brain.llm.gateway import LLMGateway
    from capsule_brain.llm.models import LLMResult
    from capsule_brain.llm.providers.base import LLMProvider
    from capsule_brain.memory.service import MemoryService
    from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
    from capsule_brain.reflection.service import ReflectionService
    from capsule_brain.reflection.repository import ReflectionRepository

    class SlowProvider(LLMProvider):
        name = "slow"
        async def generate(self, request, model_cfg):
            await asyncio.sleep(1.0)  # Exceeds the 0.05s timeout
            return LLMResult(text="RESOLVED", model="slow", provider="slow", latency_ms=1000)

    gateway = LLMGateway(
        {"enable": True, "default_model": "slow",
         "models": {"slow": {"provider": "slow", "model_name": "slow", "capabilities": ["text"]}}},
        providers={"slow": SlowProvider()},
    )

    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "mem.sqlite"))
    memory = MemoryService(bus, repository=repo)
    await memory.start()
    await gateway.start()

    reflection = ReflectionService(
        event_bus=bus,
        llm=gateway,
        memory=memory,
        cfg={
            "db_path": str(tmp_path / "ref.sqlite"),
            "max_iterations": 4,
            "session_timeout_s": 0.05,
        },
        repository=ReflectionRepository(str(tmp_path / "ref.sqlite")),
    )
    await reflection.start()

    completed_events = []
    async def capture(event):
        completed_events.append(event.payload)
    bus.subscribe("reflection.completed", capture)

    await reflection.reflect(seed="Test seed", source="test")
    await asyncio.sleep(0.01)

    assert len(completed_events) == 1
    assert completed_events[0]["stop_reason"] == "session_timeout", \
        f"Expected 'session_timeout', got {completed_events[0]['stop_reason']}"

    await reflection.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# REP-06: TaskRegistry event emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rep06_task_registry_emits_event_on_failure():
    """TaskRegistry should emit system.task.failed when a background task
    raises an unhandled exception."""
    from capsule_brain.runtime.tasks import TaskRegistry

    bus = LocalEventBus()
    await bus.start()

    failed_events = []
    async def capture(event):
        failed_events.append(event.payload)
    bus.subscribe("system.task.failed", capture)

    registry = TaskRegistry(event_bus_provider=lambda: bus)

    async def failing_task():
        raise ValueError("Task failed intentionally")

    registry.spawn(failing_task(), name="test-failing-task")

    # Wait for the task to complete and the event to be published
    await asyncio.sleep(0.1)

    assert len(failed_events) == 1, f"Expected 1 failure event, got {len(failed_events)}"
    assert failed_events[0]["task_name"] == "test-failing-task"
    assert "Task failed intentionally" in failed_events[0]["error"]
    assert failed_events[0]["error_type"] == "ValueError"

    await registry.shutdown()
    await bus.stop()


@pytest.mark.asyncio
async def test_rep06_task_registry_no_event_when_no_provider():
    """TaskRegistry without event_bus_provider should not crash on failure."""
    from capsule_brain.runtime.tasks import TaskRegistry

    registry = TaskRegistry()  # No event bus provider

    async def failing_task():
        raise ValueError("Should be handled gracefully")

    task = registry.spawn(failing_task(), name="test")
    await asyncio.sleep(0.05)
    assert task.done()
    # No exception should propagate — just logged

    await registry.shutdown()


# ---------------------------------------------------------------------------
# REP-07: Hardened teardown in app_runner
# ---------------------------------------------------------------------------

def test_rep07_app_runner_uses_timeout_on_stop():
    """run_application should use asyncio.wait_for with a 5s timeout on
    runtime.stop() to prevent hanging on force-quit."""
    # Read the source file directly to avoid importing PyQt5 (which may not
    # be installed in CI environments).
    source_path = Path(__file__).resolve().parents[2] / "src" / "capsule_brain" / "gui" / "app_runner.py"
    source = source_path.read_text()
    assert "asyncio.wait_for" in source, \
        "run_application must use asyncio.wait_for for runtime.stop()"
    assert "timeout=5.0" in source, \
        "run_application must use a 5s timeout for runtime.stop()"
    assert "TimeoutError" in source or "asyncio.TimeoutError" in source, \
        "run_application must handle asyncio.TimeoutError"
