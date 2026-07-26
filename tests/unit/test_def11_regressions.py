"""Regression tests for DEF-11-01 through DEF-11-10.

Each test targets a specific defect from the Milestone 11 audit and would
have failed against the pre-fix code.
"""
import asyncio
import json
import sqlite3
import sys
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.memory.consolidation import MemoryConsolidator
from capsule_brain.memory.models import MemoryRecord, MemoryType
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.verification.verifiers import RequiredFieldsVerifier
from capsule_brain.verification.models import VerificationStatus


# ---------------------------------------------------------------------------
# DEF-11-01: Bootstrap de-nesting — verification registers without LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_def11_01_verification_registers_without_llm(tmp_path: Path):
    """Verification and execution must register even when llm_gateway is
    disabled.  Pre-fix, they were nested inside the LLM block and skipped."""
    app = build_application({
        "memory": {"db_path": str(tmp_path / "m.sqlite")},
        "memory_consolidator": {"enable": False},
        "llm_gateway": {"enable": False},
        "verification": {"enable": True, "db_path": str(tmp_path / "v.sqlite")},
        "execution": {"enable": False},
        "goal_planner": {"db_path": str(tmp_path / "g.sqlite")},
        "redis_bridge": {"enable": False},
    })
    await app.start()
    health = await app.services.health()
    assert "verification" in health
    assert health["verification"].healthy
    # LLM-dependent services should be absent
    assert "conversation" not in health
    assert "reflection" not in health
    await app.stop()


@pytest.mark.asyncio
async def test_def11_01_conversation_enabled_without_llm_raises(tmp_path: Path):
    """Explicitly enabling conversation without LLM is a configuration error."""
    with pytest.raises(RuntimeError, match="ConversationService requires llm_gateway"):
        build_application({
            "memory": {"db_path": str(tmp_path / "m.sqlite")},
            "memory_consolidator": {"enable": False},
            "llm_gateway": {"enable": False},
            "conversation": {"enable": True, "db_path": str(tmp_path / "c.sqlite")},
            "goal_planner": {"db_path": str(tmp_path / "g.sqlite")},
            "redis_bridge": {"enable": False},
        })


# ---------------------------------------------------------------------------
# DEF-11-02: Consolidator scans oldest-first
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_def11_02_consolidator_archives_old_beyond_scan_limit(tmp_path: Path):
    """Old records past max_scan must still be archived.  Pre-fix, recent()
    returned newest-first so old records beyond the scan window were hidden."""
    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    memory = MemoryService(bus, repository=repo)
    await memory.start()

    # Create max_scan + 5 old records, then 5 new records.
    # With recent() (newest-first), only the 5 new + 995 newest old records
    # would be scanned, missing the 5 oldest. With oldest(), all 1000 old
    # records are scanned.
    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    new_ts = datetime.now(timezone.utc).isoformat()

    for i in range(1000):
        await repo.create(MemoryRecord(
            text=f"old-{i}",
            source="test",
            created_at=old_ts,
            updated_at=old_ts,
            protected=False,
        ))
    for i in range(5):
        await repo.create(MemoryRecord(
            text=f"new-{i}",
            source="test",
            created_at=new_ts,
            updated_at=new_ts,
            protected=False,
        ))

    consolidator = MemoryConsolidator(memory, {
        "archive_after_days": 7,
        "max_scan": 1000,
        "interval_sec": 9999,
    })

    archived = await consolidator.run_once()
    # All 1000 old records should be archived (they're all older than 7 days).
    # Pre-fix, only 995 would be archived (5 hidden past the scan window).
    assert archived == 1000, f"Expected 1000 archived, got {archived}"
    assert await repo.count() == 5  # Only the 5 new records remain active

    await memory.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# DEF-11-03: Windows volume path normalization
# ---------------------------------------------------------------------------

def test_def11_03_windows_volume_mount_normalization():
    """Windows drive-letter paths must be converted to POSIX format."""
    from capsule_brain.execution.container_runner import _format_volume_mount

    # Simulate Windows path with platform mock (the function only activates
    # the conversion when sys.platform == "win32")
    win_path = "C:\\Users\\test\\sandbox"
    with unittest.mock.patch.object(sys, "platform", "win32"):
        result = _format_volume_mount(win_path)
    assert result == "/c/Users/test/sandbox:/workspace:ro", result

    # POSIX paths pass through unchanged (no mock needed — we're on POSIX)
    posix_path = "/home/test/sandbox"
    result = _format_volume_mount(posix_path)
    assert result == "/home/test/sandbox:/workspace:ro", result


# ---------------------------------------------------------------------------
# DEF-11-04: Context deduplication
# ---------------------------------------------------------------------------

def test_def11_04_build_context_excludes_turn_memories():
    """OPERATOR and ASSISTANT memories must not appear in the memory section
    of _build_context since they're already in the conversation history."""
    from capsule_brain.conversation.service import ConversationService
    from capsule_brain.conversation.models import Turn, TurnRole

    # Minimal mock — we only need _build_context which is a pure function
    # over history and memories.
    svc = object.__new__(ConversationService)

    history = [
        Turn(id="t1", conversation_id="c1", role=TurnRole.USER, text="Hello"),
        Turn(id="t2", conversation_id="c1", role=TurnRole.ASSISTANT, text="Hi there"),
    ]

    memories = [
        MemoryRecord(text="Hello", type=MemoryType.OPERATOR),
        MemoryRecord(text="Hi there", type=MemoryType.ASSISTANT),
        MemoryRecord(text="Important fact", type=MemoryType.OBSERVATION),
        MemoryRecord(text="Learned lesson", type=MemoryType.REFLECTION),
    ]

    context = svc._build_context(history, memories)

    # Turn text should appear in history section
    assert "Hello" in context
    assert "Hi there" in context

    # Non-turn memories should appear in memory section
    assert "Important fact" in context
    assert "Learned lesson" in context

    # Turn memories should NOT appear in the memory section (they'd be
    # duplicating the history). Check that the memory section doesn't
    # contain [operator] or [assistant] entries.
    memory_section = context.split("Relevant recent memory:")[1]
    assert "[operator]" not in memory_section
    assert "[assistant]" not in memory_section


# ---------------------------------------------------------------------------
# DEF-11-05: RequiredFieldsVerifier fail-closed on non-JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_def11_05_required_fields_fails_on_non_json():
    """Non-JSON input to RequiredFieldsVerifier must FAIL, not SKIP."""
    verifier = RequiredFieldsVerifier()

    result = await verifier.verify(
        subject="not json at all",
        metadata={"required_fields": ["name"]},
    )
    assert result.status == VerificationStatus.FAIL, \
        f"Expected FAIL, got {result.status}"
    assert "not valid JSON" in result.summary


@pytest.mark.asyncio
async def test_def11_05_required_fields_fails_on_non_object_json():
    """JSON array (not object) must FAIL."""
    verifier = RequiredFieldsVerifier()

    result = await verifier.verify(
        subject='["not", "an", "object"]',
        metadata={"required_fields": ["name"]},
    )
    assert result.status == VerificationStatus.FAIL
    assert "not an object" in result.summary


@pytest.mark.asyncio
async def test_def11_05_required_fields_passes_on_valid_json():
    """Valid JSON with all required fields must still PASS."""
    verifier = RequiredFieldsVerifier()

    result = await verifier.verify(
        subject='{"name":"test","value":42}',
        metadata={"required_fields": ["name", "value"]},
    )
    assert result.status == VerificationStatus.PASS


# ---------------------------------------------------------------------------
# DEF-11-06: HTTPX timeout=None
# ---------------------------------------------------------------------------

def test_def11_06_httpx_client_has_no_default_timeout():
    """OpenAICompatibleProvider must disable httpx's 5s default timeout."""
    import httpx
    from capsule_brain.llm.providers.openai_compatible import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider()
    # The timeout should be disabled (None or a Timeout with all None values)
    timeout = provider.client.timeout
    if isinstance(timeout, httpx.Timeout):
        assert timeout.read is None or timeout.read == httpx.USE_CLIENT_DEFAULT
    else:
        assert timeout is None
    # Clean up
    if provider._owns_client:
        import asyncio
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            provider.client.aclose()
        )


# ---------------------------------------------------------------------------
# DEF-11-07: Stream telemetry on GeneratorExit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_def11_07_stream_records_success_on_early_exit():
    """When a consumer breaks out of streaming early, the provider should
    still be marked as healthy since it was producing chunks successfully."""
    from capsule_brain.llm.gateway import LLMGateway
    from capsule_brain.llm.models import LLMRequest, LLMResult
    from capsule_brain.llm.providers.base import LLMProvider

    class StreamingProvider(LLMProvider):
        name = "fake_stream"

        async def generate(self, request, model_cfg):
            return LLMResult(text="ok", model="fake", provider="fake", latency_ms=1)

        async def stream(self, request, model_cfg):
            for i in range(100):
                yield f"chunk-{i} "

    gateway = LLMGateway(
        {
            "enable": True,
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake_stream",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
        },
        providers={"fake_stream": StreamingProvider()},
    )
    await gateway.start()

    # Consume only the first chunk, then break out
    async for chunk in gateway.stream(LLMRequest(model="fake", prompt="test")):
        break  # GeneratorExit will be raised in the generator

    # Give the event loop a chance to process the GeneratorExit
    await asyncio.sleep(0.01)

    # The breaker should have recorded success, not failure
    breaker = gateway._breakers["fake"]
    assert breaker.failures == 0, \
        f"Expected 0 failures, got {breaker.failures}"

    await gateway.stop()


# ---------------------------------------------------------------------------
# DEF-11-09: Goal auto-completion + goal.unresolved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_def11_09_goal_auto_completes_when_all_tasks_done(tmp_path: Path):
    """Marking the last task as complete should auto-complete the goal and
    publish goal.completed."""
    from capsule_brain.core.goal_planner_v2 import GoalPlannerV2

    bus = LocalEventBus()
    await bus.start()

    async def decomposer(goal_text: str):
        return ["Task A", "Task B"]

    planner = GoalPlannerV2(
        event_bus=bus,
        decomposer=decomposer,
        cfg={"db_path": str(tmp_path / "goals.sqlite")},
    )
    await planner.start()

    completed_events = []
    bus.subscribe("goal.completed", lambda e: completed_events.append(e.payload))

    # Create a goal
    await bus.publish(EventEnvelope(
        event_type="goal.request",
        source="test",
        payload={"text": "Test goal"},
    ))

    goals = await planner.list_goals()
    assert len(goals) == 1
    goal = goals[0]
    task_ids = [t["id"] for t in goal["tasks"]]

    # Complete first task — goal should NOT auto-complete yet
    await bus.publish(EventEnvelope(
        event_type="goal.task.set_status",
        source="test",
        payload={"goal_id": goal["id"], "task_id": task_ids[0], "status": "complete"},
    ))
    assert len(completed_events) == 0

    # Complete second task — goal SHOULD auto-complete
    await bus.publish(EventEnvelope(
        event_type="goal.task.set_status",
        source="test",
        payload={"goal_id": goal["id"], "task_id": task_ids[1], "status": "complete"},
    ))
    assert len(completed_events) == 1
    assert completed_events[0]["goal_id"] == goal["id"]

    # Verify goal status is persisted as "complete"
    goals = await planner.list_goals()
    assert goals[0]["status"] == "complete"

    await planner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_def11_09_check_unresolved_goals_emits_event(tmp_path: Path):
    """check_unresolved_goals() should emit goal.unresolved for active goals
    with no completed tasks."""
    from capsule_brain.core.goal_planner_v2 import GoalPlannerV2

    bus = LocalEventBus()
    await bus.start()

    async def decomposer(goal_text: str):
        return ["Task A", "Task B"]

    planner = GoalPlannerV2(
        event_bus=bus,
        decomposer=decomposer,
        cfg={"db_path": str(tmp_path / "goals.sqlite")},
    )
    await planner.start()

    unresolved_events = []
    bus.subscribe("goal.unresolved", lambda e: unresolved_events.append(e.payload))

    # Create a goal
    await bus.publish(EventEnvelope(
        event_type="goal.request",
        source="test",
        payload={"text": "Unresolved goal"},
    ))

    # Check for unresolved goals
    await planner.check_unresolved_goals()

    assert len(unresolved_events) == 1
    assert unresolved_events[0]["text"] == "Unresolved goal"

    await planner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# DEF-11-10: Secondary indexes exist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_def11_10_execution_repository_has_started_at_index(tmp_path: Path):
    from capsule_brain.execution.repository import ExecutionRepository

    repo = ExecutionRepository(str(tmp_path / "exec.sqlite"))
    await repo.start()
    conn = repo._conn
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='execution_results'"
    ).fetchall()
    index_names = {r[0] for r in indexes}
    assert "idx_execution_started_at" in index_names
    await repo.stop()


@pytest.mark.asyncio
async def test_def11_10_verification_repository_has_indexes(tmp_path: Path):
    from capsule_brain.verification.repository import VerificationRepository

    repo = VerificationRepository(str(tmp_path / "ver.sqlite"))
    await repo.start()
    conn = repo._conn
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='verification_results'"
    ).fetchall()
    index_names = {r[0] for r in indexes}
    assert "idx_verification_created_at" in index_names
    assert "idx_verification_status" in index_names
    await repo.stop()


@pytest.mark.asyncio
async def test_def11_10_reflection_repository_has_created_at_index(tmp_path: Path):
    from capsule_brain.reflection.repository import ReflectionRepository

    repo = ReflectionRepository(str(tmp_path / "ref.sqlite"))
    await repo.start()
    conn = repo._conn
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='reflection_sessions'"
    ).fetchall()
    index_names = {r[0] for r in indexes}
    assert "idx_reflection_created_at" in index_names
    await repo.stop()


@pytest.mark.asyncio
async def test_def11_10_experience_store_has_created_at_index(tmp_path: Path):
    from capsule_brain.learning.experience_store import ExperienceStore

    store = ExperienceStore(db_path=str(tmp_path / "exp.sqlite"))
    await store.start()
    conn = store._conn
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='experiences'"
    ).fetchall()
    index_names = {r[0] for r in indexes}
    assert "idx_experience_created_at" in index_names
    await store.stop()
