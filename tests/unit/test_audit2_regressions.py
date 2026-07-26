"""Regression tests for P0/P1 fixes from the second audit.

Covers:
  - Shipped YAML config boots cleanly
  - load_config validates configuration consistency
  - Execution-backed verifiers (PythonCompileVerifier)
  - Context-aware reflection (actual artifact in seed, not IDs)
  - Digest pinning fails closed
  - Container-first execution default
  - Background event dispatch (async_dispatch)
"""
import asyncio
import sys
import unittest.mock
from pathlib import Path

import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.execution.models import ExecutionPolicy, ExecutionRequest, ExecutionResult
from capsule_brain.runtime.bootstrap import load_config, build_application
from capsule_brain.verification.models import VerificationStatus


# ---------------------------------------------------------------------------
# P0: Shipped YAML config boots
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shipped_yaml_boots_cleanly(tmp_path: Path):
    """The shipped configs/v2_runtime.yaml must boot without error."""
    cfg = load_config("configs/v2_runtime.yaml")
    # Override db paths to temp so we don't pollute the repo
    for key in ("memory", "goal_planner"):
        if key in cfg and "db_path" in cfg[key]:
            cfg[key]["db_path"] = str(tmp_path / f"{key}.sqlite")
    if "verification" in cfg:
        cfg["verification"]["db_path"] = str(tmp_path / "ver.sqlite")
    if "learning" in cfg:
        cfg["learning"]["db_path"] = str(tmp_path / "exp.sqlite")

    app = build_application(cfg)
    await app.start()
    health = await app.services.health()
    assert health["event_bus"].healthy
    assert health["verification"].healthy
    assert "conversation" not in health  # LLM disabled → conversation disabled
    await app.stop()


def test_load_config_rejects_conversation_without_llm(tmp_path: Path):
    """load_config must reject conversation.enabled=true with llm disabled."""
    import yaml
    bad_cfg = {
        "llm_gateway": {"enable": False},
        "conversation": {"enable": True},
    }
    cfg_path = tmp_path / "bad.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(bad_cfg, f)
    with pytest.raises(ValueError, match="conversation.enable is true but llm_gateway"):
        load_config(cfg_path)


def test_load_config_rejects_host_execution_without_unsafe_flag(tmp_path: Path):
    """load_config must reject host runner without unsafe_allow_host_execution."""
    import yaml
    bad_cfg = {
        "execution": {"enable": True, "runner": "host", "allow": True},
    }
    cfg_path = tmp_path / "bad.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(bad_cfg, f)
    with pytest.raises(ValueError, match="unsafe_allow_host_execution"):
        load_config(cfg_path)


# ---------------------------------------------------------------------------
# P0: Execution-backed verifiers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_python_compile_verifier_runs_code(tmp_path: Path):
    """PythonCompileVerifier should actually compile code via ExecutionService."""
    from capsule_brain.execution.service import ExecutionService
    from capsule_brain.verification.execution_verifiers import PythonCompileVerifier

    bus = LocalEventBus()
    await bus.start()
    exec_service = ExecutionService(
        event_bus=bus,
        cfg={
            "allow": True,
            "cwd_root": str(tmp_path),
            "allowed_commands": ("python",),
            "timeout_s": 10,
        },
    )
    await exec_service.start()

    verifier = PythonCompileVerifier(exec_service)

    # Valid Python should PASS
    result = await verifier.verify(
        subject="x = 1\nprint(x)\n",
        metadata={"content_type": "python"},
    )
    assert result.status == VerificationStatus.PASS, result.summary

    # Invalid Python should FAIL
    result = await verifier.verify(
        subject="def broken(:\n  pass\n",
        metadata={"content_type": "python"},
    )
    assert result.status == VerificationStatus.FAIL, result.summary

    # Non-Python content_type should SKIP
    result = await verifier.verify(
        subject="not python",
        metadata={"content_type": "json"},
    )
    assert result.status == VerificationStatus.SKIP

    await exec_service.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_verification_service_includes_execution_verifiers(tmp_path: Path):
    """VerificationService should include execution verifiers when execution_service is provided."""
    from capsule_brain.execution.service import ExecutionService
    from capsule_brain.verification.service import VerificationService

    bus = LocalEventBus()
    await bus.start()
    exec_service = ExecutionService(
        event_bus=bus,
        cfg={
            "allow": True,
            "cwd_root": str(tmp_path),
            "allowed_commands": ("python",),
        },
    )
    await exec_service.start()

    ver_service = VerificationService(
        event_bus=bus,
        cfg={"db_path": str(tmp_path / "ver.sqlite")},
        execution_service=exec_service,
    )
    await ver_service.start()

    verifier_names = [v.name for v in ver_service.verifiers]
    assert "python_compile" in verifier_names
    assert "pytest" in verifier_names
    assert "non_empty" in verifier_names  # Static verifiers still present

    await ver_service.stop()
    await exec_service.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# P0: Context-aware reflection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflection_seed_contains_actual_artifact_on_verification_failure(tmp_path: Path):
    """When verification fails, the reflection seed should contain the actual
    failed artifact (subject), not just the summary."""
    from capsule_brain.llm.models import LLMResult
    from capsule_brain.llm.providers.base import LLMProvider
    from capsule_brain.memory.service import MemoryService
    from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
    from capsule_brain.reflection.service import ReflectionService
    from capsule_brain.reflection.repository import ReflectionRepository

    class FakeProvider(LLMProvider):
        name = "fake"
        async def generate(self, request, model_cfg):
            return LLMResult(text="RESOLVED", model="fake", provider="fake", latency_ms=1)

    from capsule_brain.llm.gateway import LLMGateway
    gateway = LLMGateway(
        {"enable": True, "default_model": "fake",
         "models": {"fake": {"provider": "fake", "model_name": "fake", "capabilities": ["text"]}}},
        providers={"fake": FakeProvider()},
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
        cfg={"db_path": str(tmp_path / "ref.sqlite"), "max_iterations": 1},
        repository=ReflectionRepository(str(tmp_path / "ref.sqlite")),
    )
    await reflection.start()

    # Emit a verification failure with both summary and subject
    completed_seeds = []
    async def capture_seed(event):
        completed_seeds.append(event.payload.get("seed", ""))
    bus.subscribe("reflection.completed", capture_seed)

    await bus.publish(EventEnvelope(
        event_type="verification.failed",
        source="verification",
        payload={
            "summary": "Python syntax is invalid.",
            "subject": "def broken(:\n  pass\n",
            "source": "test",
            "metadata": {"line": 1},
        },
    ))
    await asyncio.sleep(0.1)

    # The seed should contain the actual broken code, not just the summary
    assert len(completed_seeds) == 1
    seed = completed_seeds[0]
    assert "def broken(" in seed, f"Artifact missing from seed: {seed}"
    assert "ARTIFACT" in seed
    assert "VERIFIER FAILURE" in seed

    await reflection.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_reflection_seed_contains_actual_response_on_feedback(tmp_path: Path):
    """When negative feedback is received, the reflection seed should contain
    the actual assistant response text, not just the response_id."""
    from capsule_brain.llm.gateway import LLMGateway
    from capsule_brain.llm.models import LLMResult
    from capsule_brain.llm.providers.base import LLMProvider
    from capsule_brain.memory.service import MemoryService
    from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
    from capsule_brain.conversation.repository import ConversationRepository
    from capsule_brain.conversation.models import Conversation, Turn, TurnRole, AssistantResponse
    from capsule_brain.reflection.service import ReflectionService
    from capsule_brain.reflection.repository import ReflectionRepository
    from capsule_brain.learning.experience_store import ExperienceStore
    from capsule_brain.learning.models import ExperienceRecord

    class FakeProvider(LLMProvider):
        name = "fake"
        async def generate(self, request, model_cfg):
            return LLMResult(text="RESOLVED", model="fake", provider="fake", latency_ms=1)

    gateway = LLMGateway(
        {"enable": True, "default_model": "fake",
         "models": {"fake": {"provider": "fake", "model_name": "fake", "capabilities": ["text"]}}},
        providers={"fake": FakeProvider()},
    )

    bus = LocalEventBus()
    await bus.start()
    mem_repo = SQLiteMemoryRepository(str(tmp_path / "mem.sqlite"))
    memory = MemoryService(bus, repository=mem_repo)
    await memory.start()
    await gateway.start()

    conv_repo = ConversationRepository(str(tmp_path / "conv.sqlite"))
    await conv_repo.start()

    exp_store = ExperienceStore(db_path=str(tmp_path / "exp.sqlite"))
    await exp_store.start()

    # Create a conversation with a user turn and assistant response
    conv = Conversation(id="conv-1")
    await conv_repo.create_conversation(conv)
    user_turn = Turn(id="t1", conversation_id="conv-1", role=TurnRole.USER, text="What can you do?")
    await conv_repo.create_turn(user_turn)
    asst_turn = Turn(id="t2", conversation_id="conv-1", role=TurnRole.ASSISTANT, text="I can help.")
    await conv_repo.create_turn(asst_turn)
    response = AssistantResponse(
        id="resp-1",
        conversation_id="conv-1",
        parent_turn_id="t1",
        turn_id="t2",
        text="Hello from Capsule Brain.",
    )
    await conv_repo.create_response(response)

    # Also create an experience record
    await exp_store.create_experience(ExperienceRecord(
        id="exp-1",
        response_id="resp-1",
        conversation_id="conv-1",
        user_turn_id="t1",
        assistant_turn_id="t2",
        prompt_text="What can you do?",
        response_text="Hello from Capsule Brain.",
        model_alias="fake",
        provider="fake",
        model_name="fake",
        latency_ms=10.0,
    ))

    reflection = ReflectionService(
        event_bus=bus,
        llm=gateway,
        memory=memory,
        cfg={"db_path": str(tmp_path / "ref.sqlite"), "max_iterations": 1},
        repository=ReflectionRepository(str(tmp_path / "ref.sqlite")),
        experience_store=exp_store,
        conversation_repository=conv_repo,
    )
    await reflection.start()

    # Emit negative feedback
    completed_seeds = []
    async def capture_seed(event):
        completed_seeds.append(event.payload.get("seed", ""))
    bus.subscribe("reflection.completed", capture_seed)

    await bus.publish(EventEnvelope(
        event_type="feedback.recorded",
        source="feedback",
        payload={
            "response_id": "resp-1",
            "classification": "negative",
            "text": "Too vague.",
        },
    ))
    await asyncio.sleep(0.1)

    assert len(completed_seeds) == 1
    seed = completed_seeds[0]
    # The seed should contain the actual response text, not just the ID
    assert "Hello from Capsule Brain" in seed, f"Response text missing from seed: {seed}"
    assert "Too vague" in seed
    assert "response_id=resp-1" not in seed  # Should NOT contain raw ID

    await reflection.stop()
    await exp_store.stop()
    await conv_repo.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# P0: Digest pinning fails closed
# ---------------------------------------------------------------------------

def test_digest_pinning_raises_on_resolution_failure():
    """_resolve_digest must raise when pinning is enabled and resolution fails."""
    from capsule_brain.execution.container_runner import (
        _resolve_digest,
        ContainerImageResolutionError,
    )

    with pytest.raises(ContainerImageResolutionError, match="Failed to resolve digest"):
        _resolve_digest("nonexistent_engine_xyz", "python:3.11-slim")


def test_digest_pinning_passes_already_pinned():
    """_resolve_digest should pass through already-pinned images."""
    from capsule_brain.execution.container_runner import _resolve_digest

    pinned = "python@sha256:abcdef123456"
    result = _resolve_digest("docker", pinned)
    assert result == pinned


# ---------------------------------------------------------------------------
# P1: Container-first execution default
# ---------------------------------------------------------------------------

def test_execution_default_runner_is_container():
    """The shipped YAML should default to container runner, not host."""
    cfg = load_config("configs/v2_runtime.yaml")
    assert cfg["execution"]["runner"] == "container"


# ---------------------------------------------------------------------------
# P1: Background event dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_dispatch_does_not_block_publish():
    """When async_dispatch is enabled, publish() should return before
    long-running handlers complete."""
    bus = LocalEventBus({"async_dispatch": True})
    await bus.start()

    from capsule_brain.runtime.tasks import TaskRegistry
    registry = TaskRegistry()
    bus.set_task_registry(registry)

    handler_completed = asyncio.Event()

    async def slow_handler(event):
        await asyncio.sleep(0.1)
        handler_completed.set()

    bus.subscribe("test.event", slow_handler)

    # publish() should return immediately even though the handler takes 0.1s
    import time
    start = time.perf_counter()
    await bus.publish(EventEnvelope(
        event_type="test.event",
        payload={"value": 1},
        source="test",
    ))
    elapsed = time.perf_counter() - start
    assert elapsed < 0.05, f"publish() took {elapsed:.3f}s, should be near-instant"

    # The handler should complete eventually
    await asyncio.wait_for(handler_completed.wait(), timeout=1.0)

    await registry.shutdown()
    await bus.stop()
