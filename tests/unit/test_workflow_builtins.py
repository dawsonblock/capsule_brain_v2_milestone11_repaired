import sys
import pytest
from pathlib import Path

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.execution.models import ExecutionPolicy
from capsule_brain.execution.runner import ExecutionRunner
from capsule_brain.execution.service import ExecutionService
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
from capsule_brain.workflow.builtins import (
    build_plan_generate_test_reflect_workflow,
)
from capsule_brain.workflow.models import WorkflowStatus
from capsule_brain.workflow.repository import WorkflowRepository
from capsule_brain.workflow.runner import WorkflowRunnerService


# ---------------------------------------------------------------------------
# Offline mode (no LLM, no execution)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_workflow_offline_mode_completes(tmp_path):
    """With no LLM and no execution service, the workflow still completes."""
    bus = LocalEventBus()
    await bus.start()
    repo = WorkflowRepository(str(tmp_path / "wf.sqlite"))
    runner = WorkflowRunnerService(bus, repository=repo)
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(services=None)
    )
    await runner.start()

    run = await runner.start_run(
        workflow_name="plan_generate_test_reflect",
        goal="write a hello world script",
    )

    assert run.status == WorkflowStatus.COMPLETED
    # In offline mode, plan/generate produce stubs and test is skipped.
    assert "[offline]" in run.state.plan
    assert "offline mode" in run.state.code
    assert "execution unavailable" in run.state.stderr
    # No reflect loop because test is treated as passed when execution is
    # unavailable.
    assert run.state.iterations == 0

    await runner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# With execution service (test passes)
# ---------------------------------------------------------------------------

class StubLLMProvider(LLMProvider):
    """Returns canned code for plan and generate nodes."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        # The first call is the plan node; subsequent calls are generate.
        if self.calls == 1:
            return LLMResult(
                text="1. Write a print statement",
                model="fake",
                provider="fake",
                latency_ms=1,
            )
        return LLMResult(
            text="print('hello from workflow')",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_default_workflow_with_execution_passes(tmp_path):
    """The full pipeline: LLM generates code, execution runs it, test passes."""
    bus = LocalEventBus()
    await bus.start()

    # Set up execution service with a real runner.
    root = tmp_path / "sandbox"
    root.mkdir()
    execution = ExecutionService(
        bus,
        cfg={
            "allow": True,
            "cwd_root": str(root),
            "allowed_commands": [Path(sys.executable).name],
            "timeout_s": 5,
        },
        runner=ExecutionRunner(
            ExecutionPolicy(
                allow=True,
                cwd_root=str(root),
                allowed_commands=(Path(sys.executable).name,),
                timeout_s=5,
            )
        ),
    )
    await execution.start()

    # Set up LLM gateway with a stub provider.
    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {
                "routes": {
                    "coding": {"models": ["fake"]},
                    "reflection": {"models": ["fake"]},
                }
            },
        },
        providers={"fake": StubLLMProvider()},
    )
    await gateway.start()

    services = {
        "llm_gateway": gateway,
        "execution": execution,
    }
    repo = WorkflowRepository(str(tmp_path / "wf.sqlite"))
    runner = WorkflowRunnerService(bus, repository=repo)
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(services=services)
    )
    await runner.start()

    run = await runner.start_run(
        workflow_name="plan_generate_test_reflect",
        goal="write a hello world script",
    )

    assert run.status == WorkflowStatus.COMPLETED
    assert run.state.plan == "1. Write a print statement"
    assert "print('hello from workflow')" in run.state.code
    assert "hello from workflow" in run.state.stdout
    assert run.state.extra.get("test_passed") is True
    assert run.state.iterations == 0  # no reflect loop needed

    await runner.stop()
    await gateway.stop()
    await execution.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# With execution service (test fails -> reflect loop)
# ---------------------------------------------------------------------------

class FailingCodeProvider(LLMProvider):
    """Returns broken code on the first generate, fixed code after reflect."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        if self.calls == 1:
            # Plan
            return LLMResult(
                text="1. Write a print statement",
                model="fake",
                provider="fake",
                latency_ms=1,
            )
        if self.calls == 2:
            # Generate: broken code (syntax error)
            return LLMResult(
                text="print('hello'",  # missing closing paren
                model="fake",
                provider="fake",
                latency_ms=1,
            )
        # Reflect (call 3): fixed code
        return LLMResult(
            text="print('hello')",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_default_workflow_reflect_loop_fixes_failure(tmp_path):
    """Test fails -> reflect -> test passes on the second iteration."""
    bus = LocalEventBus()
    await bus.start()

    root = tmp_path / "sandbox"
    root.mkdir()
    execution = ExecutionService(
        bus,
        cfg={
            "allow": True,
            "cwd_root": str(root),
            "allowed_commands": [Path(sys.executable).name],
            "timeout_s": 5,
        },
        runner=ExecutionRunner(
            ExecutionPolicy(
                allow=True,
                cwd_root=str(root),
                allowed_commands=(Path(sys.executable).name,),
                timeout_s=5,
            )
        ),
    )
    await execution.start()

    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {
                "routes": {
                    "coding": {"models": ["fake"]},
                    "reflection": {"models": ["fake"]},
                }
            },
        },
        providers={"fake": FailingCodeProvider()},
    )
    await gateway.start()

    services = {"llm_gateway": gateway, "execution": execution}
    repo = WorkflowRepository(str(tmp_path / "wf.sqlite"))
    runner = WorkflowRunnerService(bus, repository=repo)
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services=services, max_iterations=3
        )
    )
    await runner.start()

    run = await runner.start_run(
        workflow_name="plan_generate_test_reflect",
        goal="write a hello world script",
    )

    assert run.status == WorkflowStatus.COMPLETED
    # The reflect loop should have run once.
    assert run.state.iterations == 1
    # The final code should be the fixed version.
    assert "print('hello')" in run.state.code
    # The test should have passed on the second attempt.
    assert run.state.extra.get("test_passed") is True
    assert "hello" in run.state.stdout
    # Steps: plan, generate, test (fail), reflect, test (pass), complete
    node_sequence = [s.node_name for s in run.steps]
    assert node_sequence == ["plan", "generate", "test", "reflect", "test", "complete"]

    await runner.stop()
    await gateway.stop()
    await execution.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Human-in-the-loop approval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_workflow_with_approval_checkpoint(tmp_path):
    """require_approval=True inserts a checkpoint after test passes."""
    bus = LocalEventBus()
    await bus.start()

    root = tmp_path / "sandbox"
    root.mkdir()
    execution = ExecutionService(
        bus,
        cfg={
            "allow": True,
            "cwd_root": str(root),
            "allowed_commands": [Path(sys.executable).name],
            "timeout_s": 5,
        },
        runner=ExecutionRunner(
            ExecutionPolicy(
                allow=True,
                cwd_root=str(root),
                allowed_commands=(Path(sys.executable).name,),
                timeout_s=5,
            )
        ),
    )
    await execution.start()

    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {
                "routes": {
                    "coding": {"models": ["fake"]},
                    "reflection": {"models": ["fake"]},
                }
            },
        },
        providers={"fake": StubLLMProvider()},
    )
    await gateway.start()

    services = {"llm_gateway": gateway, "execution": execution}
    repo = WorkflowRepository(str(tmp_path / "wf.sqlite"))
    runner = WorkflowRunnerService(bus, repository=repo)
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services=services, require_approval=True
        )
    )
    await runner.start()

    approval_events = []
    bus.subscribe(
        "workflow.approval_requested",
        lambda e: approval_events.append(e.payload),
    )

    import asyncio
    task = asyncio.create_task(
        runner.start_run(
            workflow_name="plan_generate_test_reflect",
            goal="write a hello world script",
        )
    )
    await asyncio.sleep(0.2)

    # The run should be paused at the approve node.
    assert len(approval_events) == 1
    assert approval_events[0]["node"] == "approve"
    run_id = approval_events[0]["run_id"]

    paused = await runner.repository.get_run(run_id)
    assert paused.status == WorkflowStatus.PAUSED

    # Approve and let it complete.
    await runner.resume_run(run_id, approved=True)
    run = await asyncio.wait_for(task, timeout=5.0)
    assert run.status == WorkflowStatus.COMPLETED

    # Steps: plan, generate, test, approve, complete
    node_sequence = [s.node_name for s in run.steps]
    assert node_sequence == ["plan", "generate", "test", "approve", "complete"]

    await runner.stop()
    await gateway.stop()
    await execution.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Bootstrap integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bootstrap_registers_workflow_service(tmp_path):
    """build_application with workflow.enable=true registers the runner."""
    from capsule_brain.runtime.bootstrap import build_application

    cfg = {
        "workflow": {"enable": True, "db_path": str(tmp_path / "wf.sqlite")},
    }
    app = build_application(cfg)
    assert "workflow_runner" in app.services.names()
    runner = app.services.get("workflow_runner")
    assert runner is not None
    assert runner.get_workflow("plan_generate_test_reflect") is not None
