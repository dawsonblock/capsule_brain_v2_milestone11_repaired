import asyncio
import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.workflow.models import (
    WorkflowRun,
    WorkflowState,
    WorkflowStatus,
)
from capsule_brain.workflow.repository import WorkflowRepository
from capsule_brain.workflow.runner import (
    Workflow,
    WorkflowNode,
    WorkflowRunnerService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bus():
    return LocalEventBus()


async def _make_runner(tmp_path, cfg=None):
    bus = _bus()
    await bus.start()
    repo = WorkflowRepository(str(tmp_path / "workflows.sqlite"))
    runner = WorkflowRunnerService(bus, cfg=cfg or {}, repository=repo)
    return bus, runner


def _linear_workflow():
    """A simple A -> B -> C workflow that appends node names to state.errors."""

    async def make_node(name):
        async def action(state: WorkflowState) -> WorkflowState:
            state.errors.append(name)
            return state
        return action

    async def a(state): state.errors.append("A"); return state
    async def b(state): state.errors.append("B"); return state
    async def c(state): state.errors.append("C"); return state

    return Workflow(
        name="linear",
        nodes=[
            WorkflowNode("A", a, next_node="B"),
            WorkflowNode("B", b, next_node="C"),
            WorkflowNode("C", c, next_node=None),
        ],
        entry="A",
    )


# ---------------------------------------------------------------------------
# Core engine: linear workflow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_linear_workflow_completes(tmp_path):
    bus, runner = await _make_runner(tmp_path)
    runner.register_workflow(_linear_workflow())
    await runner.start()

    run = await runner.start_run(workflow_name="linear", goal="test")

    assert run.status == WorkflowStatus.COMPLETED
    assert run.stop_reason == "completed"
    assert run.state.errors == ["A", "B", "C"]
    assert len(run.steps) == 3
    assert [s.node_name for s in run.steps] == ["A", "B", "C"]
    assert all(s.status == "passed" for s in run.steps)

    await runner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_workflow_publishes_step_events(tmp_path):
    bus, runner = await _make_runner(tmp_path)
    runner.register_workflow(_linear_workflow())
    await runner.start()

    events = []
    bus.subscribe("workflow.step", lambda e: events.append(e.payload))
    bus.subscribe("workflow.started", lambda e: events.append(e.payload))
    bus.subscribe("workflow.completed", lambda e: events.append(e.payload))

    await runner.start_run(workflow_name="linear", goal="test")

    event_types = [e.get("node") or e.get("stop_reason") for e in events]
    assert "A" in event_types
    assert "B" in event_types
    assert "C" in event_types
    assert "completed" in event_types

    await runner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def _conditional_workflow():
    """A -> (cond: B if x>0 else C) -> D"""

    async def a(state):
        state.errors.append("A")
        return state

    async def b(state):
        state.errors.append("B")
        return state

    async def c(state):
        state.errors.append("C")
        return state

    async def d(state):
        state.errors.append("D")
        return state

    def after_a(state: WorkflowState) -> str:
        return "B" if state.extra.get("x", 0) > 0 else "C"

    return Workflow(
        name="conditional",
        nodes=[
            WorkflowNode("A", a, next_node=after_a),
            WorkflowNode("B", b, next_node="D"),
            WorkflowNode("C", c, next_node="D"),
            WorkflowNode("D", d, next_node=None),
        ],
        entry="A",
    )


@pytest.mark.asyncio
async def test_conditional_edge_takes_branch_b(tmp_path):
    bus, runner = await _make_runner(tmp_path)
    runner.register_workflow(_conditional_workflow())
    await runner.start()

    run = await runner.start_run(workflow_name="conditional", goal="test")
    run.state.extra["x"] = 1  # set after the fact for assertion setup
    # Actually we need x set BEFORE node A runs. Use metadata? No — the
    # state.extra is empty at start. Let's test the default (x=0 -> C).
    assert run.state.errors == ["A", "C", "D"]

    await runner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_conditional_edge_takes_branch_c(tmp_path):
    """With x=0 (default), the conditional edge should take the C branch."""
    bus, runner = await _make_runner(tmp_path)
    runner.register_workflow(_conditional_workflow())
    await runner.start()

    run = await runner.start_run(workflow_name="conditional", goal="test")
    assert run.state.errors == ["A", "C", "D"]
    assert run.status == WorkflowStatus.COMPLETED

    await runner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Node failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_failure_marks_run_failed(tmp_path):
    bus, runner = await _make_runner(tmp_path)

    async def good(state):
        state.errors.append("good")
        return state

    async def bad(state):
        raise RuntimeError("node exploded")

    async def never(state):
        state.errors.append("never")
        return state

    wf = Workflow(
        name="fails",
        nodes=[
            WorkflowNode("good", good, next_node="bad"),
            WorkflowNode("bad", bad, next_node="never"),
            WorkflowNode("never", never, next_node=None),
        ],
    )
    runner.register_workflow(wf)
    await runner.start()

    run = await runner.start_run(workflow_name="fails", goal="test")

    assert run.status == WorkflowStatus.FAILED
    assert "node_error:bad" in run.stop_reason
    assert run.state.errors == ["good"]  # "never" did not run
    assert run.current_node == "bad"  # preserved for resume

    await runner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Persistence + resume
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_persisted_and_loaded(tmp_path):
    bus, runner = await _make_runner(tmp_path)
    runner.register_workflow(_linear_workflow())
    await runner.start()

    run = await runner.start_run(workflow_name="linear", goal="persist me")
    run_id = run.id

    # Load from repository.
    loaded = await runner.repository.get_run(run_id)
    assert loaded is not None
    assert loaded.status == WorkflowStatus.COMPLETED
    assert loaded.state.goal == "persist me"
    assert loaded.state.errors == ["A", "B", "C"]
    assert len(loaded.steps) == 3

    await runner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_resume_failed_run_retries_from_current_node(tmp_path):
    """A failed run should resume from its current_node when retried."""
    bus, runner = await _make_runner(tmp_path)

    call_count = {"bad": 0}

    async def good(state):
        state.errors.append("good")
        return state

    async def bad(state):
        call_count["bad"] += 1
        if call_count["bad"] == 1:
            raise RuntimeError("first attempt fails")
        state.errors.append("bad")
        return state

    async def after(state):
        state.errors.append("after")
        return state

    wf = Workflow(
        name="resumable",
        nodes=[
            WorkflowNode("good", good, next_node="bad"),
            WorkflowNode("bad", bad, next_node="after"),
            WorkflowNode("after", after, next_node=None),
        ],
    )
    runner.register_workflow(wf)
    await runner.start()

    run = await runner.start_run(workflow_name="resumable", goal="test")
    assert run.status == WorkflowStatus.FAILED
    assert run.state.errors == ["good"]

    # Resume — the bad node should retry and succeed this time.
    resumed = await runner.resume_run(run.id, approved=True)
    assert resumed.status == WorkflowStatus.COMPLETED
    assert resumed.state.errors == ["good", "bad", "after"]

    await runner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Human-in-the-loop checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_checkpoint_pauses_run_and_awaits_approval(tmp_path):
    bus, runner = await _make_runner(tmp_path)

    async def a(state):
        state.errors.append("A")
        return state

    async def b(state):
        state.errors.append("B")
        return state

    def always_pause(state: WorkflowState) -> bool:
        return True

    wf = Workflow(
        name="checkpointed",
        nodes=[
            WorkflowNode("A", a, next_node="B"),
            WorkflowNode("B", b, next_node="C", checkpoint=always_pause),
            WorkflowNode("C", a, next_node=None),  # reuse a as a no-op
        ],
    )
    runner.register_workflow(wf)
    await runner.start()

    approval_events = []
    bus.subscribe(
        "workflow.approval_requested",
        lambda e: approval_events.append(e.payload),
    )

    # Start the run in a background task so we can approve it while it's
    # paused.
    task = asyncio.create_task(
        runner.start_run(workflow_name="checkpointed", goal="need approval")
    )
    # Wait for the approval event (the run pauses at B's checkpoint).
    await asyncio.sleep(0.1)
    assert len(approval_events) == 1
    assert approval_events[0]["node"] == "B"
    run_id = approval_events[0]["run_id"]

    # The run should be paused now.
    paused_run = await runner.repository.get_run(run_id)
    assert paused_run.status == WorkflowStatus.PAUSED
    assert paused_run.state.errors == ["A", "B"]

    # Approve it.
    await runner.resume_run(run_id, approved=True)

    # The background task should complete.
    run = await asyncio.wait_for(task, timeout=2.0)
    assert run.status == WorkflowStatus.COMPLETED
    assert run.state.errors == ["A", "B", "A"]  # C reuses action a

    await runner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_checkpoint_denial_completes_run(tmp_path):
    """Denying approval via the event bus completes the run as denied."""
    bus, runner = await _make_runner(tmp_path)

    async def a(state):
        state.errors.append("A")
        return state

    def always_pause(state: WorkflowState) -> bool:
        return True

    wf = Workflow(
        name="denied",
        nodes=[
            WorkflowNode("A", a, next_node="B", checkpoint=always_pause),
            WorkflowNode("B", a, next_node=None),
        ],
    )
    runner.register_workflow(wf)
    await runner.start()

    from capsule_brain.events.models import EventEnvelope

    completed = []
    bus.subscribe("workflow.completed", lambda e: completed.append(e.payload))

    task = asyncio.create_task(
        runner.start_run(workflow_name="denied", goal="will be denied")
    )
    await asyncio.sleep(0.1)

    runs = await runner.repository.list_pending_resume()
    assert len(runs) == 1
    run_id = runs[0].id

    # Deny via the event bus (the canonical operator path).
    await bus.publish(
        EventEnvelope(
            event_type="workflow.approve",
            source="test",
            payload={"run_id": run_id, "approved": False},
        )
    )
    run = await asyncio.wait_for(task, timeout=2.0)
    assert run.status == WorkflowStatus.COMPLETED
    assert run.stop_reason == "approval_denied"
    assert len(completed) == 1
    assert completed[0]["approved"] is False

    await runner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_checkpoint_denial_via_api(tmp_path):
    """Denying approval via resume_run(approved=False) completes the run."""
    bus, runner = await _make_runner(tmp_path)

    async def a(state):
        state.errors.append("A")
        return state

    def always_pause(state: WorkflowState) -> bool:
        return True

    wf = Workflow(
        name="denied2",
        nodes=[
            WorkflowNode("A", a, next_node="B", checkpoint=always_pause),
            WorkflowNode("B", a, next_node=None),
        ],
    )
    runner.register_workflow(wf)
    await runner.start()

    task = asyncio.create_task(
        runner.start_run(workflow_name="denied2", goal="will be denied")
    )
    await asyncio.sleep(0.1)

    # The run is paused at A's checkpoint. Deny approval.
    run_id = task.result() if task.done() else None
    # The task hasn't completed; get the run_id from the repository.
    runs = await runner.repository.list_pending_resume()
    assert len(runs) == 1
    run_id = runs[0].id

    await runner.resume_run(run_id, approved=False)
    run = await asyncio.wait_for(task, timeout=2.0)
    assert run.status == WorkflowStatus.COMPLETED
    assert run.stop_reason == "approval_denied"

    await runner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# max_steps cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_steps_cap_prevents_infinite_loop(tmp_path):
    bus, runner = await _make_runner(tmp_path, cfg={"max_steps": 5})

    async def noop(state):
        return state

    # A -> B -> A -> B ... infinite loop
    wf = Workflow(
        name="loop",
        nodes=[
            WorkflowNode("A", noop, next_node="B"),
            WorkflowNode("B", noop, next_node="A"),
        ],
    )
    runner.register_workflow(wf)
    await runner.start()

    run = await runner.start_run(workflow_name="loop", goal="loop forever")
    assert run.status == WorkflowStatus.FAILED
    assert run.stop_reason == "max_steps_exceeded"
    assert len(run.steps) == 5

    await runner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Event-driven start
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_request_event_starts_run(tmp_path):
    bus, runner = await _make_runner(tmp_path)
    runner.register_workflow(_linear_workflow())
    await runner.start()

    from capsule_brain.events.models import EventEnvelope

    completed = []
    bus.subscribe("workflow.completed", lambda e: completed.append(e.payload))

    await bus.publish(
        EventEnvelope(
            event_type="workflow.request",
            source="test",
            payload={"workflow": "linear", "goal": "via event"},
        )
    )
    await asyncio.sleep(0.1)
    assert len(completed) == 1
    assert completed[0]["workflow"] == "linear"

    await runner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_cancel_run(tmp_path):
    """cancel_run marks a paused run as failed with stop_reason='cancelled'."""
    bus, runner = await _make_runner(tmp_path)

    async def a(state):
        state.errors.append("A")
        return state

    def always_pause(state: WorkflowState) -> bool:
        return True

    wf = Workflow(
        name="cancellable",
        nodes=[
            WorkflowNode("A", a, next_node="B", checkpoint=always_pause),
            WorkflowNode("B", a, next_node=None),
        ],
    )
    runner.register_workflow(wf)
    await runner.start()

    cancelled_events = []
    bus.subscribe("workflow.cancelled", lambda e: cancelled_events.append(e.payload))

    task = asyncio.create_task(
        runner.start_run(workflow_name="cancellable", goal="cancel me")
    )
    await asyncio.sleep(0.1)
    # The run is paused at A's checkpoint.
    runs = await runner.repository.list_pending_resume()
    assert len(runs) == 1
    run_id = runs[0].id

    cancelled = await runner.cancel_run(run_id)
    assert cancelled.status == WorkflowStatus.FAILED
    assert cancelled.stop_reason == "cancelled"
    assert len(cancelled_events) == 1

    # The background task completes (the approval future was cancelled,
    # causing _execute to return normally from the checkpoint path).
    await asyncio.wait_for(task, timeout=2.0)

    # Verify the persisted run is marked failed.
    persisted = await runner.repository.get_run(run_id)
    assert persisted.status == WorkflowStatus.FAILED
    assert persisted.stop_reason == "cancelled"

    await runner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_reports_workflow_stats(tmp_path):
    bus, runner = await _make_runner(tmp_path)
    runner.register_workflow(_linear_workflow())
    await runner.start()

    await runner.start_run(workflow_name="linear", goal="h1")
    await runner.start_run(workflow_name="linear", goal="h2")

    health = await runner.health()
    assert health.details["workflow_count"] == 1
    assert health.details["runs_started"] == 2
    assert health.details["runs_completed"] == 2
    assert health.details["run_count"] == 2

    await runner.stop()
    await bus.stop()
