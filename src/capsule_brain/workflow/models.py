from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowStatus(str, Enum):
    """Lifecycle state of a single workflow run.

    Transitions:
        PENDING -> RUNNING -> (COMPLETED | FAILED | PAUSED)
        PAUSED  -> RUNNING  (on resume)
        FAILED  -> RUNNING  (on retry from the failed node)
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # Awaiting human approval at a checkpoint node
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class WorkflowStepRecord:
    """A persisted record of one node execution within a workflow run."""

    run_id: str
    node_name: str
    index: int
    started_at: str
    completed_at: str | None = None
    status: str = "running"  # running | passed | failed | skipped
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class WorkflowState:
    """Mutable state passed from node to node during a workflow run.

    This is the DAG's shared blackboard: each node reads the fields it needs
    and writes the fields it produces. The runner persists a snapshot of this
    state after every node transition so an interrupted run can resume.

    The base fields cover the common plan->generate->test->reflect cycle.
    Specialized workflows can extend this via the ``extra`` dict rather than
    subclassing, keeping serialization simple.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex)
    goal: str = ""
    plan: str = ""
    code: str = ""
    stdout: str = ""
    stderr: str = ""
    errors: list[str] = field(default_factory=list)
    iterations: int = 0
    approved: bool = False
    # The name of the node that should execute next. Set by the runner when
    # persisting state so a resumed run knows where to pick up. None means
    # "start from the entry node".
    current_node: str | None = None
    # Free-form per-workflow data.
    extra: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for persistence."""
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "plan": self.plan,
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "errors": list(self.errors),
            "iterations": self.iterations,
            "approved": self.approved,
            "current_node": self.current_node,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> "WorkflowState":
        return cls(
            run_id=snap.get("run_id", uuid4().hex),
            goal=snap.get("goal", ""),
            plan=snap.get("plan", ""),
            code=snap.get("code", ""),
            stdout=snap.get("stdout", ""),
            stderr=snap.get("stderr", ""),
            errors=list(snap.get("errors") or []),
            iterations=int(snap.get("iterations", 0)),
            approved=bool(snap.get("approved", False)),
            current_node=snap.get("current_node"),
            extra=dict(snap.get("extra") or {}),
        )


@dataclass(slots=True)
class WorkflowRun:
    """A persisted workflow run: status + state snapshot + step history."""

    id: str
    workflow_name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    completed_at: str | None = None
    state: WorkflowState = field(default_factory=WorkflowState)
    steps: list[WorkflowStepRecord] = field(default_factory=list)
    # The node that should run next when the run is resumed. Mirrors
    # state.current_node but is stored at the run level for queries that
    # don't need to deserialize the full state.
    current_node: str | None = None
    # Human-readable reason for the last status transition (e.g. "approval
    # requested at node:approve", "max_iterations exceeded").
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
