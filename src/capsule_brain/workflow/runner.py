from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.observability.tracing import get_default_tracer
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .models import (
    WorkflowRun,
    WorkflowState,
    WorkflowStatus,
    WorkflowStepRecord,
    utc_now_iso,
)
from .repository import WorkflowRepository

log = logging.getLogger(__name__)

# A node action receives the current state and returns the updated state.
# Actions may mutate the state in place and/or return a new state; the runner
# uses the returned state as the authoritative next state.
NodeAction = Callable[[WorkflowState], Awaitable[WorkflowState | None]]
# A conditional edge function inspects the state and returns the name of the
# next node to execute, or None to stop the workflow.
NextNodeResolver = Callable[[WorkflowState], str | None]
# A checkpoint guard returns True if the workflow should pause at this node
# to await human approval before proceeding. The runner publishes an
# approval-requested event and parks the run.
CheckpointGuard = Callable[[WorkflowState], bool]


class WorkflowNode:
    """A single node in a workflow DAG.

    Each node has:
    - ``name``: unique within the workflow.
    - ``action``: async callable that performs the node's work and may
      update the state.
    - ``next_node``: either a string (unconditional edge to the next node)
      or a callable (conditional edge that inspects state and picks the
      next node). None means this is a terminal node.
    - ``checkpoint``: optional guard. When it returns True, the runner
      pauses the workflow before executing the next node and emits a
      ``workflow.approval_requested`` event. The run resumes when
      ``resume_run`` is called with approval.
    """

    __slots__ = ("name", "action", "next_node", "checkpoint", "description")

    def __init__(
        self,
        name: str,
        action: NodeAction,
        next_node: str | NextNodeResolver | None = None,
        *,
        checkpoint: CheckpointGuard | None = None,
        description: str = "",
    ) -> None:
        if not name:
            raise ValueError("WorkflowNode.name must be non-empty")
        self.name = name
        self.action = action
        self.next_node = next_node
        self.checkpoint = checkpoint
        self.description = description


class Workflow:
    """A registered workflow: a named collection of nodes plus an entry node.

    The entry node is the first node executed when a run starts. Nodes are
    stored in a dict keyed by name so the runner can jump to any node
    (including for resume/retry).
    """

    __slots__ = ("name", "entry", "nodes", "description")

    def __init__(
        self,
        name: str,
        nodes: list[WorkflowNode],
        *,
        entry: str | None = None,
        description: str = "",
    ) -> None:
        if not name:
            raise ValueError("Workflow.name must be non-empty")
        if not nodes:
            raise ValueError("Workflow must have at least one node")
        self.name = name
        self.nodes: dict[str, WorkflowNode] = {n.name: n for n in nodes}
        self.entry = entry or nodes[0].name
        if self.entry not in self.nodes:
            raise ValueError(
                f"Workflow entry node {self.entry!r} not in nodes"
            )
        self.description = description

    def to_mermaid(self) -> str:
        """Render the workflow graph as a Mermaid flowchart.

        Useful for GUI visualization and debugging. Conditional edges are
        rendered with a diamond decision node.
        """
        lines = [f"flowchart {self.name}"]
        for node in self.nodes.values():
            if node.next_node is None:
                continue
            if isinstance(node.next_node, str):
                lines.append(f"  {node.name} --> {node.next_node}")
            else:
                # Conditional edge: render as a decision diamond.
                lines.append(f"  {node.name} --> {{cond}}")
                lines.append(f"  {{cond}} --> next_node(state)")
        return "\n".join(lines)


class WorkflowRunnerService(CapsuleService):
    """A managed DAG engine that runs declarative workflows on top of the
    event bus.

    The runner is the bridge between the event-driven kernel (LocalEventBus,
    MemoryService, LLMGateway, ExecutionService) and explicit multi-step
    cognitive tasks. Each node in a workflow invokes existing services or
    publishes events; the runner handles state transitions, persistence,
    human-in-the-loop checkpoints, and resume-after-restart.

    Key guarantees:
    - **Persistence**: after every node transition, the run state is flushed
      to SQLite. If the process dies, ``resume_pending_runs`` picks up at
      the exact node that was about to execute.
    - **Checkpoints**: a node with a ``checkpoint`` guard pauses the run
      before the next node, emits ``workflow.approval_requested``, and waits
      for ``resume_run(run_id, approved=True)``.
    - **Observability**: every node transition publishes a
      ``workflow.step`` event and creates a tracing span keyed by the run's
      correlation id.
    - **Bounded iteration**: a max_steps cap prevents infinite loops from a
      misconfigured conditional edge.
    """

    name = "workflow_runner"

    def __init__(
        self,
        event_bus: LocalEventBus,
        cfg: dict[str, Any] | None = None,
        repository: WorkflowRepository | None = None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.repository = repository or WorkflowRepository(
            self.cfg.get("db_path", "data/workflows.sqlite")
        )
        self._workflows: dict[str, Workflow] = {}
        # In-flight runs awaiting approval: run_id -> asyncio.Future[bool].
        # The future is resolved by resume_run() with the operator's decision.
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._unsubscribers: list = []
        self._runs_started = 0
        self._runs_completed = 0
        self._runs_failed = 0
        self._runs_paused = 0
        self._max_steps = max(1, int(self.cfg.get("max_steps", 50)))
        self._approval_timeout_s = float(
            self.cfg.get("approval_timeout_s", 0.0)  # 0 = wait forever
        )

    def register_workflow(self, workflow: Workflow) -> None:
        if workflow.name in self._workflows:
            raise ValueError(f"Workflow already registered: {workflow.name}")
        self._workflows[workflow.name] = workflow

    def get_workflow(self, name: str) -> Workflow | None:
        return self._workflows.get(name)

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        await self.repository.start()
        self._unsubscribers.extend([
            self.event_bus.subscribe(
                "workflow.request", self._on_workflow_request
            ),
            self.event_bus.subscribe(
                "workflow.approve", self._on_workflow_approve
            ),
            self.event_bus.subscribe(
                "workflow.cancel", self._on_workflow_cancel
            ),
        ])
        self.state = ServiceState.RUNNING
        # Resume any runs that were interrupted by a previous shutdown.
        if self.cfg.get("auto_resume", True):
            await self.resume_pending_runs()

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        # Cancel any in-flight approval waits so the run parks cleanly.
        for fut in self._pending_approvals.values():
            if not fut.done():
                fut.cancel()
        self._pending_approvals.clear()
        await self.repository.stop()
        self.state = ServiceState.STOPPED

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_workflow_request(self, event: EventEnvelope) -> None:
        workflow_name = str(event.payload.get("workflow", "")).strip()
        goal = str(event.payload.get("goal", "")).strip()
        if not workflow_name or workflow_name not in self._workflows:
            return
        try:
            await self.start_run(
                workflow_name=workflow_name,
                goal=goal,
                correlation_id=event.correlation_id,
                metadata=dict(event.payload.get("metadata") or {}),
            )
        except Exception as exc:
            log.exception("workflow.request handler failed")
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="workflow.failed",
                    source=self.name,
                    correlation_id=event.correlation_id,
                    payload={
                        "workflow": workflow_name,
                        "goal": goal,
                        "error": str(exc),
                    },
                )
            )

    async def _on_workflow_approve(self, event: EventEnvelope) -> None:
        run_id = str(event.payload.get("run_id", "")).strip()
        approved = bool(event.payload.get("approved", False))
        if run_id:
            await self.resume_run(run_id, approved=approved)

    async def _on_workflow_cancel(self, event: EventEnvelope) -> None:
        run_id = str(event.payload.get("run_id", "")).strip()
        if run_id:
            await self.cancel_run(run_id)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    async def start_run(
        self,
        *,
        workflow_name: str,
        goal: str,
        correlation_id: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowRun:
        """Start a new workflow run and execute it to completion or pause."""
        workflow = self._workflows.get(workflow_name)
        if workflow is None:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        state = WorkflowState(goal=goal)
        run = WorkflowRun(
            id=state.run_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.RUNNING,
            state=state,
            current_node=workflow.entry,
            metadata=dict(metadata or {}),
        )
        self._runs_started += 1
        await self.repository.save_run(run)

        await self.event_bus.publish(
            EventEnvelope(
                event_type="workflow.started",
                source=self.name,
                correlation_id=correlation_id,
                payload={
                    "run_id": run.id,
                    "workflow": workflow_name,
                    "goal": goal,
                    "entry_node": workflow.entry,
                },
            )
        )

        await self._execute(run, workflow, correlation_id=correlation_id)
        return run

    async def resume_run(
        self,
        run_id: str,
        *,
        approved: bool = True,
    ) -> WorkflowRun | None:
        """Resume a paused or failed run.

        For a paused run, ``approved`` carries the operator's decision. If
        approved, the run continues from the node after the checkpoint. If
        not approved, the run is marked COMPLETED with stop_reason
        "approval_denied".

        For a failed run, execution retries from ``current_node``.
        """
        run = await self.repository.get_run(run_id)
        if run is None:
            return None
        workflow = self._workflows.get(run.workflow_name)
        if workflow is None:
            log.warning(
                "Cannot resume run %s: workflow %s no longer registered",
                run_id,
                run.workflow_name,
            )
            return run

        # Resolve any pending approval future so the (possibly still-blocked)
        # _execute coroutine can proceed. If the run was resumed from a fresh
        # process, there is no future and we re-enter _execute directly.
        fut = self._pending_approvals.pop(run_id, None)
        if fut is not None and not fut.done():
            fut.set_result(approved)
            # The original _execute coroutine will handle the rest.
            return run

        # Re-enter execution from the current node.
        if not approved:
            run.status = WorkflowStatus.COMPLETED
            run.stop_reason = "approval_denied"
            run.completed_at = utc_now_iso()
            await self.repository.save_run(run)
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="workflow.completed",
                    source=self.name,
                    payload={
                        "run_id": run.id,
                        "workflow": run.workflow_name,
                        "stop_reason": run.stop_reason,
                        "approved": False,
                    },
                )
            )
            return run

        run.status = WorkflowStatus.RUNNING
        run.stop_reason = None
        await self.repository.save_run(run)
        await self._execute(run, workflow)
        return run

    async def cancel_run(self, run_id: str) -> WorkflowRun | None:
        """Cancel a running or paused run."""
        run = await self.repository.get_run(run_id)
        if run is None:
            return None
        if run.status in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED}:
            return run
        run.status = WorkflowStatus.FAILED
        run.stop_reason = "cancelled"
        run.completed_at = utc_now_iso()
        await self.repository.save_run(run)
        fut = self._pending_approvals.pop(run_id, None)
        if fut is not None and not fut.done():
            fut.cancel()
        await self.event_bus.publish(
            EventEnvelope(
                event_type="workflow.cancelled",
                source=self.name,
                payload={"run_id": run.id, "workflow": run.workflow_name},
            )
        )
        return run

    async def resume_pending_runs(self) -> int:
        """Resume all runs interrupted by a previous process shutdown.

        Called automatically on start() unless ``auto_resume`` is false.
        Returns the number of runs resumed.
        """
        pending = await self.repository.list_pending_resume()
        for run in pending:
            workflow = self._workflows.get(run.workflow_name)
            if workflow is None:
                log.warning(
                    "Cannot resume run %s: workflow %s not registered",
                    run.id,
                    run.workflow_name,
                )
                continue
            log.info(
                "Resuming interrupted workflow run %s at node %s",
                run.id,
                run.current_node,
            )
            # Spawn resume as a background task so startup is not blocked.
            asyncio.create_task(self.resume_run(run.id, approved=True))
        return len(pending)

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    async def _execute(
        self,
        run: WorkflowRun,
        workflow: Workflow,
        *,
        correlation_id: Any = None,
    ) -> None:
        """Execute nodes until a terminal condition is reached.

        Terminal conditions:
        - A node's next_node resolves to None (natural completion).
        - A checkpoint guard pauses the run (status -> PAUSED).
        - max_steps is exceeded (status -> FAILED).
        - A node action raises (status -> FAILED, current_node preserved).
        """
        state = run.state
        node_name = run.current_node or workflow.entry
        step_index = len(run.steps)
        tracer = get_default_tracer()

        try:
            while node_name is not None and node_name in workflow.nodes:
                if step_index >= self._max_steps:
                    run.status = WorkflowStatus.FAILED
                    run.stop_reason = "max_steps_exceeded"
                    run.completed_at = utc_now_iso()
                    await self.repository.save_run(run)
                    await self.event_bus.publish(
                        EventEnvelope(
                            event_type="workflow.failed",
                            source=self.name,
                            correlation_id=correlation_id,
                            payload={
                                "run_id": run.id,
                                "reason": run.stop_reason,
                            },
                        )
                    )
                    return

                node = workflow.nodes[node_name]
                state.current_node = node_name
                run.current_node = node_name

                step = WorkflowStepRecord(
                    run_id=run.id,
                    node_name=node_name,
                    index=step_index,
                    started_at=utc_now_iso(),
                )
                run.steps.append(step)
                await self.repository.save_step(step)
                await self.repository.save_run(run)

                await self.event_bus.publish(
                    EventEnvelope(
                        event_type="workflow.step",
                        source=self.name,
                        correlation_id=correlation_id,
                        payload={
                            "run_id": run.id,
                            "workflow": workflow.name,
                            "node": node_name,
                            "step_index": step_index,
                        },
                    )
                )

                # Execute the node action inside a tracing span.
                try:
                    with tracer.span(
                        f"workflow.node:{node_name}",
                        correlation_id=correlation_id,
                        attributes={
                            "run_id": run.id,
                            "workflow": workflow.name,
                            "node": node_name,
                            "step_index": step_index,
                        },
                    ):
                        updated = await node.action(state)
                    if updated is not None:
                        state = updated
                        run.state = state
                    step.status = "passed"
                    step.completed_at = utc_now_iso()
                    step.output = {"current_node": node_name}
                except Exception as exc:
                    step.status = "failed"
                    step.completed_at = utc_now_iso()
                    step.error = str(exc)
                    await self.repository.save_step(step)
                    run.status = WorkflowStatus.FAILED
                    run.stop_reason = f"node_error:{node_name}"
                    run.completed_at = utc_now_iso()
                    await self.repository.save_run(run)
                    await self.event_bus.publish(
                        EventEnvelope(
                            event_type="workflow.failed",
                            source=self.name,
                            correlation_id=correlation_id,
                            payload={
                                "run_id": run.id,
                                "node": node_name,
                                "error": str(exc),
                            },
                        )
                    )
                    self._runs_failed += 1
                    return

                await self.repository.save_step(step)
                await self.repository.save_run(run)

                # Check for a checkpoint guard BEFORE transitioning. If the
                # guard fires, pause the run and wait for approval.
                if node.checkpoint is not None and node.checkpoint(state):
                    run.status = WorkflowStatus.PAUSED
                    run.stop_reason = f"approval_requested:{node_name}"
                    run.current_node = self._resolve_next(node, state)
                    await self.repository.save_run(run)
                    self._runs_paused += 1
                    await self.event_bus.publish(
                        EventEnvelope(
                            event_type="workflow.approval_requested",
                            source=self.name,
                            correlation_id=correlation_id,
                            payload={
                                "run_id": run.id,
                                "workflow": workflow.name,
                                "node": node_name,
                                "next_node": run.current_node,
                                "state": state.snapshot(),
                            },
                        )
                    )
                    # Park until resume_run resolves the approval future.
                    loop = asyncio.get_event_loop()
                    approval_future: asyncio.Future[bool] = loop.create_future()
                    self._pending_approvals[run.id] = approval_future
                    try:
                        approved = await self._await_approval(
                            run.id, approval_future
                        )
                    except asyncio.CancelledError:
                        # cancel_run was called.
                        return
                    if not approved:
                        run.status = WorkflowStatus.COMPLETED
                        run.stop_reason = "approval_denied"
                        run.completed_at = utc_now_iso()
                        await self.repository.save_run(run)
                        await self.event_bus.publish(
                            EventEnvelope(
                                event_type="workflow.completed",
                                source=self.name,
                                correlation_id=correlation_id,
                                payload={
                                    "run_id": run.id,
                                    "approved": False,
                                    "stop_reason": run.stop_reason,
                                },
                            )
                        )
                        return
                    run.status = WorkflowStatus.RUNNING
                    run.stop_reason = None
                    await self.repository.save_run(run)
                    # Fall through to transition using run.current_node,
                    # which was set to the next node before pausing.

                # Resolve the next node.
                node_name = self._resolve_next(node, state)
                run.current_node = node_name
                step_index += 1

            # Natural completion: next_node resolved to None or a missing
            # node name (treated as terminal).
            run.status = WorkflowStatus.COMPLETED
            run.stop_reason = "completed"
            run.completed_at = utc_now_iso()
            await self.repository.save_run(run)
            self._runs_completed += 1
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="workflow.completed",
                    source=self.name,
                    correlation_id=correlation_id,
                    payload={
                        "run_id": run.id,
                        "workflow": workflow.name,
                        "stop_reason": run.stop_reason,
                        "iterations": state.iterations,
                    },
                )
            )
        except asyncio.CancelledError:
            # The runner was stopped mid-execution. The run is already
            # persisted at the current node; resume_pending_runs will pick
            # it up on next start.
            run.status = WorkflowStatus.FAILED
            run.stop_reason = "cancelled"
            await self.repository.save_run(run)
            raise

    async def _await_approval(
        self,
        run_id: str,
        future: asyncio.Future[bool],
    ) -> bool:
        """Await an approval future, with an optional timeout."""
        if self._approval_timeout_s > 0:
            try:
                return await asyncio.wait_for(
                    future, timeout=self._approval_timeout_s
                )
            except asyncio.TimeoutError:
                self._pending_approvals.pop(run_id, None)
                return False
        return await future

    @staticmethod
    def _resolve_next(node: WorkflowNode, state: WorkflowState) -> str | None:
        """Resolve the next node name from a node's edge spec."""
        if node.next_node is None:
            return None
        if isinstance(node.next_node, str):
            return node.next_node
        return node.next_node(state)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self) -> HealthStatus:
        state = (
            ServiceState.DEGRADED
            if self.state == ServiceState.RUNNING and self._runs_failed > 0
            else self.state
        )
        run_count = 0
        paused_count = 0
        if self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}:
            try:
                run_count = await self.repository.count()
                paused_count = await self.repository.count(
                    status=WorkflowStatus.PAUSED
                )
            except Exception:
                pass
        return HealthStatus(
            state=state,
            details={
                "workflows": sorted(self._workflows),
                "workflow_count": len(self._workflows),
                "runs_started": self._runs_started,
                "runs_completed": self._runs_completed,
                "runs_failed": self._runs_failed,
                "runs_paused": self._runs_paused,
                "pending_approvals": len(self._pending_approvals),
                "run_count": run_count,
                "paused_count": paused_count,
                "db_path": str(self.repository.db_path),
                "max_steps": self._max_steps,
            },
        )
