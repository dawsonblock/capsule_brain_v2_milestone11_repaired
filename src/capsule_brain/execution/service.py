from __future__ import annotations

from typing import Any

from capsule_brain.events.models import EventEnvelope
from capsule_brain.runtime.service import (
    CapsuleService,
    HealthStatus,
    ServiceState,
)

from .models import ExecutionPolicy, ExecutionRequest, ExecutionResult
from .repository import ExecutionRepository
from .runner import ExecutionRunner


class ExecutionService(CapsuleService):
    name = "execution"

    def __init__(
        self,
        event_bus: Any,
        cfg: dict[str, Any] | None = None,
        repository: ExecutionRepository | None = None,
        runner: Any | None = None,
        worker_pool: Any | None = None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.policy = ExecutionPolicy(
            allow=bool(self.cfg.get("allow", False)),
            timeout_s=float(self.cfg.get("timeout_s", 10.0)),
            max_output_chars=int(
                self.cfg.get("max_output_chars", 20000)
            ),
            allowed_commands=tuple(
                self.cfg.get(
                    "allowed_commands",
                    ["python", "pytest"],
                )
            ),
            cwd_root=str(self.cfg.get("cwd_root", "sandbox")),
        )
        self.runner = runner or ExecutionRunner(self.policy)
        self.repository = repository or ExecutionRepository(
            self.cfg.get("db_path", "data/execution_v2.sqlite")
        )
        # Optional bounded worker pool. When provided, execute() delegates to
        # the pool so concurrent container/host executions are capped at
        # pool.max_workers. When None (the default), execute() calls the
        # runner directly, preserving the original behavior.
        self.worker_pool = worker_pool
        self._unsubscribers: list = []
        self.runs = 0
        self.failures = 0

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        await self.repository.start()
        if self.worker_pool is not None:
            await self.worker_pool.start()
        self._unsubscribers.append(
            self.event_bus.subscribe(
                "execution.request",
                self._on_request,
            )
        )
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        if self.worker_pool is not None:
            await self.worker_pool.stop()
        await self.repository.stop()
        self.state = ServiceState.STOPPED

    async def _on_request(self, event: EventEnvelope) -> None:
        try:
            request = ExecutionRequest(
                command=list(event.payload.get("command") or []),
                cwd=event.payload.get("cwd"),
                source=str(
                    event.payload.get("source", event.source)
                ),
                metadata=dict(
                    event.payload.get("metadata") or {}
                ),
            )
            await self.execute(
                request,
                correlation_id=event.correlation_id,
            )
        except Exception as exc:
            self.failures += 1
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="execution.failed",
                    source=self.name,
                    correlation_id=event.correlation_id,
                    payload={"error": str(exc)},
                )
            )

    async def execute(
        self,
        request: ExecutionRequest,
        correlation_id: Any = None,
    ) -> ExecutionResult:
        self.runs += 1
        if self.worker_pool is not None:
            result = await self.worker_pool.submit(
                request,
                correlation_id=correlation_id,
            )
        else:
            result = await self.runner.run(request)

        if not result.passed:
            self.failures += 1

        await self.repository.save(result)

        await self.event_bus.publish(
            EventEnvelope(
                event_type=(
                    "execution.completed"
                    if result.passed
                    else "execution.failed"
                ),
                source=self.name,
                correlation_id=correlation_id,
                payload={
                    "request_id": result.request_id,
                    "command": result.command,
                    "cwd": result.cwd,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "duration_ms": result.duration_ms,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "metadata": result.metadata,
                },
            )
        )
        return result

    async def health(self) -> HealthStatus:
        state = (
            ServiceState.DEGRADED
            if self.state == ServiceState.RUNNING and self.failures > 0
            else self.state
        )
        details = {
            "runs": self.runs,
            "failures": self.failures,
            "allow": self.policy.allow,
            "timeout_s": self.policy.timeout_s,
            "cwd_root": self.policy.cwd_root,
            "allowed_commands": list(
                self.policy.allowed_commands
            ),
            "result_count": (
                await self.repository.count()
                if self.state in {
                    ServiceState.RUNNING,
                    ServiceState.DEGRADED,
                }
                else 0
            ),
            "worker_pool": self.worker_pool is not None,
        }
        if self.worker_pool is not None and self.worker_pool.is_running:
            try:
                pool_health = await self.worker_pool.health()
                details["pool"] = pool_health.details
            except Exception:
                pass
        return HealthStatus(state=state, details=details)
