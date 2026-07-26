from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.observability.tracing import get_default_tracer
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .models import ExecutionRequest, ExecutionResult

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingJob:
    request: ExecutionRequest
    correlation_id: Any
    future: asyncio.Future[ExecutionResult]
    enqueued_at: float = field(
        default_factory=lambda: asyncio.get_event_loop().time()
    )


class ExecutionWorkerPool(CapsuleService):
    """A bounded async worker pool for execution jobs.

    The pool wraps a runner (ExecutionRunner or ContainerExecutionRunner)
    and limits concurrent executions to ``max_workers``. Jobs are queued
    FIFO and dispatched to ``max_workers`` permanent worker tasks. This
    prevents Docker daemon CPU/RAM thrashing under heavy multi-task load
    and provides a natural extension point for running execution jobs on
    remote worker nodes (the runner interface is unchanged).

    The pool is a CapsuleService so it participates in the normal
    dependency-ordered lifecycle. On stop, in-flight jobs are awaited (with
    a configurable drain timeout) so partial results are not lost.
    """

    name = "execution_worker_pool"

    def __init__(
        self,
        runner: Any,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.runner = runner
        self.max_workers = max(1, int(self.cfg.get("max_workers", 2)))
        # 0 = unbounded queue. A positive value rejects submissions when the
        # queue is full, providing backpressure to upstream callers.
        self.queue_max = max(0, int(self.cfg.get("queue_max", 0)))
        self.drain_timeout_s = float(self.cfg.get("drain_timeout_s", 10.0))
        self._queue: asyncio.Queue[_PendingJob | None] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        # Metrics
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._active = 0
        self._peak_active = 0
        self._queue_depth_high_water = 0

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        self._stop_event.clear()
        # Spawn max_workers permanent consumer tasks. Each pulls jobs from
        # the queue and runs them through the wrapped runner. A None
        # sentinel signals a worker to exit cleanly on stop.
        loop = asyncio.get_event_loop()
        self._workers = [
            loop.create_task(self._worker_loop(i))
            for i in range(self.max_workers)
        ]
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        self._stop_event.set()
        # Cancel futures for any jobs still queued (callers awaiting them
        # get a CancelledError). In-flight jobs are allowed to finish during
        # the drain timeout below.
        while True:
            try:
                leftover = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if leftover is None:
                continue
            if not leftover.future.done():
                leftover.future.cancel()
        # Push one None sentinel per worker so each worker exits cleanly.
        # Each worker consumes exactly one sentinel (no drain loop) so
        # sentinels are not stolen by other workers.
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=self.drain_timeout_s,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "ExecutionWorkerPool drain timed out after %.1fs; "
                    "cancelling %d workers",
                    self.drain_timeout_s,
                    len(self._workers),
                )
                for worker in self._workers:
                    worker.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self.state = ServiceState.STOPPED

    async def submit(
        self,
        request: ExecutionRequest,
        *,
        correlation_id: Any = None,
    ) -> ExecutionResult:
        """Submit an execution job and await its result.

        The caller is suspended until a worker picks up the job and the
        runner completes. This preserves the synchronous ``execute`` API of
        ExecutionService while bounding concurrency behind the scenes.
        """
        if self.state != ServiceState.RUNNING:
            raise RuntimeError("ExecutionWorkerPool is not running")
        if self.queue_max > 0 and self._queue.qsize() >= self.queue_max:
            raise RuntimeError(
                f"ExecutionWorkerPool queue is full ({self.queue_max})"
            )

        loop = asyncio.get_event_loop()
        job = _PendingJob(
            request=request,
            correlation_id=correlation_id,
            future=loop.create_future(),
        )
        await self._queue.put(job)
        self._submitted += 1
        depth = self._queue.qsize()
        if depth > self._queue_depth_high_water:
            self._queue_depth_high_water = depth
        return await job.future

    async def _worker_loop(self, worker_id: int) -> None:
        """Permanent consumer: pull jobs and run them until a None sentinel.

        Each worker consumes exactly one None sentinel and then exits. We do
        NOT drain remaining queued jobs here — stop() cancels the futures of
        queued jobs before sending sentinels, so no worker needs to process
        them during shutdown.
        """
        while True:
            job = await self._queue.get()
            if job is None:
                return
            await self._run_job(job)

    async def _run_job(self, job: _PendingJob) -> None:
        tracer = get_default_tracer()
        self._active += 1
        if self._active > self._peak_active:
            self._peak_active = self._active
        try:
            with tracer.span(
                "execution.worker.run",
                correlation_id=job.correlation_id,
                attributes={
                    "request_id": job.request.id,
                    "command": " ".join(str(c) for c in job.request.command),
                },
            ):
                result = await self.runner.run(job.request)
            self._completed += 1
            if not job.future.done():
                job.future.set_result(result)
        except Exception as exc:
            self._failed += 1
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            self._active -= 1

    async def health(self) -> HealthStatus:
        state = (
            ServiceState.DEGRADED
            if self.state == ServiceState.RUNNING and self._failed > 0
            else self.state
        )
        return HealthStatus(
            state=state,
            details={
                "max_workers": self.max_workers,
                "queue_max": self.queue_max,
                "queue_depth": self._queue.qsize(),
                "queue_high_water": self._queue_depth_high_water,
                "active": self._active,
                "peak_active": self._peak_active,
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "runner": type(self.runner).__name__,
            },
        )
