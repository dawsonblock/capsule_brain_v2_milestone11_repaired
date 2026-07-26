from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

log = logging.getLogger(__name__)


class TaskRegistry:
    """Owns long-running asyncio tasks and guarantees observable shutdown.

    When an ``event_bus_provider`` is supplied, background task failures are
    emitted as ``system.task.failed`` events so monitoring tools and health
    checks can detect worker deaths without parsing logs.
    """

    def __init__(
        self,
        event_bus_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._event_bus_provider = event_bus_provider
        self._event_tasks: set[asyncio.Task[Any]] = set()

    def spawn(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._report_failure)
        return task

    def _report_failure(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            log.error(
                "Background task %r failed",
                task.get_name(),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._emit_task_failed_event(task, exc)

    def _emit_task_failed_event(self, task: asyncio.Task[Any], exc: BaseException) -> None:
        """Publish a system.task.failed event if an event bus is available."""
        if self._event_bus_provider is None:
            return
        try:
            bus = self._event_bus_provider()
        except Exception:
            return
        if bus is None:
            return
        # Check if the bus is running (has an is_running property)
        is_running = getattr(bus, "is_running", None)
        if callable(is_running):
            if not is_running():
                return
        elif hasattr(bus, "state"):
            from capsule_brain.runtime.service import ServiceState
            if bus.state not in {ServiceState.RUNNING, ServiceState.DEGRADED}:
                return
        try:
            from capsule_brain.events.models import EventEnvelope
            evt = asyncio.create_task(
                bus.publish(
                    EventEnvelope(
                        event_type="system.task.failed",
                        source="task_registry",
                        payload={
                            "task_name": task.get_name(),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                )
            )
            self._event_tasks.add(evt)
            evt.add_done_callback(self._event_tasks.discard)
        except Exception:
            pass

    @property
    def active_count(self) -> int:
        return sum(not task.done() for task in self._tasks)

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks if not task.done()]
        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._tasks.clear()
