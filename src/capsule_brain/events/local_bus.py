from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from capsule_brain.runtime.service import (
    CapsuleService,
    HealthStatus,
    ServiceState,
)

from .models import EventEnvelope

log = logging.getLogger(__name__)

EventHandler = Callable[[EventEnvelope], Any]


class LocalEventBus(CapsuleService):
    """Async-safe in-process event bus.

    This is the default internal bus for Capsule Brain v2. Redis can later be
    added as an external-process bridge without forcing internal events through
    a background thread.
    """

    name = "event_bus"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._publish_lock = asyncio.Lock()
        self._published = 0
        self._handler_failures = 0
        # When true, async handlers are dispatched as background tasks via the
        # task registry instead of awaited inline. This prevents long-running
        # handlers (e.g. reflection triggering multiple LLM calls) from
        # blocking event publication. Default is false for backward
        # compatibility and deterministic test ordering.
        self._async_dispatch = bool(self.cfg.get("async_dispatch", False))
        self._task_registry: Any = None

    def set_task_registry(self, registry: Any) -> None:
        """Attach a TaskRegistry for async dispatch mode.

        When set, async handlers are spawned as managed tasks instead of
        awaited inline, providing backpressure isolation for long-running
        reactions.
        """
        self._task_registry = registry
        self._async_dispatch = True

    async def start(self) -> None:
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self._handlers.clear()
        self.state = ServiceState.STOPPED

    def subscribe(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        # Duplicate subscriptions are silently ignored (deduped). The returned
        # unsubscribe is a no-op for duplicates so it cannot accidentally remove
        # the original registration.
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
        return lambda: self.unsubscribe(event_type, handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        handlers = self._handlers.get(event_type)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            self._handlers.pop(event_type, None)

    async def publish(self, event: EventEnvelope) -> None:
        # Snapshot the subscribers so a handler may subscribe/unsubscribe safely.
        async with self._publish_lock:
            handlers = list(self._handlers.get(event.event_type, ()))
            wildcard = list(self._handlers.get("*", ()))
            self._published += 1

        all_handlers = handlers + wildcard
        for handler in all_handlers:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    if self._async_dispatch and self._task_registry is not None:
                        # Dispatch as a managed background task so publish()
                        # returns immediately. This prevents long-running
                        # handlers (e.g. reflection) from blocking event
                        # publication.
                        self._task_registry.spawn(
                            _run_handler_safely(result, handler, event, self),
                            name=f"event:{event.event_type}",
                        )
                    else:
                        await result
            except Exception:
                self._handler_failures += 1
                log.exception(
                    "Event handler failed: event_type=%s handler=%r",
                    event.event_type,
                    handler,
                )

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=self.state,
            details={
                "published": self._published,
                "handler_failures": self._handler_failures,
                "subscriptions": sum(len(v) for v in self._handlers.values()),
                "async_dispatch": self._async_dispatch,
            },
        )


async def _run_handler_safely(
    awaitable: Any,
    handler: EventHandler,
    event: EventEnvelope,
    bus: LocalEventBus,
) -> None:
    """Run an async handler coroutine with error isolation."""
    try:
        await awaitable
    except Exception:
        bus._handler_failures += 1
        log.exception(
            "Async event handler failed: event_type=%s handler=%r",
            event.event_type,
            handler,
        )
