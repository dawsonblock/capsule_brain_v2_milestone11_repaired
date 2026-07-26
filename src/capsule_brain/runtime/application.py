from __future__ import annotations

import asyncio
import logging
from typing import Any

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.runtime.registry import ServiceRegistry
from capsule_brain.runtime.tasks import TaskRegistry

log = logging.getLogger(__name__)


class CapsuleApplication:
    """Top-level runtime owner for Capsule Brain v2."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = dict(cfg or {})
        self.services = ServiceRegistry()
        self.tasks = TaskRegistry(
            event_bus_provider=lambda: self.services.get("event_bus"),
        )
        self.shutdown_requested = asyncio.Event()
        self.shutdown_reason: str | None = None

    def install_core_services(self) -> None:
        self.services.register(LocalEventBus())

    def request_shutdown(self, reason: str = "requested") -> None:
        if not self.shutdown_requested.is_set():
            self.shutdown_reason = reason
            self.shutdown_requested.set()

    async def start(self) -> None:
        await self.services.start_all()

    async def wait(self) -> None:
        await self.shutdown_requested.wait()

    async def stop(self) -> None:
        # Cancel background producers before service infrastructure disappears.
        await self.tasks.shutdown()
        await self.services.stop_all()
