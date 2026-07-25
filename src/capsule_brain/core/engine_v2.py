from __future__ import annotations

from typing import Any

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.memory.service import MemoryService
from capsule_brain.runtime.registry import ServiceRegistry
from capsule_brain.runtime.tasks import TaskRegistry


class CapsuleEngineV2:
    """Thin facade over v2 runtime services.

    Memory is accessed only through MemoryService. There is intentionally no
    public `_memory` list.
    """

    def __init__(
        self,
        *,
        services: ServiceRegistry,
        tasks: TaskRegistry,
        event_bus: LocalEventBus,
        memory: MemoryService,
    ) -> None:
        self.services = services
        self.tasks = tasks
        self.events = event_bus
        self.memory = memory

    async def memory_write(self, text: str, **kwargs: Any):
        return await self.memory.write(text, **kwargs)

    async def memory_recent(self, *, limit: int = 100, include_archived: bool = False):
        return await self.memory.recent(
            limit=limit,
            include_archived=include_archived,
        )
