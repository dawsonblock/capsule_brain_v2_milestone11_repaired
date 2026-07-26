from __future__ import annotations

from typing import Any

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .models import MemoryRecord, MemoryType
from .sqlite_repository import SQLiteMemoryRepository


PROTECTED_TYPES = {
    MemoryType.OPERATOR,
    MemoryType.GOAL,
    MemoryType.FEEDBACK,
    MemoryType.SYSTEM,
}


class MemoryService(CapsuleService):
    name = "memory"

    def __init__(
        self,
        event_bus: LocalEventBus,
        cfg: dict[str, Any] | None = None,
        repository: SQLiteMemoryRepository | None = None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.repository = repository or SQLiteMemoryRepository(
            self.cfg.get("db_path", "data/memory_v2.sqlite")
        )

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        await self.repository.start()
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        await self.repository.stop()
        self.state = ServiceState.STOPPED

    async def write(
        self,
        text: str,
        *,
        type: MemoryType = MemoryType.OBSERVATION,
        source: str = "unknown",
        tags: list[str] | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        importance: float = 0.0,
        confidence: float | None = None,
        protected: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            text=text,
            type=type,
            source=source,
            tags=list(tags or []),
            conversation_id=conversation_id,
            turn_id=turn_id,
            importance=float(importance),
            confidence=confidence,
            protected=type in PROTECTED_TYPES if protected is None else protected,
            metadata=dict(metadata or {}),
        )
        await self.repository.create(record)

        await self.event_bus.publish(
            EventEnvelope(
                event_type="memory.created",
                source=self.name,
                payload={
                    "id": record.id,
                    "type": record.type.value,
                    "source": record.source,
                    "protected": record.protected,
                },
            )
        )
        return record

    async def recent(
        self,
        *,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        return await self.repository.recent(
            limit=limit,
            include_archived=include_archived,
        )

    async def archive(self, memory_id: str) -> bool:
        archived = await self.repository.archive(memory_id)
        if archived:
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="memory.archived",
                    source=self.name,
                    payload={"id": memory_id},
                )
            )
        return archived

    async def archive_batch(self, memory_ids: list[str]) -> int:
        """Archive a batch of memories in a single transaction.

        Returns the number of records actually archived (protected records
        are skipped). Publishes a single ``memory.archived_batch`` event.
        """
        count = await self.repository.archive_batch(memory_ids)
        if count > 0:
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="memory.archived_batch",
                    source=self.name,
                    payload={"count": count, "ids": list(memory_ids)},
                )
            )
        return count

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=self.state,
            details={
                "active_count": await self.repository.count()
                if self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}
                else 0,
                "total_count": await self.repository.count(include_archived=True)
                if self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}
                else 0,
                "db_path": str(self.repository.db_path),
            },
        )
