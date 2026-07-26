from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .models import MemoryRecord
from .service import MemoryService


class MemoryConsolidator(CapsuleService):
    """Archive-first memory consolidation.

    This milestone never physically deletes records. Protected memories are never
    archived automatically. Failures leave records untouched.
    """

    name = "memory_consolidator"

    def __init__(
        self,
        memory: MemoryService,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.memory = memory
        self.interval_sec = float(self.cfg.get("interval_sec", 3600))
        self.archive_after_days = float(self.cfg.get("archive_after_days", 7))
        self.max_scan = int(self.cfg.get("max_scan", 1000))
        self._task: asyncio.Task | None = None
        self._cycles = 0
        self._archived = 0
        self._errors = 0

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        self._task = asyncio.create_task(
            self._run(),
            name="memory-consolidator",
        )
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self.state = ServiceState.STOPPED

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_sec)
            await self.run_once()

    async def run_once(self) -> int:
        self._cycles += 1
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.archive_after_days
        )
        try:
            # Query oldest records first so that records beyond max_scan
            # are not permanently shielded from archival. Using recent()
            # (newest-first) would hide old records past the scan window.
            records = await self.memory.oldest(
                limit=self.max_scan,
                include_archived=False,
            )
            # Collect eligible IDs and archive them in a single transaction
            # to avoid write-lock contention from per-record BEGIN IMMEDIATE.
            eligible_ids: list[str] = []
            for record in records:
                if record.protected:
                    continue
                created = datetime.fromisoformat(record.created_at)
                if created >= cutoff:
                    continue
                eligible_ids.append(record.id)

            if not eligible_ids:
                return 0

            archived = await self.memory.archive_batch(eligible_ids)
            self._archived += archived
            return archived
        except Exception:
            # Fail closed: do not delete/archive on uncertainty.
            self._errors += 1
            return 0

    async def health(self) -> HealthStatus:
        state = (
            ServiceState.DEGRADED
            if self.state == ServiceState.RUNNING and self._errors
            else self.state
        )
        return HealthStatus(
            state=state,
            details={
                "cycles": self._cycles,
                "archived": self._archived,
                "errors": self._errors,
                "archive_after_days": self.archive_after_days,
            },
        )
