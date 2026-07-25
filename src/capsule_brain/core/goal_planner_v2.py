from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

log = logging.getLogger(__name__)


@dataclass(slots=True)
class GoalTask:
    id: str
    text: str
    status: str = "pending"


@dataclass(slots=True)
class Goal:
    id: str
    text: str
    status: str
    tasks: list[GoalTask]


class GoalPlannerV2(CapsuleService):
    """Goal planner migrated to the v2 async service architecture.

    This milestone intentionally uses a pluggable decomposer callable instead of
    binding directly to the legacy LLMAdapter.
    """

    name = "goal_planner"

    def __init__(
        self,
        event_bus: LocalEventBus,
        decomposer,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.decomposer = decomposer
        self.db_path = Path(self.cfg.get("db_path", "data/goals_v2.sqlite"))
        self._db: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._unsubscribers: list[Callable[[], None]] = []

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        async with self._lock:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._db = sqlite3.connect(self.db_path)
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tasks_json TEXT NOT NULL
                )
                """
            )
            self._db.commit()

        self._unsubscribers.extend([
            self.event_bus.subscribe("goal.request", self._on_goal_request),
            self.event_bus.subscribe("goal.task.set_status", self._on_task_status),
            self.event_bus.subscribe("goal.task.edit", self._on_task_edit),
            self.event_bus.subscribe("goal.edit", self._on_goal_edit),
        ])
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING

        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

        async with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

        self.state = ServiceState.STOPPED

    async def _on_goal_request(self, event: EventEnvelope) -> None:
        text = str(event.payload.get("text", "")).strip()
        if not text:
            return

        task_texts = await self.decomposer(text)
        tasks = [
            GoalTask(id=uuid4().hex, text=task.strip())
            for task in task_texts
            if task.strip()
        ]

        goal = Goal(
            id=uuid4().hex,
            text=text,
            status="active",
            tasks=tasks,
        )
        await self._save_goal(goal)

        await self.event_bus.publish(
            EventEnvelope(
                event_type="goal.created",
                source=self.name,
                correlation_id=event.correlation_id,
                payload=self._serialize_goal(goal),
            )
        )

    async def _on_task_status(self, event: EventEnvelope) -> None:
        goal_id = str(event.payload.get("goal_id", ""))
        task_id = str(event.payload.get("task_id", ""))
        status = str(event.payload.get("status", "")).strip()
        if not goal_id or not task_id or not status:
            return

        goal = await self._load_goal(goal_id)
        if goal is None:
            return

        for task in goal.tasks:
            if task.id == task_id:
                task.status = status
                await self._save_goal(goal)
                await self.event_bus.publish(
                    EventEnvelope(
                        event_type="goal.task.updated",
                        source=self.name,
                        correlation_id=event.correlation_id,
                        payload={
                            "goal_id": goal_id,
                            "task_id": task_id,
                            "status": status,
                        },
                    )
                )
                return

    async def _on_task_edit(self, event: EventEnvelope) -> None:
        goal_id = str(event.payload.get("goal_id", ""))
        task_id = str(event.payload.get("task_id", ""))
        new_text = str(event.payload.get("text", "")).strip()
        if not goal_id or not task_id or not new_text:
            return

        goal = await self._load_goal(goal_id)
        if goal is None:
            return

        for task in goal.tasks:
            if task.id == task_id:
                task.text = new_text
                await self._save_goal(goal)
                await self.event_bus.publish(
                    EventEnvelope(
                        event_type="goal.task.edited",
                        source=self.name,
                        correlation_id=event.correlation_id,
                        payload={
                            "goal_id": goal_id,
                            "task_id": task_id,
                            "text": new_text,
                        },
                    )
                )
                return

    async def _on_goal_edit(self, event: EventEnvelope) -> None:
        goal_id = str(event.payload.get("goal_id", ""))
        new_text = str(event.payload.get("text", "")).strip()
        if not goal_id or not new_text:
            return

        goal = await self._load_goal(goal_id)
        if goal is None:
            return

        goal.text = new_text
        await self._save_goal(goal)
        await self.event_bus.publish(
            EventEnvelope(
                event_type="goal.edited",
                source=self.name,
                correlation_id=event.correlation_id,
                payload={"goal_id": goal_id, "text": new_text},
            )
        )

    async def _save_goal(self, goal: Goal) -> None:
        async with self._lock:
            if self._db is None:
                raise RuntimeError("Goal planner is not started")

            self._db.execute(
                """
                INSERT INTO goals(id, text, status, tasks_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text,
                    status=excluded.status,
                    tasks_json=excluded.tasks_json
                """,
                (
                    goal.id,
                    goal.text,
                    goal.status,
                    json.dumps([asdict(task) for task in goal.tasks]),
                ),
            )
            self._db.commit()

    async def _load_goal(self, goal_id: str) -> Goal | None:
        async with self._lock:
            if self._db is None:
                raise RuntimeError("Goal planner is not started")

            row = self._db.execute(
                "SELECT id, text, status, tasks_json FROM goals WHERE id = ?",
                (goal_id,),
            ).fetchone()

        if row is None:
            return None

        tasks = [GoalTask(**item) for item in json.loads(row[3])]
        return Goal(id=row[0], text=row[1], status=row[2], tasks=tasks)

    async def list_goals(self) -> list[dict[str, Any]]:
        async with self._lock:
            if self._db is None:
                raise RuntimeError("Goal planner is not started")

            rows = self._db.execute(
                "SELECT id, text, status, tasks_json FROM goals ORDER BY rowid"
            ).fetchall()

        output: list[dict[str, Any]] = []
        for row in rows:
            goal = Goal(
                id=row[0],
                text=row[1],
                status=row[2],
                tasks=[GoalTask(**item) for item in json.loads(row[3])],
            )
            output.append(self._serialize_goal(goal))
        return output

    @staticmethod
    def _serialize_goal(goal: Goal) -> dict[str, Any]:
        return {
            "id": goal.id,
            "text": goal.text,
            "status": goal.status,
            "tasks": [asdict(task) for task in goal.tasks],
        }

    async def health(self) -> HealthStatus:
        goal_count = 0
        if self._db is not None:
            goals = await self.list_goals()
            goal_count = len(goals)
        return HealthStatus(
            state=self.state,
            details={
                "db_path": str(self.db_path),
                "goal_count": goal_count,
            },
        )
