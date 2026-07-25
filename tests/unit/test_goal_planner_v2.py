from pathlib import Path

import pytest

from capsule_brain.core.goal_planner_v2 import GoalPlannerV2
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope


@pytest.mark.asyncio
async def test_goal_planner_creates_goal_and_stable_task_ids(tmp_path: Path):
    bus = LocalEventBus()
    await bus.start()

    async def decomposer(goal_text: str):
        return ["First task", "Second task"]

    planner = GoalPlannerV2(
        event_bus=bus,
        decomposer=decomposer,
        cfg={"db_path": str(tmp_path / "goals.sqlite")},
    )
    await planner.start()

    created = []

    async def on_created(event):
        created.append(event.payload)

    bus.subscribe("goal.created", on_created)

    await bus.publish(
        EventEnvelope(
            event_type="goal.request",
            payload={"text": "Build the system"},
            source="test",
        )
    )

    assert len(created) == 1
    goal = created[0]
    assert goal["text"] == "Build the system"
    assert len(goal["tasks"]) == 2
    assert goal["tasks"][0]["id"] != goal["tasks"][1]["id"]

    await planner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_task_status_uses_task_id_not_text(tmp_path: Path):
    bus = LocalEventBus()
    await bus.start()

    async def decomposer(goal_text: str):
        return ["Duplicate", "Duplicate"]

    planner = GoalPlannerV2(
        event_bus=bus,
        decomposer=decomposer,
        cfg={"db_path": str(tmp_path / "goals.sqlite")},
    )
    await planner.start()

    await bus.publish(
        EventEnvelope(
            event_type="goal.request",
            payload={"text": "Goal"},
            source="test",
        )
    )

    goal = (await planner.list_goals())[0]
    first_id = goal["tasks"][0]["id"]
    second_id = goal["tasks"][1]["id"]

    await bus.publish(
        EventEnvelope(
            event_type="goal.task.set_status",
            payload={
                "goal_id": goal["id"],
                "task_id": first_id,
                "status": "complete",
            },
            source="test",
        )
    )

    updated = (await planner.list_goals())[0]
    by_id = {task["id"]: task for task in updated["tasks"]}

    assert by_id[first_id]["status"] == "complete"
    assert by_id[second_id]["status"] == "pending"

    await planner.stop()
    await bus.stop()
