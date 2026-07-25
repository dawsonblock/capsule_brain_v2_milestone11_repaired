import pytest

from capsule_brain.runtime.bootstrap import build_application


@pytest.mark.asyncio
async def test_bootstrap_starts_goal_planner_after_event_bus(tmp_path):
    app = build_application({
        "goal_planner": {"db_path": str(tmp_path / "goals.sqlite")},
        "redis_bridge": {"enable": False},
    })

    await app.start()

    health = await app.services.health()
    assert health["event_bus"].healthy
    assert health["goal_planner"].healthy

    await app.stop()
