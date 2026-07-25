import pytest

from capsule_brain.runtime.bootstrap import build_application


@pytest.mark.asyncio
async def test_bootstrap_starts_memory_before_consolidator(tmp_path):
    app = build_application({
        "memory": {"db_path": str(tmp_path / "memory.sqlite")},
        "memory_consolidator": {
            "enable": True,
            "interval_sec": 9999,
            "archive_after_days": 7,
        },
        "goal_planner": {"db_path": str(tmp_path / "goals.sqlite")},
    })

    await app.start()
    health = await app.services.health()

    assert health["memory"].healthy
    assert health["memory_consolidator"].healthy

    await app.stop()
