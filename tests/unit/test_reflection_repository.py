import pytest

from capsule_brain.reflection.models import ReflectionIteration, ReflectionSession
from capsule_brain.reflection.repository import ReflectionRepository


@pytest.mark.asyncio
async def test_reflection_session_round_trip(tmp_path):
    repo = ReflectionRepository(str(tmp_path / "reflections.sqlite"))
    await repo.start()

    session = ReflectionSession(
        seed="seed",
        source="test",
        max_iterations=2,
        iterations=[
            ReflectionIteration(
                index=0,
                critique="critique",
                revision="revision",
                evaluation="RESOLVED: fixed",
                resolved=True,
            )
        ],
        final_text="revision",
        stop_reason="resolved",
    )
    await repo.save(session)

    loaded = await repo.get(session.id)
    assert loaded is not None
    assert loaded.final_text == "revision"
    assert loaded.iterations[0].resolved is True

    await repo.stop()
