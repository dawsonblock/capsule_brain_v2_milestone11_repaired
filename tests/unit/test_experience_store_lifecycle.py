import pytest

from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.runtime.bootstrap import build_application


class FakeProvider(LLMProvider):
    name = "fake"

    async def generate(self, request, model_cfg):
        return LLMResult(
            text='{"tasks":["one"]}',
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_experience_store_registered_and_stopped_cleanly(
    tmp_path,
):
    app = build_application(
        {
            "memory": {
                "db_path": str(tmp_path / "memory.sqlite")
            },
            "memory_consolidator": {"enable": False},
            "llm_gateway": {
                "enable": True,
                "default_model": "fake",
                "models": {
                    "fake": {
                        "provider": "fake",
                        "model_name": "fake",
                        "capabilities": ["text", "json"],
                    }
                },
            },
            "conversation": {
                "enable": True,
                "db_path": str(tmp_path / "conv.sqlite"),
            },
            "learning": {
                "db_path": str(tmp_path / "exp.sqlite")
            },
            "feedback": {"enable": False},
            "goal_planner": {
                "db_path": str(tmp_path / "goals.sqlite")
            },
        },
        llm_providers={"fake": FakeProvider()},
    )

    await app.start()

    assert "experience_store" in app.services.names()
    store = app.services.get("experience_store")
    assert store is not None
    assert store._conn is not None

    await app.stop()

    assert store._conn is None
