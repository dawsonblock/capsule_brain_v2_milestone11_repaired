import pytest

from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.runtime.bootstrap import build_application


class FakeProvider(LLMProvider):
    name = "fake"
    async def generate(self, request, model_cfg):
        return LLMResult(
            text="RESOLVED: ok",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_bootstrap_wires_reflection(tmp_path):
    app = build_application(
        {
            "memory": {"db_path": str(tmp_path / "memory.sqlite")},
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
                "db_path": str(tmp_path / "conversation.sqlite"),
            },
            "learning": {"db_path": str(tmp_path / "experience.sqlite")},
            "feedback": {"enable": True},
            "reflection": {
                "enable": True,
                "db_path": str(tmp_path / "reflections.sqlite"),
            },
            "goal_planner": {"db_path": str(tmp_path / "goals.sqlite")},
        },
        llm_providers={"fake": FakeProvider()},
    )

    await app.start()
    health = await app.services.health()

    assert health["reflection"].healthy

    await app.stop()
