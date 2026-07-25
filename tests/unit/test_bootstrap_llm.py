import pytest

from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.runtime.bootstrap import build_application


class FakeProvider(LLMProvider):
    name = "fake"
    async def generate(self, request, model_cfg):
        return LLMResult(
            text='{"tasks":["A","B"]}',
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_bootstrap_wires_llm_before_goal_planner(tmp_path):
    app = build_application({
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
        "goal_planner": {"db_path": str(tmp_path / "goals.sqlite")},
    }, llm_providers={"fake": FakeProvider()})

    await app.start()
    health = await app.services.health()
    assert health["llm_gateway"].healthy
    assert health["goal_planner"].healthy
    await app.stop()
