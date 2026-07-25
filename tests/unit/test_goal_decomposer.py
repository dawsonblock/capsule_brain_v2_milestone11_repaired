import pytest

from capsule_brain.llm.goal_decomposer import StructuredGoalDecomposer
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    name = "fake"
    async def generate(self, request, model_cfg):
        return LLMResult(
            text='{"tasks":["Inspect system","Implement change","Verify tests"]}',
            model="fake-model",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_structured_goal_decomposer():
    gateway = LLMGateway({
        "default_model": "test",
        "models": {
            "test": {
                "provider": "fake",
                "model_name": "fake-model",
                "capabilities": ["text", "json"],
            }
        },
    }, providers={"fake": FakeProvider()})
    await gateway.start()

    decomposer = StructuredGoalDecomposer(gateway)
    tasks = await decomposer("Upgrade Capsule Brain")

    assert tasks == ["Inspect system", "Implement change", "Verify tests"]
    await gateway.stop()
