import pytest

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider


class MeteredProvider(LLMProvider):
    name = "fake"

    async def generate(self, request, model_cfg):
        return LLMResult(
            text="ok",
            model="metered",
            provider="fake",
            latency_ms=1,
            usage={"prompt_tokens": 1000, "completion_tokens": 500},
        )


@pytest.mark.asyncio
async def test_usage_and_cost_accounting():
    gateway = LLMGateway(
        {
            "default_model": "metered",
            "models": {
                "metered": {
                    "provider": "fake",
                    "model_name": "metered",
                    "capabilities": ["text"],
                    "input_cost_per_million": 2.0,
                    "output_cost_per_million": 4.0,
                }
            },
        },
        providers={"fake": MeteredProvider()},
    )

    await gateway.start()
    await gateway.generate(LLMRequest(prompt="x"))

    health = await gateway.health()
    usage = health.details["usage"]

    assert usage["input_tokens"] == 1000
    assert usage["output_tokens"] == 500
    assert usage["estimated_cost_usd"] == pytest.approx(0.004)

    await gateway.stop()
