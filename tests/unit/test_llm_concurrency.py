import asyncio
import pytest

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider


class TrackingProvider(LLMProvider):
    name = "fake"

    def __init__(self):
        self.active = 0
        self.max_seen = 0

    async def generate(self, request, model_cfg):
        self.active += 1
        self.max_seen = max(self.max_seen, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return LLMResult(
            text="ok",
            model="fake",
            provider="fake",
            latency_ms=20,
        )


@pytest.mark.asyncio
async def test_concurrency_limit_is_enforced():
    provider = TrackingProvider()
    gateway = LLMGateway(
        {
            "default_model": "fake",
            "max_concurrency": 2,
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
        },
        providers={"fake": provider},
    )

    await gateway.start()

    await asyncio.gather(
        *(gateway.generate(LLMRequest(prompt=str(i))) for i in range(6))
    )

    assert provider.max_seen <= 2

    await gateway.stop()
