import asyncio
import pytest

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider


class SlowProvider(LLMProvider):
    name = "fake"

    async def generate(self, request, model_cfg):
        await asyncio.sleep(10)
        return LLMResult(
            text="never",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_cancellation_propagates():
    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
        },
        providers={"fake": SlowProvider()},
    )

    await gateway.start()

    task = asyncio.create_task(
        gateway.generate(LLMRequest(prompt="cancel me"))
    )
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await gateway.stop()
