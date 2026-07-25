import pytest

from capsule_brain.llm.errors import LLMProviderError
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest
from capsule_brain.llm.providers.base import LLMProvider


class AlwaysFail(LLMProvider):
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        raise RuntimeError("fail")


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold():
    provider = AlwaysFail()
    gateway = LLMGateway(
        {
            "default_model": "bad",
            "max_attempts": 1,
            "models": {
                "bad": {
                    "provider": "fake",
                    "model_name": "bad",
                    "capabilities": ["text"],
                }
            },
            "circuit_breaker": {
                "failure_threshold": 2,
                "recovery_timeout_s": 60,
            },
        },
        providers={"fake": provider},
    )

    await gateway.start()

    for _ in range(2):
        with pytest.raises(LLMProviderError):
            await gateway.generate(LLMRequest(prompt="x"))

    calls_before = provider.calls

    with pytest.raises(LLMProviderError):
        await gateway.generate(LLMRequest(prompt="x"))

    assert provider.calls == calls_before

    health = await gateway.health()
    assert health.details["breakers"]["bad"]["open"] is True

    await gateway.stop()
