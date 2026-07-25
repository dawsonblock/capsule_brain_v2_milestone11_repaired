import asyncio
import json
import pytest

from capsule_brain.llm.errors import LLMCapabilityError, LLMTimeoutError
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, *, failures=0, delay=0.0):
        self.failures = failures
        self.delay = delay
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.calls <= self.failures:
            raise RuntimeError("transient")
        text = '{"tasks":["one","two"]}' if request.response_format == "json" else "ok"
        return LLMResult(
            text=text,
            model=model_cfg["model_name"],
            provider=self.name,
            latency_ms=1.0,
        )


def config(**overrides):
    cfg = {
        "default_model": "test",
        "timeout_s": 1,
        "max_attempts": 2,
        "backoff_s": 0,
        "models": {
            "test": {
                "provider": "fake",
                "model_name": "fake-model",
                "capabilities": ["text", "json"],
            }
        },
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.asyncio
async def test_gateway_retries_and_reports_attempts():
    provider = FakeProvider(failures=1)
    gateway = LLMGateway(config(), providers={"fake": provider})
    await gateway.start()

    result = await gateway.generate(LLMRequest(prompt="hello"))

    assert result.text == "ok"
    assert result.attempts == 2
    assert provider.calls == 2
    await gateway.stop()


@pytest.mark.asyncio
async def test_gateway_json():
    gateway = LLMGateway(config(), providers={"fake": FakeProvider()})
    await gateway.start()
    data, result = await gateway.generate_json(LLMRequest(prompt="goal"))
    assert data["tasks"] == ["one", "two"]
    await gateway.stop()


@pytest.mark.asyncio
async def test_gateway_enforces_capability():
    cfg = config()
    cfg["models"]["test"]["capabilities"] = ["text"]
    gateway = LLMGateway(cfg, providers={"fake": FakeProvider()})
    await gateway.start()
    with pytest.raises(LLMCapabilityError):
        await gateway.generate(LLMRequest(prompt="x", response_format="json"))
    await gateway.stop()


@pytest.mark.asyncio
async def test_gateway_timeout():
    gateway = LLMGateway(
        config(timeout_s=0.01, max_attempts=1),
        providers={"fake": FakeProvider(delay=0.1)},
    )
    await gateway.start()
    with pytest.raises(LLMTimeoutError):
        await gateway.generate(LLMRequest(prompt="slow"))
    await gateway.stop()
