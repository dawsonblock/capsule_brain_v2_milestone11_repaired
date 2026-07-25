import pytest

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider


class SelectiveProvider(LLMProvider):
    name = "fake"

    def __init__(self, failing_models=None):
        self.failing_models = set(failing_models or [])
        self.calls = []

    async def generate(self, request, model_cfg):
        model = model_cfg["model_name"]
        self.calls.append(model)
        if model in self.failing_models:
            raise RuntimeError(f"{model} unavailable")
        return LLMResult(
            text=f"ok:{model}",
            model=model,
            provider=self.name,
            latency_ms=1,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


@pytest.mark.asyncio
async def test_route_falls_back_to_second_model():
    provider = SelectiveProvider(failing_models={"primary-model"})
    gateway = LLMGateway(
        {
            "default_model": "primary",
            "max_attempts": 1,
            "models": {
                "primary": {
                    "provider": "fake",
                    "model_name": "primary-model",
                    "capabilities": ["text"],
                },
                "backup": {
                    "provider": "fake",
                    "model_name": "backup-model",
                    "capabilities": ["text"],
                },
            },
            "routing": {
                "routes": {
                    "chat": {"models": ["primary", "backup"]}
                }
            },
        },
        providers={"fake": provider},
    )

    await gateway.start()
    result = await gateway.generate(LLMRequest(prompt="hi"), route="chat")

    assert result.text == "ok:backup-model"
    assert provider.calls == ["primary-model", "backup-model"]

    health = await gateway.health()
    assert health.details["usage"]["fallbacks"] == 1

    await gateway.stop()
