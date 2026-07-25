import pytest

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider


class StreamProvider(LLMProvider):
    name = "fake"

    async def generate(self, request, model_cfg):
        return LLMResult(
            text="fallback",
            model="fake",
            provider="fake",
            latency_ms=1,
        )

    async def stream(self, request, model_cfg):
        yield "hello"
        yield " "
        yield "world"


@pytest.mark.asyncio
async def test_streaming_yields_chunks():
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
        providers={"fake": StreamProvider()},
    )

    await gateway.start()

    chunks = []
    async for chunk in gateway.stream(LLMRequest(prompt="x")):
        chunks.append(chunk)

    assert "".join(chunks) == "hello world"

    await gateway.stop()
