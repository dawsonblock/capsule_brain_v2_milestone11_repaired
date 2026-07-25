import pytest

from capsule_brain.conversation.repository import ConversationRepository
from capsule_brain.conversation.service import ConversationService
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository


class FakeProvider(LLMProvider):
    name = "fake"

    async def generate(self, request, model_cfg):
        assert "USER: hello" in request.prompt
        return LLMResult(
            text="Hello from Capsule Brain.",
            model="fake-model",
            provider="fake",
            latency_ms=12.5,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


@pytest.mark.asyncio
async def test_end_to_end_conversation_path(tmp_path):
    bus = LocalEventBus()
    await bus.start()

    memory = MemoryService(
        bus,
        repository=SQLiteMemoryRepository(str(tmp_path / "memory.sqlite")),
    )
    await memory.start()

    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake-model",
                    "capabilities": ["text"],
                }
            },
            "routing": {
                "routes": {
                    "chat": {"models": ["fake"]}
                }
            },
        },
        providers={"fake": FakeProvider()},
    )
    await gateway.start()

    repository = ConversationRepository(str(tmp_path / "conversation.sqlite"))
    service = ConversationService(
        bus,
        memory,
        gateway,
        repository=repository,
        cfg={"route": "chat"},
    )
    await service.start()

    gui_events = []
    bus.subscribe("gui.response", lambda event: gui_events.append(event.payload))

    response = await service.respond(
        conversation_id="conversation-1",
        text="hello",
    )

    assert response.text == "Hello from Capsule Brain."
    assert response.parent_turn_id
    assert response.turn_id
    assert response.id
    assert response.provider == "fake"
    assert response.model_name == "fake-model"
    assert gui_events[0]["response_id"] == response.id

    memories = await memory.recent(limit=10)
    types = {m.type.value for m in memories}
    assert "operator" in types
    assert "assistant" in types

    await service.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()
