import pytest

from capsule_brain.conversation.repository import ConversationRepository
from capsule_brain.conversation.service import ConversationService
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository


class FakeProvider(LLMProvider):
    name = "fake"

    async def generate(self, request, model_cfg):
        return LLMResult(
            text="event response",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_event_driven_chat_path(tmp_path):
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
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
        },
        providers={"fake": FakeProvider()},
    )
    await gateway.start()

    service = ConversationService(
        bus,
        memory,
        gateway,
        repository=ConversationRepository(str(tmp_path / "conversation.sqlite")),
    )
    await service.start()

    responses = []
    bus.subscribe(
        "conversation.response.created",
        lambda event: responses.append(event.payload),
    )

    await bus.publish(
        EventEnvelope(
            event_type="conversation.message.create",
            source="test",
            payload={
                "conversation_id": "abc",
                "text": "message",
            },
        )
    )

    assert len(responses) == 1
    assert responses[0]["conversation_id"] == "abc"
    assert responses[0]["text"] == "event response"

    await service.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()
