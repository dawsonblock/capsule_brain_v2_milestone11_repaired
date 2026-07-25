import pytest

from capsule_brain.conversation.models import AssistantResponse, Conversation, Turn, TurnRole
from capsule_brain.conversation.repository import ConversationRepository
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.learning.experience_store import ExperienceStore
from capsule_brain.learning.feedback_service import FeedbackService
from capsule_brain.learning.models import ExperienceRecord
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository


@pytest.mark.asyncio
async def test_feedback_links_to_exact_response_id(tmp_path):
    bus = LocalEventBus()
    await bus.start()

    memory = MemoryService(
        bus,
        repository=SQLiteMemoryRepository(str(tmp_path / "memory.sqlite")),
    )
    await memory.start()

    conversations = ConversationRepository(str(tmp_path / "conversation.sqlite"))
    await conversations.start()

    store = ExperienceStore(str(tmp_path / "experience.sqlite"))
    service = FeedbackService(
        bus,
        memory,
        conversations,
        store,
    )
    await service.start()

    await store.create_experience(
        ExperienceRecord(
            response_id="response-42",
            conversation_id="conversation-1",
            user_turn_id="user-1",
            assistant_turn_id="assistant-1",
            prompt_text="x",
            response_text="y",
        )
    )

    recorded = []
    bus.subscribe("feedback.recorded", lambda event: recorded.append(event.payload))

    await bus.publish(
        EventEnvelope(
            event_type="feedback.submit",
            source="test",
            payload={
                "response_id": "response-42",
                "classification": "negative",
                "text": "Wrong answer",
                "reason": "incorrect",
            },
        )
    )

    assert recorded[0]["response_id"] == "response-42"
    loaded = await store.get_by_response_id("response-42")
    assert loaded.feedback_classification == "negative"
    assert loaded.feedback_text == "Wrong answer"

    memories = await memory.recent(limit=10)
    assert any(m.type.value == "feedback" for m in memories)

    await service.stop()
    await conversations.stop()
    await memory.stop()
    await bus.stop()
