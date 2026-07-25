import pytest

from capsule_brain.conversation.models import (
    AssistantResponse,
    Conversation,
    Turn,
    TurnRole,
)
from capsule_brain.conversation.repository import (
    ConversationRepository,
)
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.learning.experience_store import ExperienceStore
from capsule_brain.learning.feedback_service import FeedbackService
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import (
    SQLiteMemoryRepository,
)


@pytest.mark.asyncio
async def test_feedback_falls_back_to_conversation_repository(
    tmp_path,
):
    bus = LocalEventBus()
    await bus.start()

    memory = MemoryService(
        bus,
        repository=SQLiteMemoryRepository(
            str(tmp_path / "memory.sqlite")
        ),
    )
    await memory.start()

    conversations = ConversationRepository(
        str(tmp_path / "conversation.sqlite")
    )
    await conversations.start()
    conversation = await conversations.create_conversation(
        Conversation()
    )
    user_turn = await conversations.create_turn(
        Turn(
            conversation_id=conversation.id,
            role=TurnRole.USER,
            text="question",
        )
    )
    assistant_turn = await conversations.create_turn(
        Turn(
            conversation_id=conversation.id,
            role=TurnRole.ASSISTANT,
            text="answer",
            parent_turn_id=user_turn.id,
        )
    )
    response = await conversations.create_response(
        AssistantResponse(
            conversation_id=conversation.id,
            parent_turn_id=user_turn.id,
            turn_id=assistant_turn.id,
            text="answer",
        )
    )

    store = ExperienceStore(
        db_path=str(tmp_path / "experience.sqlite")
    )
    await store.start()

    service = FeedbackService(
        bus,
        memory,
        conversations,
        store,
    )
    await service.start()

    recorded = []
    bus.subscribe(
        "feedback.recorded",
        lambda event: recorded.append(event.payload),
    )

    await bus.publish(
        EventEnvelope(
            event_type="feedback.submit",
            source="test",
            payload={
                "response_id": response.id,
                "classification": "negative",
                "text": "wrong",
            },
        )
    )

    assert recorded
    assert (
        recorded[0]["conversation_id"]
        == conversation.id
    )

    await service.stop()
    await store.stop()
    await conversations.stop()
    await memory.stop()
    await bus.stop()
