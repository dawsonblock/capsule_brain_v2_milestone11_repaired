import pytest

from capsule_brain.conversation.models import AssistantResponse, Conversation, Turn, TurnRole
from capsule_brain.conversation.repository import ConversationRepository


@pytest.mark.asyncio
async def test_conversation_turn_and_response_persistence(tmp_path):
    repo = ConversationRepository(str(tmp_path / "conversations.sqlite"))
    await repo.start()

    conversation = await repo.create_conversation(Conversation())
    user = await repo.create_turn(
        Turn(
            conversation_id=conversation.id,
            role=TurnRole.USER,
            text="hello",
        )
    )
    assistant = await repo.create_turn(
        Turn(
            conversation_id=conversation.id,
            role=TurnRole.ASSISTANT,
            text="hi",
            parent_turn_id=user.id,
        )
    )
    response = await repo.create_response(
        AssistantResponse(
            conversation_id=conversation.id,
            parent_turn_id=user.id,
            turn_id=assistant.id,
            text="hi",
            model_alias="test",
            provider="fake",
            model_name="fake-model",
        )
    )

    turns = await repo.recent_turns(conversation.id)
    assert [turn.role for turn in turns] == [TurnRole.USER, TurnRole.ASSISTANT]

    loaded = await repo.get_response(response.id)
    assert loaded is not None
    assert loaded.parent_turn_id == user.id
    assert loaded.turn_id == assistant.id

    await repo.stop()
