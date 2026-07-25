import pytest

from capsule_brain.learning.experience_store import ExperienceStore
from capsule_brain.learning.models import ExperienceRecord, FeedbackClass, FeedbackRecord


@pytest.mark.asyncio
async def test_experience_feedback_round_trip(tmp_path):
    store = ExperienceStore(str(tmp_path / "experience.sqlite"))
    await store.start()

    exp = ExperienceRecord(
        response_id="response-1",
        conversation_id="conversation-1",
        user_turn_id="user-1",
        assistant_turn_id="assistant-1",
        prompt_text="prompt",
        response_text="response",
        model_alias="fake",
        provider="fake",
        model_name="fake-model",
    )
    await store.create_experience(exp)

    feedback = FeedbackRecord(
        response_id="response-1",
        conversation_id="conversation-1",
        classification=FeedbackClass.POSITIVE,
        text="good",
    )
    await store.add_feedback(feedback)

    loaded = await store.get_by_response_id("response-1")
    assert loaded is not None
    assert loaded.feedback_classification == "positive"
    assert loaded.feedback_text == "good"

    await store.stop()
