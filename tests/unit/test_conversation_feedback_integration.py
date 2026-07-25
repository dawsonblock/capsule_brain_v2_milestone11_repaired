import pytest

from capsule_brain.events.models import EventEnvelope
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.runtime.bootstrap import build_application


class FakeProvider(LLMProvider):
    name = "fake"

    async def generate(self, request, model_cfg):
        return LLMResult(
            text="answer",
            model="fake-model",
            provider="fake",
            latency_ms=2,
        )


@pytest.mark.asyncio
async def test_response_creates_experience_then_feedback_attaches(tmp_path):
    app = build_application(
        {
            "memory": {"db_path": str(tmp_path / "memory.sqlite")},
            "memory_consolidator": {"enable": False},
            "llm_gateway": {
                "enable": True,
                "default_model": "fake",
                "models": {
                    "fake": {
                        "provider": "fake",
                        "model_name": "fake-model",
                        "capabilities": ["text", "json"],
                    }
                },
            },
            "conversation": {
                "enable": True,
                "db_path": str(tmp_path / "conversation.sqlite"),
            },
            "learning": {
                "db_path": str(tmp_path / "experience.sqlite"),
            },
            "feedback": {"enable": True},
            "goal_planner": {
                "db_path": str(tmp_path / "goals.sqlite"),
            },
        },
        llm_providers={"fake": FakeProvider()},
    )

    await app.start()

    bus = app.services.require("event_bus")
    conversation = app.services.require("conversation")
    feedback = app.services.require("feedback")

    response = await conversation.respond(
        conversation_id="c1",
        text="hello",
    )

    experience = await feedback.experience_store.get_by_response_id(response.id)
    assert experience is not None
    assert experience.response_text == "answer"

    await bus.publish(
        EventEnvelope(
            event_type="feedback.submit",
            source="test",
            payload={
                "response_id": response.id,
                "classification": "positive",
                "text": "correct",
            },
        )
    )

    updated = await feedback.experience_store.get_by_response_id(response.id)
    assert updated.feedback_classification == "positive"
    assert updated.feedback_text == "correct"

    await app.stop()
