from __future__ import annotations

from typing import Any

from capsule_brain.conversation.repository import ConversationRepository
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.memory.models import MemoryType
from capsule_brain.memory.service import MemoryService
from capsule_brain.runtime.service import (
    CapsuleService,
    HealthStatus,
    ServiceState,
)

from .experience_store import ExperienceStore
from .models import FeedbackClass, FeedbackRecord


class FeedbackService(CapsuleService):
    name = "feedback"

    def __init__(
        self,
        event_bus: LocalEventBus,
        memory: MemoryService,
        conversations: ConversationRepository,
        experience_store: ExperienceStore,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.memory = memory
        self.conversations = conversations
        self.experience_store = experience_store
        self._unsubscribers: list = []
        self._received = 0
        self._failures = 0

    async def start(self) -> None:
        self.state = ServiceState.STARTING

        # ExperienceStore is lifecycle-owned by ServiceRegistry in the full
        # application. This idempotent start preserves standalone service tests
        # and direct embedding without transferring shutdown ownership. We use
        # the public is_running contract rather than reaching into private
        # connection state.
        if not self.experience_store.is_running:
            await self.experience_store.start()

        self._unsubscribers.append(
            self.event_bus.subscribe(
                "feedback.submit",
                self._on_feedback,
            )
        )
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        self.state = ServiceState.STOPPED

    async def _on_feedback(self, event: EventEnvelope) -> None:
        payload = event.payload
        response_id = str(
            payload.get("response_id", "")
        ).strip()
        classification_raw = str(
            payload.get("classification", "")
        ).strip().lower()

        if not response_id or not classification_raw:
            return

        try:
            classification = FeedbackClass(
                classification_raw
            )
            experience = (
                await self.experience_store.get_by_response_id(
                    response_id
                )
            )

            if experience is not None:
                conversation_id = experience.conversation_id
            else:
                response_record = (
                    await self.conversations.get_response(
                        response_id
                    )
                )
                if response_record is None:
                    raise KeyError(
                        f"Response ID {response_id!r} not found "
                        "in experience or conversation store"
                    )
                conversation_id = (
                    response_record.conversation_id
                )

            feedback = FeedbackRecord(
                response_id=response_id,
                conversation_id=conversation_id,
                classification=classification,
                text=payload.get("text"),
                reason=payload.get("reason"),
                metadata=dict(
                    payload.get("metadata", {})
                ),
            )

            if experience is not None:
                await self.experience_store.add_feedback(
                    feedback
                )

            self._received += 1

            await self.memory.write(
                text=(
                    f"Feedback on response {response_id}: "
                    f"{classification.value}"
                    + (
                        f" — {feedback.text}"
                        if feedback.text
                        else ""
                    )
                ),
                type=MemoryType.FEEDBACK,
                source="feedback",
                conversation_id=conversation_id,
                protected=True,
                metadata={
                    "response_id": response_id,
                    "classification": (
                        classification.value
                    ),
                    "reason": feedback.reason,
                },
            )

            await self.event_bus.publish(
                EventEnvelope(
                    event_type="feedback.recorded",
                    source=self.name,
                    correlation_id=event.correlation_id,
                    payload={
                        "feedback_id": feedback.id,
                        "response_id": response_id,
                        "conversation_id": (
                            conversation_id
                        ),
                        "classification": (
                            classification.value
                        ),
                        "text": feedback.text,
                        "reason": feedback.reason,
                    },
                )
            )

        except Exception as exc:
            self._failures += 1
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="feedback.failed",
                    source=self.name,
                    correlation_id=event.correlation_id,
                    payload={
                        "response_id": response_id,
                        "error": str(exc),
                    },
                )
            )

    async def health(self) -> HealthStatus:
        state = (
            ServiceState.DEGRADED
            if (
                self.state == ServiceState.RUNNING
                and self._failures
            )
            else self.state
        )
        return HealthStatus(
            state=state,
            details={
                "received": self._received,
                "failures": self._failures,
                "experience_count": (
                    await self.experience_store.count()
                    if self.state in {
                        ServiceState.RUNNING,
                        ServiceState.DEGRADED,
                    }
                    else 0
                ),
                "db_path": str(
                    self.experience_store.db_path
                ),
            },
        )
