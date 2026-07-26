from __future__ import annotations

from typing import Any

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest
from capsule_brain.memory.models import MemoryType
from capsule_brain.memory.service import MemoryService
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .models import AssistantResponse, Turn, TurnRole
from .repository import ConversationRepository


class ConversationService(CapsuleService):
    name = "conversation"

    def __init__(
        self,
        event_bus: LocalEventBus,
        memory: MemoryService,
        llm: LLMGateway,
        cfg: dict[str, Any] | None = None,
        repository: ConversationRepository | None = None,
        experience_store=None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.memory = memory
        self.llm = llm
        self.experience_store = experience_store
        self.repository = repository or ConversationRepository(
            self.cfg.get("db_path", "data/conversations_v2.sqlite")
        )
        self.route = self.cfg.get("route", "chat")
        self.model = self.cfg.get("model")
        self.history_limit = int(self.cfg.get("history_limit", 12))
        self.memory_limit = int(self.cfg.get("memory_limit", 8))
        # Semantic memory retrieval. When enabled, the conversation service
        # embeds the user's message and retrieves the top-K semantically
        # relevant memories instead of the most recent K. Falls back to
        # chronological retrieval if semantic search is unavailable.
        self.semantic_memory = bool(self.cfg.get("semantic_memory", False))
        self.semantic_memory_limit = int(
            self.cfg.get("semantic_memory_limit", self.memory_limit)
        )
        self.system_prompt = self.cfg.get(
            "system_prompt",
            "You are Capsule Brain, a precise autonomous AI assistant.",
        )
        self._unsubscribers: list = []
        self._requests = 0
        self._failures = 0

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        await self.repository.start()
        if self.experience_store is not None:
            await self.experience_store.start()
        self._unsubscribers.append(
            self.event_bus.subscribe(
                "conversation.message.create",
                self._on_message_create,
            )
        )
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        await self.repository.stop()
        self.state = ServiceState.STOPPED

    async def _on_message_create(self, event: EventEnvelope) -> None:
        payload = event.payload
        text = str(payload.get("text", "")).strip()
        if not text:
            return

        conversation = await self.repository.ensure_conversation(
            payload.get("conversation_id")
        )
        try:
            await self.respond(
                conversation_id=conversation.id,
                text=text,
                correlation_id=str(event.correlation_id)
                if event.correlation_id else None,
            )
        except Exception as exc:
            self._failures += 1
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="conversation.response.failed",
                    source=self.name,
                    correlation_id=event.correlation_id,
                    payload={
                        "conversation_id": conversation.id,
                        "error": str(exc),
                    },
                )
            )

    async def respond(
        self,
        *,
        conversation_id: str,
        text: str,
        correlation_id: str | None = None,
    ) -> AssistantResponse:
        self._requests += 1
        conversation = await self.repository.ensure_conversation(conversation_id)

        user_turn = Turn(
            conversation_id=conversation.id,
            role=TurnRole.USER,
            text=text,
        )
        await self.repository.create_turn(user_turn)

        await self.memory.write(
            text,
            type=MemoryType.OPERATOR,
            source="operator",
            conversation_id=conversation.id,
            turn_id=user_turn.id,
            protected=True,
            metadata={"conversation_role": "user"},
        )

        history = await self.repository.recent_turns(
            conversation.id,
            limit=self.history_limit,
        )
        memory_records = await self._retrieve_memories(text)

        context = self._build_context(history, memory_records)

        result = await self.llm.generate(
            LLMRequest(
                prompt=context,
                system=self.system_prompt,
                model=self.model,
                metadata={
                    "conversation_id": conversation.id,
                    "parent_turn_id": user_turn.id,
                },
            ),
            route=self.route,
        )

        assistant_turn = Turn(
            conversation_id=conversation.id,
            role=TurnRole.ASSISTANT,
            text=result.text,
            parent_turn_id=user_turn.id,
        )
        await self.repository.create_turn(assistant_turn)

        trace = result.trace
        model_alias = trace.model_alias if trace else self.model
        route_name = trace.route if trace else self.route

        response = AssistantResponse(
            conversation_id=conversation.id,
            parent_turn_id=user_turn.id,
            turn_id=assistant_turn.id,
            text=result.text,
            model_alias=model_alias,
            provider=result.provider,
            model_name=result.model,
            latency_ms=result.latency_ms,
            attempts=result.attempts,
            usage=dict(result.usage),
            route=route_name,
        )
        await self.repository.create_response(response)

        if self.experience_store is not None:
            from capsule_brain.learning.models import ExperienceRecord
            await self.experience_store.create_experience(
                ExperienceRecord(
                    response_id=response.id,
                    conversation_id=response.conversation_id,
                    user_turn_id=response.parent_turn_id,
                    assistant_turn_id=response.turn_id,
                    prompt_text=text,
                    response_text=response.text,
                    model_alias=response.model_alias,
                    provider=response.provider,
                    model_name=response.model_name,
                    route=response.route,
                    latency_ms=response.latency_ms,
                    attempts=response.attempts,
                    usage=dict(response.usage),
                )
            )

        await self.memory.write(
            result.text,
            type=MemoryType.ASSISTANT,
            source="assistant",
            conversation_id=conversation.id,
            turn_id=assistant_turn.id,
            metadata={
                "response_id": response.id,
                "parent_turn_id": user_turn.id,
                "model_alias": response.model_alias,
                "provider": response.provider,
                "model_name": response.model_name,
                "route": response.route,
            },
        )

        await self.event_bus.publish(
            EventEnvelope(
                event_type="conversation.response.created",
                source=self.name,
                payload={
                    "response_id": response.id,
                    "conversation_id": response.conversation_id,
                    "parent_turn_id": response.parent_turn_id,
                    "turn_id": response.turn_id,
                    "text": response.text,
                    "model_alias": response.model_alias,
                    "provider": response.provider,
                    "model_name": response.model_name,
                    "latency_ms": response.latency_ms,
                    "attempts": response.attempts,
                    "usage": response.usage,
                    "route": response.route,
                },
            )
        )

        # Compatibility event for GUI layers that only need text.
        await self.event_bus.publish(
            EventEnvelope(
                event_type="gui.response",
                source=self.name,
                payload={
                    "conversation_id": response.conversation_id,
                    "response_id": response.id,
                    "text": response.text,
                },
            )
        )

        return response

    async def _retrieve_memories(self, user_text: str):
        """Retrieve memories relevant to ``user_text``.

        When semantic_memory is enabled and the memory service has a working
        embedding provider, embeds the user message and retrieves the top-K
        semantically similar memories. Falls back to chronological recent()
        retrieval on any error so a transient embedding failure never breaks
        the conversation path.
        """
        if self.semantic_memory:
            try:
                return await self.memory.search_semantic(
                    user_text,
                    limit=self.semantic_memory_limit,
                    include_archived=False,
                )
            except Exception:
                pass
        return await self.memory.recent(
            limit=self.memory_limit,
            include_archived=False,
        )

    def _build_context(self, history, memories) -> str:
        parts = ["Conversation history:"]
        for turn in history:
            parts.append(f"{turn.role.value.upper()}: {turn.text}")

        if memories:
            parts.append("\nRelevant recent memory:")
            seen: set[str] = set()
            for memory in memories:
                # Skip OPERATOR and ASSISTANT turn memories — their text
                # already appears in the Conversation history section above.
                # Including them here would duplicate context and waste tokens.
                if memory.type in {MemoryType.OPERATOR, MemoryType.ASSISTANT}:
                    continue
                key = memory.text.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                parts.append(f"- [{memory.type.value}] {memory.text}")

        parts.append("\nRespond to the latest USER message.")
        return "\n".join(parts)

    async def health(self) -> HealthStatus:
        state = (
            ServiceState.DEGRADED
            if self.state == ServiceState.RUNNING and self._failures
            else self.state
        )
        return HealthStatus(
            state=state,
            details={
                "requests": self._requests,
                "failures": self._failures,
                "db_path": str(self.repository.db_path),
                "route": self.route,
                "model": self.model,
            },
        )
