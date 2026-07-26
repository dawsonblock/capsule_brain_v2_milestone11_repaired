from __future__ import annotations

import asyncio
import logging
from typing import Any

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.memory.models import MemoryType
from capsule_brain.memory.service import MemoryService
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .models import ReflectionIteration, ReflectionSession, utc_now_iso
from .repository import ReflectionRepository
from .strategies import (
    DefaultReflectionStrategy,
    ReflectionStrategy,
    StrategyContext,
    select_strategy,
)

log = logging.getLogger(__name__)


class ReflectionService(CapsuleService):
    name = "reflection"

    def __init__(
        self,
        event_bus: LocalEventBus,
        llm: LLMGateway,
        memory: MemoryService,
        cfg: dict[str, Any] | None = None,
        repository: ReflectionRepository | None = None,
        experience_store: Any | None = None,
        conversation_repository: Any | None = None,
        verification_repository: Any | None = None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.llm = llm
        self.memory = memory
        self.repository = repository or ReflectionRepository(
            self.cfg.get("db_path", "data/reflections_v2.sqlite")
        )
        # Context resolvers for building rich reflection seeds. When available,
        # reflection gets the actual failed artifact/response, not just IDs.
        self.experience_store = experience_store
        self.conversation_repository = conversation_repository
        self.verification_repository = verification_repository
        self.model = self.cfg.get("model")
        self.route = self.cfg.get("route", "reflection")
        self.max_iterations = max(1, int(self.cfg.get("max_iterations", 4)))
        self.max_seed_chars = max(100, int(self.cfg.get("max_seed_chars", 4000)))
        # Session-level wall-clock budget. Each iteration issues 3 LLM calls
        # (critique, revise, evaluate); without a cap, degraded providers can
        # block for minutes per session.
        self.session_timeout_s = float(self.cfg.get("session_timeout_s", 60.0))
        # Adaptive reflection strategies. The default registry covers the
        # built-in strategies; callers can override via cfg["strategies"] or
        # by passing a custom dict. Strategy selection happens per-reflection
        # based on the trigger source/metadata.
        self._strategies: dict[str, ReflectionStrategy] = self._build_strategies()
        self._unsubscribers: list = []
        self._runs = 0
        self._failures = 0
        # Per-strategy invocation counts for health/metrics.
        self._strategy_runs: dict[str, int] = {}

    def _build_strategies(self) -> dict[str, ReflectionStrategy]:
        from .strategies import (
            CodeSyntaxStrategy,
            OperatorFeedbackStrategy,
            PytestFailureStrategy,
        )

        return {
            "default": DefaultReflectionStrategy(),
            "code_syntax": CodeSyntaxStrategy(),
            "pytest_failure": PytestFailureStrategy(),
            "operator_feedback": OperatorFeedbackStrategy(),
        }

    def register_strategy(self, name: str, strategy: ReflectionStrategy) -> None:
        self._strategies[name] = strategy

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        await self.repository.start()
        self._unsubscribers.extend([
            self.event_bus.subscribe("reflection.request", self._on_request),
            self.event_bus.subscribe("feedback.recorded", self._on_feedback),
            self.event_bus.subscribe("verification.failed", self._on_verification_failed),
            self.event_bus.subscribe("goal.unresolved", self._on_goal_unresolved),
        ])
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        await self.repository.stop()
        self.state = ServiceState.STOPPED

    async def _on_request(self, event: EventEnvelope) -> None:
        seed = str(event.payload.get("seed", "")).strip()
        if seed:
            await self._safe_run(seed, "operator", dict(event.payload))

    async def _on_feedback(self, event: EventEnvelope) -> None:
        classification = str(event.payload.get("classification", "")).lower()
        if classification not in {"negative", "correction"}:
            return
        text = str(event.payload.get("text") or event.payload.get("reason") or "").strip()
        response_id = str(event.payload.get("response_id", ""))

        # Resolve the actual response text so the model can reason about the
        # real artifact, not an opaque ID.
        response_text = await self._resolve_response_text(response_id)
        user_message = await self._resolve_user_message(response_id)

        parts = []
        if user_message:
            parts.append(f"USER:\n{user_message}")
        if response_text:
            parts.append(f"ASSISTANT RESPONSE:\n{response_text}")
        parts.append(f"OPERATOR FEEDBACK:\n{classification}: {text}")
        parts.append(
            "TASK: Identify why the response failed and produce a corrected response."
        )
        seed = "\n\n".join(parts)
        await self._safe_run(seed, "feedback", dict(event.payload))

    async def _on_verification_failed(self, event: EventEnvelope) -> None:
        # Build a rich seed with the actual failed artifact, not just the
        # summary. The subject (code/text) is the most important information
        # for debugging.
        summary = str(event.payload.get("summary", "")).strip()
        subject = str(event.payload.get("subject", "")).strip()
        source = str(event.payload.get("source", "")).strip()
        metadata = event.payload.get("metadata") or {}

        parts = []
        if summary:
            parts.append(f"VERIFIER FAILURE:\n{summary}")
        if subject:
            parts.append(f"ARTIFACT:\n{subject}")
        if metadata:
            parts.append(f"CHECK DETAILS:\n{metadata}")
        parts.append(
            "TASK: Analyze the failure and produce a corrected version of the artifact."
        )
        seed = "\n\n".join(parts) if parts else f"Analyze verifier failure: {event.payload}"
        await self._safe_run(seed, "verification", dict(event.payload))

    async def _on_goal_unresolved(self, event: EventEnvelope) -> None:
        seed = str(event.payload.get("goal", "")).strip()
        if not seed:
            seed = f"Analyze unresolved goal: {event.payload}"
        await self._safe_run(seed, "goal", dict(event.payload))

    async def _resolve_response_text(self, response_id: str) -> str | None:
        """Resolve a response_id to the actual assistant response text."""
        if not response_id or not response_id.strip():
            return None
        # Try ExperienceStore first (it has the full response + provenance)
        if self.experience_store is not None:
            try:
                record = await self.experience_store.get_by_response_id(response_id)
                if record is not None:
                    return record.response_text
            except Exception:
                pass
        # Fall back to ConversationRepository
        if self.conversation_repository is not None:
            try:
                response = await self.conversation_repository.get_response(response_id)
                if response is not None:
                    return response.text
            except Exception:
                pass
        return None

    async def _resolve_user_message(self, response_id: str) -> str | None:
        """Resolve the user message that preceded a given response_id."""
        if not response_id or not response_id.strip():
            return None
        if self.experience_store is not None:
            try:
                record = await self.experience_store.get_by_response_id(response_id)
                if record is not None:
                    return record.prompt_text
            except Exception:
                pass
        if self.conversation_repository is not None:
            try:
                response = await self.conversation_repository.get_response(response_id)
                if response is not None:
                    turn = await self.conversation_repository.get_turn(
                        response.parent_turn_id
                    )
                    if turn is not None:
                        return turn.text
            except Exception:
                pass
        return None

    async def _safe_run(self, seed: str, source: str, metadata: dict[str, Any]) -> None:
        try:
            await self.reflect(seed=seed, source=source, metadata=metadata)
        except Exception as exc:
            self._failures += 1
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="reflection.failed",
                    source=self.name,
                    payload={
                        "seed": seed[:300],
                        "source": source,
                        "error": str(exc),
                    },
                )
            )

    async def reflect(
        self,
        *,
        seed: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> ReflectionSession:
        self._runs += 1
        seed = seed[:self.max_seed_chars]
        session = ReflectionSession(
            seed=seed,
            source=source,
            max_iterations=self.max_iterations,
            metadata=dict(metadata or {}),
        )
        await self.repository.save(session)

        # Select an adaptive strategy based on the trigger source/metadata.
        # Falls back to the default critique->revise->evaluate loop when no
        # specialized strategy matches.
        strategy = select_strategy(
            source,
            dict(metadata or {}),
            strategies=self._strategies,
        )
        self._strategy_runs[strategy.name] = self._strategy_runs.get(strategy.name, 0) + 1
        ctx = StrategyContext(
            seed=seed,
            source=source,
            metadata=dict(metadata or {}),
            session=session,
        )

        current = seed

        try:
            async with asyncio.timeout(self.session_timeout_s):
                iterations, final_text, stop_reason = await strategy.run(
                    self.llm,
                    ctx,
                    max_iterations=self.max_iterations,
                    route=self.route,
                    model=self.model,
                )
                session.iterations = iterations
                session.final_text = final_text
                session.stop_reason = stop_reason
        except TimeoutError:
            # Session-level wall-clock budget exhausted. Preserve the latest
            # revision so partial work is not lost.
            session.final_text = current
            session.stop_reason = "session_timeout"
            log.warning(
                "Reflection session %s timed out after %.1fs",
                session.id,
                self.session_timeout_s,
            )

        session.completed_at = utc_now_iso()
        await self.repository.save(session)

        await self.memory.write(
            text=session.final_text or session.seed,
            type=MemoryType.REFLECTION,
            source="reflection",
            tags=["reflection", session.source, f"strategy:{strategy.name}"],
            metadata={
                "reflection_session_id": session.id,
                "stop_reason": session.stop_reason,
                "iterations": len(session.iterations),
                "strategy": strategy.name,
            },
        )

        await self.event_bus.publish(
            EventEnvelope(
                event_type="reflection.completed",
                source=self.name,
                payload={
                    "session_id": session.id,
                    "source": session.source,
                    "seed": session.seed,
                    "final_text": session.final_text,
                    "stop_reason": session.stop_reason,
                    "iterations": len(session.iterations),
                    "strategy": strategy.name,
                },
            )
        )

        return session

    async def health(self) -> HealthStatus:
        state = (
            ServiceState.DEGRADED
            if self.state == ServiceState.RUNNING and self._failures
            else self.state
        )
        return HealthStatus(
            state=state,
            details={
                "runs": self._runs,
                "failures": self._failures,
                "session_count": await self.repository.count()
                if self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}
                else 0,
                "max_iterations": self.max_iterations,
                "db_path": str(self.repository.db_path),
                "strategies": sorted(self._strategies),
                "strategy_runs": dict(self._strategy_runs),
            },
        )
