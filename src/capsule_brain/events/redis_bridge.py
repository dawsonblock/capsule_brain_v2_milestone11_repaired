from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterable
from typing import Any

try:
    import redis.asyncio as redis
except ImportError as exc:  # optional cross-process dependency
    raise RuntimeError(
        "RedisBridge requires the 'redis' package. Install redis>=5.0 "
        "or disable redis_bridge."
    ) from exc

from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState
from .local_bus import LocalEventBus
from .models import EventEnvelope

log = logging.getLogger(__name__)


class RedisBridge(CapsuleService):
    """Optional Redis bridge for cross-process events.

    Internal Capsule Brain services should use LocalEventBus directly.
    This bridge mirrors selected topics between Redis and the local event bus.
    """

    name = "redis_bridge"

    def __init__(
        self,
        local_bus: LocalEventBus,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.local_bus = local_bus
        self.redis_url = self.cfg.get("url", "redis://localhost:6379/0")
        self.channel_prefix = self.cfg.get("channel_prefix", "capsule")
        self.inbound_topics: set[str] = set(self.cfg.get("inbound_topics", []))
        self.outbound_topics: set[str] = set(self.cfg.get("outbound_topics", []))
        self._redis: redis.Redis | None = None
        self._pubsub = None
        self._listener_task: asyncio.Task | None = None
        self._unsubscribers: list[Callable[[], None]] = []
        self._messages_in = 0
        self._messages_out = 0
        self._errors = 0

    def _channel(self, topic: str) -> str:
        return f"{self.channel_prefix}:{topic}"

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        await self._redis.ping()

        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        if self.inbound_topics:
            await self._pubsub.subscribe(
                *(self._channel(topic) for topic in sorted(self.inbound_topics))
            )
            self._listener_task = asyncio.create_task(
                self._listen(),
                name="redis-bridge-listener",
            )

        for topic in sorted(self.outbound_topics):
            self._unsubscribers.append(
                self.local_bus.subscribe(
                    topic,
                    self._make_outbound_handler(topic),
                )
            )

        self.state = ServiceState.RUNNING

    def _make_outbound_handler(self, topic: str):
        async def handler(event: EventEnvelope) -> None:
            if self._redis is None:
                return
            try:
                payload = {
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "source": event.source,
                    "event_id": str(event.event_id),
                    "correlation_id": (
                        str(event.correlation_id) if event.correlation_id else None
                    ),
                    "created_at": event.created_at.isoformat(),
                }
                await self._redis.publish(self._channel(topic), json.dumps(payload))
                self._messages_out += 1
            except Exception:
                self._errors += 1
                log.exception("Redis outbound publish failed for %s", topic)

        return handler

    async def _listen(self) -> None:
        assert self._pubsub is not None

        try:
            async for message in self._pubsub.listen():
                if self.state in {ServiceState.STOPPING, ServiceState.STOPPED}:
                    break
                if message.get("type") != "message":
                    continue

                try:
                    channel = message["channel"]
                    prefix = f"{self.channel_prefix}:"
                    if not channel.startswith(prefix):
                        continue
                    topic = channel[len(prefix):]

                    raw = json.loads(message["data"])
                    event = EventEnvelope(
                        event_type=raw.get("event_type", topic),
                        payload=raw.get("payload", {}),
                        source=raw.get("source", "redis"),
                    )
                    await self.local_bus.publish(event)
                    self._messages_in += 1
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._errors += 1
                    log.exception("Failed processing inbound Redis message")
        except asyncio.CancelledError:
            raise
        except Exception:
            self._errors += 1
            self.state = ServiceState.DEGRADED
            log.exception("Redis bridge listener failed")

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING

        for unsubscribe in self._unsubscribers:
            try:
                unsubscribe()
            except Exception:
                log.exception("Redis bridge unsubscribe failed")
        self._unsubscribers.clear()

        if self._listener_task:
            self._listener_task.cancel()
            await asyncio.gather(self._listener_task, return_exceptions=True)
            self._listener_task = None

        if self._pubsub is not None:
            await self._pubsub.close()
            self._pubsub = None

        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

        self.state = ServiceState.STOPPED

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=self.state,
            details={
                "messages_in": self._messages_in,
                "messages_out": self._messages_out,
                "errors": self._errors,
                "inbound_topics": sorted(self.inbound_topics),
                "outbound_topics": sorted(self.outbound_topics),
            },
        )
