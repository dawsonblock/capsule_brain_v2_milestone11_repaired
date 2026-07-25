import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope


@pytest.mark.asyncio
async def test_sync_and_async_handlers_execute():
    bus = LocalEventBus()
    await bus.start()
    seen = []

    def sync_handler(event):
        seen.append(("sync", event.payload["value"]))

    async def async_handler(event):
        seen.append(("async", event.payload["value"]))

    bus.subscribe("test.event", sync_handler)
    bus.subscribe("test.event", async_handler)

    await bus.publish(
        EventEnvelope(
            event_type="test.event",
            payload={"value": 42},
            source="test",
        )
    )

    assert seen == [("sync", 42), ("async", 42)]
    await bus.stop()


@pytest.mark.asyncio
async def test_handler_failure_does_not_block_other_handlers():
    bus = LocalEventBus()
    await bus.start()
    seen = []

    def bad_handler(event):
        raise RuntimeError("boom")

    async def good_handler(event):
        seen.append(event.payload["value"])

    bus.subscribe("test.event", bad_handler)
    bus.subscribe("test.event", good_handler)

    await bus.publish(
        EventEnvelope(
            event_type="test.event",
            payload={"value": 7},
            source="test",
        )
    )

    assert seen == [7]
    health = await bus.health()
    assert health.details["handler_failures"] == 1
    await bus.stop()


@pytest.mark.asyncio
async def test_duplicate_subscription_is_deduped():
    bus = LocalEventBus()
    await bus.start()
    seen = []

    async def handler(event):
        seen.append(event.payload["value"])

    unsub_a = bus.subscribe("test.event", handler)
    unsub_b = bus.subscribe("test.event", handler)

    await bus.publish(
        EventEnvelope(
            event_type="test.event",
            payload={"value": 1},
            source="test",
        )
    )
    # Deduped: handler runs once even though subscribe was called twice.
    assert seen == [1]

    # Unsubscribing via either handle removes the single registration; the
    # other becomes a no-op rather than corrupting the registry.
    unsub_a()
    unsub_b()

    await bus.publish(
        EventEnvelope(
            event_type="test.event",
            payload={"value": 2},
            source="test",
        )
    )
    assert seen == [1]
    await bus.stop()
