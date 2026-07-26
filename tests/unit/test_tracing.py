import pytest
from uuid import uuid4

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.observability.tracing import (
    Span,
    Tracer,
    get_default_tracer,
    set_default_tracer,
)


def test_default_tracer_is_noop():
    tracer = Tracer()
    assert not tracer.enabled
    span = tracer.start_span("test")
    assert isinstance(span, Span)
    # No-op spans must not raise on any operation.
    span.set_attribute("k", "v")
    span.add_event("evt", {"x": 1})
    span.record_exception(ValueError("x"))
    span.end()
    # Idempotent end.
    span.end()


def test_span_context_manager_ends_on_exit():
    tracer = Tracer()
    with tracer.span("op") as span:
        span.set_attribute("a", 1)
    assert span._ended


def test_span_records_exception_on_raise():
    tracer = Tracer()
    with pytest.raises(ValueError, match="boom"):
        with tracer.span("op") as span:
            raise ValueError("boom")
    assert span._ended


def test_tracer_stats_count_spans():
    tracer = Tracer()
    tracer.start_span("a").end()
    with tracer.span("b"):
        pass
    stats = tracer.stats()
    assert stats["spans_started"] == 2
    assert stats["enabled"] is False


def test_otel_factory_degrades_gracefully_without_opentelemetry():
    # Even if opentelemetry is not installed, Tracer.otel() must not raise.
    tracer = Tracer.otel(service_name="test")
    assert isinstance(tracer, Tracer)
    # Span creation still works (as no-op).
    span = tracer.start_span("x")
    span.end()


def test_correlation_id_attached_to_span():
    tracer = Tracer()
    cid = uuid4()
    span = tracer.start_span("op", correlation_id=cid)
    assert span.attributes["correlation_id"] == str(cid)
    assert span.attributes["service"] == "capsule-brain"
    span.end()


@pytest.mark.asyncio
async def test_event_bus_publish_creates_span():
    """Publishing an event should create a span with the event's correlation id."""
    bus = LocalEventBus()
    await bus.start()

    captured = []

    class CapturingTracer(Tracer):
        def start_span(self, name, *, correlation_id=None, attributes=None):
            span = super().start_span(name, correlation_id=correlation_id, attributes=attributes)
            captured.append((name, span.attributes))
            return span

    set_default_tracer(CapturingTracer())
    try:
        cid = uuid4()
        await bus.publish(
            EventEnvelope(
                event_type="test.event",
                source="test",
                correlation_id=cid,
                payload={},
            )
        )
        assert any(name.startswith("event.publish") for name, _ in captured)
        # The span attributes should carry the correlation id.
        found = False
        for _, attrs in captured:
            if attrs.get("correlation_id") == str(cid):
                found = True
                break
        assert found
    finally:
        set_default_tracer(Tracer())
        await bus.stop()


@pytest.mark.asyncio
async def test_event_bus_publish_traces_handler_failures():
    bus = LocalEventBus()
    await bus.start()

    def bad_handler(event):
        raise RuntimeError("handler exploded")

    bus.subscribe("test.event", bad_handler)

    class CapturingSpan(Span):
        # Subclass to bypass __slots__ restriction on monkey-patching.
        def add_event(self, name, payload=None):
            tracer.events.append((name, payload))
            super().add_event(name, payload or {})

    class CountingTracer(Tracer):
        events: list

        def __init__(self):
            super().__init__()
            self.events = []

        def start_span(self, name, *, correlation_id=None, attributes=None):
            attrs = dict(attributes or {})
            if correlation_id is not None:
                attrs.setdefault("correlation_id", str(correlation_id))
            attrs.setdefault("service", self.service_name)
            return CapturingSpan(name, attributes=attrs)

    tracer = CountingTracer()
    set_default_tracer(tracer)
    try:
        await bus.publish(
            EventEnvelope(
                event_type="test.event",
                source="test",
                payload={},
            )
        )
        assert any(name == "handler_failed" for name, _ in tracer.events)
    finally:
        set_default_tracer(Tracer())
        await bus.stop()
