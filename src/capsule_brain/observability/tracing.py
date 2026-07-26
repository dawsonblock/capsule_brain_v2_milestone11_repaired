from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator
from uuid import UUID

log = logging.getLogger(__name__)


class Span:
    """A tracing span.

    This is a thin abstraction over OpenTelemetry spans so the rest of the
    codebase can be instrumented without a hard dependency on the
    ``opentelemetry`` package. When OTel is not configured, spans are no-ops
    whose only cost is the context-manager entry/exit.
    """

    __slots__ = ("name", "attributes", "_otel_span", "_ended")

    def __init__(
        self,
        name: str,
        *,
        otel_span: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self._otel_span = otel_span
        self.attributes: dict[str, Any] = dict(attributes or {})
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value
        if self._otel_span is not None:
            try:
                self._otel_span.set_attribute(key, value)
            except Exception:
                pass

    def add_event(self, name: str, payload: dict[str, Any] | None = None) -> None:
        if self._otel_span is not None:
            try:
                self._otel_span.add_event(name, payload or {})
            except Exception:
                pass

    def record_exception(self, exc: BaseException) -> None:
        if self._otel_span is not None:
            try:
                self._otel_span.record_exception(exc)
            except Exception:
                pass

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        if self._otel_span is not None:
            try:
                self._otel_span.end()
            except Exception:
                pass

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None and exc is not None:
            self.record_exception(exc)
        self.end()


class Tracer:
    """Tracing facade.

    The default tracer produces no-op spans. ``Tracer.otel()`` returns a
    tracer backed by OpenTelemetry if the package is installed and a tracer
    provider has been configured; otherwise it falls back to no-op.

    Correlation IDs from EventEnvelope are propagated as span attributes so a
    complete trace waterfall (Operator Event -> Memory Lookup -> LLM Route ->
    Container Execution -> Verification -> Reflection) can be reconstructed in
    Jaeger, Phoenix, or any OTel-compatible backend.
    """

    def __init__(self, *, service_name: str = "capsule-brain") -> None:
        self.service_name = service_name
        self._otel_tracer: Any = None
        self._enabled = False
        self._spans_started = 0

    @classmethod
    def otel(
        cls,
        *,
        service_name: str = "capsule-brain",
        provider: Any = None,
    ) -> "Tracer":
        """Build a tracer backed by OpenTelemetry.

        If ``provider`` is supplied it is used directly; otherwise we attempt
        to fetch the global tracer provider. When neither OTel is installed
        nor a provider is configured, the returned tracer is a no-op — this
        method never raises so callers can wire it unconditionally.
        """
        tracer = cls(service_name=service_name)
        try:
            if provider is None:
                from opentelemetry import trace  # type: ignore[import-not-found]

                provider = trace.get_tracer_provider()
            # If the provider is the default no-op provider, get_tracer returns
            # a no-op tracer, which is fine — spans will be cheap no-ops.
            tracer._otel_tracer = provider.get_tracer(service_name)
            tracer._enabled = True
        except Exception:
            # opentelemetry not installed — silently degrade to no-op.
            log.debug("OpenTelemetry not available; tracing is no-op")
        return tracer

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_span(
        self,
        name: str,
        *,
        correlation_id: UUID | str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span.

        ``correlation_id`` is attached as ``correlation_id`` so spans from
        different services handling the same operator event group into a
        single trace.
        """
        self._spans_started += 1
        attrs = dict(attributes or {})
        if correlation_id is not None:
            attrs.setdefault("correlation_id", str(correlation_id))
        attrs.setdefault("service", self.service_name)

        if self._otel_tracer is None:
            return Span(name, attributes=attrs)

        try:
            otel_span = self._otel_tracer.start_span(name)
        except Exception:
            return Span(name, attributes=attrs)
        span = Span(name, otel_span=otel_span, attributes=attrs)
        for key, value in attrs.items():
            try:
                otel_span.set_attribute(key, value)
            except Exception:
                pass
        return span

    @contextmanager
    def span(
        self,
        name: str,
        *,
        correlation_id: UUID | str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Context-managed span that ends on exit."""
        s = self.start_span(name, correlation_id=correlation_id, attributes=attributes)
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            raise
        finally:
            s.end()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "service_name": self.service_name,
            "spans_started": self._spans_started,
        }


# A process-wide default tracer. Services that want tracing should call
# ``set_default_tracer`` during bootstrap (e.g. with Tracer.otel()) and then
# use ``get_default_tracer()`` to obtain it. The default is a no-op tracer so
# the system runs fine without any OTel configuration.
_default_tracer = Tracer()


def get_default_tracer() -> Tracer:
    return _default_tracer


def set_default_tracer(tracer: Tracer) -> None:
    global _default_tracer
    _default_tracer = tracer
