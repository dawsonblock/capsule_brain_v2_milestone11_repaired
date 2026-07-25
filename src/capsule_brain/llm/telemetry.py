from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class RequestTrace:
    request_id: str = field(default_factory=lambda: uuid4().hex)
    route: str | None = None
    model_alias: str | None = None
    provider: str | None = None
    attempts: int = 0
    fallback_count: int = 0
    started_at: float = field(default_factory=perf_counter)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        self.latency_ms = (perf_counter() - self.started_at) * 1000.0
