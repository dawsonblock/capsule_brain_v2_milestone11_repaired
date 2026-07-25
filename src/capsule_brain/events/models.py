from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4


@dataclass(slots=True)
class EventEnvelope:
    event_type: str
    payload: dict[str, Any]
    source: str
    event_id: UUID = field(default_factory=uuid4)
    correlation_id: UUID | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
