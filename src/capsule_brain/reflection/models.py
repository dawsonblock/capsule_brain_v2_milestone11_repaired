from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ReflectionIteration:
    index: int
    critique: str
    revision: str
    evaluation: str
    resolved: bool = False


@dataclass(slots=True)
class ReflectionSession:
    id: str = field(default_factory=lambda: uuid4().hex)
    seed: str = ""
    source: str = "unknown"
    created_at: str = field(default_factory=utc_now_iso)
    completed_at: str | None = None
    max_iterations: int = 4
    iterations: list[ReflectionIteration] = field(default_factory=list)
    final_text: str | None = None
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
