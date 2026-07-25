from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class MemoryType(str, Enum):
    OPERATOR = "operator"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    GOAL = "goal"
    FEEDBACK = "feedback"
    DREAM = "dream"
    EXTERNAL = "external_knowledge"
    SYSTEM = "system"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryRecord:
    id: str = field(default_factory=lambda: uuid4().hex)
    text: str = ""
    type: MemoryType = MemoryType.OBSERVATION
    source: str = "unknown"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    tags: list[str] = field(default_factory=list)
    conversation_id: str | None = None
    turn_id: str | None = None
    importance: float = 0.0
    confidence: float | None = None
    archived: bool = False
    protected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
