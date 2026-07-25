from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeedbackClass(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"


@dataclass(slots=True)
class FeedbackRecord:
    id: str = field(default_factory=lambda: uuid4().hex)
    response_id: str = ""
    conversation_id: str = ""
    classification: FeedbackClass = FeedbackClass.NEUTRAL
    text: str | None = None
    reason: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperienceRecord:
    id: str = field(default_factory=lambda: uuid4().hex)
    response_id: str = ""
    conversation_id: str = ""
    user_turn_id: str = ""
    assistant_turn_id: str = ""
    prompt_text: str = ""
    response_text: str = ""
    model_alias: str | None = None
    provider: str | None = None
    model_name: str | None = None
    route: str | None = None
    latency_ms: float | None = None
    attempts: int | None = None
    usage: dict[str, int] = field(default_factory=dict)
    feedback_classification: str | None = None
    feedback_text: str | None = None
    feedback_reason: str | None = None
    verifier: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
