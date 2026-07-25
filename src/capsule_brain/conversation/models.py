from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TurnRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(slots=True)
class Conversation:
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Turn:
    id: str = field(default_factory=lambda: uuid4().hex)
    conversation_id: str = ""
    role: TurnRole = TurnRole.USER
    text: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    parent_turn_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssistantResponse:
    id: str = field(default_factory=lambda: uuid4().hex)
    conversation_id: str = ""
    parent_turn_id: str = ""
    turn_id: str = ""
    text: str = ""
    model_alias: str | None = None
    provider: str | None = None
    model_name: str | None = None
    latency_ms: float | None = None
    attempts: int | None = None
    usage: dict[str, int] = field(default_factory=dict)
    route: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
