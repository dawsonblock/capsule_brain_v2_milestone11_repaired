from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from capsule_brain.llm.telemetry import RequestTrace


class LLMCapability(str, Enum):
    TEXT = "text"
    JSON = "json"
    TOOLS = "tools"
    VISION = "vision"


@dataclass(slots=True, frozen=True)
class LLMRequest:
    prompt: str
    system: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LLMResult:
    text: str
    model: str
    provider: str
    latency_ms: float
    attempts: int = 1
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    # Authoritative per-request trace. Travels with the result so concurrent
    # callers cannot overwrite each other's provenance via shared gateway state.
    trace: "RequestTrace | None" = None
