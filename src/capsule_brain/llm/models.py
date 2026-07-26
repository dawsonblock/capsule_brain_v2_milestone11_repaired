from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from capsule_brain.llm.telemetry import RequestTrace
    from capsule_brain.llm.tools import ToolCall, ToolSpec


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
    # Tool/function calling. When ``tools`` is non-empty, the provider is
    # expected to send the tool schemas to the model and parse any tool_calls
    # in the response into ``LLMResult.tool_calls``.
    tools: list["ToolSpec"] = field(default_factory=list)
    # Prior tool results to feed back into a multi-turn tool-calling loop.
    # Each entry is a dict with ``tool_call_id``, ``name``, and ``content``.
    tool_results: list[dict[str, Any]] = field(default_factory=list)


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
    # Tool calls requested by the model. Empty when the model produced a
    # plain text answer or when no tools were supplied in the request.
    tool_calls: list["ToolCall"] = field(default_factory=list)
    # The finish reason as reported by the provider (e.g. "stop",
    # "tool_calls"). Useful for the gateway's tool-calling loop.
    finish_reason: str | None = None
