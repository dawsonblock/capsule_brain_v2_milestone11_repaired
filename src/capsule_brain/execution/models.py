from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ExecutionPolicy:
    allow: bool = False
    timeout_s: float = 10.0
    max_output_chars: int = 20000
    allowed_commands: tuple[str, ...] = ("python", "pytest")
    cwd_root: str = "sandbox"


@dataclass(slots=True)
class ExecutionRequest:
    id: str = field(default_factory=lambda: uuid4().hex)
    command: list[str] = field(default_factory=list)
    cwd: str | None = None
    source: str = "verification"
    created_at: str = field(default_factory=now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    request_id: str
    command: list[str]
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float
    started_at: str
    completed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out
