from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass(slots=True)
class VerificationCheck:
    verifier: str
    status: VerificationStatus
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    id: str = field(default_factory=lambda: uuid4().hex)
    subject: str = ""
    source: str = "unknown"
    created_at: str = field(default_factory=now)
    completed_at: str | None = None
    status: VerificationStatus = VerificationStatus.SKIP
    checks: list[VerificationCheck] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
