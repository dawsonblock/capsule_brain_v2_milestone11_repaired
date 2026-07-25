from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_timeout_s: float = 30.0


class CircuitBreaker:
    """Simple per-model circuit breaker."""

    def __init__(self, cfg: CircuitBreakerConfig | None = None) -> None:
        self.cfg = cfg or CircuitBreakerConfig()
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.cfg.recovery_timeout_s:
            self.opened_at = None
            self.failures = 0
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.cfg.failure_threshold:
            self.opened_at = time.monotonic()
