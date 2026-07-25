from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProviderHealth:
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None

    @property
    def degraded(self) -> bool:
        return self.consecutive_failures > 0


class ProviderHealthRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ProviderHealth] = {}

    def get(self, name: str) -> ProviderHealth:
        return self._items.setdefault(name, ProviderHealth())

    def success(self, name: str) -> None:
        health = self.get(name)
        health.successes += 1
        health.consecutive_failures = 0
        health.last_error = None

    def failure(self, name: str, error: Exception) -> None:
        health = self.get(name)
        health.failures += 1
        health.consecutive_failures += 1
        health.last_error = str(error)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "successes": item.successes,
                "failures": item.failures,
                "consecutive_failures": item.consecutive_failures,
                "last_error": item.last_error,
                "degraded": item.degraded,
            }
            for name, item in self._items.items()
        }
