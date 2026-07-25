from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ServiceState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class HealthStatus:
    state: ServiceState
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}


class CapsuleService(ABC):
    """Base lifecycle contract for all Capsule Brain v2 services."""

    name: str = "unnamed"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg: dict[str, Any] = dict(cfg or {})
        self.state = ServiceState.CREATED

    @property
    def is_running(self) -> bool:
        """True when the service has been started and not yet stopped.

        DEGRADED is included because a degraded service is still live and
        its dependencies may still be relied upon (with caution). This is
        the public, stable way for one service to check whether a peer it
        embeds has been started, instead of reaching into private state.
        """
        return self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}

    async def start(self) -> None:
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPED

    async def reload(self, cfg: dict[str, Any]) -> None:
        self.cfg.update(cfg)

    async def health(self) -> HealthStatus:
        return HealthStatus(self.state)
