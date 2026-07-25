from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .service import CapsuleService, HealthStatus, ServiceState

log = logging.getLogger(__name__)


class ServiceRegistry:
    """Dependency-aware service registry with deterministic startup/shutdown."""

    def __init__(self) -> None:
        self._services: dict[str, CapsuleService] = {}
        self._requires: dict[str, set[str]] = defaultdict(set)
        self._start_order: list[str] = []

    def register(
        self,
        service: CapsuleService,
        *,
        requires: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        name = service.name
        if not name or name == "unnamed":
            raise ValueError("Every service must define a stable non-empty name")
        if name in self._services:
            raise ValueError(f"Service already registered: {name}")

        self._services[name] = service
        self._requires[name] = set(requires or [])

    def get(self, name: str) -> CapsuleService | None:
        return self._services.get(name)

    def require(self, name: str) -> CapsuleService:
        service = self.get(name)
        if service is None:
            raise KeyError(f"Required service not found: {name}")
        return service

    def _resolve_start_order(self) -> list[str]:
        missing: dict[str, list[str]] = {}
        for name, deps in self._requires.items():
            absent = sorted(dep for dep in deps if dep not in self._services)
            if absent:
                missing[name] = absent
        if missing:
            raise RuntimeError(f"Missing service dependencies: {missing}")

        permanent: set[str] = set()
        temporary: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise RuntimeError(f"Service dependency cycle detected at {name!r}")

            temporary.add(name)
            for dep in sorted(self._requires[name]):
                visit(dep)
            temporary.remove(name)
            permanent.add(name)
            order.append(name)

        for name in sorted(self._services):
            visit(name)

        return order

    async def start_all(self) -> None:
        order = self._resolve_start_order()
        started: list[str] = []

        try:
            for name in order:
                service = self._services[name]
                service.state = ServiceState.STARTING
                log.info("Starting service: %s", name)
                await service.start()
                if service.state == ServiceState.STARTING:
                    service.state = ServiceState.RUNNING
                started.append(name)
        except Exception:
            log.exception("Service startup failed; rolling back")
            for started_name in reversed(started):
                try:
                    await self._services[started_name].stop()
                except Exception:
                    log.exception("Rollback stop failed for %s", started_name)
            raise

        self._start_order = started

    async def stop_all(self) -> None:
        for name in reversed(self._start_order):
            service = self._services[name]
            try:
                service.state = ServiceState.STOPPING
                log.info("Stopping service: %s", name)
                await service.stop()
                if service.state == ServiceState.STOPPING:
                    service.state = ServiceState.STOPPED
            except Exception:
                service.state = ServiceState.FAILED
                log.exception("Service stop failed: %s", name)

        self._start_order.clear()

    async def health(self) -> dict[str, HealthStatus]:
        output: dict[str, HealthStatus] = {}
        for name, service in self._services.items():
            try:
                output[name] = await service.health()
            except Exception as exc:
                output[name] = HealthStatus(
                    state=ServiceState.FAILED,
                    message=str(exc),
                )
        return output

    def names(self) -> tuple[str, ...]:
        return tuple(self._services)
