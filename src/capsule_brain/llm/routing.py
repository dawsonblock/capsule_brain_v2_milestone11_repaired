from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class RoutePlan:
    name: str
    models: tuple[str, ...]


class ModelRouter:
    """Rule-based router with deterministic fallback order."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        self.routes: dict[str, RoutePlan] = {}

        for name, route_cfg in cfg.get("routes", {}).items():
            models = tuple(route_cfg.get("models", ()))
            if models:
                self.routes[name] = RoutePlan(name=name, models=models)

    def resolve(
        self,
        route: str | None,
        explicit_model: str | None,
        default_model: str,
    ) -> RoutePlan:
        if explicit_model:
            return RoutePlan(name="explicit", models=(explicit_model,))

        if route and route in self.routes:
            return self.routes[route]

        return RoutePlan(name="default", models=(default_model,))
