from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class UsageTotals:
    requests: int = 0
    failures: int = 0
    fallbacks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageTracker:
    def __init__(self) -> None:
        self.totals = UsageTotals()

    def record_success(
        self,
        *,
        usage: dict[str, int],
        model_cfg: dict[str, Any],
        fallback_count: int,
    ) -> None:
        self.totals.requests += 1
        self.totals.fallbacks += fallback_count

        input_tokens = int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        output_tokens = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )

        self.totals.input_tokens += input_tokens
        self.totals.output_tokens += output_tokens

        in_rate = float(model_cfg.get("input_cost_per_million", 0.0))
        out_rate = float(model_cfg.get("output_cost_per_million", 0.0))
        self.totals.estimated_cost_usd += (
            input_tokens / 1_000_000.0 * in_rate
            + output_tokens / 1_000_000.0 * out_rate
        )

    def record_failure(self) -> None:
        self.totals.requests += 1
        self.totals.failures += 1
