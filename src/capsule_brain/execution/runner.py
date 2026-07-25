from __future__ import annotations

import asyncio
import time

from ._bounded import run_bounded
from .models import ExecutionResult, now
from .policy import validate_request


class ExecutionRunner:
    """Restricted verification runner.

    Synchronous subprocess execution is isolated with asyncio.to_thread so it
    cannot block the main event loop. Output is streamed through reader threads
    capped at ``max_output_chars`` so a noisy child cannot exhaust host memory.
    """

    def __init__(self, policy):
        self.policy = policy

    async def run(self, request):
        cwd, _ = validate_request(self.policy, request)
        started_at = now()
        started = time.perf_counter()

        raw = await asyncio.to_thread(
            run_bounded,
            request.command,
            cwd=str(cwd),
            timeout_s=self.policy.timeout_s,
            max_output_chars=self.policy.max_output_chars,
        )
        cap = self.policy.max_output_chars

        def decode(value):
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")[:cap]
            return str(value or "")[:cap]

        return ExecutionResult(
            request_id=request.id,
            command=list(request.command),
            cwd=str(cwd),
            exit_code=raw["exit_code"],
            stdout=decode(raw["stdout"]),
            stderr=decode(raw["stderr"]),
            timed_out=raw["timed_out"],
            duration_ms=(time.perf_counter() - started) * 1000.0,
            started_at=started_at,
            completed_at=now(),
            metadata=dict(request.metadata),
        )
