import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.execution.models import (
    ExecutionRequest,
    ExecutionResult,
)
from capsule_brain.execution.repository import ExecutionRepository
from capsule_brain.execution.service import ExecutionService


class FailingRunner:
    async def run(self, request):
        return ExecutionResult(
            request_id=request.id,
            command=request.command,
            cwd=request.cwd or "",
            exit_code=1,
            stdout="",
            stderr="failure",
            timed_out=False,
            duration_ms=1.0,
            started_at="start",
            completed_at="end",
        )


@pytest.mark.asyncio
async def test_nonzero_exit_increments_failure_metric(tmp_path):
    bus = LocalEventBus()
    await bus.start()

    service = ExecutionService(
        bus,
        {
            "allow": True,
            "cwd_root": str(tmp_path),
            "allowed_commands": ["python"],
        },
        repository=ExecutionRepository(
            str(tmp_path / "execution.sqlite")
        ),
        runner=FailingRunner(),
    )
    await service.start()

    result = await service.execute(
        ExecutionRequest(
            command=["python", "x.py"],
            cwd=str(tmp_path),
        )
    )

    assert result.passed is False
    assert service.failures == 1

    await service.stop()
    await bus.stop()
