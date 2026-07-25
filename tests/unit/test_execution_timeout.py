import sys
import pytest
from pathlib import Path

from capsule_brain.execution.models import ExecutionPolicy, ExecutionRequest
from capsule_brain.execution.runner import ExecutionRunner

@pytest.mark.asyncio
async def test_execution_timeout_returns_structured_result(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "slow.py").write_text("import time; time.sleep(5)")

    runner = ExecutionRunner(
        ExecutionPolicy(
            allow=True,
            cwd_root=str(root),
            allowed_commands=(Path(sys.executable).name,),
            timeout_s=0.05,
        )
    )

    result = await runner.run(
        ExecutionRequest(
            command=[sys.executable, "slow.py"],
            cwd=str(root),
        )
    )

    assert result.timed_out is True
    assert result.passed is False
    assert result.exit_code is None
