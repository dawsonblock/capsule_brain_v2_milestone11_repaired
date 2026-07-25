from capsule_brain.execution.container_runner import (
    ContainerExecutionRunner,
    _is_digest_pinned,
)
from capsule_brain.execution.models import ExecutionPolicy


def test_container_runner_security_defaults():
    runner = ContainerExecutionRunner(
        ExecutionPolicy(
            allow=True,
            allowed_commands=("python",),
        )
    )

    assert runner.memory == "512m"
    # Swap defaults to memory so --memory cannot be bypassed via swap.
    assert runner.memory_swap == runner.memory
    assert runner.cpus == "1.0"
    assert runner.pids_limit == 128
    assert runner.nofile_limit == 1024
    assert runner.user != "0:0"
    assert runner.pin_image_digest is True


def test_digest_pinning_detection():
    assert _is_digest_pinned("python:3.11-slim") is False
    assert _is_digest_pinned(
        "python@sha256:abcdef"
    ) is True
    assert _is_digest_pinned("sha256:abcdef") is True
