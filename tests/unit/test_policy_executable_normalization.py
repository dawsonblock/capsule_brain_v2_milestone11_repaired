from pathlib import Path

import pytest

import capsule_brain.execution.policy as policy_module
from capsule_brain.execution.models import (
    ExecutionPolicy,
    ExecutionRequest,
)


def test_windows_executable_extension_handling(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        policy_module.sys,
        "platform",
        "win32",
    )

    policy = ExecutionPolicy(
        allow=True,
        cwd_root=str(tmp_path),
        allowed_commands=("python", "pytest"),
    )
    request = ExecutionRequest(
        command=["python.exe", "script.py"],
        cwd=str(tmp_path),
    )

    cwd, executable = policy_module.validate_request(
        policy,
        request,
    )

    assert cwd == Path(tmp_path).resolve()
    assert executable == "python"


def test_executable_path_is_rejected(tmp_path):
    policy = ExecutionPolicy(
        allow=True,
        cwd_root=str(tmp_path),
        allowed_commands=("python",),
    )

    with pytest.raises(
        policy_module.ExecutionPolicyError
    ):
        policy_module.validate_request(
            policy,
            ExecutionRequest(
                command=["/tmp/malicious/python", "x.py"],
                cwd=str(tmp_path),
            ),
        )
