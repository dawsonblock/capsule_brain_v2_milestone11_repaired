from __future__ import annotations

import os
import sys
from pathlib import Path

from .models import ExecutionPolicy, ExecutionRequest


class ExecutionPolicyError(RuntimeError):
    pass


def _normalize_executable_name(value: str) -> str:
    name = Path(value).name
    if sys.platform == "win32" and Path(name).suffix.lower() in {
        ".exe",
        ".bat",
        ".cmd",
    }:
        return Path(name).stem
    return name


def validate_request(
    policy: ExecutionPolicy,
    request: ExecutionRequest,
) -> tuple[Path, str]:
    if not policy.allow:
        raise ExecutionPolicyError("Execution is disabled by policy.")

    if not request.command:
        raise ExecutionPolicyError("Execution command is empty.")

    raw_executable = request.command[0]
    executable_path = Path(raw_executable)

    # Caller-controlled executable paths are rejected. The sole host-mode
    # compatibility exception is the exact interpreter running Capsule Brain;
    # this avoids allowing a lookalike binary such as /tmp/malicious/python.
    has_explicit_path = executable_path.parent not in {Path("."), Path("")}
    if has_explicit_path:
        requested = executable_path.resolve()
        current_python = Path(sys.executable).resolve()
        if requested != current_python:
            raise ExecutionPolicyError(
                "Executable paths are not allowed; use an allowlisted "
                "command name or the current Python interpreter."
            )

    executable_name = _normalize_executable_name(raw_executable)
    allowed = {
        _normalize_executable_name(command)
        for command in policy.allowed_commands
    }

    if executable_name not in allowed:
        raise ExecutionPolicyError(
            f"Command not allowed: {executable_name}"
        )

    root = Path(policy.cwd_root).resolve()
    cwd = Path(request.cwd or policy.cwd_root).resolve()

    try:
        cwd.relative_to(root)
    except ValueError as exc:
        raise ExecutionPolicyError(
            "Working directory escapes sandbox root."
        ) from exc

    cwd.mkdir(parents=True, exist_ok=True)
    return cwd, executable_name
