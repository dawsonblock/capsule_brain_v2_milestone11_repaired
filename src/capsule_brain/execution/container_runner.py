from __future__ import annotations

import asyncio
import shutil
import subprocess
import time

from ._bounded import run_bounded
from .models import ExecutionRequest, ExecutionResult, now
from .policy import ExecutionPolicyError, validate_request


def _is_digest_pinned(image: str) -> bool:
    """True if ``image`` is already pinned to an immutable digest.

    Accepts both ``repo@sha256:...`` and bare ``sha256:...`` forms.
    """
    return "@sha256:" in image or image.startswith("sha256:")


def _resolve_digest(engine: str, image: str) -> str:
    """Resolve a floating tag to an immutable digest.

    Returns the original image unchanged if the engine cannot resolve it
    (e.g. offline); pinning is a hardening measure, not a gate.
    """
    if _is_digest_pinned(image):
        return image
    try:
        completed = subprocess.run(
            [engine, "image", "inspect", "--format", "{{index .RepoDigests 0}}", image],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return image
    if completed.returncode != 0:
        return image
    digest = completed.stdout.decode("utf-8", errors="replace").strip()
    if not digest or "@" not in digest:
        return image
    return digest


class ContainerExecutionRunner:
    """Container-backed execution runner.

    Docker/Podman is invoked as a subprocess. The workspace is read-only,
    networking is disabled, execution is non-root, and CPU/RAM/PID limits
    are enforced.
    """

    def __init__(
        self,
        policy,
        *,
        engine: str = "docker",
        image: str = "python:3.11-slim",
        memory: str = "512m",
        memory_swap: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 128,
        nofile_limit: int = 1024,
        user: str = "65534:65534",
        pin_image_digest: bool = True,
    ) -> None:
        self.policy = policy
        self.engine = engine
        self.image = image
        self.memory = memory
        # Swap is capped equal to memory by default so --memory cannot be
        # bypassed via swap-backed allocation. Set memory_swap to "0" to
        # disable swap entirely, or to a larger value to allow it.
        self.memory_swap = memory_swap
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.nofile_limit = nofile_limit
        self.user = user
        # When enabled, floating tags are resolved to an immutable digest
        # before each run so a rebuilt image cannot silently change behavior.
        self.pin_image_digest = pin_image_digest

    async def run(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        cwd, _ = validate_request(
            self.policy,
            request,
        )

        if shutil.which(self.engine) is None:
            raise ExecutionPolicyError(
                f"Container engine not found: {self.engine}"
            )

        image = self.image
        if self.pin_image_digest:
            image = await asyncio.to_thread(
                _resolve_digest, self.engine, image
            )

        started_at = now()
        started = time.perf_counter()

        command = [
            self.engine,
            "run",
            "--rm",
            "--init",
            "--network",
            "none",
            "--user",
            self.user,
            "--memory",
            self.memory,
            "--memory-swap",
            self.memory_swap,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "--ulimit",
            f"nofile={self.nofile_limit}:{self.nofile_limit}",
            "--read-only",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--workdir",
            "/workspace",
            "--volume",
            f"{cwd}:/workspace:ro",
            image,
            *request.command,
        ]

        def execute_container():
            return run_bounded(
                command,
                timeout_s=self.policy.timeout_s,
                max_output_chars=self.policy.max_output_chars,
            )

        raw = await asyncio.to_thread(
            execute_container
        )
        cap = self.policy.max_output_chars

        def decode(value) -> str:
            if isinstance(value, bytes):
                return value.decode(
                    "utf-8",
                    errors="replace",
                )[:cap]
            return str(value or "")[:cap]

        return ExecutionResult(
            request_id=request.id,
            command=list(request.command),
            cwd=str(cwd),
            exit_code=raw["exit_code"],
            stdout=decode(raw["stdout"]),
            stderr=decode(raw["stderr"]),
            timed_out=raw["timed_out"],
            duration_ms=(
                time.perf_counter() - started
            )
            * 1000.0,
            started_at=started_at,
            completed_at=now(),
            metadata={
                **dict(request.metadata),
                "runner": "container",
                "engine": self.engine,
                # Record the image actually used (may be a resolved digest),
                # plus the original configured tag for traceability.
                "image": image,
                "configured_image": self.image,
            },
        )
