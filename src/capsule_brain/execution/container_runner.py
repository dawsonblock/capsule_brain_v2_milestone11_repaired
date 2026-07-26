from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ._bounded import run_bounded
from .models import ExecutionRequest, ExecutionResult, now
from .policy import ExecutionPolicyError, validate_request


class ContainerImageResolutionError(ExecutionPolicyError):
    """Raised when digest pinning is enabled but resolution fails.

    A security control named ``pin_image_digest`` must fail closed — if the
    digest cannot be resolved, the run must not proceed with a mutable tag.
    """


def _format_volume_mount(host_path: str) -> str:
    """Format a host path as a Docker/Podman read-only volume mount.

    On Windows, drive letters (``C:\\path``) produce a colon that breaks the
    ``--volume src:dst:opts`` parser. Convert to the POSIX-style path that
    Docker Desktop on Windows accepts (``/c/path/to``).
    """
    if sys.platform == "win32" and len(host_path) >= 2 and host_path[1] == ":":
        drive = host_path[0].lower()
        rest = host_path[2:].replace("\\", "/")
        return f"/{drive}{rest}:/workspace:ro"
    return f"{host_path}:/workspace:ro"


def _is_digest_pinned(image: str) -> bool:
    """True if ``image`` is already pinned to an immutable digest.

    Accepts both ``repo@sha256:...`` and bare ``sha256:...`` forms.
    """
    return "@sha256:" in image or image.startswith("sha256:")


def _resolve_digest(engine: str, image: str) -> str:
    """Resolve a floating tag to an immutable digest.

    Raises ``ContainerImageResolutionError`` if resolution fails. Pinning is
    a security control — it must fail closed, not silently fall back to a
    mutable tag.
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
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise ContainerImageResolutionError(
            f"Failed to resolve digest for {image!r}: {exc}. "
            f"Set pin_image_digest=false to allow mutable tags."
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ContainerImageResolutionError(
            f"Failed to resolve digest for {image!r}: "
            f"engine returned exit code {completed.returncode}. "
            f"Set pin_image_digest=false to allow mutable tags."
        )
    digest = completed.stdout.decode("utf-8", errors="replace").strip()
    if not digest or "@" not in digest:
        raise ContainerImageResolutionError(
            f"Engine returned no digest for {image!r}: got {digest!r}. "
            f"Set pin_image_digest=false to allow mutable tags."
        )
    return digest


def _kill_container(engine: str, cid: str) -> None:
    """Kill and remove a container by ID. Best-effort — errors are logged."""
    try:
        subprocess.run(
            [engine, "kill", cid],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        pass
    try:
        subprocess.run(
            [engine, "rm", "-f", cid],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        pass


class ContainerExecutionRunner:
    """Container-backed execution runner.

    Docker/Podman is invoked as a subprocess. The workspace is read-only,
    networking is disabled, execution is non-root, and CPU/RAM/PID limits
    are enforced. On timeout, the container is explicitly killed and removed
    via ``--cidfile`` tracking — killing the Docker CLI process alone does
    not guarantee the backing container is stopped.
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
        # before each run. Resolution MUST succeed — failure is a hard error.
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

        # Use a CID file so we can explicitly kill/remove the container on
        # timeout. Killing the Docker CLI process does not guarantee the
        # backing container is stopped.
        cid_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".cid",
            delete=False,
            prefix="capsule_exec_",
        )
        cid_file.close()
        cid_path = cid_file.name

        command = [
            self.engine,
            "run",
            "--rm",
            "--init",
            "--cidfile", cid_path,
            "--network",
            "none",
            # Suppress .pyc writes — the workspace is mounted read-only and
            # Python's default __pycache__ writes would fail with EACCES.
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONPYCACHEPREFIX=/tmp/pycache",
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
            _format_volume_mount(str(cwd)),
            image,
            *request.command,
        ]

        def execute_container():
            return run_bounded(
                command,
                timeout_s=self.policy.timeout_s,
                max_output_chars=self.policy.max_output_chars,
            )

        raw = await asyncio.to_thread(execute_container)

        # If the process timed out, explicitly kill and remove the container
        # using the CID from the cidfile. This is the only reliable way to
        # ensure the backing container is stopped — killing the Docker CLI
        # client process leaves the container running under the daemon.
        if raw["timed_out"]:
            cid = _read_cid(cid_path)
            if cid:
                await asyncio.to_thread(_kill_container, self.engine, cid)

        # Clean up the CID file
        try:
            os.unlink(cid_path)
        except OSError:
            pass

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


def _read_cid(cid_path: str) -> str | None:
    """Read the container ID from a cidfile. Returns None if empty/missing."""
    try:
        with open(cid_path) as f:
            cid = f.read().strip()
            return cid if cid else None
    except (OSError, IOError):
        return None
