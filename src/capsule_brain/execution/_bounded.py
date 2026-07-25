from __future__ import annotations

import subprocess
import threading
from typing import Mapping


def _read_capped(pipe, sink: bytearray, cap: int) -> None:
    """Read from ``pipe`` into ``sink`` until EOF, keeping at most ``cap`` bytes.

    Reading continues after the cap is reached so the child process is not
    blocked on a full pipe; excess bytes are discarded. This bounds host
    memory to roughly ``2 * cap`` regardless of how much the child emits.
    """
    capped = False
    while True:
        chunk = pipe.read(4096)
        if not chunk:
            break
        if capped:
            continue
        if len(sink) + len(chunk) > cap:
            sink.extend(chunk[: cap - len(sink)])
            capped = True
        else:
            sink.extend(chunk)


def run_bounded(
    command: list[str],
    *,
    cwd: str | None = None,
    timeout_s: float,
    max_output_chars: int,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Run ``command`` capturing stdout/stderr with a hard per-stream byte cap.

    Unlike ``subprocess.run(..., capture_output=True)``, output is streamed
    through reader threads so a child that emits gigabytes cannot exhaust host
    memory: each stream retains at most ``max_output_chars`` bytes.

    Returns a dict with ``exit_code``, ``stdout`` (bytes), ``stderr`` (bytes),
    and ``timed_out`` (bool). On timeout the process is killed and
    ``exit_code`` is reported as ``None`` since the process did not complete
    normally.
    """
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    stdout = bytearray()
    stderr = bytearray()
    t_out = threading.Thread(
        target=_read_capped,
        args=(proc.stdout, stdout, max_output_chars),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_read_capped,
        args=(proc.stderr, stderr, max_output_chars),
        daemon=True,
    )
    t_out.start()
    t_err.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        timed_out = True
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    t_out.join(timeout=10)
    t_err.join(timeout=10)

    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except Exception:
            pass

    return {
        # A timed-out process did not complete; report no exit code rather
        # than the kill signal's status so callers can distinguish timeout
        # from a real non-zero exit.
        "exit_code": None if timed_out else proc.returncode,
        "stdout": bytes(stdout),
        "stderr": bytes(stderr),
        "timed_out": timed_out,
    }
