from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from .models import ExecutionResult


class ExecutionRepository:
    def __init__(self, db_path: str = "data/execution_v2.sqlite") -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._conn is not None:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_results (
                    request_id TEXT PRIMARY KEY,
                    command_json TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    exit_code INTEGER,
                    stdout TEXT NOT NULL,
                    stderr TEXT NOT NULL,
                    timed_out INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_execution_started_at "
                "ON execution_results(started_at)"
            )
            self._conn.commit()

    async def stop(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
                self._conn = None

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ExecutionRepository is not started")
        return self._conn

    async def save(self, result: ExecutionResult) -> ExecutionResult:
        async with self._lock:
            conn = self._require_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO execution_results (
                    request_id, command_json, cwd, exit_code, stdout, stderr,
                    timed_out, duration_ms, started_at, completed_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.request_id,
                    json.dumps(result.command),
                    result.cwd,
                    result.exit_code,
                    result.stdout,
                    result.stderr,
                    int(result.timed_out),
                    result.duration_ms,
                    result.started_at,
                    result.completed_at,
                    json.dumps(result.metadata),
                ),
            )
            conn.commit()
        return result

    async def count(self) -> int:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM execution_results"
            ).fetchone()
        return int(row["n"])
