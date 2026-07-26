from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .models import VerificationResult


class VerificationRepository:
    def __init__(
        self,
        db_path: str = "data/verification_v2.sqlite",
    ) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._conn is not None:
                return

            self.db_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(
                "PRAGMA journal_mode=WAL;"
            )
            self._conn.execute(
                "PRAGMA synchronous=NORMAL;"
            )
            self._conn.execute(
                "PRAGMA busy_timeout=5000;"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_results (
                    id TEXT PRIMARY KEY,
                    subject TEXT,
                    source TEXT,
                    created_at TEXT,
                    completed_at TEXT,
                    status TEXT,
                    checks_json TEXT,
                    metadata_json TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_verification_created_at "
                "ON verification_results(created_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_verification_status "
                "ON verification_results(status)"
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
            raise RuntimeError(
                "VerificationRepository is not started"
            )
        return self._conn

    async def save(
        self,
        result: VerificationResult,
    ) -> VerificationResult:
        async with self._lock:
            conn = self._require_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO verification_results
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.subject,
                    result.source,
                    result.created_at,
                    result.completed_at,
                    result.status.value,
                    json.dumps(
                        [
                            asdict(check)
                            for check in result.checks
                        ],
                        default=str,
                    ),
                    json.dumps(result.metadata),
                ),
            )
            conn.commit()
        return result

    async def count(self) -> int:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM verification_results
                """
            ).fetchone()
        return int(row["n"])
