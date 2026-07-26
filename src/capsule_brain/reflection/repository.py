from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .models import ReflectionIteration, ReflectionSession


class ReflectionRepository:
    def __init__(self, db_path: str = "data/reflections_v2.sqlite") -> None:
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
                CREATE TABLE IF NOT EXISTS reflection_sessions (
                    id TEXT PRIMARY KEY,
                    seed TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    max_iterations INTEGER NOT NULL,
                    iterations_json TEXT NOT NULL,
                    final_text TEXT,
                    stop_reason TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reflection_created_at "
                "ON reflection_sessions(created_at)"
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
            raise RuntimeError("ReflectionRepository is not started")
        return self._conn

    async def save(self, session: ReflectionSession) -> ReflectionSession:
        async with self._lock:
            conn = self._require_conn()
            conn.execute(
                """
                INSERT INTO reflection_sessions (
                    id, seed, source, created_at, completed_at, max_iterations,
                    iterations_json, final_text, stop_reason, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    completed_at=excluded.completed_at,
                    iterations_json=excluded.iterations_json,
                    final_text=excluded.final_text,
                    stop_reason=excluded.stop_reason,
                    metadata_json=excluded.metadata_json
                """,
                (
                    session.id,
                    session.seed,
                    session.source,
                    session.created_at,
                    session.completed_at,
                    session.max_iterations,
                    json.dumps([asdict(i) for i in session.iterations]),
                    session.final_text,
                    session.stop_reason,
                    json.dumps(session.metadata),
                ),
            )
            conn.commit()
        return session

    async def get(self, session_id: str) -> ReflectionSession | None:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM reflection_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        return ReflectionSession(
            id=row["id"],
            seed=row["seed"],
            source=row["source"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            max_iterations=row["max_iterations"],
            iterations=[
                ReflectionIteration(**item)
                for item in json.loads(row["iterations_json"])
            ],
            final_text=row["final_text"],
            stop_reason=row["stop_reason"],
            metadata=json.loads(row["metadata_json"]),
        )

    async def count(self) -> int:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM reflection_sessions"
            ).fetchone()
        return int(row["n"])
