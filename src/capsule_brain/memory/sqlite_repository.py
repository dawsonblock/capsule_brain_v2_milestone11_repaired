from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from .models import MemoryRecord, MemoryType, utc_now_iso
from .repository import MemoryRepository


class SQLiteMemoryRepository(MemoryRepository):
    """Transactional SQLite/WAL memory repository.

    A single asyncio lock protects connection use because sqlite3 connections are
    synchronous and not safe for concurrent coroutine access without serialization.
    """

    def __init__(self, db_path: str = "data/memory_v2.sqlite") -> None:
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
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    conversation_id TEXT,
                    turn_id TEXT,
                    importance REAL NOT NULL,
                    confidence REAL,
                    archived INTEGER NOT NULL,
                    protected INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source)"
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
            raise RuntimeError("Memory repository is not started")
        return self._conn

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            text=row["text"],
            type=MemoryType(row["type"]),
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            tags=json.loads(row["tags_json"]),
            conversation_id=row["conversation_id"],
            turn_id=row["turn_id"],
            importance=float(row["importance"]),
            confidence=row["confidence"],
            archived=bool(row["archived"]),
            protected=bool(row["protected"]),
            metadata=json.loads(row["metadata_json"]),
        )

    async def create(self, record: MemoryRecord) -> MemoryRecord:
        async with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO memories (
                        id, text, type, source, created_at, updated_at,
                        tags_json, conversation_id, turn_id, importance,
                        confidence, archived, protected, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.text,
                        record.type.value,
                        record.source,
                        record.created_at,
                        record.updated_at,
                        json.dumps(record.tags),
                        record.conversation_id,
                        record.turn_id,
                        record.importance,
                        record.confidence,
                        int(record.archived),
                        int(record.protected),
                        json.dumps(record.metadata),
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return record

    async def get(self, memory_id: str) -> MemoryRecord | None:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    async def recent(
        self,
        *,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        async with self._lock:
            conn = self._require_conn()
            if include_archived:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE archived = 0
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    async def oldest(
        self,
        *,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        """Return records ordered oldest-first.

        Used by the consolidator so that records beyond ``limit`` are not
        permanently shielded from archival — unlike ``recent()`` which returns
        newest-first and would hide old records past the scan window.
        """
        async with self._lock:
            conn = self._require_conn()
            if include_archived:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE archived = 0
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    async def update(self, record: MemoryRecord) -> MemoryRecord:
        updated = replace(record, updated_at=utc_now_iso())
        async with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    """
                    UPDATE memories SET
                        text = ?,
                        type = ?,
                        source = ?,
                        updated_at = ?,
                        tags_json = ?,
                        conversation_id = ?,
                        turn_id = ?,
                        importance = ?,
                        confidence = ?,
                        archived = ?,
                        protected = ?,
                        metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        updated.text,
                        updated.type.value,
                        updated.source,
                        updated.updated_at,
                        json.dumps(updated.tags),
                        updated.conversation_id,
                        updated.turn_id,
                        updated.importance,
                        updated.confidence,
                        int(updated.archived),
                        int(updated.protected),
                        json.dumps(updated.metadata),
                        updated.id,
                    ),
                )
                if cur.rowcount != 1:
                    raise KeyError(updated.id)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return updated

    async def archive(self, memory_id: str) -> bool:
        async with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    """
                    UPDATE memories
                    SET archived = 1, updated_at = ?
                    WHERE id = ? AND protected = 0
                    """,
                    (utc_now_iso(), memory_id),
                )
                conn.commit()
                return cur.rowcount == 1
            except Exception:
                conn.rollback()
                raise

    async def archive_batch(self, memory_ids: list[str]) -> int:
        """Archive multiple memories in a single transaction.

        This avoids the write-lock contention that arises from calling
        ``archive()`` in a loop, where each call opens a separate
        ``BEGIN IMMEDIATE`` transaction. Protected records are skipped
        (the ``WHERE protected = 0`` clause ensures this).
        """
        if not memory_ids:
            return 0
        async with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" for _ in memory_ids)
                cur = conn.execute(
                    f"""
                    UPDATE memories
                    SET archived = 1, updated_at = ?
                    WHERE id IN ({placeholders}) AND protected = 0
                    """,
                    [utc_now_iso()] + list(memory_ids),
                )
                count = cur.rowcount
                conn.commit()
                return count
            except Exception:
                conn.rollback()
                raise

    async def count(self, *, include_archived: bool = False) -> int:
        async with self._lock:
            conn = self._require_conn()
            if include_archived:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM memories"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM memories WHERE archived = 0"
                ).fetchone()
        return int(row["n"])
