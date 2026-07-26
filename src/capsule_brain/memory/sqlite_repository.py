from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from .embeddings import (
    cosine_similarity,
    deserialize_vector,
    serialize_vector,
    try_load_sqlite_vec,
)
from .models import MemoryRecord, MemoryType, utc_now_iso
from .repository import MemoryRepository


class SQLiteMemoryRepository(MemoryRepository):
    """Transactional SQLite/WAL memory repository.

    A single asyncio lock protects connection use because sqlite3 connections are
    synchronous and not safe for concurrent coroutine access without serialization.

    Semantic vector search is supported via an optional ``memory_embeddings``
    table. When the ``sqlite-vec`` extension is loadable, similarity search is
    executed inside SQLite; otherwise the repository falls back to an
    in-Python cosine-similarity scan over stored embeddings. Both paths share
    the same ``search_semantic`` API so callers do not need to know which
    backend is active.
    """

    def __init__(self, db_path: str = "data/memory_v2.sqlite") -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        # Whether the sqlite-vec extension was successfully loaded. Set during
        # start() so search_semantic can pick the optimal path.
        self._vec_available: bool = False

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
            # Embeddings table. The vector is stored as a serialized blob so
            # the in-Python fallback can decode it without any extension. When
            # sqlite-vec is available we additionally register a virtual table
            # that mirrors this data for native KNN search.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_embeddings_dim "
                "ON memory_embeddings(dimension)"
            )
            self._vec_available = try_load_sqlite_vec(self._conn)
            if self._vec_available:
                self._init_vec_virtual_table()
            self._conn.commit()

    def _init_vec_virtual_table(self) -> None:
        """Create the sqlite-vec virtual table used for native KNN search.

        The dimension is fixed at creation time. We use 0 as a sentinel and
        let the first indexed embedding define the real dimension; sqlite-vec
        requires the dimension at CREATE time, so we defer virtual-table
        creation until the first embedding is indexed (see _ensure_vec_table).
        """
        # Intentionally a no-op here; the table is created lazily once we
        # know the embedding dimension. This avoids a chicken-and-egg problem
        # where the table is created with a wrong dimension before any
        # embedding has been seen.
        return None

    def _ensure_vec_table(self, dimension: int) -> None:
        """Lazily create the sqlite-vec virtual table for ``dimension``.

        Called the first time an embedding of a given dimension is indexed.
        Subsequent calls with the same dimension are no-ops.
        """
        if not self._vec_available or self._conn is None:
            return
        # sqlite-vec virtual tables are named vec_<dim> so multiple dimensions
        # can coexist if a deployment upgrades its embedding model.
        table = f"vec_memories_{dimension}"
        try:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
                "USING vec0(memory_id TEXT PRIMARY KEY, embedding float[{dimension}])"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            # If the virtual table cannot be created (e.g. dimension mismatch
            # from a prior schema), silently fall back to the in-Python path.
            self._vec_available = False

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

    async def index_embedding(
        self,
        memory_id: str,
        embedding: list[float],
    ) -> None:
        """Persist (or replace) the embedding vector for ``memory_id``.

        Idempotent: re-indexing overwrites the prior vector. When sqlite-vec
        is available the vector is also upserted into the dimension-specific
        virtual table for native KNN search.
        """
        if not embedding:
            raise ValueError("embedding must be a non-empty list of floats")
        dimension = len(embedding)
        blob = serialize_vector(embedding)
        async with self._lock:
            conn = self._require_conn()
            if self._vec_available:
                self._ensure_vec_table(dimension)
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO memory_embeddings (
                        memory_id, embedding, dimension, created_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        embedding = excluded.embedding,
                        dimension = excluded.dimension,
                        created_at = excluded.created_at
                    """,
                    (memory_id, blob, dimension, utc_now_iso()),
                )
                if self._vec_available:
                    table = f"vec_memories_{dimension}"
                    # sqlite-vec upsert: delete-then-insert avoids needing
                    # ON CONFLICT support in the virtual table.
                    conn.execute(
                        f"DELETE FROM {table} WHERE memory_id = ?",
                        (memory_id,),
                    )
                    conn.execute(
                        f"INSERT INTO {table} (memory_id, embedding) VALUES (?, ?)",
                        (memory_id, _vec_pack(embedding)),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    async def search_semantic(
        self,
        query_embedding: list[float],
        *,
        limit: int = 8,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        """Return the top-K memories most similar to ``query_embedding``.

        Uses the sqlite-vec virtual table when available; otherwise falls back
        to an in-Python cosine-similarity scan over the ``memory_embeddings``
        table. Both paths join back to ``memories`` so archived filtering and
        row decoding are identical.
        """
        if not query_embedding:
            raise ValueError("query_embedding must be a non-empty list")
        limit = max(1, int(limit))
        dimension = len(query_embedding)

        async with self._lock:
            conn = self._require_conn()
            if self._vec_available:
                records = self._vec_search(
                    conn, query_embedding, dimension, limit, include_archived
                )
                if records is not None:
                    return records
                # Fall through to the in-Python path if the virtual table for
                # this dimension does not exist yet (no embeddings indexed).
            return self._python_search(
                conn, query_embedding, dimension, limit, include_archived
            )

    def _vec_search(
        self,
        conn: sqlite3.Connection,
        query_embedding: list[float],
        dimension: int,
        limit: int,
        include_archived: bool,
    ) -> list[MemoryRecord] | None:
        """Native KNN search via sqlite-vec. Returns None if the virtual
        table for this dimension does not exist (caller falls back)."""
        table = f"vec_memories_{dimension}"
        try:
            rows = conn.execute(
                f"SELECT memory_id FROM {table} "
                "WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                (_vec_pack(query_embedding), limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return None
        if not rows:
            return []
        ids = [row["memory_id"] for row in rows]
        # Preserve KNN order when fetching the full records.
        placeholders = ",".join("?" for _ in ids)
        archived_clause = "" if include_archived else "AND archived = 0"
        ordered = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders}) {archived_clause}",
            ids,
        ).fetchall()
        by_id = {row["id"]: row for row in ordered}
        return [
            self._row_to_record(by_id[memory_id])
            for memory_id in ids
            if memory_id in by_id
        ]

    def _python_search(
        self,
        conn: sqlite3.Connection,
        query_embedding: list[float],
        dimension: int,
        limit: int,
        include_archived: bool,
    ) -> list[MemoryRecord]:
        """In-Python cosine-similarity fallback.

        Loads embeddings of the matching dimension and computes similarity in
        Python. Suitable for small-to-medium memory stores; for large stores
        install sqlite-vec to enable native KNN search.
        """
        archived_clause = "" if include_archived else "AND m.archived = 0"
        rows = conn.execute(
            f"""
            SELECT m.*, e.embedding
            FROM memories m
            JOIN memory_embeddings e ON e.memory_id = m.id
            WHERE e.dimension = ? {archived_clause}
            """,
            (dimension,),
        ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            candidate = deserialize_vector(row["embedding"])
            score = cosine_similarity(query_embedding, candidate)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_record(row) for _, row in scored[:limit]]

    async def count_embeddings(self) -> int:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM memory_embeddings"
            ).fetchone()
        return int(row["n"])


def _vec_pack(vec: list[float]) -> bytes:
    """Pack a float vector into the little-endian float32 blob sqlite-vec expects."""
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)
