from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .models import AssistantResponse, Conversation, Turn, TurnRole, utc_now_iso


class ConversationRepository:
    def __init__(self, db_path: str = "data/conversations_v2.sqlite") -> None:
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

            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    title TEXT,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    parent_turn_id TEXT,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );

                CREATE TABLE IF NOT EXISTS assistant_responses (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    parent_turn_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    model_alias TEXT,
                    provider TEXT,
                    model_name TEXT,
                    latency_ms REAL,
                    attempts INTEGER,
                    usage_json TEXT NOT NULL,
                    route TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id),
                    FOREIGN KEY(parent_turn_id) REFERENCES turns(id),
                    FOREIGN KEY(turn_id) REFERENCES turns(id)
                );

                CREATE INDEX IF NOT EXISTS idx_turns_conversation
                    ON turns(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_responses_parent
                    ON assistant_responses(parent_turn_id);
                """
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
            raise RuntimeError("Conversation repository is not started")
        return self._conn

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        async with self._lock:
            conn = self._require_conn()
            conn.execute(
                """
                INSERT INTO conversations
                (id, created_at, updated_at, title, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation.id,
                    conversation.created_at,
                    conversation.updated_at,
                    conversation.title,
                    json.dumps(conversation.metadata),
                ),
            )
            conn.commit()
        return conversation

    async def ensure_conversation(self, conversation_id: str | None = None) -> Conversation:
        async with self._lock:
            conn = self._require_conn()
            if conversation_id:
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if row is not None:
                    return Conversation(
                        id=row["id"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        title=row["title"],
                        metadata=json.loads(row["metadata_json"]),
                    )

            conversation = Conversation(id=conversation_id) if conversation_id else Conversation()
            conn.execute(
                """
                INSERT INTO conversations
                (id, created_at, updated_at, title, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation.id,
                    conversation.created_at,
                    conversation.updated_at,
                    conversation.title,
                    json.dumps(conversation.metadata),
                ),
            )
            conn.commit()
            return conversation

    async def create_turn(self, turn: Turn) -> Turn:
        async with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO turns
                    (id, conversation_id, role, text, created_at, parent_turn_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.id,
                        turn.conversation_id,
                        turn.role.value,
                        turn.text,
                        turn.created_at,
                        turn.parent_turn_id,
                        json.dumps(turn.metadata),
                    ),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (utc_now_iso(), turn.conversation_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return turn

    async def create_response(self, response: AssistantResponse) -> AssistantResponse:
        async with self._lock:
            conn = self._require_conn()
            conn.execute(
                """
                INSERT INTO assistant_responses (
                    id, conversation_id, parent_turn_id, turn_id, text,
                    model_alias, provider, model_name, latency_ms, attempts,
                    usage_json, route, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.id,
                    response.conversation_id,
                    response.parent_turn_id,
                    response.turn_id,
                    response.text,
                    response.model_alias,
                    response.provider,
                    response.model_name,
                    response.latency_ms,
                    response.attempts,
                    json.dumps(response.usage),
                    response.route,
                    response.created_at,
                    json.dumps(response.metadata),
                ),
            )
            conn.commit()
        return response

    async def recent_turns(
        self,
        conversation_id: str,
        *,
        limit: int = 20,
    ) -> list[Turn]:
        async with self._lock:
            conn = self._require_conn()
            rows = conn.execute(
                """
                SELECT * FROM turns
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()

        rows = list(reversed(rows))
        return [
            Turn(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=TurnRole(row["role"]),
                text=row["text"],
                created_at=row["created_at"],
                parent_turn_id=row["parent_turn_id"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    async def get_response(self, response_id: str) -> AssistantResponse | None:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM assistant_responses WHERE id = ?",
                (response_id,),
            ).fetchone()

        if row is None:
            return None

        return AssistantResponse(
            id=row["id"],
            conversation_id=row["conversation_id"],
            parent_turn_id=row["parent_turn_id"],
            turn_id=row["turn_id"],
            text=row["text"],
            model_alias=row["model_alias"],
            provider=row["provider"],
            model_name=row["model_name"],
            latency_ms=row["latency_ms"],
            attempts=row["attempts"],
            usage=json.loads(row["usage_json"]),
            route=row["route"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    async def get_turn(self, turn_id: str) -> Turn | None:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM turns WHERE id = ?",
                (turn_id,),
            ).fetchone()

        if row is None:
            return None

        return Turn(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=TurnRole(row["role"]),
            text=row["text"],
            created_at=row["created_at"],
            parent_turn_id=row["parent_turn_id"],
            metadata=json.loads(row["metadata_json"]),
        )
