from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from capsule_brain.runtime.service import (
    CapsuleService,
    HealthStatus,
    ServiceState,
)

from .models import ExperienceRecord, FeedbackRecord, utc_now_iso


class ExperienceStore(CapsuleService):
    name = "experience_store"

    def __init__(
        self,
        db_path: str = "data/experience_v2.sqlite",
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.db_path = Path(self.cfg.get("db_path", db_path))
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._conn is not None:
                return

            self.state = ServiceState.STARTING
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._create_tables()
            self.state = ServiceState.RUNNING

    def _create_tables(self) -> None:
        conn = self._require_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiences (
                id TEXT PRIMARY KEY,
                response_id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL,
                user_turn_id TEXT NOT NULL,
                assistant_turn_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                model_alias TEXT,
                provider TEXT,
                model_name TEXT,
                route TEXT,
                latency_ms REAL,
                attempts INTEGER,
                usage_json TEXT NOT NULL,
                feedback_classification TEXT,
                feedback_text TEXT,
                feedback_reason TEXT,
                verifier_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                response_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                classification TEXT NOT NULL,
                text TEXT,
                reason TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_experience_conversation
                ON experiences(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_experience_created_at
                ON experiences(created_at);
            CREATE INDEX IF NOT EXISTS idx_feedback_response
                ON feedback(response_id);
            """
        )
        conn.commit()

    async def stop(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
                self._conn = None
            self.state = ServiceState.STOPPED

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ExperienceStore is not started")
        return self._conn

    async def create_experience(
        self,
        record: ExperienceRecord,
    ) -> ExperienceRecord:
        async with self._lock:
            conn = self._require_conn()
            conn.execute(
                """
                INSERT INTO experiences (
                    id, response_id, conversation_id, user_turn_id,
                    assistant_turn_id, prompt_text, response_text, model_alias,
                    provider, model_name, route, latency_ms, attempts,
                    usage_json, feedback_classification, feedback_text,
                    feedback_reason, verifier_json, created_at, updated_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.response_id,
                    record.conversation_id,
                    record.user_turn_id,
                    record.assistant_turn_id,
                    record.prompt_text,
                    record.response_text,
                    record.model_alias,
                    record.provider,
                    record.model_name,
                    record.route,
                    record.latency_ms,
                    record.attempts,
                    json.dumps(record.usage),
                    record.feedback_classification,
                    record.feedback_text,
                    record.feedback_reason,
                    json.dumps(record.verifier),
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()
        return record

    async def add_feedback(self, feedback: FeedbackRecord) -> None:
        async with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO feedback (
                        id, response_id, conversation_id, classification,
                        text, reason, created_at, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback.id,
                        feedback.response_id,
                        feedback.conversation_id,
                        feedback.classification.value,
                        feedback.text,
                        feedback.reason,
                        feedback.created_at,
                        json.dumps(feedback.metadata),
                    ),
                )
                cur = conn.execute(
                    """
                    UPDATE experiences
                    SET feedback_classification = ?,
                        feedback_text = ?,
                        feedback_reason = ?,
                        updated_at = ?
                    WHERE response_id = ?
                    """,
                    (
                        feedback.classification.value,
                        feedback.text,
                        feedback.reason,
                        utc_now_iso(),
                        feedback.response_id,
                    ),
                )
                if cur.rowcount != 1:
                    raise KeyError(feedback.response_id)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    async def get_by_response_id(
        self,
        response_id: str,
    ) -> ExperienceRecord | None:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM experiences WHERE response_id = ?",
                (response_id,),
            ).fetchone()

        if row is None:
            return None

        return ExperienceRecord(
            id=row["id"],
            response_id=row["response_id"],
            conversation_id=row["conversation_id"],
            user_turn_id=row["user_turn_id"],
            assistant_turn_id=row["assistant_turn_id"],
            prompt_text=row["prompt_text"],
            response_text=row["response_text"],
            model_alias=row["model_alias"],
            provider=row["provider"],
            model_name=row["model_name"],
            route=row["route"],
            latency_ms=row["latency_ms"],
            attempts=row["attempts"],
            usage=json.loads(row["usage_json"]),
            feedback_classification=row["feedback_classification"],
            feedback_text=row["feedback_text"],
            feedback_reason=row["feedback_reason"],
            verifier=json.loads(row["verifier_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    async def count(self) -> int:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM experiences"
            ).fetchone()
        return int(row["n"])

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=self.state,
            details={
                "db_path": str(self.db_path),
                "experience_count": (
                    await self.count()
                    if self.state == ServiceState.RUNNING
                    else 0
                ),
            },
        )
