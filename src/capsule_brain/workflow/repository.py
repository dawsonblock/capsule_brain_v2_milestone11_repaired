from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import (
    WorkflowRun,
    WorkflowState,
    WorkflowStatus,
    WorkflowStepRecord,
    utc_now_iso,
)


class WorkflowRepository:
    """SQLite/WAL persistence for workflow runs.

    A single asyncio lock serializes connection access because sqlite3
    connections are synchronous. The schema stores one row per run plus one
    row per step, so a run can be resumed at the exact node where it stopped
    by reading the run's ``current_node``.
    """

    def __init__(self, db_path: str = "data/workflows.sqlite") -> None:
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
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id TEXT PRIMARY KEY,
                    workflow_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    current_node TEXT,
                    stop_reason TEXT,
                    state_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_runs_status "
                "ON workflow_runs(status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_runs_name "
                "ON workflow_runs(workflow_name)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_steps (
                    run_id TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    error TEXT,
                    PRIMARY KEY (run_id, step_index),
                    FOREIGN KEY(run_id) REFERENCES workflow_runs(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_steps_run "
                "ON workflow_steps(run_id)"
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
            raise RuntimeError("WorkflowRepository is not started")
        return self._conn

    async def save_run(self, run: WorkflowRun) -> WorkflowRun:
        """Upsert a workflow run (with its current state snapshot)."""
        run.updated_at = utc_now_iso()
        async with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO workflow_runs (
                        id, workflow_name, status, created_at, updated_at,
                        completed_at, current_node, stop_reason,
                        state_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status = excluded.status,
                        updated_at = excluded.updated_at,
                        completed_at = excluded.completed_at,
                        current_node = excluded.current_node,
                        stop_reason = excluded.stop_reason,
                        state_json = excluded.state_json,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        run.id,
                        run.workflow_name,
                        run.status.value,
                        run.created_at,
                        run.updated_at,
                        run.completed_at,
                        run.current_node,
                        run.stop_reason,
                        json.dumps(run.state.snapshot()),
                        json.dumps(run.metadata),
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return run

    async def save_step(self, step: WorkflowStepRecord) -> WorkflowStepRecord:
        """Upsert a single step record. Steps are keyed by (run_id, index)."""
        async with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO workflow_steps (
                        run_id, node_name, step_index, started_at,
                        completed_at, status, output_json, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, step_index) DO UPDATE SET
                        completed_at = excluded.completed_at,
                        status = excluded.status,
                        output_json = excluded.output_json,
                        error = excluded.error
                    """,
                    (
                        step.run_id,
                        step.node_name,
                        step.index,
                        step.started_at,
                        step.completed_at,
                        step.status,
                        json.dumps(step.output),
                        step.error,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return step

    async def get_run(self, run_id: str) -> WorkflowRun | None:
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM workflow_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            step_rows = conn.execute(
                "SELECT * FROM workflow_steps WHERE run_id = ? "
                "ORDER BY step_index ASC",
                (run_id,),
            ).fetchall()
        return self._row_to_run(row, step_rows)

    async def list_runs(
        self,
        *,
        status: WorkflowStatus | None = None,
        workflow_name: str | None = None,
        limit: int = 100,
    ) -> list[WorkflowRun]:
        """List runs, optionally filtered by status and/or workflow name.

        Returns runs without their step history (steps are loaded on demand
        via get_run) to keep listing cheap.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if workflow_name is not None:
            clauses.append("workflow_name = ?")
            params.append(workflow_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))

        async with self._lock:
            conn = self._require_conn()
            rows = conn.execute(
                f"SELECT * FROM workflow_runs {where} "
                "ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_run(row, []) for row in rows]

    async def list_pending_resume(self) -> list[WorkflowRun]:
        """Return runs that were interrupted and can be resumed.

        A run is resumable if it is PAUSED (awaiting approval) or FAILED with
        a known current_node. PENDING runs that never started are excluded —
        they have no state to resume from.
        """
        async with self._lock:
            conn = self._require_conn()
            rows = conn.execute(
                """
                SELECT * FROM workflow_runs
                WHERE status IN (?, ?)
                  AND current_node IS NOT NULL
                ORDER BY created_at ASC
                """,
                (WorkflowStatus.PAUSED.value, WorkflowStatus.FAILED.value),
            ).fetchall()
        return [self._row_to_run(row, []) for row in rows]

    async def count(self, *, status: WorkflowStatus | None = None) -> int:
        async with self._lock:
            conn = self._require_conn()
            if status is None:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM workflow_runs"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM workflow_runs WHERE status = ?",
                    (status.value,),
                ).fetchone()
        return int(row["n"])

    @staticmethod
    def _row_to_run(
        row: sqlite3.Row,
        step_rows: list[sqlite3.Row],
    ) -> WorkflowRun:
        state_snap = json.loads(row["state_json"])
        steps = [
            WorkflowStepRecord(
                run_id=sr["run_id"],
                node_name=sr["node_name"],
                index=sr["step_index"],
                started_at=sr["started_at"],
                completed_at=sr["completed_at"],
                status=sr["status"],
                output=json.loads(sr["output_json"]),
                error=sr["error"],
            )
            for sr in step_rows
        ]
        return WorkflowRun(
            id=row["id"],
            workflow_name=row["workflow_name"],
            status=WorkflowStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            current_node=row["current_node"],
            stop_reason=row["stop_reason"],
            state=WorkflowState.from_snapshot(state_snap),
            steps=steps,
            metadata=json.loads(row["metadata_json"]),
        )
