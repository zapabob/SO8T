#!/usr/bin/env python3
"""SQLite-backed progress store for long-running pipelines.

Design goals:
- Durable state across power loss / process crash (boot-time resumption).
- Minimal dependencies (stdlib sqlite3).
- Safe concurrent reads (monitor UI) while writer appends events.

The store is append-only: every state change is recorded as an event.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    timestamp: str
    phase: str
    message: str
    checkpoint_path: Optional[str] = None
    metrics_json: Optional[str] = None

    @staticmethod
    def build(
        *,
        run_id: str,
        phase: str,
        message: str,
        checkpoint_path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> "ProgressEvent":
        return ProgressEvent(
            run_id=run_id,
            timestamp=timestamp or utc_now_iso(),
            phase=phase,
            message=message,
            checkpoint_path=checkpoint_path,
            metrics_json=json.dumps(metrics, ensure_ascii=False) if metrics is not None else None,
        )


class PipelineProgressStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS progress_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    message TEXT NOT NULL,
                    checkpoint_path TEXT,
                    metrics_json TEXT
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_progress_events_run_id_ts ON progress_events(run_id, timestamp);"
            )
            self._conn.commit()

    def append(self, event: ProgressEvent) -> None:
        payload = asdict(event)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO progress_events(run_id, timestamp, phase, message, checkpoint_path, metrics_json)
                VALUES(:run_id, :timestamp, :phase, :message, :checkpoint_path, :metrics_json);
                """,
                payload,
            )
            self._conn.commit()

    def latest(self, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = "SELECT run_id, timestamp, phase, message, checkpoint_path, metrics_json FROM progress_events"
        params: tuple[Any, ...] = ()
        if run_id:
            query += " WHERE run_id = ?"
            params = (run_id,)
        query += " ORDER BY id DESC LIMIT 1"
        with self._lock:
            cur = self._conn.execute(query, params)
            row = cur.fetchone()
        if not row:
            return None
        metrics = None
        if row[5]:
            try:
                metrics = json.loads(row[5])
            except Exception:
                metrics = row[5]
        return {
            "run_id": row[0],
            "timestamp": row[1],
            "phase": row[2],
            "message": row[3],
            "checkpoint_path": row[4],
            "metrics": metrics,
        }

    def latest_checkpoint(self, run_id: Optional[str] = None) -> Optional[str]:
        query = """
            SELECT checkpoint_path
            FROM progress_events
            WHERE checkpoint_path IS NOT NULL
        """
        params: tuple[Any, ...] = ()
        if run_id:
            query += " AND run_id = ?"
            params = (run_id,)
        query += " ORDER BY id DESC LIMIT 1"
        with self._lock:
            cur = self._conn.execute(query, params)
            row = cur.fetchone()
        if not row:
            return None
        return row[0]


# ---------------------------------------------------------------------------
# Compatibility layer (used by boot_pipeline_launcher.py)
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path("logs/pipeline_progress.sqlite")
_DEFAULT_STORE: Optional[PipelineProgressStore] = None


def init_db(db_path: Path = _DEFAULT_DB_PATH) -> None:
    """Initialize the global SQLite store (idempotent)."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None or _DEFAULT_STORE.db_path != db_path:
        _DEFAULT_STORE = PipelineProgressStore(db_path)


def _require_store() -> PipelineProgressStore:
    if _DEFAULT_STORE is None:
        init_db()
    assert _DEFAULT_STORE is not None
    return _DEFAULT_STORE


def record_run(run_id: str, git_commit_hash: str = "unknown") -> None:
    store = _require_store()
    store.append(
        ProgressEvent.build(
            run_id=run_id,
            phase="run_start",
            message="pipeline run started",
            metrics={"git_commit": git_commit_hash},
        )
    )


def record_checkpoint(
    run_id: str,
    checkpoint_type: str,
    *,
    step: Optional[int],
    checkpoint_path: str,
    is_rolling: bool,
) -> None:
    store = _require_store()
    store.append(
        ProgressEvent.build(
            run_id=run_id,
            phase="checkpoint",
            message=f"checkpoint captured ({checkpoint_type})",
            checkpoint_path=checkpoint_path,
            metrics={"step": step, "checkpoint_type": checkpoint_type, "is_rolling": is_rolling},
        )
    )


def log_progress(run_id: str, phase: str, message: str) -> None:
    store = _require_store()
    store.append(ProgressEvent.build(run_id=run_id, phase=phase, message=message))


def complete_run(run_id: str) -> None:
    store = _require_store()
    store.append(
        ProgressEvent.build(run_id=run_id, phase="run_complete", message="pipeline run completed")
    )


def fail_run(run_id: str, reason: str) -> None:
    store = _require_store()
    store.append(
        ProgressEvent.build(
            run_id=run_id, phase="run_failed", message="pipeline run failed", metrics={"reason": reason}
        )
    )


def get_run_status(run_id: str) -> Optional[Dict[str, Any]]:
    store = _require_store()
    return store.latest(run_id=run_id)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect pipeline progress sqlite store.")
    parser.add_argument("--db", type=Path, default=Path("logs/pipeline_progress.sqlite"))
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    store = PipelineProgressStore(args.db)
    latest = store.latest(args.run_id)
    if latest is None:
        print("No events found.")
        return
    print(json.dumps(latest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
