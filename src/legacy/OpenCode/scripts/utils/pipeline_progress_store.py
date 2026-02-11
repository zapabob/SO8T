"""
Pipeline Progress Store - SQLite-based persistent tracking for pipeline execution.

Stores run_id, timestamp, phase, seed, dataset_version, checkpoint_path, and metrics
to logs/pipeline_progress.sqlite for resumability and observability.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "logs" / "pipeline_progress.sqlite"


def _get_connection() -> sqlite3.Connection:
    """Get database connection, ensuring parent directory exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema if it doesn't exist."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            seed INTEGER,
            dataset_version TEXT,
            config_hash TEXT,
            git_commit_hash TEXT,
            status TEXT DEFAULT 'running',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            step INTEGER,
            checkpoint_path TEXT,
            metrics TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            is_rolling INTEGER DEFAULT 0,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS progress_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            message TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
    """)

    conn.commit()
    conn.close()


def record_run(
    run_id: str,
    seed: Optional[int] = None,
    dataset_version: Optional[str] = None,
    config_hash: Optional[str] = None,
    git_commit_hash: Optional[str] = None,
) -> bool:
    """Record the start of a new pipeline run."""
    conn = _get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO runs (run_id, timestamp, seed, dataset_version, config_hash, git_commit_hash, status)
            VALUES (?, ?, ?, ?, ?, ?, 'running')
        """,
            (
                run_id,
                datetime.now().isoformat(),
                seed,
                dataset_version,
                config_hash,
                git_commit_hash,
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def record_checkpoint(
    run_id: str,
    phase: str,
    step: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    is_rolling: bool = False,
):
    """Record a checkpoint with phase, step, path, and metrics."""
    conn = _get_connection()
    cursor = conn.cursor()

    metrics_json = json.dumps(metrics) if metrics else None

    cursor.execute(
        """
        INSERT INTO checkpoints (run_id, phase, step, checkpoint_path, metrics, is_rolling)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (run_id, phase, step, checkpoint_path, metrics_json, 1 if is_rolling else 0),
    )

    cursor.execute(
        """
        UPDATE runs SET updated_at = datetime('now') WHERE run_id = ?
    """,
        (run_id,),
    )

    conn.commit()
    conn.close()


def log_progress(run_id: str, phase: str, message: str):
    """Log a progress message."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO progress_log (run_id, phase, message)
        VALUES (?, ?, ?)
    """,
        (run_id, phase, message),
    )

    conn.commit()
    conn.close()


def get_latest_checkpoint(run_id: str) -> Optional[Dict[str, Any]]:
    """Get the most recent checkpoint for a run."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM checkpoints
        WHERE run_id = ?
        ORDER BY id DESC
        LIMIT 1
    """,
        (run_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if row:
        result = dict(row)
        if result.get("metrics"):
            result["metrics"] = json.loads(result["metrics"])
        return result
    return None


def get_latest_rolling_checkpoint(run_id: str) -> Optional[Dict[str, Any]]:
    """Get the most recent rolling checkpoint for a run."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM checkpoints
        WHERE run_id = ? AND is_rolling = 1
        ORDER BY id DESC
        LIMIT 1
    """,
        (run_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if row:
        result = dict(row)
        if result.get("metrics"):
            result["metrics"] = json.loads(result["metrics"])
        return result
    return None


def get_run_status(run_id: str) -> Optional[Dict[str, Any]]:
    """Get the current status of a run."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def complete_run(run_id: str):
    """Mark a run as completed."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE runs SET status = 'completed', updated_at = datetime('now')
        WHERE run_id = ?
    """,
        (run_id,),
    )

    conn.commit()
    conn.close()


def fail_run(run_id: str, error_message: Optional[str] = None):
    """Mark a run as failed with optional error message."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE runs SET status = 'failed', updated_at = datetime('now')
        WHERE run_id = ?
    """,
        (run_id,),
    )

    if error_message:
        log_progress(run_id, "ERROR", error_message)

    conn.commit()
    conn.close()


def get_all_runs() -> List[Dict[str, Any]]:
    """Get all runs ordered by most recent first."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM runs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_run_history(run_id: str) -> List[Dict[str, Any]]:
    """Get full history (checkpoints and logs) for a run."""
    conn = _get_connection()
    cursor = conn.cursor()

    checkpoints = cursor.execute(
        "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()

    logs = cursor.execute(
        "SELECT * FROM progress_log WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()

    conn.close()

    return {
        "checkpoints": [dict(row) for row in checkpoints],
        "logs": [dict(row) for row in logs],
    }


def delete_run(run_id: str):
    """Delete a run and all its associated data."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM progress_log WHERE run_id = ?", (run_id,))
    cursor.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
    cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    conn.commit()
    conn.close()


def get_last_successful_step(run_id: str, phase: str) -> Optional[int]:
    """Get the last successful step for a given phase in a run."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT MAX(step) as last_step FROM checkpoints
        WHERE run_id = ? AND phase = ?
        ORDER BY timestamp DESC
    """,
        (run_id, phase),
    )

    row = cursor.fetchone()
    conn.close()

    return row["last_step"] if row else None


def get_latest_rolling_checkpoint_any() -> Optional[Dict[str, Any]]:
    """Get the most recent rolling checkpoint across all runs."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM checkpoints
        WHERE is_rolling = 1
        ORDER BY timestamp DESC
        LIMIT 1
    """
    )

    row = cursor.fetchone()
    conn.close()

    if row:
        result = dict(row)
        if result.get("metrics"):
            result["metrics"] = json.loads(result["metrics"])
        return result
    return None


def get_current_run_id() -> Optional[str]:
    """Get the most recent running run_id."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT run_id FROM runs
        WHERE status = 'running'
        ORDER BY created_at DESC
        LIMIT 1
    """
    )

    row = cursor.fetchone()
    conn.close()

    return row["run_id"] if row else None


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_PATH}")

    test_run_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    record_run(test_run_id, seed=42, dataset_version="v1.0", config_hash="abc123")
    record_checkpoint(
        test_run_id,
        "training",
        step=100,
        checkpoint_path="/ckpt/100",
        metrics={"loss": 0.5},
    )
    log_progress(test_run_id, "training", "Checkpoint saved")

    print(f"Test run created: {test_run_id}")
    print(f"Latest checkpoint: {get_latest_checkpoint(test_run_id)}")
    print(f"Run history: {get_run_history(test_run_id)}")

    complete_run(test_run_id)
    print(f"Run completed: {get_run_status(test_run_id)}")
