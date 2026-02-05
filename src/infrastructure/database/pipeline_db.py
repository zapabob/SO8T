"""
Pipeline DB Logger for SO8T Moonshot v3.0

SQLite helper for logging pipeline runs, datasets, checkpoints, and metrics.
Safe to use even when DB is missing (no-ops with warnings).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    run_id: str
    pipeline_name: str
    model_name: Optional[str]
    base_model: Optional[str]


class PipelineDB:
    def __init__(self, db_path: str = "so8t_memory.db") -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _db_available(self) -> bool:
        return self.db_path.exists()

    def start_run(
        self,
        pipeline_name: str,
        model_name: Optional[str] = None,
        base_model: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[str]:
        if not self._db_available():
            logger.warning("[DB] so8t_memory.db not found. Skipping pipeline run logging.")
            return None

        run_id = f"{pipeline_name}-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO pipeline_runs (run_id, pipeline_name, model_name, base_model, status, notes)
                    VALUES (?, ?, ?, ?, 'running', ?)
                    """,
                    (run_id, pipeline_name, model_name, base_model, notes),
                )
                conn.commit()
            return run_id
        except Exception as exc:
            logger.error(f"[DB] Failed to start run: {exc}")
            return None

    def end_run(self, run_id: str, status: str = "completed", notes: Optional[str] = None) -> None:
        if not self._db_available() or not run_id:
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE pipeline_runs
                    SET status = ?, end_time = CURRENT_TIMESTAMP, notes = COALESCE(?, notes)
                    WHERE run_id = ?
                    """,
                    (status, notes, run_id),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[DB] Failed to end run: {exc}")

    def log_dataset(
        self,
        run_id: Optional[str],
        dataset_id: str,
        source_type: str,
        category: Optional[str] = None,
        local_path: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        sample_count: Optional[int] = None,
        acquired_via: Optional[str] = None,
    ) -> None:
        if not self._db_available():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO dataset_registry
                    (run_id, dataset_id, source_type, category, local_path, file_size_bytes, sample_count, acquired_via)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, dataset_id, source_type, category, local_path, file_size_bytes, sample_count, acquired_via),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[DB] Failed to log dataset: {exc}")

    def log_checkpoint(
        self,
        run_id: Optional[str],
        phase: str,
        checkpoint_path: str,
        resume_count: int = 0,
        notes: Optional[str] = None,
    ) -> None:
        if not self._db_available():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO checkpoint_registry
                    (run_id, phase, checkpoint_path, resume_count, notes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, phase, checkpoint_path, resume_count, notes),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[DB] Failed to log checkpoint: {exc}")

    def log_metric(
        self,
        session_id: Optional[str],
        metric_type: str,
        metric_value: float,
        status: Optional[str] = None,
        details: Optional[str] = None,
        threshold_value: Optional[float] = None,
    ) -> None:
        if not self._db_available():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO model_metrics
                    (session_id, metric_type, metric_value, threshold_value, status, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, metric_type, metric_value, threshold_value, status, details),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[DB] Failed to log metric: {exc}")

    def log_resource(
        self,
        run_id: Optional[str],
        cpu_percent: Optional[float] = None,
        memory_gb: Optional[float] = None,
        gpu_memory_gb: Optional[float] = None,
        disk_free_gb: Optional[float] = None,
    ) -> None:
        if not self._db_available():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO resource_metrics
                    (run_id, cpu_percent, memory_gb, gpu_memory_gb, disk_free_gb)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, cpu_percent, memory_gb, gpu_memory_gb, disk_free_gb),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[DB] Failed to log resource metrics: {exc}")


def get_file_size_bytes(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None
