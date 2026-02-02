"""
SQLite backed audit trail for SO8T decision making.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


class SQLMemoryManager:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        with sqlite3.connect(self.db_path) as conn, schema_path.open("r", encoding="utf-8") as fh:
            conn.executescript(fh.read())

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    def log_decision(
        self,
        conversation_id: str,
        user_input: str,
        model_output: str,
        decision: str,
        verifier_score: float,
        latency_ms: Optional[int] = None,
        policy_match_score: Optional[float] = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO decision_log(conversation_id, user_input, model_output, decision, verifier_score, latency_ms, policy_match_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, user_input, model_output, decision, verifier_score, latency_ms, policy_match_score),
            )

    def log_audit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO audit_log(event_type, payload) VALUES (?, ?)",
                (event_type, json.dumps(payload, ensure_ascii=False)),
            )

    def fetch_recent_decisions(self, limit: int = 50) -> Iterable[Dict[str, Any]]:
        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT created_at, decision, verifier_score, conversation_id FROM decision_log ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            for row in cursor.fetchall():
                yield {
                    "created_at": row[0],
                    "decision": row[1],
                    "verifier_score": row[2],
                    "conversation_id": row[3],
                }
