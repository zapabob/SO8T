#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import implementation logs into so8t_memory.db knowledge_base table.

Usage:
  py -3 scripts/utils/import_implementation_logs.py
  py -3 scripts/utils/import_implementation_logs.py --db so8t_memory.db --logs-dir _docs
"""

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime


def ensure_database(db_path: Path, schema_path: Path) -> None:
    """Create database schema if DB is missing or empty."""
    if db_path.exists() and db_path.stat().st_size > 0:
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    schema_sql = schema_path.read_text(encoding="utf-8")
    cur.executescript(schema_sql)

    # database_info is not part of create_schema.sql, add if missing
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS database_info (
            id INTEGER PRIMARY KEY,
            version TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
        """
    )
    cur.execute("SELECT COUNT(*) FROM database_info")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO database_info (version, description) VALUES (?, ?)",
            ("1.1.0", "SO8T Moonshot v3.0 Pipeline Database - Dataset/Run/Checkpoint Metrics"),
        )
    conn.commit()
    conn.close()


def load_logs(logs_dir: Path) -> list[tuple[str, str]]:
    """Collect implementation logs from _docs directory."""
    # Prefer dated implementation logs (YYYY-mm-dd*.md)
    log_files = sorted(logs_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*.md"))
    # Fallback to explicit implementation log filenames
    log_files += sorted(logs_dir.glob("IMPLEMENTATION_LOG*.md"))
    logs = []
    for path in log_files:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="ignore")
        logs.append((path.name, content))
    return logs


def upsert_logs(conn: sqlite3.Connection, logs: list[tuple[str, str]]) -> int:
    cur = conn.cursor()
    inserted = 0
    for filename, content in logs:
        cur.execute(
            """
            SELECT id FROM knowledge_base
            WHERE topic = ? AND source_type = 'document'
            """,
            (filename,),
        )
        row = cur.fetchone()
        if row:
            cur.execute(
                """
                UPDATE knowledge_base
                SET content = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (content, row[0]),
            )
        else:
            cur.execute(
                """
                INSERT INTO knowledge_base
                (topic, content, source_type, source_id, confidence, created_at, updated_at)
                VALUES (?, ?, 'document', NULL, 1.0, ?, ?)
                """,
                (filename, content, datetime.now().isoformat(), datetime.now().isoformat()),
            )
            inserted += 1
    conn.commit()
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Import implementation logs into so8t_memory.db")
    parser.add_argument("--db", type=str, default="so8t_memory.db", help="Database path")
    parser.add_argument("--logs-dir", type=str, default="_docs", help="Directory containing implementation logs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    db_path = (project_root / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
    logs_dir = (project_root / args.logs_dir).resolve() if not Path(args.logs_dir).is_absolute() else Path(args.logs_dir)
    schema_path = project_root / "database" / "create_schema.sql"

    if not logs_dir.exists():
        raise SystemExit(f"Logs directory not found: {logs_dir}")

    ensure_database(db_path, schema_path)
    logs = load_logs(logs_dir)
    if not logs:
        print(f"No implementation logs found in {logs_dir}")
        return 0

    conn = sqlite3.connect(str(db_path))
    inserted = upsert_logs(conn, logs)
    conn.close()
    print(f"Imported {len(logs)} logs ({inserted} new, {len(logs) - inserted} updated) into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
