from pathlib import Path

from safety_sql.sqlmm import SQLMemoryManager


def test_sql_memory_manager_logs(tmp_path: Path):
    db_path = tmp_path / "audit.db"
    manager = SQLMemoryManager(db_path)
    manager.log_decision(
        conversation_id="c1",
        user_input="hello",
        model_output="allow",
        decision="ALLOW",
        verifier_score=0.8,
    )
    rows = list(manager.fetch_recent_decisions())
    assert len(rows) == 1
    assert rows[0]["decision"] == "ALLOW"
