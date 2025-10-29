PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS identity_contract (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    agent_version TEXT NOT NULL,
    fingerprint TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_state (
    id INTEGER PRIMARY KEY,
    policy_name TEXT NOT NULL,
    revision INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    conversation_id TEXT,
    user_input TEXT NOT NULL,
    model_output TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('ALLOW','ESCALATE','DENY')),
    verifier_score REAL NOT NULL,
    latency_ms INTEGER,
    policy_match_score REAL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decision_log_conversation ON decision_log(conversation_id);
CREATE INDEX IF NOT EXISTS idx_decision_log_decision ON decision_log(decision);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
