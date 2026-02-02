-- SO8T Safety Pipeline Database Schema
-- SQLite database for conversation history and knowledge base management

-- Enable WAL mode for better concurrent access
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;

-- Conversation history table
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_input TEXT NOT NULL,
    safety_judgment TEXT CHECK(safety_judgment IN ('ALLOW', 'ESCALATION', 'DENY')),
    model_response TEXT NOT NULL,
    rotation_state BLOB,  -- SO(8)群の状態保存
    confidence_score REAL,
    processing_time_ms INTEGER,
    input_type TEXT CHECK(input_type IN ('text', 'image', 'multimodal')),
    ocr_text TEXT,  -- OCRで抽出されたテキスト（画像入力の場合）
    ocr_confidence REAL,  -- OCR信頼度
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge base table for storing learned information
CREATE TABLE knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,  -- ベクトル埋め込み（SO(8)群構造）
    embedding_dim INTEGER DEFAULT 4096,
    source_type TEXT CHECK(source_type IN ('conversation', 'document', 'manual', 'distillation')),
    source_id INTEGER,  -- 参照元のID
    confidence REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Safety patterns table for storing dangerous patterns
CREATE TABLE safety_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT CHECK(pattern_type IN ('harmful_content', 'illegal_activity', 'personal_info', 'bias', 'misinformation')),
    pattern_text TEXT NOT NULL,
    severity_level INTEGER CHECK(severity_level BETWEEN 1 AND 5),  -- 1=低, 5=高
    action_required TEXT CHECK(action_required IN ('ALLOW', 'ESCALATION', 'DENY')),
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Model performance metrics table
CREATE TABLE model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    metric_type TEXT CHECK(metric_type IN ('safety_accuracy', 'response_time', 'confidence', 'rotation_stability')),
    metric_value REAL NOT NULL,
    threshold_value REAL,
    status TEXT CHECK(status IN ('pass', 'fail', 'warning')),
    details TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- SO(8) group state tracking table
CREATE TABLE so8_group_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    layer_index INTEGER NOT NULL,
    rotation_matrix BLOB NOT NULL,  -- 8x8回転行列
    rotation_angles BLOB NOT NULL,  -- 8次元回転角度
    group_stability REAL,  -- 群の安定性指標
    pet_penalty REAL,  -- PET正則化ペナルティ
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_conversation_session ON conversation_history(session_id);
CREATE INDEX idx_conversation_timestamp ON conversation_history(timestamp);
CREATE INDEX idx_conversation_safety ON conversation_history(safety_judgment);
CREATE INDEX idx_conversation_type ON conversation_history(input_type);

CREATE INDEX idx_knowledge_topic ON knowledge_base(topic);
CREATE INDEX idx_knowledge_source ON knowledge_base(source_type, source_id);
CREATE INDEX idx_knowledge_confidence ON knowledge_base(confidence);
CREATE INDEX idx_knowledge_accessed ON knowledge_base(last_accessed);

CREATE INDEX idx_safety_pattern_type ON safety_patterns(pattern_type);
CREATE INDEX idx_safety_severity ON safety_patterns(severity_level);
CREATE INDEX idx_safety_action ON safety_patterns(action_required);

CREATE INDEX idx_metrics_session ON model_metrics(session_id);
CREATE INDEX idx_metrics_type ON model_metrics(metric_type);
CREATE INDEX idx_metrics_timestamp ON model_metrics(timestamp);

CREATE INDEX idx_so8_session ON so8_group_states(session_id);
CREATE INDEX idx_so8_layer ON so8_group_states(layer_index);
CREATE INDEX idx_so8_timestamp ON so8_group_states(timestamp);

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_conversation_timestamp 
    AFTER UPDATE ON conversation_history
    BEGIN
        UPDATE conversation_history 
        SET updated_at = CURRENT_TIMESTAMP 
        WHERE id = NEW.id;
    END;

CREATE TRIGGER update_knowledge_timestamp 
    AFTER UPDATE ON knowledge_base
    BEGIN
        UPDATE knowledge_base 
        SET updated_at = CURRENT_TIMESTAMP 
        WHERE id = NEW.id;
    END;

CREATE TRIGGER update_safety_timestamp 
    AFTER UPDATE ON safety_patterns
    BEGIN
        UPDATE safety_patterns 
        SET updated_at = CURRENT_TIMESTAMP 
        WHERE id = NEW.id;
    END;

-- Insert initial safety patterns
INSERT INTO safety_patterns (pattern_type, pattern_text, severity_level, action_required, description) VALUES ('harmful_content', '爆弾|爆発物|武器|殺人|自殺', 5, 'DENY', '危険な内容の検出');
INSERT INTO safety_patterns (pattern_type, pattern_text, severity_level, action_required, description) VALUES ('illegal_activity', '違法|犯罪|麻薬|覚醒剤|密売', 5, 'DENY', '違法行為の検出');
INSERT INTO safety_patterns (pattern_type, pattern_text, severity_level, action_required, description) VALUES ('personal_info', '個人情報|住所|電話番号|クレジットカード', 4, 'ESCALATION', '個人情報の検出');
INSERT INTO safety_patterns (pattern_type, pattern_text, severity_level, action_required, description) VALUES ('bias', '差別|偏見|ヘイト|人種差別|性差別', 4, 'ESCALATION', '偏見・差別的内容の検出');
INSERT INTO safety_patterns (pattern_type, pattern_text, severity_level, action_required, description) VALUES ('misinformation', 'デマ|フェイク|偽情報|陰謀論', 3, 'ESCALATION', '誤情報の検出');

-- Create views for common queries
CREATE VIEW recent_conversations AS
SELECT 
    session_id,
    COUNT(*) as message_count,
    MAX(timestamp) as last_activity,
    AVG(confidence_score) as avg_confidence,
    COUNT(CASE WHEN safety_judgment = 'DENY' THEN 1 END) as deny_count
FROM conversation_history
GROUP BY session_id
ORDER BY last_activity DESC;

CREATE VIEW safety_statistics AS
SELECT 
    safety_judgment,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time
FROM conversation_history
GROUP BY safety_judgment;

CREATE VIEW knowledge_topics AS
SELECT 
    topic,
    COUNT(*) as reference_count,
    AVG(confidence) as avg_confidence,
    MAX(last_accessed) as last_accessed
FROM knowledge_base
GROUP BY topic
ORDER BY reference_count DESC;

-- Vacuum and analyze for optimal performance
VACUUM;
ANALYZE;
