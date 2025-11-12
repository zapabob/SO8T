"""
SO8T Compliance Logger

This module provides comprehensive logging for compliance, audit, and inference tracking.
It records safety judgments (ALLOW/ESCALATION/DENY), audit trails, and inference processes
for regulatory compliance and transparency.
"""

import sqlite3
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import uuid
import getpass
import socket

logger = logging.getLogger(__name__)


class SO8TComplianceLogger:
    """
    SO8T Compliance Logger
    
    Features:
    - Safety judgment logging (ALLOW/ESCALATION/DENY)
    - Audit trail tracking (who, what, when, where)
    - Inference process logging (inputs, outputs, reasoning)
    - Compliance reporting
    - Data retention management
    - GDPR/regulatory compliance support
    """
    
    def __init__(self, db_path: str = "database/so8t_compliance.db"):
        """
        Initialize compliance logger
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self.conn = None
        self.current_user = getpass.getuser()
        self.current_hostname = socket.gethostname()
        self._connect()
        self._create_tables()
        
    def _connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Enable WAL mode for better concurrency
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA cache_size=10000")
            
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys=ON")
            
            logger.info(f"Connected to compliance database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            
            # Safety judgments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS safety_judgments (
                    judgment_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    judgment TEXT NOT NULL CHECK(judgment IN ('ALLOW', 'ESCALATION', 'DENY')),
                    confidence_score REAL NOT NULL,
                    safety_score REAL,
                    reasoning TEXT,
                    model_version TEXT,
                    hostname TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    request_metadata TEXT
                )
            """)
            
            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    action_result TEXT NOT NULL CHECK(action_result IN ('SUCCESS', 'FAILURE', 'DENIED')),
                    details TEXT,
                    ip_address TEXT,
                    hostname TEXT,
                    user_agent TEXT,
                    security_level TEXT,
                    compliance_tags TEXT
                )
            """)
            
            # Inference log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_log (
                    inference_id TEXT PRIMARY KEY,
                    judgment_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    processing_time_ms REAL,
                    temperature REAL,
                    top_p REAL,
                    max_tokens INTEGER,
                    input_hash TEXT NOT NULL,
                    output_hash TEXT NOT NULL,
                    input_summary TEXT,
                    output_summary TEXT,
                    reasoning_steps TEXT,
                    group_structure_state TEXT,
                    triality_weights TEXT,
                    pet_loss REAL,
                    safety_head_output TEXT,
                    task_head_output TEXT,
                    authority_head_output TEXT,
                    FOREIGN KEY (judgment_id) REFERENCES safety_judgments(judgment_id)
                )
            """)
            
            # Escalation log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escalation_log (
                    escalation_id TEXT PRIMARY KEY,
                    judgment_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    escalation_reason TEXT NOT NULL,
                    escalation_type TEXT NOT NULL CHECK(escalation_type IN ('SAFETY', 'AUTHORITY', 'COMPLEXITY', 'ETHICAL')),
                    priority TEXT NOT NULL CHECK(priority IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                    assigned_to TEXT,
                    status TEXT DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'IN_REVIEW', 'APPROVED', 'REJECTED', 'ESCALATED_FURTHER')),
                    resolution TEXT,
                    resolution_timestamp TIMESTAMP,
                    resolved_by TEXT,
                    human_judgment TEXT CHECK(human_judgment IN ('ALLOW', 'DENY', NULL)),
                    override_reason TEXT,
                    FOREIGN KEY (judgment_id) REFERENCES safety_judgments(judgment_id)
                )
            """)
            
            # Compliance report table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id TEXT PRIMARY KEY,
                    report_type TEXT NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL,
                    generated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    generated_by TEXT NOT NULL,
                    total_requests INTEGER,
                    allow_count INTEGER,
                    escalation_count INTEGER,
                    deny_count INTEGER,
                    avg_confidence REAL,
                    avg_processing_time_ms REAL,
                    compliance_score REAL,
                    report_data TEXT,
                    report_format TEXT
                )
            """)
            
            # Data retention policy table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_retention_policy (
                    policy_id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    retention_days INTEGER NOT NULL,
                    last_cleanup TIMESTAMP,
                    next_cleanup TIMESTAMP,
                    records_deleted INTEGER DEFAULT 0,
                    policy_status TEXT DEFAULT 'ACTIVE' CHECK(policy_status IN ('ACTIVE', 'SUSPENDED', 'INACTIVE'))
                )
            """)
            
            # Create indexes separately
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_safety_judgment ON safety_judgments(judgment)",
                "CREATE INDEX IF NOT EXISTS idx_safety_timestamp ON safety_judgments(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_safety_user ON safety_judgments(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_safety_session ON safety_judgments(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)",
                "CREATE INDEX IF NOT EXISTS idx_audit_result ON audit_log(action_result)",
                "CREATE INDEX IF NOT EXISTS idx_inference_timestamp ON inference_log(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_inference_session ON inference_log(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_inference_judgment ON inference_log(judgment_id)",
                "CREATE INDEX IF NOT EXISTS idx_escalation_timestamp ON escalation_log(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_escalation_status ON escalation_log(status)",
                "CREATE INDEX IF NOT EXISTS idx_escalation_priority ON escalation_log(priority)",
                "CREATE INDEX IF NOT EXISTS idx_report_timestamp ON compliance_reports(generated_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_report_type ON compliance_reports(report_type)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            self.conn.commit()
            logger.info("Compliance database tables and indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def _compute_hash(self, text: str) -> str:
        """
        Compute SHA256 hash of text
        
        Args:
            text: Text to hash
            
        Returns:
            Hex digest of hash
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def log_safety_judgment(
        self,
        session_id: str,
        input_text: str,
        judgment: str,
        confidence_score: float,
        safety_score: Optional[float] = None,
        reasoning: Optional[str] = None,
        model_version: Optional[str] = None,
        user_id: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a safety judgment
        
        Args:
            session_id: Session ID
            input_text: Input text that was judged
            judgment: Safety judgment (ALLOW/ESCALATION/DENY)
            confidence_score: Confidence score (0-1)
            safety_score: Safety score (0-1)
            reasoning: Reasoning for the judgment
            model_version: Model version
            user_id: User ID (defaults to current user)
            request_metadata: Additional request metadata
            
        Returns:
            Judgment ID
        """
        try:
            if judgment not in ['ALLOW', 'ESCALATION', 'DENY']:
                raise ValueError(f"Invalid judgment: {judgment}")
            
            judgment_id = str(uuid.uuid4())
            user_id = user_id or self.current_user
            input_hash = self._compute_hash(input_text)
            
            # Serialize metadata
            metadata_json = json.dumps(request_metadata) if request_metadata else None
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO safety_judgments (
                    judgment_id, user_id, session_id, input_text, input_hash,
                    judgment, confidence_score, safety_score, reasoning,
                    model_version, hostname, request_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                judgment_id, user_id, session_id, input_text, input_hash,
                judgment, confidence_score, safety_score, reasoning,
                model_version, self.current_hostname, metadata_json
            ))
            
            self.conn.commit()
            logger.info(f"Logged safety judgment: {judgment_id} ({judgment})")
            
            # Log audit trail
            self.log_audit_action(
                user_id=user_id,
                action="SAFETY_JUDGMENT",
                resource_type="inference",
                resource_id=judgment_id,
                action_result="SUCCESS",
                details=f"Judgment: {judgment}, Confidence: {confidence_score:.3f}"
            )
            
            return judgment_id
            
        except Exception as e:
            logger.error(f"Failed to log safety judgment: {e}")
            raise
    
    def log_audit_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        action_result: str,
        resource_id: Optional[str] = None,
        details: Optional[str] = None,
        security_level: Optional[str] = None,
        compliance_tags: Optional[List[str]] = None
    ) -> str:
        """
        Log an audit action
        
        Args:
            user_id: User ID
            action: Action performed
            resource_type: Type of resource
            action_result: Result (SUCCESS/FAILURE/DENIED)
            resource_id: Resource ID
            details: Additional details
            security_level: Security level
            compliance_tags: Compliance tags
            
        Returns:
            Audit ID
        """
        try:
            if action_result not in ['SUCCESS', 'FAILURE', 'DENIED']:
                raise ValueError(f"Invalid action result: {action_result}")
            
            audit_id = str(uuid.uuid4())
            
            # Serialize compliance tags
            tags_json = json.dumps(compliance_tags) if compliance_tags else None
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (
                    audit_id, user_id, action, resource_type, resource_id,
                    action_result, details, hostname, security_level, compliance_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_id, user_id, action, resource_type, resource_id,
                action_result, details, self.current_hostname, security_level, tags_json
            ))
            
            self.conn.commit()
            logger.debug(f"Logged audit action: {audit_id} ({action})")
            
            return audit_id
            
        except Exception as e:
            logger.error(f"Failed to log audit action: {e}")
            raise
    
    def log_inference(
        self,
        session_id: str,
        model_name: str,
        model_version: str,
        input_text: str,
        output_text: str,
        judgment_id: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        reasoning_steps: Optional[List[str]] = None,
        group_structure_state: Optional[Dict[str, Any]] = None,
        triality_weights: Optional[Dict[str, float]] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an inference process
        
        Args:
            session_id: Session ID
            model_name: Model name
            model_version: Model version
            input_text: Input text
            output_text: Output text
            judgment_id: Associated judgment ID
            processing_time_ms: Processing time in milliseconds
            reasoning_steps: List of reasoning steps
            group_structure_state: SO8T group structure state
            triality_weights: Triality head weights
            generation_params: Generation parameters
            
        Returns:
            Inference ID
        """
        try:
            inference_id = str(uuid.uuid4())
            input_hash = self._compute_hash(input_text)
            output_hash = self._compute_hash(output_text)
            
            # Create summaries (first 200 chars)
            input_summary = input_text[:200] + "..." if len(input_text) > 200 else input_text
            output_summary = output_text[:200] + "..." if len(output_text) > 200 else output_text
            
            # Serialize complex data
            reasoning_json = json.dumps(reasoning_steps) if reasoning_steps else None
            group_state_json = json.dumps(group_structure_state) if group_structure_state else None
            triality_json = json.dumps(triality_weights) if triality_weights else None
            
            # Extract generation params
            temperature = generation_params.get('temperature') if generation_params else None
            top_p = generation_params.get('top_p') if generation_params else None
            max_tokens = generation_params.get('max_tokens') if generation_params else None
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO inference_log (
                    inference_id, judgment_id, session_id, model_name, model_version,
                    processing_time_ms, temperature, top_p, max_tokens,
                    input_hash, output_hash, input_summary, output_summary,
                    reasoning_steps, group_structure_state, triality_weights
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inference_id, judgment_id, session_id, model_name, model_version,
                processing_time_ms, temperature, top_p, max_tokens,
                input_hash, output_hash, input_summary, output_summary,
                reasoning_json, group_state_json, triality_json
            ))
            
            self.conn.commit()
            logger.info(f"Logged inference: {inference_id}")
            
            return inference_id
            
        except Exception as e:
            logger.error(f"Failed to log inference: {e}")
            raise
    
    def log_escalation(
        self,
        judgment_id: str,
        escalation_reason: str,
        escalation_type: str,
        priority: str,
        assigned_to: Optional[str] = None
    ) -> str:
        """
        Log an escalation
        
        Args:
            judgment_id: Associated judgment ID
            escalation_reason: Reason for escalation
            escalation_type: Type (SAFETY/AUTHORITY/COMPLEXITY/ETHICAL)
            priority: Priority (LOW/MEDIUM/HIGH/CRITICAL)
            assigned_to: User assigned to review
            
        Returns:
            Escalation ID
        """
        try:
            if escalation_type not in ['SAFETY', 'AUTHORITY', 'COMPLEXITY', 'ETHICAL']:
                raise ValueError(f"Invalid escalation type: {escalation_type}")
            
            if priority not in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                raise ValueError(f"Invalid priority: {priority}")
            
            escalation_id = str(uuid.uuid4())
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO escalation_log (
                    escalation_id, judgment_id, escalation_reason,
                    escalation_type, priority, assigned_to
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                escalation_id, judgment_id, escalation_reason,
                escalation_type, priority, assigned_to
            ))
            
            self.conn.commit()
            logger.info(f"Logged escalation: {escalation_id} ({escalation_type}, {priority})")
            
            # Log audit trail
            self.log_audit_action(
                user_id=self.current_user,
                action="ESCALATION_CREATED",
                resource_type="escalation",
                resource_id=escalation_id,
                action_result="SUCCESS",
                details=f"Type: {escalation_type}, Priority: {priority}",
                security_level=priority
            )
            
            return escalation_id
            
        except Exception as e:
            logger.error(f"Failed to log escalation: {e}")
            raise
    
    def resolve_escalation(
        self,
        escalation_id: str,
        resolution: str,
        human_judgment: str,
        resolved_by: str,
        override_reason: Optional[str] = None
    ):
        """
        Resolve an escalation
        
        Args:
            escalation_id: Escalation ID
            resolution: Resolution description
            human_judgment: Human judgment (ALLOW/DENY)
            resolved_by: User who resolved
            override_reason: Reason for override if any
        """
        try:
            if human_judgment not in ['ALLOW', 'DENY']:
                raise ValueError(f"Invalid human judgment: {human_judgment}")
            
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE escalation_log
                SET status = 'APPROVED',
                    resolution = ?,
                    resolution_timestamp = CURRENT_TIMESTAMP,
                    resolved_by = ?,
                    human_judgment = ?,
                    override_reason = ?
                WHERE escalation_id = ?
            """, (resolution, resolved_by, human_judgment, override_reason, escalation_id))
            
            self.conn.commit()
            logger.info(f"Resolved escalation: {escalation_id} ({human_judgment})")
            
            # Log audit trail
            self.log_audit_action(
                user_id=resolved_by,
                action="ESCALATION_RESOLVED",
                resource_type="escalation",
                resource_id=escalation_id,
                action_result="SUCCESS",
                details=f"Human judgment: {human_judgment}",
                security_level="HIGH"
            )
            
        except Exception as e:
            logger.error(f"Failed to resolve escalation: {e}")
            raise
    
    def get_compliance_statistics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get compliance statistics
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing statistics
        """
        try:
            cursor = self.conn.cursor()
            
            # Build WHERE clause for date range
            where_clause = "WHERE 1=1"
            params = []
            if start_date:
                where_clause += " AND DATE(timestamp) >= ?"
                params.append(start_date)
            if end_date:
                where_clause += " AND DATE(timestamp) <= ?"
                params.append(end_date)
            
            # Safety judgment statistics
            cursor.execute(f"""
                SELECT 
                    judgment,
                    COUNT(*) as count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(safety_score) as avg_safety_score
                FROM safety_judgments
                {where_clause}
                GROUP BY judgment
            """, params)
            
            judgment_stats = {}
            for row in cursor.fetchall():
                judgment_stats[row['judgment']] = {
                    'count': row['count'],
                    'avg_confidence': row['avg_confidence'],
                    'avg_safety_score': row['avg_safety_score']
                }
            
            # Escalation statistics
            cursor.execute(f"""
                SELECT 
                    escalation_type,
                    priority,
                    status,
                    COUNT(*) as count
                FROM escalation_log
                {where_clause}
                GROUP BY escalation_type, priority, status
            """, params)
            
            escalation_stats = []
            for row in cursor.fetchall():
                escalation_stats.append({
                    'type': row['escalation_type'],
                    'priority': row['priority'],
                    'status': row['status'],
                    'count': row['count']
                })
            
            # Audit statistics
            cursor.execute(f"""
                SELECT 
                    action,
                    action_result,
                    COUNT(*) as count
                FROM audit_log
                {where_clause}
                GROUP BY action, action_result
            """, params)
            
            audit_stats = []
            for row in cursor.fetchall():
                audit_stats.append({
                    'action': row['action'],
                    'result': row['action_result'],
                    'count': row['count']
                })
            
            # Total counts
            cursor.execute(f"""
                SELECT COUNT(*) as total FROM safety_judgments {where_clause}
            """, params)
            total_judgments = cursor.fetchone()['total']
            
            return {
                'period': {
                    'start_date': start_date or 'inception',
                    'end_date': end_date or 'now'
                },
                'total_judgments': total_judgments,
                'judgment_breakdown': judgment_stats,
                'escalation_breakdown': escalation_stats,
                'audit_breakdown': audit_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get compliance statistics: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Closed compliance database connection")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Create compliance logger
    compliance_logger = SO8TComplianceLogger()
    
    # Log a safety judgment
    session_id = str(uuid.uuid4())
    judgment_id = compliance_logger.log_safety_judgment(
        session_id=session_id,
        input_text="ユーザーの個人情報を教えてください",
        judgment="DENY",
        confidence_score=0.95,
        safety_score=0.15,
        reasoning="個人情報の開示要求のため、プライバシー保護の観点から拒否",
        model_version="SO8T-1.0.0"
    )
    
    # Log inference
    inference_id = compliance_logger.log_inference(
        session_id=session_id,
        model_name="SO8T-Distilled-Safety",
        model_version="1.0.0",
        input_text="ユーザーの個人情報を教えてください",
        output_text="申し訳ございませんが、個人情報の開示はできません。",
        judgment_id=judgment_id,
        processing_time_ms=125.5,
        reasoning_steps=[
            "入力解析: 個人情報要求を検出",
            "安全性評価: 低安全スコア (0.15)",
            "判定: DENY",
            "理由生成: プライバシー保護"
        ],
        triality_weights={
            "task": 0.3,
            "safety": 0.9,
            "authority": 0.6
        }
    )
    
    # Get statistics
    stats = compliance_logger.get_compliance_statistics()
    print("\nCompliance Statistics:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Close
    compliance_logger.close()

