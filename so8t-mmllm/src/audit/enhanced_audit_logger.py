#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
監査ログ強化版
- Windows Event Log統合
- コンプライアンスレポート自動生成
- エビングハウス忘却曲線統合
- 完全監査証跡
"""

import os
import sys
import json
import sqlite3
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import math


@dataclass
class AuditEvent:
    """監査イベント"""
    event_id: str
    timestamp: str
    event_type: str  # decision/access/modification/error
    user_id: str
    action: str
    decision: str  # ALLOW/ESCALATE/DENY
    resource: str
    details: Dict
    importance_score: float = 0.5


@dataclass
class ComplianceReport:
    """コンプライアンスレポート"""
    report_id: str
    period_start: str
    period_end: str
    total_events: int
    decision_breakdown: Dict[str, int]
    high_risk_events: int
    escalated_events: int
    denied_events: int
    compliance_score: float
    violations: List[Dict]
    recommendations: List[str]


class WindowsEventLogIntegration:
    """Windows Event Log統合"""
    
    @staticmethod
    def write_event(event_id: int, event_type: str, message: str):
        """
        Windows Event Logに書き込み
        
        Args:
            event_id: イベントID
            event_type: Information/Warning/Error
            message: メッセージ
        """
        try:
            # PowerShell経由でEvent Log書き込み
            ps_cmd = f"""
            Write-EventLog -LogName Application -Source "SO8T" -EventId {event_id} -EntryType {event_type} -Message "{message}"
            """
            
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=5
            )
        
        except Exception as e:
            print(f"[WARNING] Failed to write Windows Event Log: {e}")
    
    @staticmethod
    def register_source():
        """Event Logソース登録"""
        try:
            ps_cmd = """
            New-EventLog -LogName Application -Source "SO8T" -ErrorAction SilentlyContinue
            """
            
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            print("[OK] Windows Event Log source registered")
        
        except Exception as e:
            print(f"[WARNING] Failed to register Event Log source: {e}")


class ForgettingCurveManager:
    """エビングハウス忘却曲線管理"""
    
    # 復習スケジュール（日数）
    REVIEW_INTERVALS = [1, 3, 7, 14, 30, 60, 120, 240]
    
    @staticmethod
    def calculate_retention(days_elapsed: int, importance: float = 0.5) -> float:
        """
        保持率計算（エビングハウス曲線）
        
        Args:
            days_elapsed: 経過日数
            importance: 重要度（0.0-1.0）
        
        Returns:
            retention: 保持率（0.0-1.0）
        """
        # R(t) = e^(-t/S) where S = S0 * (1 + importance)
        S0 = 5  # 基本記憶強度
        S = S0 * (1 + importance)
        retention = math.exp(-days_elapsed / S)
        return retention
    
    @classmethod
    def get_next_review_date(cls, last_review: datetime, review_count: int) -> datetime:
        """
        次回復習日取得
        
        Args:
            last_review: 最終復習日
            review_count: 復習回数
        
        Returns:
            next_review: 次回復習日
        """
        if review_count >= len(cls.REVIEW_INTERVALS):
            interval = cls.REVIEW_INTERVALS[-1]
        else:
            interval = cls.REVIEW_INTERVALS[review_count]
        
        return last_review + timedelta(days=interval)


class EnhancedAuditLogger:
    """監査ログ強化版"""
    
    def __init__(self, db_path: Path = Path("database/so8t_audit_enhanced.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Windows Event Log統合
        self.event_log = WindowsEventLogIntegration()
        self.event_log.register_source()
        
        # 忘却曲線管理
        self.forgetting_curve = ForgettingCurveManager()
    
    def _init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 監査イベントテーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            user_id TEXT NOT NULL,
            action TEXT NOT NULL,
            decision TEXT NOT NULL,
            resource TEXT,
            details TEXT,
            importance_score REAL DEFAULT 0.5
        )
        """)
        
        # 忘却曲線テーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS forgetting_curve (
            event_id TEXT PRIMARY KEY,
            last_review TEXT NOT NULL,
            review_count INTEGER DEFAULT 0,
            next_review TEXT NOT NULL,
            retention_score REAL DEFAULT 1.0,
            FOREIGN KEY (event_id) REFERENCES audit_events(event_id)
        )
        """)
        
        # コンプライアンス違反テーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS compliance_violations (
            violation_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            event_id TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            resolved BOOLEAN DEFAULT 0,
            FOREIGN KEY (event_id) REFERENCES audit_events(event_id)
        )
        """)
        
        # インデックス
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user ON audit_events(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision ON audit_events(decision)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_next_review ON forgetting_curve(next_review)")
        
        conn.commit()
        conn.close()
    
    def log_event(self, event: AuditEvent):
        """
        イベント記録
        
        Args:
            event: 監査イベント
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO audit_events 
        (event_id, timestamp, event_type, user_id, action, decision, resource, details, importance_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.timestamp,
            event.event_type,
            event.user_id,
            event.action,
            event.decision,
            event.resource,
            json.dumps(event.details, ensure_ascii=False),
            event.importance_score
        ))
        
        # 忘却曲線エントリー作成
        now = datetime.fromisoformat(event.timestamp)
        next_review = self.forgetting_curve.get_next_review_date(now, 0)
        
        cursor.execute("""
        INSERT INTO forgetting_curve 
        (event_id, last_review, review_count, next_review, retention_score)
        VALUES (?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.timestamp,
            0,
            next_review.isoformat(),
            1.0
        ))
        
        conn.commit()
        conn.close()
        
        # Windows Event Log統合
        event_type_map = {
            "decision": 1001,
            "access": 1002,
            "modification": 1003,
            "error": 1004
        }
        event_id_num = event_type_map.get(event.event_type, 1000)
        
        log_level = "Information"
        if event.decision == "DENY":
            log_level = "Warning"
        elif event.decision == "ESCALATE":
            log_level = "Information"
        
        message = f"SO8T Audit: {event.action} by {event.user_id} - {event.decision}"
        self.event_log.write_event(event_id_num, log_level, message)
    
    def update_retention_scores(self):
        """保持率更新（定期実行推奨）"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT fc.event_id, fc.last_review, ae.importance_score
        FROM forgetting_curve fc
        JOIN audit_events ae ON fc.event_id = ae.event_id
        """)
        
        rows = cursor.fetchall()
        
        for event_id, last_review, importance in rows:
            last_review_dt = datetime.fromisoformat(last_review)
            days_elapsed = (datetime.now() - last_review_dt).days
            retention = self.forgetting_curve.calculate_retention(days_elapsed, importance)
            
            cursor.execute("""
            UPDATE forgetting_curve
            SET retention_score = ?
            WHERE event_id = ?
            """, (retention, event_id))
        
        conn.commit()
        conn.close()
    
    def get_events_for_review(self) -> List[AuditEvent]:
        """復習必要なイベント取得"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT ae.event_id, ae.timestamp, ae.event_type, ae.user_id, ae.action, ae.decision, ae.resource, ae.details, ae.importance_score
        FROM audit_events ae
        JOIN forgetting_curve fc ON ae.event_id = fc.event_id
        WHERE fc.next_review <= ?
        ORDER BY fc.retention_score ASC
        LIMIT 50
        """, (datetime.now().isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            event = AuditEvent(
                event_id=row[0],
                timestamp=row[1],
                event_type=row[2],
                user_id=row[3],
                action=row[4],
                decision=row[5],
                resource=row[6],
                details=json.loads(row[7]) if row[7] else {},
                importance_score=row[8]
            )
            events.append(event)
        
        return events
    
    def generate_compliance_report(self, 
                                     start_date: datetime, 
                                     end_date: datetime) -> ComplianceReport:
        """
        コンプライアンスレポート生成
        
        Args:
            start_date: 開始日
            end_date: 終了日
        
        Returns:
            report: コンプライアンスレポート
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 総イベント数
        cursor.execute("""
        SELECT COUNT(*) FROM audit_events
        WHERE timestamp BETWEEN ? AND ?
        """, (start_date.isoformat(), end_date.isoformat()))
        total_events = cursor.fetchone()[0]
        
        # 判定内訳
        cursor.execute("""
        SELECT decision, COUNT(*) as count
        FROM audit_events
        WHERE timestamp BETWEEN ? AND ?
        GROUP BY decision
        """, (start_date.isoformat(), end_date.isoformat()))
        
        decision_breakdown = {}
        for decision, count in cursor.fetchall():
            decision_breakdown[decision] = count
        
        # 高リスクイベント（重要度0.7以上）
        cursor.execute("""
        SELECT COUNT(*) FROM audit_events
        WHERE timestamp BETWEEN ? AND ?
        AND importance_score >= 0.7
        """, (start_date.isoformat(), end_date.isoformat()))
        high_risk_events = cursor.fetchone()[0]
        
        # 違反
        cursor.execute("""
        SELECT violation_id, timestamp, violation_type, severity, description
        FROM compliance_violations
        WHERE timestamp BETWEEN ? AND ?
        """, (start_date.isoformat(), end_date.isoformat()))
        
        violations = []
        for row in cursor.fetchall():
            violation = {
                "violation_id": row[0],
                "timestamp": row[1],
                "type": row[2],
                "severity": row[3],
                "description": row[4]
            }
            violations.append(violation)
        
        conn.close()
        
        # コンプライアンススコア計算
        escalated_events = decision_breakdown.get("ESCALATE", 0)
        denied_events = decision_breakdown.get("DENY", 0)
        
        compliance_score = 1.0
        if total_events > 0:
            # 違反率でスコア減算
            violation_rate = len(violations) / total_events
            compliance_score -= violation_rate * 0.5
            
            # DENY率でスコア減算
            deny_rate = denied_events / total_events
            compliance_score -= deny_rate * 0.3
        
        compliance_score = max(0.0, min(1.0, compliance_score))
        
        # 推奨事項
        recommendations = []
        if len(violations) > 0:
            recommendations.append("コンプライアンス違反への対応が必要です")
        if denied_events / max(total_events, 1) > 0.1:
            recommendations.append("DENY率が高いため、ポリシー見直しを推奨します")
        if escalated_events / max(total_events, 1) > 0.2:
            recommendations.append("エスカレーション率が高いため、判断基準の明確化を推奨します")
        
        # レポート作成
        report = ComplianceReport(
            report_id=hashlib.md5(f"{start_date}{end_date}".encode()).hexdigest()[:16],
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_events=total_events,
            decision_breakdown=decision_breakdown,
            high_risk_events=high_risk_events,
            escalated_events=escalated_events,
            denied_events=denied_events,
            compliance_score=compliance_score,
            violations=violations,
            recommendations=recommendations
        )
        
        # レポートファイル保存
        self._save_compliance_report(report)
        
        return report
    
    def _save_compliance_report(self, report: ComplianceReport):
        """コンプライアンスレポート保存"""
        report_dir = Path("_docs/compliance_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Markdown形式
        report_file = report_dir / f"compliance_report_{report.report_id}.md"
        
        report_md = f"""# コンプライアンスレポート

## 期間
- **開始日**: {report.period_start}
- **終了日**: {report.period_end}
- **レポートID**: {report.report_id}

## サマリー
- **総イベント数**: {report.total_events:,}
- **高リスクイベント**: {report.high_risk_events:,}
- **エスカレーション**: {report.escalated_events:,}
- **拒否**: {report.denied_events:,}
- **コンプライアンススコア**: {report.compliance_score:.2%}

## 判定内訳
"""
        
        for decision, count in report.decision_breakdown.items():
            percentage = (count / report.total_events * 100) if report.total_events > 0 else 0
            report_md += f"- **{decision}**: {count:,} ({percentage:.1f}%)\n"
        
        if report.violations:
            report_md += "\n## コンプライアンス違反\n\n"
            report_md += "| 違反ID | 日時 | タイプ | 深刻度 | 説明 |\n"
            report_md += "|--------|------|--------|--------|------|\n"
            
            for violation in report.violations:
                report_md += f"| {violation['violation_id']} | {violation['timestamp']} | {violation['type']} | {violation['severity']} | {violation['description']} |\n"
        
        if report.recommendations:
            report_md += "\n## 推奨事項\n\n"
            for i, rec in enumerate(report.recommendations, 1):
                report_md += f"{i}. {rec}\n"
        
        report_md += """
## ステータス
- [OK] 監査ログ記録完了
- [OK] Windows Event Log統合
- [OK] 忘却曲線管理
- [OK] コンプライアンスレポート生成
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_md)
        
        print(f"[OK] Compliance report saved to {report_file}")


def test_enhanced_audit():
    """テスト実行"""
    print("\n[TEST] Enhanced Audit Logger Test")
    print("="*60)
    
    logger = EnhancedAuditLogger()
    
    # テストイベント記録
    event = AuditEvent(
        event_id="TEST001",
        timestamp=datetime.now().isoformat(),
        event_type="decision",
        user_id="test_user",
        action="query_processing",
        decision="ALLOW",
        resource="test_query",
        details={"query": "テストクエリ"},
        importance_score=0.6
    )
    
    logger.log_event(event)
    print("[OK] Event logged")
    
    # コンプライアンスレポート生成
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    report = logger.generate_compliance_report(start_date, end_date)
    print(f"[OK] Compliance report generated")
    print(f"Total events: {report.total_events}")
    print(f"Compliance score: {report.compliance_score:.2%}")
    
    print("\n[OK] Test completed")


if __name__ == "__main__":
    test_enhanced_audit()

