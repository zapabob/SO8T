#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
電子カルテ管理システム（EMR: Electronic Medical Record）
- OCR統合（Tesseract）
- 画像認識（OpenCV）
- マルチモーダル統合（画像+テキスト）
- 個人情報保護（極秘扱い）
- ESCALATE優先（医療判断）
"""

import os
import sys
import cv2
import numpy as np
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pytesseract


@dataclass
class MedicalRecord:
    """カルテレコード"""
    record_id: str
    patient_id_hash: str  # ハッシュ化（個人情報保護）
    timestamp: str
    record_type: str  # text/image/multimodal
    text_content: str
    image_path: Optional[str]
    ocr_content: Optional[str]
    diagnosis: Optional[str]
    classification: str = "極秘"  # 常に極秘
    metadata: Dict = None


@dataclass
class DiagnosisAssistance:
    """診断支援結果"""
    record_id: str
    timestamp: str
    findings: List[str]
    suggestions: List[str]
    decision: str  # ALLOW（一般情報）/ESCALATE（医師判断必要）/DENY（回答不可）
    confidence: float
    explanation: str


class ElectronicMedicalRecordSystem:
    """電子カルテ管理システム"""
    
    def __init__(self, db_path: Path = Path("database/so8t_emr.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Tesseract設定（Windows環境）
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def _init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_records (
            record_id TEXT PRIMARY KEY,
            patient_id_hash TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            record_type TEXT NOT NULL,
            text_content TEXT,
            image_path TEXT,
            ocr_content TEXT,
            diagnosis TEXT,
            classification TEXT DEFAULT '極秘',
            metadata TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnosis_assistance (
            assistance_id TEXT PRIMARY KEY,
            record_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            findings TEXT,
            suggestions TEXT,
            decision TEXT NOT NULL,
            confidence REAL,
            explanation TEXT,
            FOREIGN KEY (record_id) REFERENCES medical_records(record_id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_audit (
            audit_id TEXT PRIMARY KEY,
            record_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            access_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            allowed BOOLEAN NOT NULL,
            reason TEXT,
            FOREIGN KEY (record_id) REFERENCES medical_records(record_id)
        )
        """)
        
        # インデックス
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient ON medical_records(patient_id_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON medical_records(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_timestamp ON access_audit(timestamp)")
        
        conn.commit()
        conn.close()
    
    def add_record(self, record: MedicalRecord, user_id: str) -> bool:
        """カルテ追加"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            INSERT INTO medical_records 
            (record_id, patient_id_hash, timestamp, record_type, text_content, 
             image_path, ocr_content, diagnosis, classification, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.record_id,
                record.patient_id_hash,
                record.timestamp,
                record.record_type,
                record.text_content,
                record.image_path,
                record.ocr_content,
                record.diagnosis,
                record.classification,
                json.dumps(record.metadata, ensure_ascii=False) if record.metadata else None
            ))
            
            conn.commit()
            
            # アクセス監査ログ
            self._log_access(record.record_id, user_id, "create", True, "")
            
            print(f"[OK] Medical record added: {record.record_id}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Failed to add record: {e}")
            conn.rollback()
            return False
        
        finally:
            conn.close()
    
    def process_medical_image(self, image_path: Path, patient_id: str) -> MedicalRecord:
        """医療画像処理（OCR）"""
        print(f"[PROCESS] Processing medical image: {image_path}")
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 前処理（ノイズ除去、コントラスト調整）
        denoised = cv2.fastNlMeansDenoising(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # OCR実行
        try:
            ocr_text = pytesseract.image_to_string(enhanced, lang='jpn')
        except Exception as e:
            print(f"[WARNING] OCR failed: {e}")
            ocr_text = ""
        
        # カルテレコード作成
        patient_id_hash = hashlib.sha256(patient_id.encode()).hexdigest()[:32]
        
        record = MedicalRecord(
            record_id=f"EMR_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}",
            patient_id_hash=patient_id_hash,
            timestamp=datetime.now().isoformat(),
            record_type="multimodal",
            text_content="",
            image_path=str(image_path),
            ocr_content=ocr_text,
            diagnosis=None,
            classification="極秘",
            metadata={
                "image_size": f"{image.shape[1]}x{image.shape[0]}",
                "processed_at": datetime.now().isoformat()
            }
        )
        
        print(f"[OK] Medical image processed")
        return record
    
    def _log_access(self, record_id: str, user_id: str, access_type: str, allowed: bool, reason: str):
        """アクセス監査ログ"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        audit_id = hashlib.md5(f"{record_id}_{user_id}_{datetime.now()}".encode()).hexdigest()[:16]
        
        cursor.execute("""
        INSERT INTO access_audit 
        (audit_id, record_id, user_id, access_type, timestamp, allowed, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            audit_id,
            record_id,
            user_id,
            access_type,
            datetime.now().isoformat(),
            allowed,
            reason
        ))
        
        conn.commit()
        conn.close()
    
    def provide_diagnosis_assistance(self, record_id: str) -> DiagnosisAssistance:
        """診断支援（必ずESCALATE）"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT text_content, ocr_content, diagnosis
        FROM medical_records
        WHERE record_id = ?
        """, (record_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Record not found: {record_id}")
        
        text_content, ocr_content, diagnosis = row
        
        # 簡易的な所見（実際は高度な医療AIが必要）
        findings = []
        if ocr_content:
            findings.append("OCRテキスト抽出成功")
        if diagnosis:
            findings.append(f"既存診断: {diagnosis}")
        
        # 診断支援は必ず医師判断にエスカレーション
        assistance = DiagnosisAssistance(
            record_id=record_id,
            timestamp=datetime.now().isoformat(),
            findings=findings,
            suggestions=["医師による確認が必要です", "専門的な診断が推奨されます"],
            decision="ESCALATE",
            confidence=0.5,
            explanation="医療判断は専門医の確認が必須です。AIによる診断は参考情報のみです。"
        )
        
        # 診断支援ログ保存
        self._save_diagnosis_assistance(assistance)
        
        return assistance
    
    def _save_diagnosis_assistance(self, assistance: DiagnosisAssistance):
        """診断支援ログ保存"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        assistance_id = hashlib.md5(f"{assistance.record_id}_{datetime.now()}".encode()).hexdigest()[:16]
        
        cursor.execute("""
        INSERT INTO diagnosis_assistance 
        (assistance_id, record_id, timestamp, findings, suggestions, decision, confidence, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assistance_id,
            assistance.record_id,
            assistance.timestamp,
            json.dumps(assistance.findings, ensure_ascii=False),
            json.dumps(assistance.suggestions, ensure_ascii=False),
            assistance.decision,
            assistance.confidence,
            assistance.explanation
        ))
        
        conn.commit()
        conn.close()


def test_emr_system():
    """テスト実行"""
    print("\n[TEST] Electronic Medical Record System Test")
    print("="*60)
    
    emr = ElectronicMedicalRecordSystem()
    
    # テストレコード作成
    record = MedicalRecord(
        record_id="TEST_EMR_001",
        patient_id_hash=hashlib.sha256("PATIENT001".encode()).hexdigest()[:32],
        timestamp=datetime.now().isoformat(),
        record_type="text",
        text_content="患者の症状について記録します。",
        image_path=None,
        ocr_content=None,
        diagnosis="経過観察",
        classification="極秘",
        metadata={"doctor_id": "DOC001"}
    )
    
    emr.add_record(record, user_id="DOC001")
    
    # 診断支援
    print("\n[ASSIST] Requesting diagnosis assistance...")
    assistance = emr.provide_diagnosis_assistance(record.record_id)
    print(f"Decision: {assistance.decision}")
    print(f"Explanation: {assistance.explanation}")
    
    print("\n[OK] Test completed")


if __name__ == "__main__":
    test_emr_system()

