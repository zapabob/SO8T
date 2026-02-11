#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画像検知エージェント（監視カメラ・異物混入・不審者検知）
- OpenCV統合
- YOLO物体検知（オプション）
- 危険予知、異常検知
- カルテ画像OCR
"""

import os
import sys
import cv2
import numpy as np
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import time


@dataclass
class DetectionResult:
    """検知結果"""
    detection_id: str
    timestamp: str
    detection_type: str  # suspicious_person/foreign_object/danger/abnormal
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    image_path: str
    decision: str  # ALLOW/ESCALATE/DENY
    description: str


class VisualDetectionAgent:
    """画像検知エージェント"""
    
    def __init__(self, db_path: Path = Path("database/so8t_visual_detection.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # OpenCV設定
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # 監視カメラ
        self.camera = None
        self.monitoring = False
        self.monitor_thread = None
    
    def _init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            detection_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            detection_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox TEXT,
            image_path TEXT,
            decision TEXT NOT NULL,
            description TEXT
        )
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_type ON detections(detection_type)
        """)
        
        conn.commit()
        conn.close()
    
    def log_detection(self, result: DetectionResult):
        """検知ログ記録"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO detections 
        (detection_id, timestamp, detection_type, confidence, bbox, image_path, decision, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.detection_id,
            result.timestamp,
            result.detection_type,
            result.confidence,
            json.dumps(result.bbox),
            result.image_path,
            result.decision,
            result.description
        ))
        
        conn.commit()
        conn.close()
    
    def detect_suspicious_person(self, image: np.ndarray) -> List[DetectionResult]:
        """不審者検知"""
        detections = []
        
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 顔検知
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # 簡易的な不審者判定（実際はより高度な分析が必要）
            # 例: 顔の位置、サイズ、時間帯などで判定
            confidence = 0.7  # 仮の確信度
            
            result = DetectionResult(
                detection_id=f"SUSP_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}",
                timestamp=datetime.now().isoformat(),
                detection_type="suspicious_person",
                confidence=confidence,
                bbox=(int(x), int(y), int(w), int(h)),
                image_path="",
                decision="ESCALATE" if confidence > 0.8 else "ALLOW",
                description=f"人物検知: 確信度{confidence:.2f}"
            )
            
            detections.append(result)
            self.log_detection(result)
        
        return detections
    
    def detect_foreign_object(self, image: np.ndarray, reference_image: Optional[np.ndarray] = None) -> List[DetectionResult]:
        """異物混入検知（差分検出）"""
        detections = []
        
        if reference_image is None:
            return detections
        
        # 差分検出
        diff = cv2.absdiff(image, reference_image)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 閾値処理
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # 輪郭検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 小さすぎる変化は無視
            if area < 100:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            confidence = min(area / 10000, 1.0)  # 面積ベース確信度
            
            result = DetectionResult(
                detection_id=f"FOBJ_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}",
                timestamp=datetime.now().isoformat(),
                detection_type="foreign_object",
                confidence=confidence,
                bbox=(int(x), int(y), int(w), int(h)),
                image_path="",
                decision="ESCALATE",
                description=f"異物検知: 面積{area:.0f}px"
            )
            
            detections.append(result)
            self.log_detection(result)
        
        return detections
    
    def detect_danger(self, image: np.ndarray) -> List[DetectionResult]:
        """危険検知（色ベース検出例：赤色検知）"""
        detections = []
        
        # HSV変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 赤色範囲（火災、血液など）
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 500:  # 小さい領域は無視
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            confidence = min(area / 5000, 1.0)
            
            result = DetectionResult(
                detection_id=f"DANG_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}",
                timestamp=datetime.now().isoformat(),
                detection_type="danger",
                confidence=confidence,
                bbox=(int(x), int(y), int(w), int(h)),
                image_path="",
                decision="ESCALATE",
                description=f"危険検知（赤色領域）: 面積{area:.0f}px"
            )
            
            detections.append(result)
            self.log_detection(result)
        
        return detections
    
    def process_image(self, image_path: Path, detection_types: List[str] = None) -> List[DetectionResult]:
        """画像処理"""
        if detection_types is None:
            detection_types = ["suspicious_person", "danger"]
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            return []
        
        all_detections = []
        
        # 検知実行
        if "suspicious_person" in detection_types:
            detections = self.detect_suspicious_person(image)
            all_detections.extend(detections)
        
        if "danger" in detection_types:
            detections = self.detect_danger(image)
            all_detections.extend(detections)
        
        # 検知結果を画像に描画（オプション）
        output_dir = Path("outputs/visual_detection")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for detection in all_detections:
            x, y, w, h = detection.bbox
            color = (0, 0, 255) if detection.decision == "ESCALATE" else (0, 255, 0)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, detection.detection_type, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        output_path = output_dir / f"detected_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(str(output_path), image)
        
        return all_detections
    
    def start_camera_monitoring(self, camera_id: int = 0, interval: int = 5):
        """監視カメラ監視開始"""
        print(f"[START] Camera monitoring (camera: {camera_id}, interval: {interval}s)")
        
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            print(f"[ERROR] Failed to open camera {camera_id}")
            return
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                ret, frame = self.camera.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    time.sleep(interval)
                    continue
                
                # 検知実行
                detections = []
                detections.extend(self.detect_suspicious_person(frame))
                detections.extend(self.detect_danger(frame))
                
                if detections:
                    print(f"[ALERT] Detected {len(detections)} events")
                    for detection in detections:
                        print(f"  - {detection.detection_type}: {detection.description}")
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_camera_monitoring(self):
        """監視カメラ監視停止"""
        print("[STOP] Stopping camera monitoring")
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if self.camera:
            self.camera.release()
    
    def generate_report(self, output_path: Path = None):
        """レポート生成"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT detection_type, decision, COUNT(*) as count
        FROM detections
        GROUP BY detection_type, decision
        """)
        
        stats = cursor.fetchall()
        conn.close()
        
        if output_path is None:
            output_path = Path("_docs") / f"{datetime.now().strftime('%Y-%m-%d')}_visual_detection_report.md"
        
        output_path.parent.mkdir(exist_ok=True)
        
        report = f"""# 画像検知レポート

## 検知概要
- **レポート日時**: {datetime.now().isoformat()}
- **総検知数**: {sum(row[2] for row in stats)}

## 種類別統計

| 検知タイプ | 判定 | 検知数 |
|-----------|------|--------|
"""
        
        for detection_type, decision, count in stats:
            report += f"| {detection_type} | {decision} | {count} |\n"
        
        report += """
## 検知機能
- 不審者検知（顔認識ベース）
- 異物混入検知（差分検出）
- 危険検知（色ベース）
- カルテ画像OCR（準備中）

## 次のステップ
- [READY] YOLOベース高精度検知
- [READY] マルチカメラ統合
- [READY] リアルタイム処理最適化
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Report saved to {output_path}")


def test_visual_detection():
    """テスト実行"""
    print("\n[TEST] Visual Detection Agent Test")
    print("="*60)
    
    agent = VisualDetectionAgent()
    
    # テスト画像生成（ダミー）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 255), -1)  # 赤色矩形
    
    test_image_path = Path("test_image.jpg")
    cv2.imwrite(str(test_image_path), test_image)
    
    # 検知テスト
    print("\n[DETECT] Processing test image...")
    detections = agent.process_image(test_image_path, ["danger"])
    print(f"[OK] Found {len(detections)} detections")
    
    for detection in detections:
        print(f"  - {detection.detection_type}: {detection.description}")
    
    # レポート生成
    print("\n[REPORT] Generating report...")
    agent.generate_report()
    
    # クリーンアップ
    if test_image_path.exists():
        test_image_path.unlink()
    
    print("\n[OK] Test completed")


if __name__ == "__main__":
    test_visual_detection()
