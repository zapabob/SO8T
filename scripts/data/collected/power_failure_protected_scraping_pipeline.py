#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
電源断保護・重複防止機能付きスクレイピングパイプライン

チェックポイント自動保存、緊急保存、セッション管理、重複防止機能を実装

Usage:
    python scripts/pipelines/power_failure_protected_scraping_pipeline.py --config configs/so8t_thinking_controlled_scraping_config.yaml
"""

import sys
import json
import logging
import argparse
import signal
import pickle
import hashlib
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from collections import deque
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "audit"))

# 監査ログインポート
try:
    from scripts.audit.scraping_audit_logger import ScrapingAuditLogger, ScrapingSession, ScrapingEvent
    AUDIT_LOGGER_AVAILABLE = True
except ImportError:
    AUDIT_LOGGER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Scraping audit logger not available")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/power_failure_protected_scraping_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """チェックポイントデータ"""
    session_id: str
    timestamp: str
    browser_states: Dict[int, Dict]
    collected_urls: Set[str]
    collected_keywords: Set[str]
    samples_collected: int
    last_activity: str
    checkpoint_type: str  # auto/emergency/manual


@dataclass
class SessionState:
    """セッション状態"""
    session_id: str
    started_at: str
    browser_index: int
    keyword: str
    status: str
    samples_collected: int
    last_activity: str
    collected_urls: Set[str]
    collected_keywords: Set[str]


class PowerFailureRecovery:
    """電源断保護・重複防止システム"""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 10,
        checkpoint_interval: float = 300.0,  # 5分
        audit_logger: Optional[ScrapingAuditLogger] = None
    ):
        """
        初期化
        
        Args:
            checkpoint_dir: チェックポイントディレクトリ
            max_checkpoints: 最大チェックポイント数
            checkpoint_interval: チェックポイント保存間隔（秒）
            audit_logger: 監査ロガー
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.audit_logger = audit_logger
        
        # セッション管理
        self.sessions: Dict[str, SessionState] = {}
        self.checkpoints: deque = deque(maxlen=max_checkpoints)
        
        # 重複防止
        self.url_hashes: Set[str] = set()
        self.keyword_session_map: Dict[str, Set[str]] = {}
        
        # 自動チェックポイントスレッド
        self.checkpoint_thread: Optional[threading.Thread] = None
        self.running = False
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("Power Failure Recovery System Initialized")
        logger.info("="*80)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Checkpoint interval: {checkpoint_interval} seconds")
        logger.info(f"Max checkpoints: {max_checkpoints}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"[SIGNAL] Received signal {signum}, performing emergency save...")
            self.emergency_save()
            logger.info("[OK] Emergency save completed")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def start_auto_checkpoint(self):
        """自動チェックポイント開始"""
        if self.running:
            return
        
        self.running = True
        
        def checkpoint_loop():
            while self.running:
                time.sleep(self.checkpoint_interval)
                if self.running:
                    self.save_checkpoint(checkpoint_type="auto")
        
        self.checkpoint_thread = threading.Thread(target=checkpoint_loop, daemon=True)
        self.checkpoint_thread.start()
        logger.info("[OK] Auto checkpoint started")
    
    def stop_auto_checkpoint(self):
        """自動チェックポイント停止"""
        self.running = False
        if self.checkpoint_thread:
            self.checkpoint_thread.join(timeout=5.0)
        logger.info("[OK] Auto checkpoint stopped")
    
    def create_session(
        self,
        session_id: str,
        browser_index: int,
        keyword: str
    ) -> SessionState:
        """
        セッション作成
        
        Args:
            session_id: セッションID
            browser_index: ブラウザインデックス
            keyword: キーワード
            
        Returns:
            session: セッション状態
        """
        session = SessionState(
            session_id=session_id,
            started_at=datetime.now().isoformat(),
            browser_index=browser_index,
            keyword=keyword,
            status="active",
            samples_collected=0,
            last_activity=datetime.now().isoformat(),
            collected_urls=set(),
            collected_keywords=set()
        )
        
        self.sessions[session_id] = session
        
        # キーワード・セッションマッピング
        if keyword not in self.keyword_session_map:
            self.keyword_session_map[keyword] = set()
        self.keyword_session_map[keyword].add(session_id)
        
        # 監査ログ記録
        if self.audit_logger:
            scraping_session = ScrapingSession(
                session_id=session_id,
                started_at=session.started_at,
                browser_index=browser_index,
                keyword=keyword,
                status="active",
                samples_collected=0,
                last_activity=session.last_activity
            )
            self.audit_logger.log_scraping_session(scraping_session)
            
            event = ScrapingEvent(
                event_id=f"{session_id}_start",
                session_id=session_id,
                timestamp=session.started_at,
                event_type="scraping_start",
                keyword=keyword
            )
            self.audit_logger.log_scraping_event(event)
        
        logger.info(f"[SESSION] Created: {session_id} (browser: {browser_index}, keyword: {keyword})")
        return session
    
    def update_session(
        self,
        session_id: str,
        url: Optional[str] = None,
        samples_collected: Optional[int] = None,
        status: Optional[str] = None
    ):
        """
        セッション更新
        
        Args:
            session_id: セッションID
            url: URL（オプション）
            samples_collected: 収集サンプル数（オプション）
            status: ステータス（オプション）
        """
        if session_id not in self.sessions:
            logger.warning(f"[SESSION] Session not found: {session_id}")
            return
        
        session = self.sessions[session_id]
        
        if url:
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            session.collected_urls.add(url)
            self.url_hashes.add(url_hash)
        
        if samples_collected is not None:
            session.samples_collected = samples_collected
        
        if status:
            session.status = status
        
        session.last_activity = datetime.now().isoformat()
        
        # 監査ログ記録
        if self.audit_logger:
            scraping_session = ScrapingSession(
                session_id=session_id,
                started_at=session.started_at,
                browser_index=session.browser_index,
                keyword=session.keyword,
                status=session.status,
                samples_collected=session.samples_collected,
                last_activity=session.last_activity
            )
            self.audit_logger.log_scraping_session(scraping_session)
    
    def check_duplicate(self, url: str, keyword: str, session_id: str) -> bool:
        """
        重複チェック
        
        Args:
            url: URL
            keyword: キーワード
            session_id: セッションID
            
        Returns:
            is_duplicate: 重複しているかどうか
        """
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        # 監査ログで重複チェック
        if self.audit_logger:
            is_duplicate = self.audit_logger.check_duplicate(url, keyword, session_id)
            if is_duplicate:
                logger.debug(f"[DUPLICATE] Detected: {url_hash[:16]}... (keyword: {keyword})")
                return True
        
        # ローカル重複チェック
        if url_hash in self.url_hashes:
            logger.debug(f"[DUPLICATE] Detected locally: {url_hash[:16]}...")
            return True
        
        return False
    
    def register_duplicate(self, url: str, keyword: str, session_id: str):
        """
        重複を登録
        
        Args:
            url: URL
            keyword: キーワード
            session_id: セッションID
        """
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        self.url_hashes.add(url_hash)
        
        if session_id in self.sessions:
            self.sessions[session_id].collected_urls.add(url)
        
        # 監査ログ記録
        if self.audit_logger:
            self.audit_logger.register_duplicate(url, keyword, session_id)
            
            event = ScrapingEvent(
                event_id=f"{session_id}_duplicate_{int(time.time())}",
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                event_type="duplicate_detected",
                url=url,
                keyword=keyword
            )
            self.audit_logger.log_scraping_event(event)
    
    def save_checkpoint(self, checkpoint_type: str = "auto") -> bool:
        """
        チェックポイント保存
        
        Args:
            checkpoint_type: チェックポイントタイプ（auto/emergency/manual）
            
        Returns:
            success: 成功フラグ
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}_{checkpoint_type}.pkl"
            
            # ブラウザ状態を収集
            browser_states = {}
            for session_id, session in self.sessions.items():
                browser_states[session.browser_index] = {
                    'session_id': session_id,
                    'keyword': session.keyword,
                    'status': session.status,
                    'samples_collected': session.samples_collected,
                    'last_activity': session.last_activity
                }
            
            checkpoint_data = CheckpointData(
                session_id=f"checkpoint_{timestamp}",
                timestamp=datetime.now().isoformat(),
                browser_states=browser_states,
                collected_urls=set().union(*[s.collected_urls for s in self.sessions.values()]),
                collected_keywords=set().union(*[s.collected_keywords for s in self.sessions.values()]),
                samples_collected=sum(s.samples_collected for s in self.sessions.values()),
                last_activity=datetime.now().isoformat(),
                checkpoint_type=checkpoint_type
            )
            
            # セッション状態をシリアライズ可能な形式に変換
            checkpoint_dict = asdict(checkpoint_data)
            checkpoint_dict['collected_urls'] = list(checkpoint_dict['collected_urls'])
            checkpoint_dict['collected_keywords'] = list(checkpoint_dict['collected_keywords'])
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_dict, f)
            
            self.checkpoints.append(checkpoint_file)
            
            # 監査ログ記録
            if self.audit_logger:
                event = ScrapingEvent(
                    event_id=f"checkpoint_{timestamp}",
                    session_id="system",
                    timestamp=datetime.now().isoformat(),
                    event_type="checkpoint_saved",
                    details={
                        'checkpoint_type': checkpoint_type,
                        'checkpoint_file': str(checkpoint_file),
                        'sessions_count': len(self.sessions),
                        'samples_collected': checkpoint_data.samples_collected
                    }
                )
                self.audit_logger.log_scraping_event(event)
            
            logger.info(f"[CHECKPOINT] Saved: {checkpoint_file.name} (type: {checkpoint_type})")
            return True
            
        except Exception as e:
            logger.error(f"[CHECKPOINT] Failed to save checkpoint: {e}")
            return False
    
    def emergency_save(self) -> bool:
        """
        緊急保存
        
        Returns:
            success: 成功フラグ
        """
        logger.warning("[EMERGENCY] Performing emergency save...")
        return self.save_checkpoint(checkpoint_type="emergency")
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        最新チェックポイント読み込み
        
        Returns:
            checkpoint_data: チェックポイントデータ（Noneの場合は読み込み失敗）
        """
        try:
            checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"), key=lambda p: p.stat().st_mtime)
            
            if not checkpoint_files:
                logger.info("[CHECKPOINT] No checkpoint files found")
                return None
            
            latest_checkpoint = checkpoint_files[-1]
            
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"[CHECKPOINT] Loaded: {latest_checkpoint.name}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"[CHECKPOINT] Failed to load checkpoint: {e}")
            return None
    
    def recover_sessions(self) -> Dict[str, SessionState]:
        """
        セッション復旧
        
        Returns:
            sessions: 復旧したセッションの辞書
        """
        checkpoint_data = self.load_latest_checkpoint()
        
        if not checkpoint_data:
            logger.info("[RECOVERY] No checkpoint found, starting fresh")
            return {}
        
        logger.info("[RECOVERY] Recovering sessions from checkpoint...")
        
        recovered_sessions = {}
        
        # ブラウザ状態からセッションを復旧
        browser_states = checkpoint_data.get('browser_states', {})
        collected_urls = set(checkpoint_data.get('collected_urls', []))
        collected_keywords = set(checkpoint_data.get('collected_keywords', []))
        
        for browser_index, browser_state in browser_states.items():
            session_id = browser_state.get('session_id')
            if not session_id:
                continue
            
            session = SessionState(
                session_id=session_id,
                started_at=browser_state.get('started_at', datetime.now().isoformat()),
                browser_index=browser_index,
                keyword=browser_state.get('keyword', 'unknown'),
                status=browser_state.get('status', 'active'),
                samples_collected=browser_state.get('samples_collected', 0),
                last_activity=browser_state.get('last_activity', datetime.now().isoformat()),
                collected_urls=collected_urls.copy(),
                collected_keywords=collected_keywords.copy()
            )
            
            recovered_sessions[session_id] = session
            
            # URLハッシュを復元
            for url in collected_urls:
                url_hash = hashlib.sha256(url.encode()).hexdigest()
                self.url_hashes.add(url_hash)
        
        self.sessions = recovered_sessions
        
        # 監査ログ記録
        if self.audit_logger:
            recovery_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.audit_logger.log_power_failure_recovery(
                recovery_id=recovery_id,
                failure_timestamp=checkpoint_data.get('timestamp', datetime.now().isoformat()),
                recovery_timestamp=datetime.now().isoformat(),
                session_id="system",
                checkpoint_path=str(self.checkpoint_dir),
                samples_lost=0,
                samples_recovered=sum(s.samples_collected for s in recovered_sessions.values()),
                recovery_status="success",
                details={
                    'sessions_recovered': len(recovered_sessions),
                    'checkpoint_file': checkpoint_data.get('session_id', 'unknown')
                }
            )
        
        logger.info(f"[RECOVERY] Recovered {len(recovered_sessions)} sessions")
        return recovered_sessions
    
    def cleanup_old_checkpoints(self):
        """古いチェックポイントを削除"""
        try:
            checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"), key=lambda p: p.stat().st_mtime)
            
            if len(checkpoint_files) > self.max_checkpoints:
                files_to_delete = checkpoint_files[:-self.max_checkpoints]
                for file in files_to_delete:
                    file.unlink()
                    logger.debug(f"[CLEANUP] Deleted old checkpoint: {file.name}")
                
                logger.info(f"[CLEANUP] Deleted {len(files_to_delete)} old checkpoints")
                
        except Exception as e:
            logger.error(f"[CLEANUP] Failed to cleanup checkpoints: {e}")


def main():
    """テスト実行"""
    logger.info("="*80)
    logger.info("Power Failure Recovery System Test")
    logger.info("="*80)
    
    checkpoint_dir = Path("D:/webdataset/checkpoints/power_failure_recovery")
    recovery = PowerFailureRecovery(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=10,
        checkpoint_interval=60.0  # テスト用に1分
    )
    
    # セッション作成
    session1 = recovery.create_session("test_session_001", browser_index=0, keyword="Python")
    session2 = recovery.create_session("test_session_002", browser_index=1, keyword="JavaScript")
    
    # セッション更新
    recovery.update_session("test_session_001", url="https://example.com", samples_collected=10)
    recovery.update_session("test_session_002", url="https://example2.com", samples_collected=5)
    
    # チェックポイント保存
    recovery.save_checkpoint(checkpoint_type="manual")
    logger.info("[OK] Checkpoint saved")
    
    # 重複チェック
    is_duplicate = recovery.check_duplicate("https://example.com", "Python", "test_session_001")
    logger.info(f"[OK] Duplicate check: {is_duplicate}")
    
    logger.info("[OK] Test completed")


if __name__ == "__main__":
    main()

