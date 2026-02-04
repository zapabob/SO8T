#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
進捗管理ユーティリティ

30分間隔でMD形式ログを生成し、フェーズ進捗を追跡

Usage:
    from src.utils.progress_manager import ProgressManager
    
    manager = ProgressManager(session_id="20250127_120000")
    manager.update_phase_status("phase1", "running", progress=0.5)
    manager.log_progress()
"""

import json
import logging
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PhaseStatus:
    """フェーズ状態"""
    phase_name: str
    status: str  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0  # 0.0-1.0
    metrics: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self):
        return {
            'phase_name': self.phase_name,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'progress': self.progress,
            'metrics': self.metrics,
            'error_message': self.error_message,
            'duration_seconds': (self.end_time - self.start_time) if self.end_time and self.start_time else None
        }


class ProgressManager:
    """進捗管理メインクラス"""
    
    def __init__(self, session_id: str, log_interval: int = 1800):
        """
        Args:
            session_id: セッションID
            log_interval: ログ生成間隔（秒、デフォルト30分）
        """
        self.session_id = session_id
        self.log_interval = log_interval
        self.start_time = time.time()
        
        # 出力ディレクトリ
        self.logs_dir = PROJECT_ROOT / "_docs" / "progress_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # フェーズ状態管理
        self.phases: Dict[str, PhaseStatus] = {}
        self.lock = threading.Lock()
        
        # ログ生成スレッド
        self.log_thread = None
        self.running = False
        
        logger.info("="*80)
        logger.info("Progress Manager Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Log interval: {self.log_interval} seconds ({self.log_interval/60:.1f} minutes)")
        logger.info(f"Logs directory: {self.logs_dir}")
    
    def start_logging(self):
        """ログ生成スレッドを開始"""
        if self.running:
            logger.warning("Logging thread already running")
            return
        
        self.running = True
        self.log_thread = threading.Thread(target=self._log_loop, daemon=True)
        self.log_thread.start()
        logger.info("Progress logging thread started")
    
    def stop_logging(self):
        """ログ生成スレッドを停止"""
        self.running = False
        if self.log_thread:
            self.log_thread.join(timeout=5.0)
        logger.info("Progress logging thread stopped")
    
    def _log_loop(self):
        """ログ生成ループ"""
        while self.running:
            try:
                self.log_progress()
                time.sleep(self.log_interval)
            except Exception as e:
                logger.error(f"Error in log loop: {e}")
                time.sleep(60)  # エラー時は1分待機
    
    def update_phase_status(
        self,
        phase_name: str,
        status: str,
        progress: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """
        フェーズ状態を更新
        
        Args:
            phase_name: フェーズ名
            status: 状態（pending, running, completed, failed）
            progress: 進捗（0.0-1.0）
            metrics: メトリクス辞書
            error_message: エラーメッセージ（失敗時）
        """
        with self.lock:
            if phase_name not in self.phases:
                self.phases[phase_name] = PhaseStatus(
                    phase_name=phase_name,
                    status=status,
                    start_time=time.time()
                )
            
            phase = self.phases[phase_name]
            phase.status = status
            phase.progress = max(0.0, min(1.0, progress))
            
            if status == "running" and phase.start_time is None:
                phase.start_time = time.time()
            
            if status in ["completed", "failed"]:
                phase.end_time = time.time()
            
            if metrics:
                phase.metrics.update(metrics)
            
            if error_message:
                phase.error_message = error_message
            
            logger.info(f"Phase '{phase_name}' status updated: {status} (progress: {progress:.1%})")
    
    def log_progress(self):
        """進捗ログを生成"""
        with self.lock:
            log_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': time.time() - self.start_time,
                'phases': {name: phase.to_dict() for name, phase in self.phases.items()}
            }
        
        # JSON形式で保存
        log_file = self.logs_dir / f"{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        # MD形式で保存
        md_file = self.logs_dir / f"{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        md_content = self._generate_markdown_log(log_data)
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Progress log saved: {md_file}")
    
    def _generate_markdown_log(self, log_data: Dict) -> str:
        """Markdown形式のログを生成"""
        elapsed_hours = log_data['elapsed_seconds'] / 3600
        elapsed_str = f"{elapsed_hours:.2f} hours ({log_data['elapsed_seconds']:.0f} seconds)"
        
        content = f"""# SO8T Complete Pipeline Progress Log

## Session Information

- **Session ID**: {log_data['session_id']}
- **Timestamp**: {log_data['timestamp']}
- **Elapsed Time**: {elapsed_str}

## Phase Status

"""
        
        for phase_name, phase_data in log_data['phases'].items():
            status_emoji = {
                'pending': '⏳',
                'running': '🔄',
                'completed': '✅',
                'failed': '❌'
            }.get(phase_data['status'], '❓')
            
            progress_bar = self._generate_progress_bar(phase_data['progress'])
            
            duration_str = ""
            if phase_data['duration_seconds']:
                duration_hours = phase_data['duration_seconds'] / 3600
                duration_str = f" ({duration_hours:.2f} hours)"
            
            content += f"""### {status_emoji} Phase {phase_name.upper()}

- **Status**: {phase_data['status']}
- **Progress**: {progress_bar} {phase_data['progress']:.1%}
- **Duration**: {duration_str if duration_str else "N/A"}
"""
            
            if phase_data['start_time']:
                start_dt = datetime.fromtimestamp(phase_data['start_time'])
                content += f"- **Start Time**: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if phase_data['end_time']:
                end_dt = datetime.fromtimestamp(phase_data['end_time'])
                content += f"- **End Time**: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if phase_data['metrics']:
                content += "\n**Metrics:**\n"
                for key, value in phase_data['metrics'].items():
                    if isinstance(value, (int, float)):
                        content += f"- {key}: {value:.4f}\n"
                    else:
                        content += f"- {key}: {value}\n"
            
            if phase_data['error_message']:
                content += f"\n**Error**: {phase_data['error_message']}\n"
            
            content += "\n"
        
        # サマリー
        total_phases = len(log_data['phases'])
        completed_phases = sum(1 for p in log_data['phases'].values() if p['status'] == 'completed')
        failed_phases = sum(1 for p in log_data['phases'].values() if p['status'] == 'failed')
        running_phases = sum(1 for p in log_data['phases'].values() if p['status'] == 'running')
        
        # ゼロ除算を防ぐ
        overall_progress = (completed_phases / total_phases * 100) if total_phases > 0 else 0.0
        
        content += f"""## Summary

- **Total Phases**: {total_phases}
- **Completed**: {completed_phases}
- **Running**: {running_phases}
- **Failed**: {failed_phases}
- **Overall Progress**: {overall_progress:.1f}% ({completed_phases}/{total_phases})

---
*Generated automatically by ProgressManager*
"""
        
        return content
    
    def _generate_progress_bar(self, progress: float, width: int = 20) -> str:
        """進捗バーを生成"""
        filled = int(progress * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """進捗サマリーを取得"""
        with self.lock:
            total_phases = len(self.phases)
            if total_phases == 0:
                return {
                    'total_phases': 0,
                    'completed': 0,
                    'running': 0,
                    'failed': 0,
                    'overall_progress': 0.0,
                    'elapsed_seconds': time.time() - self.start_time
                }
            
            completed = sum(1 for p in self.phases.values() if p.status == 'completed')
            running = sum(1 for p in self.phases.values() if p.status == 'running')
            failed = sum(1 for p in self.phases.values() if p.status == 'failed')
            
            return {
                'total_phases': total_phases,
                'completed': completed,
                'running': running,
                'failed': failed,
                'overall_progress': completed / total_phases,
                'elapsed_seconds': time.time() - self.start_time,
                'phases': {name: phase.to_dict() for name, phase in self.phases.items()}
            }


def main():
    """テスト用メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progress Manager Test")
    parser.add_argument("--session-id", type=str, default=None, help="Session ID")
    args = parser.parse_args()
    
    session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    manager = ProgressManager(session_id=session_id, log_interval=60)  # テスト用に1分間隔
    manager.start_logging()
    
    # テストデータ
    manager.update_phase_status("phase1", "running", progress=0.3)
    time.sleep(2)
    manager.update_phase_status("phase1", "completed", progress=1.0, metrics={"accuracy": 0.95})
    
    manager.update_phase_status("phase2", "running", progress=0.5)
    time.sleep(2)
    
    manager.log_progress()
    manager.stop_logging()
    
    summary = manager.get_progress_summary()
    print("\nProgress Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

