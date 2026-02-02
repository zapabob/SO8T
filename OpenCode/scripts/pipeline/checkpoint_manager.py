#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
チェックポイント管理モジュール

3分間隔でチェックポイントを保存し、最大5つのローリングストックを維持します。
電源投入時に既に収集したデータを読み込まないようにします。

Usage:
    python scripts/pipelines/checkpoint_manager.py --checkpoint-dir D:/webdataset/checkpoints
"""

import sys
import json
import logging
import pickle
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/checkpoint_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """チェックポイント管理クラス（3分間隔、5つローリングストック）"""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_interval: float = 180.0,  # 3分 = 180秒
        max_checkpoints: int = 5,
        resume_on_startup: bool = False  # 電源投入時に既収集データを読み込まない
    ):
        """
        初期化
        
        Args:
            checkpoint_dir: チェックポイントディレクトリ
            checkpoint_interval: チェックポイント保存間隔（秒）
            max_checkpoints: 最大チェックポイント数
            resume_on_startup: 起動時にチェックポイントから復旧するか
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.resume_on_startup = resume_on_startup
        
        # チェックポイント管理
        self.checkpoint_deque = deque(maxlen=max_checkpoints)
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter = 0
        
        # セッションID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("="*80)
        logger.info("Checkpoint Manager Initialized")
        logger.info("="*80)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Checkpoint interval: {self.checkpoint_interval} seconds")
        logger.info(f"Max checkpoints: {self.max_checkpoints}")
        logger.info(f"Resume on startup: {self.resume_on_startup}")
        logger.info(f"Session ID: {self.session_id}")
    
    def should_save_checkpoint(self) -> bool:
        """
        チェックポイントを保存すべきか判定
        
        Returns:
            True: 保存すべき、False: 保存不要
        """
        current_time = time.time()
        elapsed = current_time - self.last_checkpoint_time
        
        return elapsed >= self.checkpoint_interval
    
    def save_checkpoint(
        self,
        samples: List[Dict],
        visited_urls: Dict[str, bool],
        collected_count: int,
        phase: str = "scraping",
        additional_data: Optional[Dict] = None
    ) -> bool:
        """
        チェックポイントを保存
        
        Args:
            samples: 収集済みサンプル
            visited_urls: 訪問済みURL辞書
            collected_count: 収集カウント
            phase: 現在のフェーズ（scraping, cleaning, dataset, rag, cog）
            additional_data: 追加データ
        
        Returns:
            True: 保存成功、False: 保存失敗
        """
        if not self.should_save_checkpoint():
            return False
        
        try:
            logger.info(f"[CHECKPOINT] Saving checkpoint {self.checkpoint_counter} (phase: {phase})...")
            
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}_{self.checkpoint_counter:04d}.pkl"
            
            checkpoint_data = {
                'session_id': self.session_id,
                'checkpoint_id': self.checkpoint_counter,
                'checkpoint_time': datetime.now().isoformat(),
                'phase': phase,
                'samples': samples,
                'visited_urls': dict(visited_urls),
                'collected_count': collected_count,
                'additional_data': additional_data or {}
            }
            
            # チェックポイントを保存
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # チェックポイントリストに追加（FIFO、最大5個）
            self.checkpoint_deque.append(str(checkpoint_file))
            
            # 古いチェックポイント削除（6個目以降）
            if len(self.checkpoint_deque) > self.max_checkpoints:
                old_checkpoint = Path(self.checkpoint_deque[0])
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"[CLEANUP] Deleted old checkpoint: {old_checkpoint.name}")
            
            self.last_checkpoint_time = time.time()
            self.checkpoint_counter += 1
            
            logger.info(f"[OK] Checkpoint saved: {checkpoint_file.name}")
            logger.info(f"[INFO] Active checkpoints: {len(self.checkpoint_deque)}/{self.max_checkpoints}")
            logger.info(f"[INFO] Samples in checkpoint: {len(samples)}")
            
            return True
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to save checkpoint: {e}", exc_info=True)
            return False
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        最新のチェックポイントを読み込み
        
        Returns:
            checkpoint_data: チェックポイントデータ（存在しない場合はNone）
        """
        if not self.resume_on_startup:
            logger.info("[CHECKPOINT] Resume on startup disabled, skipping checkpoint load")
            return None
        
        # チェックポイントディレクトリから最新のチェックポイントを検索
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not checkpoint_files:
            logger.info("[CHECKPOINT] No checkpoint found")
            return None
        
        latest_checkpoint = checkpoint_files[0]
        logger.info(f"[CHECKPOINT] Loading latest checkpoint: {latest_checkpoint.name}")
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"[OK] Checkpoint loaded: {checkpoint_data.get('collected_count', 0)} samples")
            logger.info(f"[OK] Checkpoint time: {checkpoint_data.get('checkpoint_time', 'unknown')}")
            logger.info(f"[OK] Checkpoint phase: {checkpoint_data.get('phase', 'unknown')}")
            
            return checkpoint_data
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to load checkpoint: {e}", exc_info=True)
            return None
    
    def clear_checkpoints(self):
        """すべてのチェックポイントをクリア"""
        logger.info("[CHECKPOINT] Clearing all checkpoints...")
        
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint_file.unlink()
                logger.info(f"[CLEANUP] Deleted checkpoint: {checkpoint_file.name}")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to delete checkpoint {checkpoint_file.name}: {e}")
        
        self.checkpoint_deque.clear()
        self.checkpoint_counter = 0
        self.last_checkpoint_time = time.time()
        
        logger.info("[OK] All checkpoints cleared")
    
    def get_checkpoint_info(self) -> Dict:
        """
        チェックポイント情報を取得
        
        Returns:
            info: チェックポイント情報
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        
        return {
            'total_checkpoints': len(checkpoint_files),
            'active_checkpoints': len(self.checkpoint_deque),
            'max_checkpoints': self.max_checkpoints,
            'checkpoint_interval': self.checkpoint_interval,
            'last_checkpoint_time': datetime.fromtimestamp(self.last_checkpoint_time).isoformat(),
            'session_id': self.session_id,
            'resume_on_startup': self.resume_on_startup
        }


def main():
    """メイン関数（テスト用）"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint Manager")
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=Path('D:/webdataset/checkpoints/unified_pipeline'),
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear all checkpoints'
    )
    
    args = parser.parse_args()
    
    manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        resume_on_startup=False  # テスト時はFalse
    )
    
    if args.clear:
        manager.clear_checkpoints()
    else:
        info = manager.get_checkpoint_info()
        print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()





