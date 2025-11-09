#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コーディング特化再学習パイプライン

コーディングデータに特化したSO8Tモデルの再学習を実行します。
QLoRA 8bitファインチューニングとコーディングタスク用の評価指標を含みます。

Usage:
    python scripts/pipelines/coding_focused_retraining_pipeline.py --config configs/coding_focused_retraining_config.yaml
"""

import sys
import json
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 既存の学習スクリプトをインポート
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/coding_focused_retraining_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CodingFocusedRetrainingPipeline:
    """コーディング特化再学習パイプライン"""
    
    def __init__(self, config_path: Path):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = Path(config_path)
        
        # 設定を読み込み
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # パスを解決
        self.base_model_path = Path(self.config['model']['base_model_path'])
        self.coding_dataset_path = Path(self.config['data']['coding_dataset_path'])
        self.output_dir = Path(self.config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("="*80)
        logger.info("Coding Focused Retraining Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Base model: {self.base_model_path}")
        logger.info(f"Coding dataset: {self.coding_dataset_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def validate_dataset(self) -> bool:
        """データセットを検証"""
        logger.info("[VALIDATE] Validating coding dataset...")
        
        if not self.coding_dataset_path.exists():
            logger.error(f"[ERROR] Dataset path does not exist: {self.coding_dataset_path}")
            return False
        
        # JSONLファイルを検索
        jsonl_files = list(self.coding_dataset_path.glob("coding_training_*.jsonl"))
        
        if not jsonl_files:
            logger.error(f"[ERROR] No coding training files found in {self.coding_dataset_path}")
            return False
        
        total_samples = 0
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
                total_samples += count
                logger.info(f"[VALIDATE] Found {count} samples in {jsonl_file.name}")
        
        logger.info(f"[VALIDATE] Total samples: {total_samples}")
        
        if total_samples == 0:
            logger.error("[ERROR] No samples found in dataset")
            return False
        
        return True
    
    def run_training(self) -> bool:
        """再学習を実行"""
        logger.info("[TRAIN] Starting coding-focused retraining...")
        
        try:
            # 既存の学習スクリプトを呼び出し
            from train_so8t_phi3_qlora import main as train_main
            import argparse as arg_parser
            
            # 学習スクリプト用の引数を作成
            train_args = arg_parser.Namespace()
            train_args.config = str(self.config_path)
            train_args.resume = None
            
            # 学習を実行
            train_main(train_args)
            
            logger.info("[TRAIN] Training completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"[ERROR] Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_pipeline(self) -> Dict:
        """パイプラインを実行"""
        logger.info("[PIPELINE] Starting coding-focused retraining pipeline...")
        
        results = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'validation': False,
            'training': False,
            'end_time': None,
            'output_dir': str(self.output_dir)
        }
        
        # 1. データセット検証
        if not self.validate_dataset():
            logger.error("[PIPELINE] Dataset validation failed")
            results['end_time'] = datetime.now().isoformat()
            return results
        
        results['validation'] = True
        
        # 2. 再学習実行
        if not self.run_training():
            logger.error("[PIPELINE] Training failed")
            results['end_time'] = datetime.now().isoformat()
            return results
        
        results['training'] = True
        results['end_time'] = datetime.now().isoformat()
        
        logger.info("[PIPELINE] Pipeline completed successfully")
        return results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Coding Focused Retraining Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    
    args = parser.parse_args()
    
    pipeline = CodingFocusedRetrainingPipeline(config_path=args.config)
    results = pipeline.run_pipeline()
    
    # 結果を保存
    results_file = pipeline.output_dir / f"coding_retraining_results_{pipeline.session_id}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[COMPLETE] Results saved to {results_file}")


if __name__ == '__main__':
    main()

