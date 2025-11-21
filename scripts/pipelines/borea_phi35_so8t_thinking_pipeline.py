#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5 SO8T/thinking統合パイプライン

Step 1: /thinkデータセット作成
Step 2: SO8T統合学習
Step 3: 焼き込み処理
Step 4: GGUF変換
"""

import os
import sys
import subprocess
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/borea_phi35_so8t_thinking_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BoreaSO8TThinkingPipeline:
    """Borea-Phi-3.5 SO8T/thinking統合パイプライン"""
    
    def __init__(
        self,
        config_path: Path,
        skip_steps: Optional[list] = None
    ):
        """
        Args:
            config_path: 設定ファイルパス
            skip_steps: スキップするステップのリスト
        """
        self.config_path = Path(config_path)
        self.skip_steps = set(skip_steps or [])
        
        # 設定ファイル読み込み
        import yaml
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # パス設定
        self.base_output_dir = Path(self.config.get("pipeline", {}).get("base_output_dir", "D:/webdataset"))
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_output_dir / "borea_phi35_so8t_thinking" / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("Borea-Phi-3.5 SO8T/thinking Pipeline")
        logger.info("="*80)
        logger.info(f"Config: {config_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Session ID: {self.session_id}")
    
    def step1_create_thinking_dataset(self) -> Optional[Path]:
        """Step 1: /think形式データセット作成"""
        if "step1" in self.skip_steps:
            logger.info("[SKIP] Step 1: Create thinking dataset")
            return None
        
        logger.info("="*80)
        logger.info("Step 1: Create /think format dataset")
        logger.info("="*80)
        
        # 入力データセットパス
        input_datasets = self.config.get("pipeline", {}).get("input_datasets", [])
        if not input_datasets:
            logger.error("No input datasets specified in config")
            return None
        
        # 出力パス
        output_dataset = self.output_dir / "thinking_sft_dataset.jsonl"
        
        # スクリプト実行
        script_path = PROJECT_ROOT / "scripts" / "data" / "create_thinking_sft_dataset.py"
        
        cmd = [
            "py", "-3",
            str(script_path),
            "--inputs"
        ] + [str(Path(d)) for d in input_datasets] + [
            "--output", str(output_dataset)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"[OK] Step 1 completed: {output_dataset}")
            return output_dataset
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Step 1 failed: {e.stderr}")
            return None
    
    def step2_train_so8t(self, dataset_path: Path) -> Optional[Path]:
        """Step 2: SO8T統合学習"""
        if "step2" in self.skip_steps:
            logger.info("[SKIP] Step 2: SO8T training")
            return None
        
        logger.info("="*80)
        logger.info("Step 2: SO8T integrated training")
        logger.info("="*80)
        
        # 学習出力ディレクトリ
        training_output = self.output_dir / "training"
        
        # スクリプト実行
        script_path = PROJECT_ROOT / "scripts" / "training" / "train_borea_phi35_so8t_thinking.py"
        
        cmd = [
            "py", "-3",
            str(script_path),
            "--config", str(self.config_path),
            "--dataset", str(dataset_path),
            "--output-dir", str(training_output)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"[OK] Step 2 completed: {training_output}")
            
            # 最終モデルパス
            final_model = training_output / "final_model"
            if final_model.exists():
                return final_model
            else:
                logger.warning(f"Final model not found: {final_model}")
                return training_output
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Step 2 failed: {e.stderr}")
            return None
    
    def step3_bake_so8t(self, trained_model_path: Path) -> Optional[Path]:
        """Step 3: 焼き込み処理"""
        if "step3" in self.skip_steps:
            logger.info("[SKIP] Step 3: SO8T baking")
            return None
        
        logger.info("="*80)
        logger.info("Step 3: SO8T baking")
        logger.info("="*80)
        
        # 焼き込み出力ディレクトリ
        baked_output = self.output_dir / "baked_model"
        
        # スクリプト実行
        script_path = PROJECT_ROOT / "scripts" / "training" / "bake_borea_phi35_so8t.py"
        
        cmd = [
            "py", "-3",
            str(script_path),
            "--model-path", str(trained_model_path),
            "--output-path", str(baked_output)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"[OK] Step 3 completed: {baked_output}")
            return baked_output
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Step 3 failed: {e.stderr}")
            return None
    
    def step4_convert_gguf(self, baked_model_path: Path) -> Optional[Path]:
        """Step 4: GGUF変換"""
        if "step4" in self.skip_steps:
            logger.info("[SKIP] Step 4: GGUF conversion")
            return None
        
        logger.info("="*80)
        logger.info("Step 4: GGUF conversion")
        logger.info("="*80)
        
        # GGUF出力ディレクトリ
        gguf_output = Path("D:/webdataset/gguf_models/borea_phi35_so8t_thinking")
        
        # スクリプト実行
        script_path = PROJECT_ROOT / "scripts" / "conversion" / "convert_borea_so8t_to_gguf.py"
        
        model_name = self.config.get("pipeline", {}).get("model_name", "borea_phi35_so8t_thinking")
        
        cmd = [
            "py", "-3",
            str(script_path),
            "--model-path", str(baked_model_path),
            "--output-dir", str(gguf_output),
            "--model-name", model_name
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"[OK] Step 4 completed: {gguf_output}")
            return gguf_output
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Step 4 failed: {e.stderr}")
            return None
    
    def run(self) -> bool:
        """パイプライン実行"""
        logger.info("Starting pipeline execution...")
        
        # Step 1: /thinkデータセット作成
        dataset_path = self.step1_create_thinking_dataset()
        if dataset_path is None and "step1" not in self.skip_steps:
            logger.error("Step 1 failed, aborting")
            return False
        
        # Step 2: SO8T統合学習
        trained_model_path = self.step2_train_so8t(dataset_path) if dataset_path else None
        if trained_model_path is None and "step2" not in self.skip_steps:
            logger.error("Step 2 failed, aborting")
            return False
        
        # Step 3: 焼き込み処理
        baked_model_path = self.step3_bake_so8t(trained_model_path) if trained_model_path else None
        if baked_model_path is None and "step3" not in self.skip_steps:
            logger.error("Step 3 failed, aborting")
            return False
        
        # Step 4: GGUF変換
        gguf_output = self.step4_convert_gguf(baked_model_path) if baked_model_path else None
        if gguf_output is None and "step4" not in self.skip_steps:
            logger.error("Step 4 failed, aborting")
            return False
        
        # 完了メタデータ保存
        metadata = {
            "session_id": self.session_id,
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir),
            "steps_completed": {
                "step1": dataset_path is not None,
                "step2": trained_model_path is not None,
                "step3": baked_model_path is not None,
                "step4": gguf_output is not None
            },
            "paths": {
                "dataset": str(dataset_path) if dataset_path else None,
                "trained_model": str(trained_model_path) if trained_model_path else None,
                "baked_model": str(baked_model_path) if baked_model_path else None,
                "gguf_output": str(gguf_output) if gguf_output else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = self.output_dir / "pipeline_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("="*80)
        logger.info("[SUCCESS] Pipeline completed")
        logger.info("="*80)
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Borea-Phi-3.5 SO8T/thinking integrated pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file path (YAML)"
    )
    parser.add_argument(
        "--skip-steps",
        type=str,
        nargs="+",
        help="Steps to skip (step1, step2, step3, step4)"
    )
    
    args = parser.parse_args()
    
    pipeline = BoreaSO8TThinkingPipeline(
        config_path=args.config,
        skip_steps=args.skip_steps
    )
    
    success = pipeline.run()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()













































































