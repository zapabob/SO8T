#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOONSHOT Pipeline with tqdm & logging progress display
AEGIS Autonomous A/B Testing System with real-time progress monitoring

使用方法:
    py -3 scripts/pipeline/moonshot_pipeline.py --phase env_check
    py -3 scripts/pipeline/moonshot_pipeline.py --phase dataset_creation
    py -3 scripts/pipeline/moonshot_pipeline.py --phase lm_eval_setup
    py -3 scripts/pipeline/moonshot_pipeline.py --full  # 全フェーズ実行
"""

import sys
import os
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# tqdm for progress bars
from tqdm import tqdm

# Windows cp932エンコーディング対策
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class MoonshotPipeline:
    """MOONSHOTパイプライン実行クラス"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.log_file = self.project_root / "ab_test_automation.log"

        # ロギング設定
        self._setup_logging()

        # Phase定義
        self.phases = {
            "env_check": {
                "name": "Environment Check",
                "script": "tests/simple_rlpo_test.py",
                "description": "SO8Tコンポーネント検証"
            },
            "dataset_creation": {
                "name": "AEGIS Dataset Creation",
                "script": "scripts/data/create_aegis_high_quality_dataset.py",
                "description": "高品質データセット作成"
            },
            "lm_eval_setup": {
                "name": "LM-Eval Setup",
                "script": "scripts/evaluation/setup_lm_eval_elyza.py",
                "description": "評価環境構築"
            },
            "rlpo_training": {
                "name": "SO(8) RLPO Training",
                "script": "scripts/training/rlpo_science_nsfw_automated.py",
                "description": "SO(8)理論RLPO学習",
                "args": ["--max_steps", "10000", "--checkpoint_interval", "180"]
            },
            "baseline_conversion": {
                "name": "Baseline GGUF Conversion",
                "script": "scripts/conversion/convert_hf_to_gguf.py",
                "description": "ベースラインモデル変換",
                "condition": lambda: (self.project_root / "models" / "phi-3.5-mini-instruct").exists()
            },
            "aegis_conversion": {
                "name": "AEGIS GGUF Conversion",
                "script": "scripts/utils/task_manager.py",
                "description": "AEGISモデル変換",
                "args": ["gguf"],
                "condition": lambda: (self.project_root / "checkpoints" / "rlpo_science_nsfw_automated" / "final_model").exists()
            },
            "ab_test": {
                "name": "A/B Test Execution",
                "script": "scripts/evaluation/run_comprehensive_abc_benchmark.py",
                "description": "業界標準ABCテスト実行",
                "args": ["--num_samples", "100", "--num_seeds", "10"]
            },
            "statistical_analysis": {
                "name": "Statistical Analysis",
                "script": "scripts/evaluation/statistical_abc_analysis.py",
                "description": "統計的ABC解析",
                "args": ["--results_file", "results/abc_testing/comprehensive_abc_results.json"]
            },
            "visualization": {
                "name": "Statistical Visualization",
                "script": "scripts/evaluation/visualize_abc_benchmark_statistics.py",
                "description": "統計可視化グラフ生成",
                "args": [
                    "--results_file", "results/abc_testing/comprehensive_abc_results.json",
                    "--statistical_analysis_file", "results/abc_testing/statistical_analysis.json"
                ]
            },
            "hf_upload_prep": {
                "name": "HF Upload Preparation",
                "script": "scripts/evaluation/prepare_hf_upload.py",
                "description": "HFアップロード準備"
            }
        }

    def _setup_logging(self):
        """ロギング設定"""
        # ファイルロガー
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # コンソールロガー
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)

        # ルートロガー設定
        self.logger = logging.getLogger('MOONSHOT')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def run_single_phase(self, phase_key: str) -> bool:
        """単一フェーズ実行"""
        if phase_key not in self.phases:
            self.logger.error(f"[ERROR] Unknown phase: {phase_key}")
            return False

        phase = self.phases[phase_key]

        # 条件チェック
        if 'condition' in phase and not phase['condition']():
            self.logger.info(f"[SKIP] {phase['name']}: Condition not met")
            return True

        self.logger.info(f"[START] {phase['name']}: {phase['description']}")

        try:
            # コマンド構築
            cmd = ["py", "-3"]

            if phase_key == "aegis_conversion":
                # 特殊な引数が必要な場合
                cmd.extend([
                    phase["script"],
                    "gguf",
                    "--model_path", str(self.project_root / "checkpoints" / "rlpo_science_nsfw_automated" / "final_model"),
                    "--quantization", "q8_0",
                    "--output_file", str(self.project_root / "D:/webdataset/gguf_models/aegis_phi35_so8t/aegis_phi35_so8t_Q8_0.gguf")
                ])
            else:
                cmd.append(phase["script"])
                if "args" in phase:
                    cmd.extend(phase["args"])

            self.logger.info(f"[EXEC] {' '.join(cmd)}")

            # プロセス実行
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=False,  # リアルタイム出力
                text=True,
                encoding='utf-8'
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                self.logger.info(f"[SUCCESS] {phase['name']} completed in {elapsed:.1f}s")
                return True
            else:
                self.logger.error(f"[ERROR] {phase['name']} failed with code {result.returncode}")
                return False

        except Exception as e:
            self.logger.error(f"[ERROR] {phase['name']} failed: {e}")
            return False

    def run_full_pipeline(self):
        """全フェーズ実行"""
        self.logger.info("[START] MOONSHOT Full Pipeline Execution")
        self.logger.info("=" * 80)

        # tqdm進捗バー
        phase_keys = list(self.phases.keys())
        total_phases = len(phase_keys)

        with tqdm(total=total_phases, desc="[MOONSHOT] Pipeline Progress",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]') as pbar:

            success_count = 0

            for i, phase_key in enumerate(phase_keys, 1):
                # フェーズ名をプログレスバーに表示
                pbar.set_description(f"[MOONSHOT] Phase {i}/{total_phases}: {self.phases[phase_key]['name']}")

                # フェーズ実行
                if self.run_single_phase(phase_key):
                    success_count += 1
                    pbar.update(1)
                else:
                    pbar.set_description(f"[MOONSHOT] FAILED at Phase {i}: {self.phases[phase_key]['name']}")
                    break

        # 結果表示
        self.logger.info("=" * 80)
        if success_count == total_phases:
            self.logger.info("[SUCCESS] MOONSHOT Pipeline completed successfully!")
            self.logger.info("[INFO] All phases executed without errors")
        else:
            self.logger.error(f"[ERROR] MOONSHOT Pipeline failed at phase {success_count + 1}/{total_phases}")

        return success_count == total_phases

def main():
    parser = argparse.ArgumentParser(description="MOONSHOT Pipeline with tqdm & logging")
    parser.add_argument("--phase", choices=[
        "env_check", "dataset_creation", "lm_eval_setup", "rlpo_training",
        "baseline_conversion", "aegis_conversion", "ab_test", "statistical_analysis",
        "visualization", "hf_upload_prep"
    ], help="Execute single phase")
    parser.add_argument("--full", action="store_true", help="Execute full pipeline")
    parser.add_argument("--list", action="store_true", help="List all phases")

    args = parser.parse_args()

    pipeline = MoonshotPipeline()

    if args.list:
        print("[INFO] Available MOONSHOT phases:")
        for key, phase in pipeline.phases.items():
            condition_met = True
            if 'condition' in phase:
                condition_met = phase['condition']()
            status = "[READY]" if condition_met else "[SKIP]"
            print(f"  {key}: {phase['name']} - {phase['description']} {status}")

    elif args.phase:
        success = pipeline.run_single_phase(args.phase)
        sys.exit(0 if success else 1)

    elif args.full:
        success = pipeline.run_full_pipeline()
        sys.exit(0 if success else 1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

