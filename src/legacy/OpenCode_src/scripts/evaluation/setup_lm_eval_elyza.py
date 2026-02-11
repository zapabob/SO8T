#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lm-eval-harnessとELYZA-100セットアップスクリプト

A/Bテスト用の評価環境を構築する
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

# Set UTF-8 encoding for Windows compatibility
os.environ['PYTHONIOENCODING'] = 'utf-8'
from typing import Dict, List, Any

# tqdm for progress bars
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class LMEvalSetup:
    """lm-eval-harnessとELYZA-100セットアップクラス"""

    def __init__(self, lm_eval_dir: str = "lm-evaluation-harness"):
        self.lm_eval_dir = Path(lm_eval_dir)
        self.eval_data_dir = Path("data/evaluation")
        self.eval_data_dir.mkdir(parents=True, exist_ok=True)

    def setup_elyza_100(self):
        """ELYZA-100データセットのダウンロードとセットアップ"""
        print("[DOWNLOAD] Setting up ELYZA-100 dataset...")

        try:
            from datasets import load_dataset

            # ELYZA-100データセットをダウンロード
            print("Downloading ELYZA-100 from Hugging Face...")
            dataset = load_dataset("elyza/ELYZA-tasks-100")

            # データセットをJSONL形式で保存
            output_file = self.eval_data_dir / "elyza_100.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:
                for split_name, split_data in dataset.items():
                    print(f"Processing {split_name} split...")
                    for item in tqdm(split_data, desc=f"Processing {split_name}"):
                        # ELYZA-100のフォーマットを統一
                        sample = {
                            "task_id": item.get("task_id", ""),
                            "input": item.get("input", ""),
                            "output": item.get("output", ""),
                            "eval_aspect": item.get("eval_aspect", ""),
                            "split": split_name
                        }
                        json.dump(sample, f, ensure_ascii=False)
                        f.write('\n')

            print(f"[SUCCESS] ELYZA-100 saved to {output_file}")

        except ImportError:
            print("[ERROR] datasets library not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            self.setup_elyza_100()  # 再試行

        except Exception as e:
            print(f"[ERROR] Failed to setup ELYZA-100: {e}")
            raise

    def setup_lm_eval_config(self):
        """lm-eval-harnessの設定ファイル作成"""
        print("[CONFIG] Creating lm-eval configuration...")

        # カスタムタスク設定
        custom_tasks_config = {
            "group": "custom_aegis_tasks",
            "task": "elyza_100",
            "dataset_path": str(self.eval_data_dir / "elyza_100.jsonl"),
            "output_type": "generate_until",
            "test_split": "test",
            "doc_to_text": "{{input}}",
            "doc_to_target": "{{output}}",
            "generation_kwargs": {
                "until": ["\n"],
                "do_sample": False,
                "temperature": 0.0
            },
            "metric_list": [
                {"metric": "exact_match", "aggregation": "mean"}
            ]
        }

        # 設定ファイルを保存
        config_file = self.eval_data_dir / "custom_tasks.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump([custom_tasks_config], f, allow_unicode=True, default_flow_style=False)

        print(f"[SUCCESS] lm-eval config saved to {config_file}")

    def verify_lm_eval_setup(self):
        """lm-eval-harnessのセットアップ確認"""
        print("[VERIFY] Verifying lm-eval setup...")

        try:
            # lm-eval-harnessのインポートテスト
            sys.path.insert(0, str(self.lm_eval_dir))

            import lm_eval
            print("[SUCCESS] lm-eval-harness import successful")

            # 利用可能なタスク一覧を取得
            tasks = lm_eval.list_tasks()
            print(f"[INFO] Available tasks: {len(tasks)} tasks loaded")

            # ELYZA-100タスクの確認
            if "elyza_100" in tasks:
                print("[SUCCESS] ELYZA-100 task registered")
            else:
                print("[WARNING] ELYZA-100 task not found in registry")

        except ImportError as e:
            print(f"[ERROR] lm-eval import failed: {e}")
            return False

        return True

    def create_ab_test_config(self):
        """A/Bテスト設定ファイル作成"""
        print("[ABTEST] Creating A/B test configuration...")

        ab_config = {
            "baseline_model": {
                "name": "baseline_phi35_bf16",
                "path": "microsoft/phi-3.5-mini-instruct",
                "quantization": "bf16",
                "gguf_path": "D:/webdataset/gguf_models/baseline_phi35_bf16.gguf"
            },
            "aegis_model": {
                "name": "aegis_phi35_so8t",
                "path": "checkpoints/rlpo_science_nsfw_automated/final_model",
                "quantization": "q8_0",
                "gguf_path": "D:/webdataset/gguf_models/aegis_phi35_so8t/aegis_phi35_so8t_Q8_0.gguf"
            },
            "evaluation_tasks": [
                "elyza_100",
                "arc_challenge",
                "hellaswag",
                "truthfulqa_mc2",
                "winogrande",
                "gsm8k"
            ],
            "batch_size": 8,
            "num_fewshot": [0, 5, 10],
            "output_dir": "results/ab_test_results"
        }

        config_file = self.eval_data_dir / "ab_test_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(ab_config, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] A/B test config saved to {config_file}")

    def run(self):
        """メイン実行関数"""
        print("[START] Setting up lm-eval-harness and ELYZA-100 for A/B testing")
        print("=" * 70)

        try:
            # ELYZA-100セットアップ
            self.setup_elyza_100()

            # lm-eval設定
            self.setup_lm_eval_config()

            # A/Bテスト設定
            self.create_ab_test_config()

            # セットアップ検証
            if self.verify_lm_eval_setup():
                print("\n[SUCCESS] lm-eval and ELYZA-100 setup completed successfully!")
                print("[READY] Ready for A/B testing with statistical analysis")
            else:
                print("\n[ERROR] Setup verification failed")
                return False

        except Exception as e:
            print(f"\n[ERROR] Setup failed: {e}")
            return False

        return True

def main():
    parser = argparse.ArgumentParser(description="Setup lm-eval-harness and ELYZA-100")
    parser.add_argument("--lm_eval_dir", type=str, default="lm-evaluation-harness",
                       help="Path to lm-evaluation-harness directory")
    parser.add_argument("--eval_data_dir", type=str, default="data/evaluation",
                       help="Directory to store evaluation data")

    args = parser.parse_args()

    setup = LMEvalSetup(args.lm_eval_dir)
    success = setup.run()

    if success:
        print("\n[SUCCESS] All components ready for A/B testing!")
        print("Next: Run scripts/evaluation/run_llama_cpp_ab_test.py")
    else:
        print("\n[ERROR] Setup failed. Please check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
