#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama.cpp.pythonを使用したA/Bテスト評価スクリプト

GGUFモデルをllama.cppで読み込み、全問解き評価を実行
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

# tqdm for progress bars
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import checkpoint manager
try:
    from scripts.utils.checkpoint_manager import create_task_manager
    checkpoint_available = True
except ImportError:
    checkpoint_available = False

class LlamaCppABTester:
    """llama.cpp.pythonを使用したA/Bテスト評価クラス"""

    def __init__(self, config_file: str = "data/evaluation/ab_test_config.json"):
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.results_dir = Path(self.config["output_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # チェックポイントマネージャー
        if checkpoint_available:
            self.checkpoint_manager = create_task_manager(
                "llama_cpp_ab_test",
                save_interval_sec=180,  # 3分毎
                max_checkpoints=5
            )

    def install_llama_cpp_python(self):
        """llama.cpp.pythonのインストール"""
        print("📦 Installing llama.cpp.python...")

        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "llama-cpp-python",
                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            ])
            print("✅ llama.cpp.python installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install llama.cpp.python: {e}")
            return False

    def load_model(self, model_config: Dict[str, Any]):
        """GGUFモデルの読み込み"""
        try:
            from llama_cpp import Llama

            gguf_path = model_config["gguf_path"]
            if not Path(gguf_path).exists():
                raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

            print(f"🔄 Loading model: {model_config['name']} from {gguf_path}")

            # GGUFモデルを読み込み
            llm = Llama(
                model_path=str(gguf_path),
                n_ctx=4096,  # コンテキスト長
                n_threads=8,  # スレッド数
                verbose=False
            )

            print("✅ Model loaded successfully"            return llm

        except ImportError:
            print("❌ llama-cpp-python not available")
            if self.install_llama_cpp_python():
                return self.load_model(model_config)
            return None
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None

    def load_evaluation_data(self, task_name: str) -> List[Dict[str, Any]]:
        """評価データの読み込み"""
        if task_name == "elyza_100":
            data_file = Path("data/evaluation/elyza_100.jsonl")
        else:
            # 他のタスクはlm-eval-harnessから取得
            print(f"⚠️ Task {task_name} not directly supported, skipping")
            return []

        if not data_file.exists():
            print(f"❌ Evaluation data not found: {data_file}")
            return []

        samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        print(f"📋 Loaded {len(samples)} samples for {task_name}")
        return samples

    def evaluate_sample(self, llm, sample: Dict[str, Any], task_name: str) -> Dict[str, Any]:
        """単一サンプルの評価"""
        input_text = sample.get("input", "")
        expected_output = sample.get("output", "")

        try:
            # モデルによる推論
            start_time = time.time()

            response = llm(
                input_text,
                max_tokens=512,
                temperature=0.0,  # 決定論的出力
                echo=False
            )

            inference_time = time.time() - start_time

            generated_text = response["choices"][0]["text"].strip()

            # 評価（完全一致）
            exact_match = generated_text == expected_output

            return {
                "task_id": sample.get("task_id", ""),
                "input": input_text,
                "expected_output": expected_output,
                "generated_output": generated_text,
                "exact_match": exact_match,
                "inference_time": inference_time,
                "model_response": response
            }

        except Exception as e:
            print(f"❌ Error evaluating sample: {e}")
            return {
                "task_id": sample.get("task_id", ""),
                "input": input_text,
                "expected_output": expected_output,
                "generated_output": "",
                "exact_match": False,
                "inference_time": 0.0,
                "error": str(e)
            }

    def evaluate_model(self, model_config: Dict[str, Any], task_name: str, num_fewshot: int = 0) -> Dict[str, Any]:
        """モデルの評価実行"""
        print(f"🔬 Evaluating {model_config['name']} on {task_name} (fewshot={num_fewshot})")

        # モデル読み込み
        llm = self.load_model(model_config)
        if llm is None:
            return {"error": "Failed to load model"}

        # データ読み込み
        samples = self.load_evaluation_data(task_name)
        if not samples:
            return {"error": "No evaluation data found"}

        # 評価実行
        results = []
        correct_count = 0

        for sample in tqdm(samples, desc=f"Evaluating {model_config['name']}"):
            result = self.evaluate_sample(llm, sample, task_name)
            results.append(result)

            if result.get("exact_match", False):
                correct_count += 1

        # 結果集計
        accuracy = correct_count / len(samples) if samples else 0

        evaluation_result = {
            "model_name": model_config["name"],
            "task_name": task_name,
            "num_fewshot": num_fewshot,
            "total_samples": len(samples),
            "correct_count": correct_count,
            "accuracy": accuracy,
            "results": results,
            "timestamp": time.time()
        }

        return evaluation_result

    def run_ab_test(self):
        """A/Bテスト実行"""
        print("🆚 Starting A/B Test Evaluation")
        print("=" * 50)

        # チェックポイント開始
        if checkpoint_available:
            self.checkpoint_manager.start()

        results = {
            "baseline": {},
            "aegis": {},
            "comparison": {},
            "timestamp": time.time()
        }

        try:
            baseline_config = self.config["baseline_model"]
            aegis_config = self.config["aegis_model"]
            tasks = self.config["evaluation_tasks"]
            fewshot_settings = self.config["num_fewshot"]

            # 各モデル・各タスク・各fewshot設定で評価
            for model_config, model_key in [(baseline_config, "baseline"), (aegis_config, "aegis")]:
                print(f"\n🔬 Evaluating {model_key.upper()} model...")
                results[model_key] = {}

                for task in tasks:
                    results[model_key][task] = {}

                    for fewshot in fewshot_settings:
                        print(f"  📊 {task} (fewshot={fewshot})")

                        eval_result = self.evaluate_model(model_config, task, fewshot)
                        results[model_key][task][str(fewshot)] = eval_result

                        # 中間結果保存
                        self.save_results(results, "intermediate")

            # 比較分析
            results["comparison"] = self.compare_models(results)

            # 最終結果保存
            self.save_results(results, "final")

            print("
🎉 A/B Test completed!"            print(f"📊 Results saved to {self.results_dir}")

            # チェックポイント完了
            if checkpoint_available:
                self.checkpoint_manager.mark_completed()

        except Exception as e:
            print(f"❌ A/B test failed: {e}")
            if checkpoint_available:
                self.checkpoint_manager.save_checkpoint()
            raise

        return results

    def compare_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """モデル比較分析"""
        comparison = {}

        baseline = results.get("baseline", {})
        aegis = results.get("aegis", {})

        for task in self.config["evaluation_tasks"]:
            comparison[task] = {}

            for fewshot in self.config["num_fewshot"]:
                fewshot_str = str(fewshot)

                baseline_result = baseline.get(task, {}).get(fewshot_str, {})
                aegis_result = aegis.get(task, {}).get(fewshot_str, {})

                baseline_acc = baseline_result.get("accuracy", 0)
                aegis_acc = aegis_result.get("accuracy", 0)

                comparison[task][fewshot_str] = {
                    "baseline_accuracy": baseline_acc,
                    "aegis_accuracy": aegis_acc,
                    "improvement": aegis_acc - baseline_acc,
                    "improvement_percent": (aegis_acc - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
                }

        return comparison

    def save_results(self, results: Dict[str, Any], suffix: str = "final"):
        """結果保存"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ab_test_results_{suffix}_{timestamp}.json"

        output_file = self.results_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run A/B Test with llama.cpp.python")
    parser.add_argument("--config", type=str, default="data/evaluation/ab_test_config.json",
                       help="A/B test configuration file")
    parser.add_argument("--checkpoint", action="store_true",
                       help="Enable checkpointing")

    args = parser.parse_args()

    tester = LlamaCppABTester(args.config)
    results = tester.run_ab_test()

    if results:
        print("\n✅ A/B test completed successfully!")
        print("📊 Next: Run statistical analysis with scripts/evaluation/analyze_ab_test_stats.py")
    else:
        print("\n❌ A/B test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
