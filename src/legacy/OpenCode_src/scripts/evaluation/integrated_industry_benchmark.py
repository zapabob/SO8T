#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合業界標準ベンチマークスクリプト
lm-evaluation-harnessを使用してmodelA（Borea-Phi3.5-instinct-jp）とAEGISを自動比較
GGUFモデルをOllama経由で実行
"""

import argparse
import json
import subprocess
import sys
import time
import signal
import atexit
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import logging.handlers
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import stats

try:
    import torch
except ImportError:
    torch = None

# ロギング設定
LOG_DIR = Path("D:/webdataset/benchmark_results/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

file_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / 'integrated_industry_benchmark.log',
    maxBytes=10*1024*1024,
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 設定
DATA_DIR = Path("D:/webdataset/benchmark_datasets")
RESULTS_DIR = Path("D:/webdataset/benchmark_results/industry_standard")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_INTERVAL = 180  # 3分（秒）
MAX_CHECKPOINTS = 5
PLAY_AUDIO_SCRIPT = Path("scripts/utils/play_audio_notification.ps1")

# モデル設定
# lm-evaluation-harnessはollamaバックエンドをサポートしていないため、
# openai-chat-completionsバックエンド経由でOllama APIを使用
MODELS = {
    "modelA": {
        "name": "Borea-Phi3.5-instinct-jp",
        "ollama_name": "Borea-Phi3.5-instinct-jp:latest",  # モデルが存在しない場合は model-a:q8_0 を使用
        "runner": "openai-chat-completions",  # Ollama API互換のopenai-chat-completionsバックエンドを使用
        "fallback": "model-a:q8_0"  # フォールバックモデル
    },
    "AEGIS": {
        "name": "aegis-adjusted",
        "ollama_name": "aegis-adjusted:latest",  # モデルが存在しない場合は aegis-0.8:latest を使用
        "runner": "openai-chat-completions",  # Ollama API互換のopenai-chat-completionsバックエンドを使用
        "fallback": "aegis-0.8:latest"  # フォールバックモデル
    }
}

# ベンチマークタスク設定
BENCHMARK_TASKS = {
    "mmlu": {
        "lm_eval_task": "mmlu",
        "description": "MMLU (Massive Multitask Language Understanding)"
    },
    "gsm8k": {
        "lm_eval_task": "gsm8k",
        "description": "GSM8K (Grade School Math 8K)"
    },
    "arc_challenge": {
        "lm_eval_task": "arc:challenge",
        "description": "ARC Challenge"
    },
    "arc_easy": {
        "lm_eval_task": "arc:easy",
        "description": "ARC Easy"
    },
    "hellaswag": {
        "lm_eval_task": "hellaswag",
        "description": "HellaSwag"
    },
    "winogrande": {
        "lm_eval_task": "winogrande",
        "description": "Winogrande"
    }
}

# ELYZA-100とFinal Examはカスタム実装が必要（後で追加）


class IntegratedIndustryBenchmark:
    """統合業界標準ベンチマーク実行クラス"""

    def __init__(self, output_dir: Path = RESULTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter = 0
        self.all_results = []
        self.completed_tests = set()
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self.signal_handler)
        
        atexit.register(self.cleanup)
        
        # チェックポイント復旧
        self.load_checkpoint()

    def signal_handler(self, signum, frame):
        """シグナルハンドラー（電源断対応）"""
        logger.warning(f"Received signal {signum}, saving checkpoint...")
        self.save_checkpoint("interrupted")
        sys.exit(0)

    def cleanup(self):
        """クリーンアップ処理"""
        try:
            self.save_checkpoint("final")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def should_save_checkpoint(self) -> bool:
        """チェックポイント保存判定"""
        current_time = time.time()
        return (current_time - self.last_checkpoint_time) >= CHECKPOINT_INTERVAL

    def cleanup_old_checkpoints(self):
        """古いチェックポイントを削除（ローリングストック）"""
        try:
            checkpoints = sorted(
                self.checkpoint_dir.glob("industry_benchmark_checkpoint_*.json"),
                key=lambda x: x.stat().st_mtime
            )
            
            if len(checkpoints) > MAX_CHECKPOINTS:
                checkpoints_to_delete = checkpoints[:-MAX_CHECKPOINTS]
                for checkpoint in checkpoints_to_delete:
                    checkpoint.unlink()
                    logger.debug(f"Deleted old checkpoint: {checkpoint}")
                
                logger.info(f"Cleaned up {len(checkpoints_to_delete)} old checkpoints")
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")

    def save_checkpoint(self, suffix: str = ""):
        """チェックポイント保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"industry_benchmark_checkpoint_{timestamp}_{suffix}.json"
            
            checkpoint_data = {
                "timestamp": timestamp,
                "completed_tests": list(self.completed_tests),
                "results": self.all_results,
                "checkpoint_counter": self.checkpoint_counter
            }
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.last_checkpoint_time = time.time()
            self.checkpoint_counter += 1
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            self.cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        """チェックポイント復旧"""
        try:
            checkpoints = sorted(
                self.checkpoint_dir.glob("industry_benchmark_checkpoint_*.json"),
                key=lambda x: x.stat().st_mtime
            )
            
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                logger.info(f"Loading checkpoint: {latest_checkpoint}")
                
                with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                self.completed_tests = set(checkpoint_data.get("completed_tests", []))
                self.all_results = checkpoint_data.get("results", [])
                self.checkpoint_counter = checkpoint_data.get("checkpoint_counter", 0)
                
                logger.info(f"Recovered {len(self.all_results)} results from checkpoint")
                logger.info(f"Completed tests: {sorted(self.completed_tests)}")
                
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    def run_lm_eval_benchmark(
        self,
        model_key: str,
        task_key: str,
        limit: Optional[int] = None
    ) -> Dict:
        """lm-evaluation-harnessを使用してベンチマークを実行"""
        model_config = MODELS[model_key]
        task_config = BENCHMARK_TASKS[task_key]
        
        test_id = f"{model_key}_{task_key}"
        if test_id in self.completed_tests:
            logger.info(f"Skipping already completed test: {test_id}")
            return None
        
        logger.info(f"Running benchmark: {model_key} on {task_key}")
        
        # 実行ディレクトリ作成
        run_dir = self.output_dir / f"{model_key}_{task_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # lm-evalコマンド構築
        # Ollama APIをopenai-chat-completionsバックエンド経由で使用
        model_runner = model_config["runner"]
        # Ollama APIはOpenAI互換APIなので、openai-chat-completionsバックエンドを使用
        # base_urlをOllamaのAPIエンドポイントに設定（/v1エンドポイントを使用）
        # api_keyは不要だが、lm-evalが要求する場合は"ollama"を指定
        model_args_value = f"model={model_config['ollama_name']},base_url=http://127.0.0.1:11434/v1"
        
        cmd = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            model_runner,
            "--model_args",
            model_args_value,
            "--tasks",
            task_config["lm_eval_task"],
            "--batch_size",
            "4",
            "--device",
            "cuda:0" if torch and torch.cuda.is_available() else "cpu",
            "--output_path",
            str(run_dir),
            "--log_samples"
        ]
        
        if limit:
            cmd.extend(["--limit", str(limit)])
        
        # 実行
        log_path = run_dir / "lm_eval_stdout.log"
        start_time = time.time()
        
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                    log_file.flush()
                
                process.wait()
                exit_code = process.returncode
            
            elapsed_time = time.time() - start_time
            
            # 結果読み込み
            results_path = run_dir / "results.json"
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                
                # 結果サマリー抽出
                summary = {}
                for task_name, metrics in results_data.get("results", {}).items():
                    numeric_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            numeric_metrics[key] = float(value)
                    summary[task_name] = numeric_metrics
                
                result = {
                    "model": model_key,
                    "model_name": model_config["name"],
                    "task": task_key,
                    "task_name": task_config["lm_eval_task"],
                    "elapsed_time": elapsed_time,
                    "exit_code": exit_code,
                    "summary": summary,
                    "run_dir": str(run_dir),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                self.all_results.append(result)
                self.completed_tests.add(test_id)
                
                if self.should_save_checkpoint():
                    self.save_checkpoint("progress")
                
                logger.info(f"Benchmark completed: {model_key} on {task_key} (time: {elapsed_time:.1f}s)")
                return result
            else:
                logger.error(f"Results file not found: {results_path}")
                return None
                
        except Exception as e:
            logger.error(f"Benchmark failed: {model_key} on {task_key}: {e}")
            return None

    def run_benchmark_suite(self, limit: Optional[int] = None):
        """全ベンチマークスイート実行"""
        logger.info("Starting integrated industry benchmark suite...")
        
        total_tests = len(MODELS) * len(BENCHMARK_TASKS)
        completed = 0
        
        with tqdm(total=total_tests, desc="Benchmark Progress") as pbar:
            for model_key in MODELS.keys():
                for task_key in BENCHMARK_TASKS.keys():
                    result = self.run_lm_eval_benchmark(model_key, task_key, limit)
                    completed += 1
                    pbar.update(1)
                    
                    if result:
                        pbar.set_postfix({
                            "model": model_key,
                            "task": task_key,
                            "status": "OK"
                        })
                    else:
                        pbar.set_postfix({
                            "model": model_key,
                            "task": task_key,
                            "status": "SKIP/FAIL"
                        })
        
        logger.info(f"Benchmark suite completed: {completed}/{total_tests} tests")
        self.save_checkpoint("final")

    def compare_models(self) -> Dict:
        """モデル比較結果を生成"""
        if not self.all_results:
            logger.warning("No results to compare")
            return {}
        
        comparison = {}
        
        for task_key in BENCHMARK_TASKS.keys():
            task_results = [r for r in self.all_results if r["task"] == task_key]
            
            if len(task_results) < 2:
                continue
            
            model_a_results = [r for r in task_results if r["model"] == "modelA"]
            aegis_results = [r for r in task_results if r["model"] == "AEGIS"]
            
            if not model_a_results or not aegis_results:
                continue
            
            # スコア抽出（acc, exact_match等）
            def extract_score(result):
                summary = result.get("summary", {})
                for task_name, metrics in summary.items():
                    score = metrics.get("acc", metrics.get("exact_match", metrics.get("acc_norm", 0.0)))
                    return score
                return 0.0
            
            model_a_score = extract_score(model_a_results[0])
            aegis_score = extract_score(aegis_results[0])
            
            comparison[task_key] = {
                "modelA": model_a_score,
                "AEGIS": aegis_score,
                "difference": aegis_score - model_a_score,
                "modelA_time": model_a_results[0].get("elapsed_time", 0.0),
                "AEGIS_time": aegis_results[0].get("elapsed_time", 0.0)
            }
        
        return comparison

    def calculate_statistics(self) -> Dict:
        """統計分析を実行"""
        comparison = self.compare_models()
        
        if not comparison:
            return {}
        
        stats_dict = {}
        
        for task_key, comp_data in comparison.items():
            model_a_score = comp_data["modelA"]
            aegis_score = comp_data["AEGIS"]
            diff = comp_data["difference"]
            
            # 95%信頼区間（簡易版、サンプル数が少ない場合は警告）
            n = len([r for r in self.all_results if r["task"] == task_key])
            if n >= 2:
                std_err = abs(diff) / np.sqrt(n) if n > 0 else 0.0
                ci_95 = 1.96 * std_err
            else:
                ci_95 = 0.0
            
            stats_dict[task_key] = {
                "modelA_mean": model_a_score,
                "AEGIS_mean": aegis_score,
                "difference_mean": diff,
                "difference_std": 0.0,  # 単一実行のため0
                "ci_95_lower": diff - ci_95,
                "ci_95_upper": diff + ci_95,
                "n_samples": n
            }
        
        return stats_dict

    def perform_significance_test(self) -> Dict:
        """統計的有意差検定（t-test）"""
        comparison = self.compare_models()
        
        if not comparison:
            return {}
        
        significance_results = {}
        
        for task_key, comp_data in comparison.items():
            # 単一実行のため、有意差検定は簡易版
            diff = comp_data["difference"]
            abs_diff = abs(diff)
            
            # 効果量ベースの簡易判定（0.05を閾値とする）
            significant = abs_diff > 0.05
            
            significance_results[task_key] = {
                "difference": diff,
                "significant": significant,
                "p_value": None,  # 単一実行のため計算不可
                "effect_size": abs_diff
            }
        
        return significance_results

    def save_results(self):
        """結果をJSON形式で保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"integrated_benchmark_results_{timestamp}.json"
        
        comparison = self.compare_models()
        statistics = self.calculate_statistics()
        significance = self.perform_significance_test()
        
        output_data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "models": MODELS,
                "tasks": BENCHMARK_TASKS,
                "total_results": len(self.all_results)
            },
            "results": self.all_results,
            "comparison": comparison,
            "statistics": statistics,
            "significance": significance
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_path}")
        return results_path


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Industry Standard Benchmark"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (None = all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh (ignore checkpoints)"
    )
    
    args = parser.parse_args()
    
    benchmark = IntegratedIndustryBenchmark(output_dir=args.output_dir)
    
    if args.fresh:
        benchmark.completed_tests = set()
        benchmark.all_results = []
        logger.info("Starting fresh (checkpoints ignored)")
    
    # ベンチマーク実行
    benchmark.run_benchmark_suite(limit=args.limit)
    
    # 結果保存
    results_path = benchmark.save_results()
    
    # 音声通知
    if PLAY_AUDIO_SCRIPT.exists():
        try:
            subprocess.run(
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(PLAY_AUDIO_SCRIPT),
                ],
                check=False,
            )
        except Exception:
            pass
    
    logger.info("Integrated industry benchmark completed!")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

