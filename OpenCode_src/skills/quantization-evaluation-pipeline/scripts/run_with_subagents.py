#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
サブエージェントを使用したGGUF量子化評価パイプライン実行スクリプト
並列処理による効率化とリアルタイム監視
"""

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import uuid
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SubAgentOrchestrator:
    """
    サブエージェントオーケストレーター
    GGUF量子化評価パイプラインの並列実行を管理
    """

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.output_dir = Path("quantization_evaluation_output")
        self.progress_file = self.output_dir / f"pipeline_progress_{pipeline_id}.json"
        self.lock = threading.Lock()

        # Phase定義
        self.phases = [
            {"id": "imatrix_collection", "name": "imatrixデータ収集", "weight": 0.2},
            {"id": "quantization", "name": "GGUF量子化実行", "weight": 0.3},
            {"id": "evaluation", "name": "統計的評価", "weight": 0.3},
            {"id": "visualization", "name": "結果可視化", "weight": 0.1},
            {"id": "documentation", "name": "学術文書生成", "weight": 0.1}
        ]

        # 初期化
        self._initialize_progress_tracking()

    def _initialize_progress_tracking(self):
        """進捗追跡初期化"""
        initial_progress = {
            "pipeline_id": self.pipeline_id,
            "start_time": time.time(),
            "current_phase": 0,
            "total_phases": len(self.phases),
            "percentage": 0.0,
            "phase_name": "initialization",
            "phase_start_time": time.time(),
            "elapsed_seconds": 0,
            "estimated_remaining": "計算中...",
            "status": "running",
            "process_id": os.getpid(),
            "subagents": {},
            "errors": []
        }

        with self.lock:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(initial_progress, f, indent=2, ensure_ascii=False)

    def update_progress(self, phase_index: int, phase_progress: float = 1.0, status: str = "running", error: str = None):
        """進捗更新"""
        with self.lock:
            try:
                # 現在の進捗読み込み
                if self.progress_file.exists():
                    with open(self.progress_file, 'r', encoding='utf-8') as f:
                        progress = json.load(f)
                else:
                    progress = self._initialize_progress_tracking()

                # 進捗計算
                completed_weight = sum(phase["weight"] for phase in self.phases[:phase_index])
                current_weight = self.phases[phase_index]["weight"] * phase_progress
                total_percentage = (completed_weight + current_weight) * 100

                # 時間計算
                current_time = time.time()
                elapsed = current_time - progress["start_time"]

                # ETA計算
                if total_percentage > 0:
                    total_estimated_time = elapsed / (total_percentage / 100)
                    remaining_time = total_estimated_time - elapsed
                    eta = self._format_time(remaining_time)
                else:
                    eta = "計算中..."

                # 進捗更新
                progress.update({
                    "current_phase": phase_index,
                    "percentage": total_percentage,
                    "phase_name": self.phases[phase_index]["name"],
                    "elapsed_seconds": int(elapsed),
                    "estimated_remaining": eta,
                    "status": status
                })

                # エラー記録
                if error:
                    progress["errors"].append({
                        "phase": self.phases[phase_index]["name"],
                        "error": error,
                        "timestamp": time.time()
                    })

                # 進捗保存
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress, f, indent=2, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Progress update failed: {e}")

    def _format_time(self, seconds: float) -> str:
        """時間フォーマット"""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}分{remaining_seconds}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}時間{minutes}分"

    def run_imatrix_collection_agent(self, model_path: str) -> bool:
        """imatrix収集サブエージェント"""
        logger.info("Starting imatrix collection subagent")

        try:
            self.update_progress(0, 0.1, "running")

            cmd = [
                "python", "scripts/quantization/collect_imatrix_data.py",
                "--model", model_path,
                "--output", "imatrix_data/model.imatrix",
                "--samples", "100000"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            if result.returncode == 0:
                self.update_progress(0, 1.0, "running")
                logger.info("imatrix collection completed")
                return True
            else:
                error_msg = f"imatrix collection failed: {result.stderr}"
                self.update_progress(0, 1.0, "failed", error_msg)
                logger.error(error_msg)
                return False

        except subprocess.TimeoutExpired:
            error_msg = "imatrix collection timed out"
            self.update_progress(0, 1.0, "failed", error_msg)
            return False
        except Exception as e:
            error_msg = f"imatrix collection error: {e}"
            self.update_progress(0, 1.0, "failed", error_msg)
            return False

    def run_quantization_agent(self, model_path: str, quantizations: List[str]) -> bool:
        """量子化サブエージェント"""
        logger.info("Starting quantization subagent")

        try:
            self.update_progress(1, 0.0, "running")
            total_formats = len(quantizations)

            for i, quant_format in enumerate(quantizations):
                progress = (i / total_formats) * 0.9  # 90%まで
                self.update_progress(1, progress, "running")

                logger.info(f"Quantizing to {quant_format}")

                cmd = [
                    "python", "scripts/quantization/quantize_with_imatrix.py",
                    "--model", model_path,
                    "--imatrix", "imatrix_data/model.imatrix",
                    "--format", quant_format,
                    "--output", f"quantized_models/model_{quant_format}.gguf"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                if result.returncode != 0:
                    error_msg = f"Quantization to {quant_format} failed: {result.stderr}"
                    self.update_progress(1, 1.0, "failed", error_msg)
                    return False

            self.update_progress(1, 1.0, "running")
            logger.info("Quantization completed")
            return True

        except subprocess.TimeoutExpired:
            error_msg = "Quantization timed out"
            self.update_progress(1, 1.0, "failed", error_msg)
            return False
        except Exception as e:
            error_msg = f"Quantization error: {e}"
            self.update_progress(1, 1.0, "failed", error_msg)
            return False

    def run_evaluation_agent(self, quantizations: List[str], benchmarks: List[str], runs: int) -> bool:
        """評価サブエージェント"""
        logger.info("Starting evaluation subagent")

        try:
            self.update_progress(2, 0.0, "running")

            total_evaluations = len(quantizations) * len(benchmarks)
            completed = 0

            for quant_format in quantizations:
                model_path = f"quantized_models/model_{quant_format}.gguf"

                for benchmark in benchmarks:
                    logger.info(f"Evaluating {quant_format} on {benchmark}")

                    # 複数回実行
                    for run in range(runs):
                        cmd = [
                            "python", "scripts/evaluation/statistical_benchmark_evaluation.py",
                            "--model", model_path,
                            "--benchmark", benchmark,
                            "--run", str(run)
                        ]

                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                        if result.returncode != 0:
                            error_msg = f"Evaluation failed for {quant_format} on {benchmark}: {result.stderr}"
                            self.update_progress(2, 1.0, "failed", error_msg)
                            return False

                    completed += 1
                    progress = completed / total_evaluations
                    self.update_progress(2, progress, "running")

            self.update_progress(2, 1.0, "running")
            logger.info("Evaluation completed")
            return True

        except Exception as e:
            error_msg = f"Evaluation error: {e}"
            self.update_progress(2, 1.0, "failed", error_msg)
            return False

    def run_visualization_agent(self) -> bool:
        """可視化サブエージェント"""
        logger.info("Starting visualization subagent")

        try:
            self.update_progress(3, 0.3, "running")

            # 性能比較グラフ生成
            cmd1 = [
                "python", "scripts/visualization/generate_quantization_comparison.py",
                "--results", "evaluation_results/quantization_comparison.json",
                "--output", "charts/quantization_performance.png"
            ]

            result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)

            if result1.returncode != 0:
                error_msg = f"Performance chart generation failed: {result1.stderr}"
                self.update_progress(3, 1.0, "failed", error_msg)
                return False

            self.update_progress(3, 0.7, "running")

            # サイズ vs 性能グラフ生成
            cmd2 = [
                "python", "scripts/visualization/generate_size_performance_tradeoff.py",
                "--results", "evaluation_results/quantization_comparison.json",
                "--output", "charts/size_performance_tradeoff.png"
            ]

            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=300)

            if result2.returncode != 0:
                error_msg = f"Size-performance chart generation failed: {result2.stderr}"
                self.update_progress(3, 1.0, "failed", error_msg)
                return False

            self.update_progress(3, 1.0, "running")
            logger.info("Visualization completed")
            return True

        except Exception as e:
            error_msg = f"Visualization error: {e}"
            self.update_progress(3, 1.0, "failed", error_msg)
            return False

    def run_documentation_agent(self) -> bool:
        """文書生成サブエージェント"""
        logger.info("Starting documentation subagent")

        try:
            self.update_progress(4, 0.5, "running")

            # スコアカード生成
            cmd = [
                "python", "scripts/documentation/generate_academic_scorecard.py",
                "--results", "evaluation_results/quantization_comparison.json",
                "--methodology", "methodology/quantization_methodology.md",
                "--output", "scorecards/quantization_evaluation.md"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                error_msg = f"Documentation generation failed: {result.stderr}"
                self.update_progress(4, 1.0, "failed", error_msg)
                return False

            self.update_progress(4, 1.0, "running")
            logger.info("Documentation completed")
            return True

        except Exception as e:
            error_msg = f"Documentation error: {e}"
            self.update_progress(4, 1.0, "failed", error_msg)
            return False

    def execute_pipeline(self, model_path: str, quantizations: List[str],
                        benchmarks: List[str], runs: int) -> Dict[str, Any]:
        """パイプライン実行"""
        logger.info("Starting subagent-based quantization evaluation pipeline")

        results = {
            "pipeline_id": self.pipeline_id,
            "start_time": time.time(),
            "phases": []
        }

        try:
            # Phase 1: imatrix収集
            phase_start = time.time()
            success = self.run_imatrix_collection_agent(model_path)
            if not success:
                raise RuntimeError("imatrix collection failed")

            results["phases"].append({
                "name": "imatrix_collection",
                "success": True,
                "duration": time.time() - phase_start
            })

            # Phase 2: 量子化
            phase_start = time.time()
            success = self.run_quantization_agent(model_path, quantizations)
            if not success:
                raise RuntimeError("quantization failed")

            results["phases"].append({
                "name": "quantization",
                "success": True,
                "duration": time.time() - phase_start
            })

            # Phase 3: 評価
            phase_start = time.time()
            success = self.run_evaluation_agent(quantizations, benchmarks, runs)
            if not success:
                raise RuntimeError("evaluation failed")

            results["phases"].append({
                "name": "evaluation",
                "success": True,
                "duration": time.time() - phase_start
            })

            # Phase 4: 可視化
            phase_start = time.time()
            success = self.run_visualization_agent()
            if not success:
                raise RuntimeError("visualization failed")

            results["phases"].append({
                "name": "visualization",
                "success": True,
                "duration": time.time() - phase_start
            })

            # Phase 5: 文書生成
            phase_start = time.time()
            success = self.run_documentation_agent()
            if not success:
                raise RuntimeError("documentation failed")

            results["phases"].append({
                "name": "documentation",
                "success": True,
                "duration": time.time() - phase_start
            })

            # 完了
            self.update_progress(4, 1.0, "completed")

            results["status"] = "completed"
            results["total_duration"] = time.time() - results["start_time"]
            results["end_time"] = time.time()

            logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.update_progress(len(self.phases) - 1, 1.0, "failed", str(e))

            results["status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = time.time()

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Subagent-based GGUF Quantization Evaluation Pipeline')
    parser.add_argument('--model', required=True, help='Path to base model')
    parser.add_argument('--quantizations', nargs='+', default=['bf16', 'q8_0', 'q4_k_m'],
                       help='Quantization formats')
    parser.add_argument('--benchmarks', nargs='+',
                       default=['gsm8k', 'math', 'arc_challenge', 'elyza_tasks_100'],
                       help='Benchmarks to evaluate')
    parser.add_argument('--runs', type=int, default=5, help='Runs per evaluation')
    parser.add_argument('--pipeline-id', help='Pipeline ID (auto-generated if not provided)')

    args = parser.parse_args()

    # Pipeline ID生成
    pipeline_id = args.pipeline_id or str(uuid.uuid4())[:8]
    logger.info(f"Pipeline ID: {pipeline_id}")

    # オーケストレーター初期化
    orchestrator = SubAgentOrchestrator(pipeline_id)

    try:
        # PowerShell進捗監視スクリプト起動（バックグラウンド）
        logger.info("Starting PowerShell progress monitor...")

        ps_cmd = [
            "powershell.exe", "-ExecutionPolicy", "Bypass", "-File",
            "scripts/monitor_quantization_progress.ps1",
            "-PipelineId", pipeline_id
        ]

        # PowerShellプロセスをバックグラウンドで起動
        ps_process = subprocess.Popen(
            ps_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        logger.info("PowerShell monitor started (PID: {})".format(ps_process.pid))

        # パイプライン実行
        results = orchestrator.execute_pipeline(
            args.model, args.quantizations, args.benchmarks, args.runs
        )

        # 結果保存
        output_file = f"quantization_evaluation_output/pipeline_results_{pipeline_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if results["status"] == "completed":
            print("🎉 Subagent-based quantization evaluation pipeline completed!")
            print(f"📊 Pipeline ID: {pipeline_id}")
            print(f"📈 Results saved to: {output_file}")
            print("📋 Progress monitor should show completion status"
        else:
            print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            exit(1)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("⏹️  Pipeline interrupted")
        exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()