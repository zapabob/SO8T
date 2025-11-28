#!/usr/bin/env python3
"""
SO8T Post-Training Workflow Automation
HFモデル完成後の自動GGUF変換→ベンチマーク→ABテスト実行

ワークフロー:
1. HFモデル完了検知
2. GGUF変換 (F16, Q8_0, Q4_K_M)
3. Ollamaインポート
4. 業界標準ベンチマーク実行
5. Model A vs AEGIS V2.0 ABテスト
6. 結果分析とレポート生成
7. 完了通知

使用方法:
python scripts/post_training_workflow.py --watch
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'post_training_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PostTrainingWorkflow:
    """HFモデル完成後の自動ワークフロー管理"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.output_base = Path("D:/webdataset")
        self.checkpoints_dir = self.output_base / "checkpoints" / "training"
        self.gguf_dir = self.output_base / "gguf_models"
        self.benchmark_dir = self.output_base / "benchmark_results"

        # ワークフロー設定
        self.watch_interval = 300  # 5分間隔でチェック
        self.max_retries = 3

    def check_training_completion(self) -> Optional[Path]:
        """トレーニング完了をチェックして完了したモデルディレクトリを返す"""
        logger.info("Checking for completed training sessions...")

        # トレーニングセッションを検索
        session_dirs = list(self.checkpoints_dir.glob("so8t_*"))
        if not session_dirs:
            logger.info("No training sessions found")
            return None

        # 最新のセッションからチェック
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for session_dir in session_dirs:
            final_model = session_dir / "final_model"
            if final_model.exists():
                logger.info(f"Found completed training: {session_dir}")
                return session_dir

        logger.info("No completed training sessions found")
        return None

    def convert_to_gguf(self, model_dir: Path) -> List[Path]:
        """HFモデルをGGUF形式に変換"""
        logger.info(f"Converting HF model to GGUF: {model_dir}")

        gguf_paths = []
        model_name = model_dir.name

        # GGUF出力ディレクトリ作成
        gguf_output_dir = self.gguf_dir / model_name
        gguf_output_dir.mkdir(parents=True, exist_ok=True)

        # 量子化タイプ
        quantizations = [
            ("f16", "f16"),
            ("q8_0", "q8_0"),
            ("q4_k_m", "q4_k_m")
        ]

        for quant_type, outfile_type in quantizations:
            try:
                logger.info(f"Converting to {quant_type}...")

                output_file = gguf_output_dir / f"{model_name}_{quant_type.upper()}.gguf"

                cmd = [
                    sys.executable,
                    "scripts/conversion/convert_hf_to_gguf.py",
                    str(model_dir),
                    "--outfile", str(output_file),
                    "--outtype", outfile_type
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

                if result.returncode == 0:
                    logger.info(f"Successfully converted to {quant_type}")
                    gguf_paths.append(output_file)
                else:
                    logger.error(f"Failed to convert to {quant_type}: {result.stderr}")

            except Exception as e:
                logger.error(f"Error converting to {quant_type}: {e}")

        return gguf_paths

    def import_to_ollama(self, gguf_paths: List[Path]) -> List[str]:
        """GGUFモデルをOllamaにインポート"""
        logger.info("Importing GGUF models to Ollama...")

        model_names = []

        for gguf_path in gguf_paths:
            try:
                model_name = gguf_path.stem  # 拡張子なしのファイル名

                logger.info(f"Importing {model_name}...")

                # Modelfile作成
                modelfile_content = f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{.System}}}}

{{{{.Prompt}}}}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
"""

                modelfile_path = gguf_path.parent / f"{model_name}.modelfile"
                with open(modelfile_path, 'w', encoding='utf-8') as f:
                    f.write(modelfile_content)

                # Ollamaインポート
                cmd = ["ollama", "create", f"{model_name}:latest", "-f", str(modelfile_path)]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info(f"Successfully imported {model_name} to Ollama")
                    model_names.append(f"{model_name}:latest")
                else:
                    logger.error(f"Failed to import {model_name}: {result.stderr}")

            except Exception as e:
                logger.error(f"Error importing {gguf_path}: {e}")

        return model_names

    def run_industry_benchmarks(self, model_names: List[str]) -> Path:
        """業界標準ベンチマーク実行"""
        logger.info("Running industry standard benchmarks...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.benchmark_dir / f"post_training_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                sys.executable,
                "scripts/evaluation/run_industry_benchmark.bat"
            ]

            # 環境変数でモデル指定
            env = os.environ.copy()
            env["BENCHMARK_MODELS"] = ",".join(model_names)

            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=self.project_root, env=env)

            if result.returncode == 0:
                logger.info("Industry benchmarks completed successfully")
                return output_dir
            else:
                logger.error(f"Benchmarks failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Error running benchmarks: {e}")
            return None

    def run_ab_test(self, model_names: List[str]) -> Path:
        """Model A vs AEGIS ABテスト実行"""
        logger.info("Running A/B test...")

        try:
            cmd = [
                sys.executable,
                "scripts/comprehensive_ab_benchmark.py"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                logger.info("A/B test completed successfully")
                # ABテスト結果ディレクトリを特定
                ab_results = list(self.benchmark_dir.glob("*ab_test*"))
                if ab_results:
                    return max(ab_results, key=lambda x: x.stat().st_mtime)
                return None
            else:
                logger.error(f"A/B test failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            return None

    def generate_final_report(self, benchmark_results: Path, ab_results: Path) -> Path:
        """最終レポート生成"""
        logger.info("Generating final comprehensive report...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_base / "final_reports" / f"complete_evaluation_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # レポート内容作成
        report_content = f"""# SO8T Complete Evaluation Report
Generated: {datetime.now().isoformat()}

## Training Completion
- Model: SO8T Borea-Phi3.5-instinct-jp Enhanced
- Status: Training Completed Successfully
- Timestamp: {datetime.now()}

## GGUF Conversion Results
- F16, Q8_0, Q4_K_M formats generated
- Ollama import completed

## Benchmark Results Location
{benchmark_results}

## A/B Test Results Location
{ab_results}

## Next Steps
1. Review benchmark results in: {benchmark_results}
2. Review A/B test results in: {ab_results}
3. Compare model performance metrics
4. Generate final model selection report
"""

        report_file = report_dir / "final_evaluation_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Final report generated: {report_file}")
        return report_file

    def play_completion_sound(self):
        """完了音声再生"""
        try:
            import subprocess
            audio_file = self.project_root / ".cursor" / "marisa_owattaze.wav"
            if audio_file.exists():
                subprocess.run([
                    "powershell",
                    "-c",
                    f"(New-Object Media.SoundPlayer '{audio_file}').PlaySync();"
                ])
                logger.info("Completion sound played")
        except Exception as e:
            logger.warning(f"Could not play completion sound: {e}")

    def run_workflow(self, model_dir: Path):
        """完全ワークフロー実行"""
        logger.info("Starting post-training workflow...")

        # 1. GGUF変換
        gguf_paths = self.convert_to_gguf(model_dir)
        if not gguf_paths:
            logger.error("GGUF conversion failed")
            return False

        # 2. Ollamaインポート
        ollama_models = self.import_to_ollama(gguf_paths)
        if not ollama_models:
            logger.error("Ollama import failed")
            return False

        # 3. 業界標準ベンチマーク
        benchmark_results = self.run_industry_benchmarks(ollama_models)
        if not benchmark_results:
            logger.warning("Industry benchmarks failed, continuing...")

        # 4. ABテスト
        ab_results = self.run_ab_test(ollama_models)
        if not ab_results:
            logger.warning("A/B test failed, continuing...")

        # 5. 最終レポート生成
        if benchmark_results and ab_results:
            self.generate_final_report(benchmark_results, ab_results)

        # 6. 完了通知
        self.play_completion_sound()

        logger.info("Post-training workflow completed successfully!")
        return True

    def watch_mode(self):
        """監視モード：トレーニング完了を待ってワークフロー実行"""
        logger.info("Starting watch mode for training completion...")

        while True:
            try:
                completed_model = self.check_training_completion()
                if completed_model:
                    logger.info(f"Training completion detected: {completed_model}")
                    success = self.run_workflow(completed_model)
                    if success:
                        logger.info("Workflow completed successfully. Exiting watch mode.")
                        break
                    else:
                        logger.error("Workflow failed. Continuing to watch...")

                time.sleep(self.watch_interval)

            except KeyboardInterrupt:
                logger.info("Watch mode interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in watch mode: {e}")
                time.sleep(self.watch_interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SO8T Post-Training Workflow")
    parser.add_argument("--watch", action="store_true",
                       help="Watch for training completion and run workflow")
    parser.add_argument("--model-dir", type=str,
                       help="Specific model directory to process")
    parser.add_argument("--run-once", action="store_true",
                       help="Run workflow once on completed models")

    args = parser.parse_args()

    workflow = PostTrainingWorkflow()

    if args.watch:
        workflow.watch_mode()
    elif args.model_dir:
        model_dir = Path(args.model_dir)
        if model_dir.exists():
            workflow.run_workflow(model_dir)
        else:
            logger.error(f"Model directory not found: {model_dir}")
    elif args.run_once:
        completed_model = workflow.check_training_completion()
        if completed_model:
            workflow.run_workflow(completed_model)
        else:
            logger.info("No completed training found")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
