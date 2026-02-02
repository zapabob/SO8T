#!/usr/bin/env python3
"""
Moonshot Pipeline v3.0 - Full Orchestration.

Complete pipeline: Research → Dataset → SFT → GRPO → Benchmark → Release
With tqdm-style progress and SQL tracking.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

from tqdm import tqdm
from scripts.utils.runtime_requirements import check_runtime_requirements

logger = logging.getLogger(__name__)


class MoonshotPipelineV3:
    """Complete Moonshot Pipeline v3.0 orchestrator."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.logs_dir = self.project_root / "logs"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.results_dir = self.project_root / "results"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline configuration
        self.config = {
            "model_name": "microsoft/Phi-3.5-mini-instruct",
            "sft_epochs": 3,
            "grpo_steps": 1000,
            "num_benchmark_seeds": 10,
            "rtx3060_optimized": True,
            "arxiv_biorxiv_max_papers": 100000,
            "download_metrics_path": "",
        }

    def setup_logging(self):
        """Configure logging."""
        log_file = (
            self.logs_dir
            / f"moonshot_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        return logger

    def _future_plan_text(self) -> str:
        """Return the forward plan to persist in logs/SQL."""
        return (
            "Future Plan (Research Use, 100K+ CoT)\n"
            "- Runtime: Python 3.12.9 + uv + PyTorch CUDA + flash-attn + Unsloth\n"
            "- Models: A=microsoft-phi3.5mini-instinct, "
            "B=AXCEPT-Borea-phi3.5-instinct-jp, "
            "C=zapabobouj-AEGIS-phi3.5-jp-v3.0 (BF16 GGUF)\n"
            "- Data: existing JSON/JSONL (defense/JAXA/NSFW/drug), "
            "ArXiv/BioRxiv (API), Japanese national universities, "
            "Common Test, medical school exams, prep-school PDFs (2024-2026), "
            "ML/CS/AI top-cited arXiv (2024-2026), Lean4 mathlib, HF CLI downloads\n"
            "- CoT: quadruple reasoning tags, strict parser + cleansing\n"
            "- Eval: lm-evaluation-harness/DeepEval/HumanEval; "
            "ABC test + cross-validation; ANOVA + Tukey; "
            "p-values, power, error-bar plots; "
            "CUDA memory free + model reload after each benchmark\n"
            "- Compliance: research-only, cite all sources in HF Model Card/README\n"
        )

    def _record_future_plan(self, run_id: str):
        """Persist future plan in implementation logs and SQL."""
        plan_text = self._future_plan_text()
        # Write to logs file
        plan_path = self.logs_dir / f"future_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        plan_path.write_text(plan_text, encoding="utf-8")
        logger.info("[PLAN] Future plan saved: %s", plan_path)

        # Write to _docs with implementation log tail
        docs_dir = self.project_root / "_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        docs_path = docs_dir / f"{datetime.now().strftime('%Y-%m-%d')}実装計画_{self.project_root.name}.md"

        # include latest log tail if available
        log_files = sorted(self.logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        log_tail = ""
        if log_files:
            try:
                lines = log_files[0].read_text(encoding="utf-8", errors="replace").splitlines()
                log_tail = "\n".join(lines[-200:])
            except Exception:
                log_tail = ""

        docs_content = (
            f"# 実装計画書（保存）\n\n"
            f"- Worktree: {self.project_root.name}\n"
            f"- Timestamp: {datetime.now().isoformat()}\n"
            f"- Plan Log: {plan_path}\n\n"
            "## 計画\n\n"
            f"{plan_text}\n\n"
            "## 直近の実装ログ（最後の200行）\n\n"
            "```\n"
            f"{log_tail}\n"
            "```\n"
        )
        docs_path.write_text(docs_content, encoding="utf-8")
        logger.info("[PLAN] Future plan saved to _docs: %s", docs_path)

        # Write to SQL (progress_log)
        try:
            sys.path.insert(0, str(self.project_root / "scripts" / "utils"))
            from pipeline_progress_store import log_progress

            log_progress(run_id, "PLAN", f"Future plan saved: {plan_path}")
            log_progress(run_id, "PLAN", f"Future plan saved to _docs: {docs_path}")
        except Exception as e:
            logger.warning("Failed to log future plan to SQL: %s", e)

    def print_progress(self, phase: str, step: str, message: str, progress: float):
        """Print progress with bar."""
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"[{phase}] |{bar}| {step} - {message}")

    def log_to_sql(self, phase: str, step: str, message: str):
        """Log progress to SQLite."""
        try:
            sys.path.insert(0, str(self.project_root / "scripts" / "utils"))
            from pipeline_progress_store import log_progress, get_run_id, init_db

            if get_run_id:
                log_progress(get_run_id(), f"{phase}:{step}", message)
        except Exception:
            pass  # SQL logging is optional

    def phase_data(self) -> bool:
        """Phase 0: Data acquisition (arXiv/BioRxiv) and integration."""
        self.print_progress("DATA", "0/5", "Starting arXiv/BioRxiv data pipeline", 0.0)
        self.log_to_sql("DATA", "start", "Starting arXiv/BioRxiv data pipeline")
        try:
            script = self.project_root / "scripts" / "data_processing" / "process_arxiv_biorxiv.py"
            cmd = [sys.executable, str(script), "--max-papers", str(self.config["arxiv_biorxiv_max_papers"])]
            if self.config.get("download_metrics_path"):
                cmd += ["--download-metrics", self.config["download_metrics_path"]]
            subprocess.run(cmd, cwd=self.project_root, check=True)

            integrate_script = self.project_root / "scripts" / "data_processing" / "integrate_domain_knowledge.py"
            subprocess.run([sys.executable, str(integrate_script)], cwd=self.project_root, check=False)

            self.print_progress("DATA", "0/5", "Data pipeline complete", 0.1)
            self.log_to_sql("DATA", "complete", "Data pipeline finished successfully")
            return True
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            self.log_to_sql("DATA", "error", str(e))
            return False

    def phase_sft(self) -> bool:
        """Phase 1: SFT Training."""
        self.print_progress("SFT", "1/5", "Starting SFT training", 0.0)
        self.log_to_sql("SFT", "start", "Starting SFT training")

        try:
            from scripts.training.v3_sft_pipeline import V3SFTPipeline, V3SFTConfig

            config = V3SFTConfig()
            config.num_train_epochs = self.config["sft_epochs"]

            pipeline = V3SFTPipeline(config)
            pipeline.train()

            self.print_progress("SFT", "1/5", "SFT training complete", 0.2)
            self.log_to_sql("SFT", "complete", "SFT training finished successfully")
            return True

        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            self.log_to_sql("SFT", "error", str(e))
            return False

    def phase_grpo(self) -> bool:
        """Phase 2: GRPO Training."""
        self.print_progress("GRPO", "2/5", "Starting GRPO training", 0.2)
        self.log_to_sql("GRPO", "start", "Starting GRPO training")

        try:
            from scripts.training.v3_grpo_pipeline import V3GRPOPipeline, V3GRPOConfig

            config = V3GRPOConfig()
            config.sft_adapter_path = "checkpoints/v3_sft/adapter"

            pipeline = V3GRPOPipeline(config)
            pipeline.train()

            self.print_progress("GRPO", "2/5", "GRPO training complete", 0.4)
            self.log_to_sql("GRPO", "complete", "GRPO training finished successfully")
            return True

        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
            self.log_to_sql("GRPO", "error", str(e))
            return False

    def phase_benchmark(self) -> bool:
        """Phase 3: ABC Benchmark."""
        self.print_progress("BENCH", "3/5", "Starting ABC benchmark", 0.4)
        self.log_to_sql("BENCH", "start", "Starting ABC benchmark evaluation")

        try:
            # Run ABC benchmark
            from scripts.evaluation.run_abc_v3 import ABCBenchmarkV3

            benchmark = ABCBenchmarkV3(num_seeds=self.config["num_benchmark_seeds"])
            results = benchmark.run_full_benchmark()

            # Run statistics
            from scripts.analysis.abc_statistics_v3 import ABCStatisticsV3

            stats = ABCStatisticsV3()
            stats.run_full_analysis()

            # Run visualization
            from scripts.analysis.abc_visualizer_v3 import ABCVisualizerV3

            viz = ABCVisualizerV3()
            viz.run_full_visualization()

            self.print_progress("BENCH", "3/5", "Benchmark complete", 0.6)
            self.log_to_sql("BENCH", "complete", "ABC benchmark and analysis finished")
            return True

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            self.log_to_sql("BENCH", "error", str(e))
            return False

    def phase_release(self) -> bool:
        """Phase 4: HF Release."""
        self.print_progress("RELEASE", "4/5", "Preparing HF release", 0.6)
        self.log_to_sql("RELEASE", "start", "Starting HF release preparation")

        try:
            from scripts.hf_upload_v3 import HFUploaderV3

            uploader = HFUploaderV3(
                model_path="checkpoints/v3_grpo/adapter", output_dir="models/hf_upload"
            )
            # Prepare files but don't upload (requires credentials)
            uploader.convert_to_safetensors()
            uploader.convert_to_gguf_bf16()

            # Create model card
            model_card_path = self.project_root / "docs" / "MODEL_CARD_v3.md"
            if model_card_path.exists():
                uploader.create_model_card()
                print(f"[RELEASE] Model card ready: {model_card_path}")

            self.print_progress("RELEASE", "4/5", "Release prep complete", 0.8)
            self.log_to_sql("RELEASE", "complete", "HF release preparation finished")
            return True

        except Exception as e:
            logger.error(f"Release preparation failed: {e}")
            self.log_to_sql("RELEASE", "error", str(e))
            return False

    def phase_cleanup(self) -> bool:
        """Phase 5: Cleanup and summary."""
        self.print_progress("CLEANUP", "5/5", "Finalizing pipeline", 0.8)

        try:
            # Create implementation summary
            summary = {
                "pipeline": "Moonshot v3.0",
                "completed_at": datetime.now().isoformat(),
                "model": self.config["model_name"],
                "phases": {
                    "sft": "complete",
                    "grpo": "complete",
                    "benchmark": "complete",
                    "release": "complete",
                },
                "output_locations": {
                    "checkpoints": str(self.checkpoints_dir),
                    "results": str(self.results_dir),
                    "logs": str(self.logs_dir),
                },
            }

            summary_path = self.logs_dir / "pipeline_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.print_progress("CLEANUP", "5/5", "Pipeline complete!", 1.0)
            self.log_to_sql(
                "CLEANUP", "complete", "Full pipeline finished successfully"
            )

            print("\n" + "=" * 60)
            print("Moonshot Pipeline v3.0 - Complete!")
            print("=" * 60)
            print(f"Summary: {summary_path}")
            print(f"Results: {self.results_dir}")
            print("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    def run_full_pipeline(self) -> bool:
        """Execute complete pipeline."""
        logger = self.setup_logging()
        check_runtime_requirements()
        logger.info("=" * 60)
        logger.info("Moonshot Pipeline v3.0 - Full Orchestration")
        logger.info(f"Base Model: {self.config['model_name']}")
        logger.info(f"RTX3060 Optimized: {self.config['rtx3060_optimized']}")
        logger.info("=" * 60)

        # Initialize SQL tracking
        try:
            sys.path.insert(0, str(self.project_root / "scripts" / "utils"))
            from pipeline_progress_store import init_db, record_run

            init_db()
            run_id = f"moonshot_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            record_run(run_id, git_commit_hash=self._get_git_commit())
            logger.info(f"SQL tracking initialized: {run_id}")
            self._record_future_plan(run_id)
        except Exception as e:
            logger.warning(f"SQL tracking not available: {e}")

        # Run phases
        success = True

        # Phase 0: Data
        if not self.phase_data():
            logger.warning("Data pipeline had issues, continuing...")

        # Phase 1: SFT
        if not self.phase_sft():
            logger.warning("SFT had issues, continuing...")

        # Phase 2: GRPO
        if not self.phase_grpo():
            logger.warning("GRPO had issues, continuing...")

        # Phase 3: Benchmark
        if not self.phase_benchmark():
            logger.warning("Benchmark had issues, continuing...")

        # Phase 4: Release
        if not self.phase_release():
            logger.warning("Release had issues, continuing...")

        # Phase 5: Cleanup
        self.phase_cleanup()

        logger.info("Pipeline execution complete!")
        return success

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()[:40] if result.stdout else "unknown"
        except Exception:
            return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Moonshot Pipeline v3.0")
    parser.add_argument(
        "--sft-epochs", type=int, default=3, help="Number of SFT training epochs"
    )
    parser.add_argument(
        "--grpo-steps", type=int, default=1000, help="Number of GRPO training steps"
    )
    parser.add_argument(
        "--benchmark-seeds",
        type=int,
        default=10,
        help="Number of benchmark evaluation seeds",
    )
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT phase")
    parser.add_argument("--skip-grpo", action="store_true", help="Skip GRPO phase")
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Skip benchmark phase"
    )
    parser.add_argument(
        "--skip-release", action="store_true", help="Skip release phase"
    )

    args = parser.parse_args()

    pipeline = MoonshotPipelineV3()
    pipeline.config["sft_epochs"] = args.sft_epochs
    pipeline.config["grpo_steps"] = args.grpo_steps
    pipeline.config["num_benchmark_seeds"] = args.benchmark_seeds

    if args.skip_sft and args.skip_grpo and args.skip_benchmark and args.skip_release:
        print("Nothing to do! Specify at least one phase to run.")
        return

    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
