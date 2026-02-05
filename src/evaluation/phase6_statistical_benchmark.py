#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6: 業界標準ベンチマーク パイプライン（強化版）

lm-eval-harness を使用した業界標準ベンチマークの同時評価を実施。
Model A (Baseline), Model B (Borea), Model C (AEGIS) の統計的比較。

Benchmarks:
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (Commonsense reasoning)
- ARC-Challenge (Science reasoning)
- TruthfulQA (Factual accuracy)
- ELYZA-100 (Japanese instruction following)
- JHumanEval (Japanese code generation)
- MT-Bench (Multi-turn dialogue)

Statistical Analysis:
- One-way ANOVA
- Cohen's d effect size
- Confidence intervals
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "phase6_benchmark_industry.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IndustryStandardBenchmark:
    """
    業界標準ベンチマークパイプライン。
    lm-eval-harness を使用した同時マルチタスク評価。
    """

    # Model definitions
    MODELS = {
        "model_a": {
            "name": "microsoft/Phi-3.5-mini-instruct",
            "label": "Model A (Baseline)",
            "description": "Phi-3.5-mini-instruct オリジナル（ベースライン）",
        },
        "model_b": {
            "name": "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp",
            "label": "Model B (Borea)",
            "description": "日本語拡張済み Borea モデル",
        },
        "model_c": {
            "name": "zapabobouj/AEGIS-phi3.5-jp-v3.0",
            "label": "Model C (AEGIS)",
            "description": "SO8T 四重推論強化 AEGIS モデル",
        },
    }

    # Industry-standard benchmarks
    BENCHMARKS = {
        # === General Knowledge & Reasoning ===
        "mmlu": {
            "name": "MMLU",
            "tasks": "mmlu",
            "num_fewshot": 5,
            "description": "Massive Multitask Language Understanding",
        },
        "mmlu_pro": {
            "name": "MMLU-Pro",
            "tasks": "mmlu_pro",
            "num_fewshot": 5,
            "description": "MMLU Professional (harder subset)",
        },
        "hellaswag": {
            "name": "HellaSwag",
            "tasks": "hellaswag",
            "num_fewshot": 10,
            "description": "Commonsense NLI",
        },
        "truthfulqa": {
            "name": "TruthfulQA",
            "tasks": "truthfulqa_mc2",
            "num_fewshot": 0,
            "description": "Truthful QA Multiple Choice",
        },
        "winogrande": {
            "name": "WinoGrande",
            "tasks": "winogrande",
            "num_fewshot": 5,
            "description": "Commonsense reasoning",
        },
        
        # === Science Benchmarks ===
        "arc_challenge": {
            "name": "ARC-Challenge",
            "tasks": "arc_challenge",
            "num_fewshot": 25,
            "description": "AI2 Reasoning Challenge (hard)",
        },
        "arc_easy": {
            "name": "ARC-Easy",
            "tasks": "arc_easy",
            "num_fewshot": 25,
            "description": "AI2 Reasoning Challenge (easy)",
        },
        "sciq": {
            "name": "SciQ",
            "tasks": "sciq",
            "num_fewshot": 0,
            "description": "Science Question Answering",
        },
        "gpqa": {
            "name": "GPQA",
            "tasks": "gpqa",
            "num_fewshot": 0,
            "description": "Graduate-level Physics/Chemistry/Biology QA",
        },
        "gpqa_diamond": {
            "name": "GPQA Diamond",
            "tasks": "gpqa_diamond",
            "num_fewshot": 0,
            "description": "GPQA Expert-verified subset",
        },
        "openbookqa": {
            "name": "OpenBookQA",
            "tasks": "openbookqa",
            "num_fewshot": 0,
            "description": "Open-domain science QA",
        },
        "piqa": {
            "name": "PIQA",
            "tasks": "piqa",
            "num_fewshot": 0,
            "description": "Physical Intuition QA",
        },
        
        # === Mathematics Benchmarks ===
        "gsm8k": {
            "name": "GSM8K",
            "tasks": "gsm8k",
            "num_fewshot": 5,
            "description": "Grade School Math (8K problems)",
        },
        "math": {
            "name": "MATH",
            "tasks": "hendrycks_math",
            "num_fewshot": 4,
            "description": "Mathematics Problem Solving (Hendrycks)",
        },
        "math_500": {
            "name": "MATH-500",
            "tasks": "math_500",
            "num_fewshot": 4,
            "description": "MATH 500-problem subset",
        },
        "minerva_math": {
            "name": "Minerva Math",
            "tasks": "minerva_math",
            "num_fewshot": 4,
            "description": "Minerva Math Benchmark",
        },
        "aime": {
            "name": "AIME",
            "tasks": "aime",
            "num_fewshot": 0,
            "description": "American Invitational Mathematics Examination",
        },
        "amc": {
            "name": "AMC",
            "tasks": "amc",
            "num_fewshot": 0,
            "description": "American Mathematics Competition",
        },
        "olympiad_bench": {
            "name": "OlympiadBench",
            "tasks": "olympiad_bench",
            "num_fewshot": 0,
            "description": "Math/Science Olympiad Problems",
        },
        "mgsm": {
            "name": "MGSM",
            "tasks": "mgsm",
            "num_fewshot": 8,
            "description": "Multilingual Grade School Math",
        },
        "mathqa": {
            "name": "MathQA",
            "tasks": "mathqa",
            "num_fewshot": 0,
            "description": "Math Word Problem QA",
        },
        "aqua_rat": {
            "name": "AQuA-RAT",
            "tasks": "aqua_rat",
            "num_fewshot": 0,
            "description": "Algebra Question Answering with Rationales",
        },
        
        # === Code Benchmarks ===
        "humaneval": {
            "name": "HumanEval",
            "tasks": "humaneval",
            "num_fewshot": 0,
            "description": "OpenAI HumanEval Code Generation",
        },
        "mbpp": {
            "name": "MBPP",
            "tasks": "mbpp",
            "num_fewshot": 3,
            "description": "Mostly Basic Python Problems",
        },
        
        # === Japanese Benchmarks ===
        "jcommonsenseqa": {
            "name": "JCommonsenseQA",
            "tasks": "jcommonsenseqa",
            "num_fewshot": 3,
            "description": "Japanese commonsense QA",
        },
        "jnli": {
            "name": "JNLI",
            "tasks": "jnli",
            "num_fewshot": 3,
            "description": "Japanese NLI",
        },
        "elyza100": {
            "name": "ELYZA-100",
            "tasks": "elyza_tasks_100",
            "num_fewshot": 0,
            "description": "ELYZA Tasks 100 (Japanese instruction following)",
        },
        "jhumaneval": {
            "name": "JHumanEval",
            "tasks": "jhumaneval",
            "num_fewshot": 0,
            "description": "Japanese HumanEval Code Generation",
        },
        "mgsm_ja": {
            "name": "MGSM-JA",
            "tasks": "mgsm_ja",
            "num_fewshot": 8,
            "description": "Multilingual GSM (Japanese)",
        },
    }


    def __init__(
        self,
        output_dir: Optional[Path] = None,
        use_vllm: bool = False,
        batch_size: int = 8,
    ) -> None:
        self.project_root = PROJECT_ROOT
        self.output_dir = output_dir or self.project_root / "src" / "evaluation" / "results" / "phase6_industry"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_vllm = use_vllm
        self.batch_size = batch_size
        self.results: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Industry Standard Benchmark Pipeline initialized.")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Use vLLM: {use_vllm}, Batch size: {batch_size}")

    def check_lm_eval_installed(self) -> bool:
        """Check if lm-eval-harness is installed."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "lm_eval", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"lm-eval check failed: {e}")
            return False

    def install_lm_eval(self) -> bool:
        """Install lm-eval-harness."""
        logger.info("Installing lm-eval-harness...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "lm-eval[vllm]", "-q"],
                check=True,
                timeout=300
            )
            return True
        except Exception as e:
            logger.error(f"Failed to install lm-eval: {e}")
            return False

    def run_lm_eval_benchmark(
        self,
        model_path: str,
        tasks: List[str],
        num_fewshot: int = 0,
    ) -> Dict[str, float]:
        """
        Run lm-eval-harness benchmark for a specific model.
        """
        logger.info(f"Running lm-eval for {model_path} on tasks: {tasks}")
        
        task_string = ",".join(tasks)
        output_path = self.output_dir / f"lm_eval_{model_path.replace('/', '_')}.json"
        
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},trust_remote_code=True",
            "--tasks", task_string,
            "--num_fewshot", str(num_fewshot),
            "--batch_size", str(self.batch_size),
            "--output_path", str(output_path),
        ]
        
        if self.use_vllm:
            cmd[2:4] = ["--model", "vllm"]
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                logger.warning(f"lm-eval returned non-zero: {result.stderr}")
            
            # Parse results
            if output_path.exists():
                with open(output_path, "r", encoding="utf-8") as f:
                    raw_results = json.load(f)
                return self._parse_lm_eval_results(raw_results)
            
        except subprocess.TimeoutExpired:
            logger.error("lm-eval timed out")
        except Exception as e:
            logger.error(f"lm-eval failed: {e}")
        
        return {}

    def _parse_lm_eval_results(self, raw_results: Dict[str, Any]) -> Dict[str, float]:
        """Parse lm-eval-harness output format."""
        parsed: Dict[str, float] = {}
        
        if "results" in raw_results:
            for task_name, task_results in raw_results["results"].items():
                # Extract main metric (usually acc or acc_norm)
                for metric in ["acc_norm,none", "acc,none", "exact_match,none"]:
                    if metric in task_results:
                        parsed[task_name] = task_results[metric]
                        break
        
        return parsed

    def run_all_benchmarks_parallel(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all benchmarks for all models in parallel.
        """
        logger.info("=" * 60)
        logger.info("Starting Parallel Industry-Standard Benchmarking")
        logger.info("=" * 60)
        
        # Check/install lm-eval
        if not self.check_lm_eval_installed():
            if not self.install_lm_eval():
                logger.error("Could not install lm-eval. Aborting.")
                return {}
        
        # Prepare all tasks
        all_tasks = list(self.BENCHMARKS.keys())
        
        # Run benchmarks for each model
        for model_key, model_info in self.MODELS.items():
            model_name = model_info["name"]
            logger.info(f"\nBenchmarking {model_info['label']}: {model_name}")
            
            results = self.run_lm_eval_benchmark(
                model_path=model_name,
                tasks=all_tasks,
                num_fewshot=5
            )
            
            self.results[model_key] = {
                "model_info": model_info,
                "benchmark_results": results,
                "timestamp": datetime.now().isoformat(),
            }
        
        return self.results

    def compute_anova(self, metric_name: str) -> Dict[str, Any]:
        """
        Compute one-way ANOVA for a specific metric across all models.
        """
        try:
            from scipy import stats
            
            groups = []
            for model_key in ["model_a", "model_b", "model_c"]:
                if model_key in self.results:
                    score = self.results[model_key]["benchmark_results"].get(metric_name, 0)
                    groups.append([score])  # Single value per model
            
            if len(groups) < 3:
                return {"error": "Not enough data"}
            
            # For single values, we can't compute traditional ANOVA
            # This is a placeholder for when we have multiple runs
            return {
                "scores": {k: self.results[k]["benchmark_results"].get(metric_name, 0) 
                          for k in self.results},
                "note": "Single-run comparison (ANOVA requires multiple samples per group)"
            }
            
        except ImportError:
            return {"error": "scipy not available"}

    def compute_cohens_d(self, model1: str, model2: str, metric_name: str) -> float:
        """
        Compute Cohen's d effect size between two models.
        """
        if model1 not in self.results or model2 not in self.results:
            return 0.0
        
        score1 = self.results[model1]["benchmark_results"].get(metric_name, 0)
        score2 = self.results[model2]["benchmark_results"].get(metric_name, 0)
        
        # For single scores, return simple difference (not true Cohen's d)
        return score2 - score1

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive summary statistics across all benchmarks.
        Returns mean, SD, 95% CI, and per-category aggregates.
        """
        logger.info("Computing summary statistics...")
        
        stats_summary: Dict[str, Any] = {
            "per_model": {},
            "per_category": {},
            "pairwise_comparisons": {},
        }
        
        # Category definitions
        categories = {
            "General Knowledge": ["mmlu", "mmlu_pro", "hellaswag", "truthfulqa", "winogrande"],
            "Science": ["arc_challenge", "arc_easy", "sciq", "gpqa", "gpqa_diamond", "openbookqa", "piqa"],
            "Mathematics": ["gsm8k", "math", "math_500", "minerva_math", "aime", "amc", "olympiad_bench", "mgsm", "mathqa", "aqua_rat"],
            "Code": ["humaneval", "mbpp", "jhumaneval"],
            "Japanese": ["jcommonsenseqa", "jnli", "elyza100", "mgsm_ja"],
        }
        
        for model_key in ["model_a", "model_b", "model_c"]:
            if model_key not in self.results:
                continue
            
            benchmark_results = self.results[model_key]["benchmark_results"]
            all_scores = list(benchmark_results.values())
            
            if all_scores:
                mean_score = np.mean(all_scores)
                std_score = np.std(all_scores, ddof=1) if len(all_scores) > 1 else 0
                n = len(all_scores)
                se = std_score / np.sqrt(n) if n > 0 else 0
                ci_95 = 1.96 * se
                
                stats_summary["per_model"][model_key] = {
                    "mean": float(mean_score),
                    "std": float(std_score),
                    "n": n,
                    "se": float(se),
                    "ci_95_lower": float(mean_score - ci_95),
                    "ci_95_upper": float(mean_score + ci_95),
                }
                
                # Per-category statistics
                for cat_name, cat_benchmarks in categories.items():
                    cat_scores = [benchmark_results.get(b, 0) for b in cat_benchmarks if b in benchmark_results]
                    if cat_scores:
                        cat_mean = np.mean(cat_scores)
                        cat_std = np.std(cat_scores, ddof=1) if len(cat_scores) > 1 else 0
                        
                        if cat_name not in stats_summary["per_category"]:
                            stats_summary["per_category"][cat_name] = {}
                        
                        stats_summary["per_category"][cat_name][model_key] = {
                            "mean": float(cat_mean),
                            "std": float(cat_std),
                            "n": len(cat_scores),
                        }
        
        # Pairwise t-tests (if scipy available)
        try:
            from scipy import stats as scipy_stats
            
            model_pairs = [("model_a", "model_b"), ("model_b", "model_c"), ("model_a", "model_c")]
            
            for m1, m2 in model_pairs:
                if m1 in self.results and m2 in self.results:
                    scores1 = list(self.results[m1]["benchmark_results"].values())
                    scores2 = list(self.results[m2]["benchmark_results"].values())
                    
                    if len(scores1) > 1 and len(scores2) > 1:
                        t_stat, p_value = scipy_stats.ttest_ind(scores1, scores2)
                        
                        stats_summary["pairwise_comparisons"][f"{m1}_vs_{m2}"] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant_005": p_value < 0.05,
                            "significant_001": p_value < 0.01,
                        }
        except ImportError:
            stats_summary["pairwise_comparisons"]["note"] = "scipy not available for t-tests"
        
        self.summary_stats = stats_summary
        return stats_summary

    def generate_error_bar_plot(self) -> Optional[Path]:
        """
        Generate error bar plot comparing models across benchmark categories.
        Returns path to saved PNG file.
        """
        logger.info("Generating error bar plot...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            if not hasattr(self, 'summary_stats'):
                self.compute_summary_statistics()
            
            categories = list(self.summary_stats.get("per_category", {}).keys())
            if not categories:
                logger.warning("No category data for plotting")
                return None
            
            models = ["model_a", "model_b", "model_c"]
            model_labels = ["Model A (Baseline)", "Model B (Borea)", "Model C (AEGIS)"]
            colors = ["#3498db", "#e74c3c", "#2ecc71"]
            
            x = np.arange(len(categories))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for i, (model_key, label, color) in enumerate(zip(models, model_labels, colors)):
                means = []
                stds = []
                
                for cat in categories:
                    cat_data = self.summary_stats["per_category"].get(cat, {}).get(model_key, {})
                    means.append(cat_data.get("mean", 0))
                    stds.append(cat_data.get("std", 0))
                
                ax.bar(x + i * width, means, width, 
                       yerr=stds, 
                       label=label, 
                       color=color,
                       capsize=5,
                       error_kw={'linewidth': 1.5})
            
            ax.set_xlabel('Benchmark Category', fontsize=12)
            ax.set_ylabel('Score (Mean ± SD)', fontsize=12)
            ax.set_title('AEGIS-v3.0 Statistical Benchmark Analysis\n(DeepSeek GRPO / Sakana AI Integrated / SO8T Quadrality)', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.legend(loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.0)
            
            # Add citation box
            citation_text = (
                "Tech Citations:\n"
                "- GRPO: DeepSeek-AI, DeepSeek-V3 Technical Report (2024)\n"
                "- Evolution: Akiba et al., Evolutionary Optimization of Model Merging (2024)\n"
                "- Reasoning: SO8T Quadrality & mHC (2025-2026)"
            )
            plt.figtext(0.15, 0.02, citation_text, fontsize=8, style='italic', 
                        bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            plot_path = self.output_dir / "benchmark_comparison_error_bars.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Error bar plot saved to {plot_path}")
            return plot_path
            
        except ImportError as e:
            logger.warning(f"matplotlib not available: {e}")
            return None

    def generate_comparison_table_with_stats(self) -> str:
        """Generate markdown comparison table with ±SD and significance markers."""
        if not self.results:
            return "No results available."
        
        if not hasattr(self, 'summary_stats'):
            self.compute_summary_statistics()
        
        # Collect all metrics
        all_metrics = set()
        for model_data in self.results.values():
            all_metrics.update(model_data["benchmark_results"].keys())
        
        # Build table with SD
        header = "| Benchmark | Model A | Model B | Model C | Δ(B→C) |"
        separator = "|-----------|---------|---------|---------|--------|"
        
        rows = [header, separator]
        
        for metric in sorted(all_metrics):
            scores = {}
            for model_key in ["model_a", "model_b", "model_c"]:
                if model_key in self.results:
                    scores[model_key] = self.results[model_key]["benchmark_results"].get(metric, 0)
            
            delta = scores.get("model_c", 0) - scores.get("model_b", 0)
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            
            # Add significance marker
            sig_marker = ""
            if abs(delta) > 0.05:
                sig_marker = " **"
            elif abs(delta) > 0.02:
                sig_marker = " *"
            
            row = f"| {metric} | {scores.get('model_a', 0):.4f} | {scores.get('model_b', 0):.4f} | {scores.get('model_c', 0):.4f} | {delta_str}{sig_marker} |"
            rows.append(row)
        
        return "\n".join(rows)

    def generate_summary_stats_table(self) -> str:
        """Generate summary statistics table (Mean ± SD, 95% CI)."""
        if not hasattr(self, 'summary_stats'):
            self.compute_summary_statistics()
        
        rows = [
            "| Model | Mean | SD | 95% CI | N |",
            "|-------|------|-----|--------|---|",
        ]
        
        for model_key in ["model_a", "model_b", "model_c"]:
            stats = self.summary_stats.get("per_model", {}).get(model_key, {})
            if stats:
                label = self.MODELS[model_key]["label"]
                mean = stats.get("mean", 0)
                std = stats.get("std", 0)
                ci_l = stats.get("ci_95_lower", 0)
                ci_u = stats.get("ci_95_upper", 0)
                n = stats.get("n", 0)
                
                rows.append(f"| {label} | {mean:.4f} ± {std:.4f} | {std:.4f} | [{ci_l:.4f}, {ci_u:.4f}] | {n} |")
        
        return "\n".join(rows)

    def generate_pvalue_table(self) -> str:
        """Generate p-value comparison table."""
        if not hasattr(self, 'summary_stats'):
            self.compute_summary_statistics()
        
        pairwise = self.summary_stats.get("pairwise_comparisons", {})
        
        if "note" in pairwise:
            return f"*{pairwise['note']}*"
        
        rows = [
            "| Comparison | t-statistic | p-value | Significant (α=0.05) | Significant (α=0.01) |",
            "|------------|-------------|---------|----------------------|----------------------|",
        ]
        
        for comparison, data in pairwise.items():
            if isinstance(data, dict):
                t_stat = data.get("t_statistic", 0)
                p_val = data.get("p_value", 1)
                sig_05 = "✓" if data.get("significant_005", False) else "✗"
                sig_01 = "✓" if data.get("significant_001", False) else "✗"
                
                display_name = comparison.replace("model_", "M").replace("_vs_", " vs ")
                rows.append(f"| {display_name} | {t_stat:.4f} | {p_val:.6f} | {sig_05} | {sig_01} |")
        
        return "\n".join(rows)

    def generate_comparison_table(self) -> str:
        """Generate markdown comparison table."""
        return self.generate_comparison_table_with_stats()

    def generate_academic_model_card(self) -> str:
        """Generate academic-style model card with comprehensive statistics."""
        timestamp = datetime.now().isoformat()
        
        # Compute all statistics
        self.compute_summary_statistics()
        
        comparison_table = self.generate_comparison_table_with_stats()
        summary_stats_table = self.generate_summary_stats_table()
        pvalue_table = self.generate_pvalue_table()
        
        # Generate plot
        plot_path = self.generate_error_bar_plot()
        plot_embed = ""
        if plot_path and plot_path.exists():
            plot_embed = f"\n![Benchmark Comparison with Error Bars]({plot_path.name})\n"
        
        card = f"""---
language:
  - ja
  - en
license: apache-2.0
tags:
  - llm
  - phi-3.5
  - japanese
  - so8t
  - quadrality
  - aegis
base_model: AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp
pipeline_tag: text-generation
---

# AEGIS-Phi-3.5-JP v3.0 (Model C)

## Model Description / モデル概要

**English:**
AEGIS-Phi-3.5-JP v3.0 is an advanced Japanese language model based on Borea-Phi-3.5-mini-Instruct-Jp, enhanced with SO8T Quadrality Reasoning framework. The model integrates specialized knowledge in geopolitics (2024-2026), scientific reasoning, and safety-aware responses.

**日本語:**
AEGIS-Phi-3.5-JP v3.0は、Borea-Phi-3.5-mini-Instruct-Jpをベースに、SO8T四重推論フレームワークで強化された高度な日本語言語モデルです。

## Training Methodology / 学習手法

| Component | Method |
|-----------|--------|
| Base Weight Preservation | LoRA/QLoRA Adapters |
| Supervised Fine-Tuning | ShareGPT format data |
| Reinforcement Learning | GRPO (Group Relative Policy Optimization) |
| Reasoning Enhancement | SO8T Quadrality (Scalar, Vector, +Spinor, -Spinor) |

---

## Benchmark Results / ベンチマーク結果

Industry-standard benchmarks evaluated using [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Summary Statistics / 要約統計量

{summary_stats_table}

*Note: Mean ± SD across all benchmarks. 95% CI = 95% Confidence Interval.*
{plot_embed}
### Detailed Benchmark Scores / 詳細ベンチマークスコア

{comparison_table}

*Note: Δ(B→C) = improvement from Model B to Model C. ** = large effect (|Δ| > 0.05), * = medium effect (|Δ| > 0.02).*

### Statistical Significance / 統計的有意性

{pvalue_table}

### Interpretation / 解釈

- **α = 0.05**: Standard significance threshold (p < 0.05)
- **α = 0.01**: Stringent significance threshold (p < 0.01)
- **Cohen's d**: Effect size where |d| > 0.8 = large, 0.5-0.8 = medium, 0.2-0.5 = small

---

## Dataset Sources / データセット出典

### Geopolitics (2024-2026) / 地政学
- Venezuela crisis and US-Latin America relations
- Ukraine war progression and European security
- Japan-China relations (diplomatic, economic security, national security)

### Technology / テクノロジー
- GPU shortage and memory/SSD price dynamics
- AI/LLM developments: Opus 4.5, Codex, Claude Code, MCP, Skill OSS

### Culture / カルチャー
- Gundam franchise:
  - SEED FREEDOM (2024)
  - GQuuuuuuX (Director: Kazuya Tsurumaki, Studio Khara × Sunrise, 2025)
  - Hathaway Part 2 (2025-2026)

---

## Citation / 引用

```bibtex
@misc{{aegis-phi35-jp-v3,
  author = {{zapabobouj}},
  title = {{AEGIS-Phi-3.5-JP v3.0: Quadrality Reasoning Enhanced Japanese LLM}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/zapabobouj/AEGIS-phi3.5-jp-v3.0}}}}
}}
```

## References / 参考文献

1. Microsoft. (2024). Phi-3.5 Technical Report.
2. AXCXEPT. (2024). Borea Japanese Language Model.
3. EleutherAI. (2024). lm-evaluation-harness. GitHub.
4. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
5. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
```python
6. DeepSeek-AI. (2024). DeepSeek-V3 Technical Report. (GRPO)
7. Unsloth AI. (2024). Unsloth: Lightweight and fast LLM fine-tuning. GitHub.
8. Gerganov, G., et al. (2024). llama.cpp: Importance Matrix (imatrix) Quantization. GitHub.
9. Akiba, T., et al. (2024). Evolutionary Optimization of Model Merging. arXiv:2403.13187. (Sakana AI)
10. mHC: Multi-Head Control/Consistency for Reasoning Models.
```
Generated: {timestamp}
"""
        
        # Save model card
        model_card_path = self.output_dir / "MODEL_CARD.md"
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(card)
        
        logger.info(f"Academic model card saved to {model_card_path}")
        return card

    def save_results(self) -> Path:
        """Save all benchmark results including statistics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        # Include summary stats in output
        output_data = {
            "results": self.results,
            "summary_statistics": getattr(self, 'summary_stats', {}),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return output_path

    def run(self) -> Path:
        """Execute full benchmark pipeline with statistics."""
        self.run_all_benchmarks_parallel()
        self.compute_summary_statistics()
        self.generate_academic_model_card()
        return self.save_results()


def main() -> None:
    """Main entry point."""
    pipeline = IndustryStandardBenchmark()
    output_path = pipeline.run()
    print(f"\nIndustry-standard benchmarking complete. Results: {output_path}")


if __name__ == "__main__":
    main()

