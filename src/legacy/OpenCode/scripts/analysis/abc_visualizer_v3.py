#!/usr/bin/env python3
"""
ABC Benchmark Visualization v3.0.

Generates error bar plots and summary visualizations for ABC benchmark results.
Output: PNG plots + Markdown summary tables.
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ABCVisualizerV3:
    """Visualization for ABC benchmark results."""

    def __init__(
        self,
        results_path: str = "results/abc_testing/abc_results_v3.json",
        stats_path: str = "results/abc_testing/statistical_analysis_v3.json",
    ):
        self.project_root = Path(__file__).parent.parent.parent
        self.results_path = self.project_root / results_path
        self.stats_path = self.project_root / stats_path
        self.output_dir = self.project_root / "results/abc_testing/figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.load_data()

    def load_data(self):
        """Load results and statistics."""
        if self.results_path.exists():
            with open(self.results_path, "r", encoding="utf-8") as f:
                self.results = json.load(f)

        if self.stats_path.exists():
            with open(self.stats_path, "r", encoding="utf-8") as f:
                self.stats = json.load(f)

    def plot_error_bars(self, save: bool = True) -> plt.Figure:
        """Generate error bar plot with 95% CI."""
        fig, ax = plt.subplots(figsize=(12, 6))

        summary = self.stats.get("summary_statistics", {})
        benchmarks = self.results.get("metadata", {}).get("benchmarks", {})
        models = ["A", "B", "C"]
        model_names = {"A": "Phi-3.5-instinct", "B": "Borea-phi3.5", "C": "AEGIS-v3.0"}
        colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
        markers = {"A": "o", "B": "s", "C": "^"}

        x_positions = np.arange(len(benchmarks))
        width = 0.25

        for i, model in enumerate(models):
            means = []
            errors_lower = []
            errors_upper = []

            for bench_key in benchmarks.keys():
                stats_data = summary.get(model, {}).get(bench_key, {})
                mean = stats_data.get("mean", 0)
                ci = stats_data.get("ci_95", [mean, mean])

                means.append(mean)
                errors_lower.append(mean - ci[0])
                errors_upper.append(ci[1] - mean)

            x = x_positions + i * width
            ax.errorbar(
                x,
                means,
                yerr=[errors_lower, errors_upper],
                label=model_names[model],
                color=colors[model],
                marker=markers[model],
                markersize=8,
                capsize=5,
                linewidth=2,
                linestyle="-",
            )

        ax.set_xticks(x_positions + width)
        ax.set_xticklabels(list(benchmarks.values()), fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlabel("Benchmark", fontsize=12)
        ax.set_title("ABC Benchmark Results (95% CI Error Bars)", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "abc_error_bars.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved error bar plot to: {output_path}")

        return fig

    def plot_comparison_heatmap(self, save: bool = True) -> plt.Figure:
        """Generate model comparison heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))

        summary = self.stats.get("summary_statistics", {})
        benchmarks = self.results.get("metadata", {}).get("benchmarks", {})
        models = ["A", "B", "C"]

        # Create comparison matrix (Model C vs others)
        matrix = []
        bench_labels = []

        for bench_key, bench_name in benchmarks.items():
            row = []
            for model in models:
                stats_data = summary.get(model, {}).get(bench_key, {})
                row.append(stats_data.get("mean", 0))
            matrix.append(row)
            bench_labels.append(bench_name)

        matrix = np.array(matrix)

        # Create heatmap
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(bench_labels)))
        ax.set_xticklabels(models, fontsize=12)
        ax.set_yticklabels(bench_labels, fontsize=10)

        # Add value annotations
        for i in range(len(bench_labels)):
            for j in range(len(models)):
                value = matrix[i, j]
                text_color = "white" if value < 0.3 or value > 0.7 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                )

        ax.set_title("ABC Benchmark Accuracy Heatmap", fontsize=14)
        ax.set_xlabel("Model")
        ax.set_ylabel("Benchmark")

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "abc_heatmap.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved heatmap to: {output_path}")

        return fig

    def plot_significance_matrix(self, save: bool = True) -> plt.Figure:
        """Generate significance matrix from t-tests."""
        fig, ax = plt.subplots(figsize=(8, 6))

        pairwise = self.stats.get("pairwise_ttests", {})
        benchmarks = self.results.get("metadata", {}).get("benchmarks", {})
        comparisons = ["A_vs_B", "A_vs_C", "B_vs_C"]
        comp_labels = ["A vs B", "A vs C", "B vs C"]

        matrix = []
        for bench_key in benchmarks.keys():
            row = []
            for comp in comparisons:
                p_val = pairwise.get(comp, {}).get(bench_key, {}).get("p_value", 1.0)
                sig = 1 if p_val < 0.05 else 0
                row.append(sig)
            matrix.append(row)

        matrix = np.array(matrix)

        # Create heatmap
        im = ax.imshow(matrix, cmap="Greens", aspect="auto", vmin=0, vmax=1)

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Significant (p < 0.05)", rotation=-90, va="bottom")

        ax.set_xticks(np.arange(len(comparisons)))
        ax.set_yticks(np.arange(len(benchmarks)))
        ax.set_xticklabels(comp_labels, fontsize=11)
        ax.set_yticklabels(list(benchmarks.values()), fontsize=10)

        # Annotate
        for i in range(len(benchmarks)):
            for j in range(len(comparisons)):
                val = matrix[i, j]
                text = "Yes" if val == 1 else "No"
                text_color = "white" if val == 1 else "black"
                ax.text(
                    j, i, text, ha="center", va="center", color=text_color, fontsize=10
                )

        ax.set_title(
            "Statistical Significance Matrix (Welch t-test, α=0.05)", fontsize=14
        )
        ax.set_xlabel("Comparison")
        ax.set_ylabel("Benchmark")

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "abc_significance.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved significance matrix to: {output_path}")

        return fig

    def generate_markdown_report(self) -> str:
        """Generate markdown report with tables and figure links."""
        summary = self.stats.get("summary_statistics", {})
        benchmarks = self.results.get("metadata", {}).get("benchmarks", {})
        anova = self.stats.get("anova_results", {})
        pairwise = self.stats.get("pairwise_ttests", {})

        md = "# ABC Benchmark Results v3.0\n\n"

        # Summary Table
        md += "## Summary Statistics (Mean ± 95% CI)\n\n"
        md += "| Benchmark | Model A | Model B | Model C | ANOVA p | η² |\n"
        md += "|-----------|---------|---------|---------|---------|-----|\n"

        for bench_key, bench_name in benchmarks.items():
            row = f"| **{bench_name}** "
            for model in ["A", "B", "C"]:
                stats_data = summary.get(model, {}).get(bench_key, {})
                mean = stats_data.get("mean", 0)
                ci = stats_data.get("ci_95", [0, 0])
                row += f"| {mean:.3f} ± {((ci[1] - ci[0]) / 2):.3f} "
            anova_data = anova.get(bench_key, {})
            md += f"| {anova_data.get('p_value', 1.0):.4f} | {anova_data.get('eta_squared', 0):.3f} |\n"

        md += "\n## Pairwise Comparisons (Welch's t-test)\n\n"

        for comp in ["A_vs_B", "A_vs_C", "B_vs_C"]:
            md += f"### {comp.replace('_', ' ')}\n\n"
            md += "| Benchmark | t-stat | p-value | Cohen's d | Significant |\n"
            md += "|-----------|--------|---------|-----------|-------------|\n"

            for bench_key in benchmarks.keys():
                data = pairwise.get(comp, {}).get(bench_key, {})
                md += f"| {benchmarks[bench_key]} | {data.get('t_statistic', 0):.3f} | "
                md += (
                    f"{data.get('p_value', 1.0):.4f} | {data.get('cohens_d', 0):.3f} | "
                )
                md += (
                    "Yes | No"[not data.get("significant", False) :: _german] * 2
                    + " |\n"
                )

            md += "\n"

        md += "## Visualizations\n\n"
        md += "![Error Bars](figures/abc_error_bars.png)\n\n"
        md += "![Heatmap](figures/abc_heatmap.png)\n\n"
        md += "![Significance Matrix](figures/abc_significance.png)\n\n"

        return md

    def run_full_visualization(self):
        """Generate all visualizations and markdown report."""
        logger.info("=" * 60)
        logger.info("Starting ABC Visualization v3.0")
        logger.info("=" * 60)

        logger.info("Generating error bar plot...")
        self.plot_error_bars()

        logger.info("Generating heatmap...")
        self.plot_comparison_heatmap()

        logger.info("Generating significance matrix...")
        self.plot_significance_matrix()

        logger.info("Generating markdown report...")
        md_report = self.generate_markdown_report()

        md_path = self.output_dir.parent / "abc_results_v3.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        logger.info(f"Saved markdown report to: {md_path}")

        logger.info("Visualization complete!")


def main():
    parser = argparse.ArgumentParser(description="ABC Visualization v3.0")
    parser.add_argument(
        "--results", type=str, default="results/abc_testing/abc_results_v3.json"
    )
    parser.add_argument(
        "--stats", type=str, default="results/abc_testing/statistical_analysis_v3.json"
    )

    args = parser.parse_args()

    viz = ABCVisualizerV3(results_path=args.results, stats_path=args.stats)
    viz.run_full_visualization()


if __name__ == "__main__":
    main()
