#!/usr/bin/env python3
"""
ABC Statistics Analysis v3.0.

Statistical analysis for ABC benchmark results:
- Summary statistics (mean, std, 95% CI)
- Welch's t-test for pairwise comparisons
- Holm-Bonferroni correction for multiple comparisons
- One-way ANOVA with effect size (η²)
- Significance level: α = 0.05
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SummaryStats:
    """Summary statistics for a benchmark."""

    mean: float
    std: float
    n: int
    ci_95_lower: float
    ci_95_upper: float
    sem: float


@dataclass
class TTestResult:
    """Result of Welch's t-test."""

    t_statistic: float
    p_value: float
    degrees_of_freedom: float
    mean_difference: float
    significant: bool
    effect_size_cohens_d: float


@dataclass
class ANOVAResult:
    """Result of one-way ANOVA."""

    f_statistic: float
    p_value: float
    effect_size_eta_squared: float
    significant: bool


class ABCStatisticsV3:
    """Statistical analysis for ABC benchmark results."""

    def __init__(self, results_path: str = "results/abc_testing/abc_results_v3.json"):
        self.project_root = Path(__file__).parent.parent.parent
        self.results_path = self.project_root / results_path
        self.results: Dict[str, Any] = {}
        self.load_results()

    def load_results(self):
        """Load benchmark results from JSON."""
        if self.results_path.exists():
            with open(self.results_path, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            logger.info(f"Loaded results from {self.results_path}")
        else:
            logger.warning(f"Results file not found: {self.results_path}")

    def compute_summary_stats(self, scores: List[float]) -> SummaryStats:
        """Compute summary statistics with 95% CI."""
        n = len(scores)
        if n == 0:
            return SummaryStats(0, 0, 0, 0, 0, 0)

        mean = np.mean(scores)
        std = np.std(scores, ddof=1)  # Sample standard deviation
        sem = std / np.sqrt(n)  # Standard error of mean

        # 95% CI using t-distribution
        df = n - 1
        t_critical = stats.t.ppf(0.975, df)  # Two-tailed 95%
        ci_margin = t_critical * sem

        return SummaryStats(
            mean=mean,
            std=std,
            n=n,
            ci_95_lower=mean - ci_margin,
            ci_95_upper=mean + ci_margin,
            sem=sem,
        )

    def get_benchmark_scores(self, model_key: str, benchmark: str) -> List[float]:
        """Extract scores for a specific model and benchmark."""
        if model_key not in self.results.get("results", {}):
            return []

        scores = []
        for seed_key, seed_data in self.results["results"][model_key].items():
            if benchmark in seed_data:
                scores.append(seed_data[benchmark])

        return scores

    def welch_ttest(self, scores1: List[float], scores2: List[float]) -> TTestResult:
        """Perform Welch's t-test for unequal variances."""
        if len(scores1) == 0 or len(scores2) == 0:
            return TTestResult(0, 1.0, 0, 0, False, 0)

        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
        n1, n2 = len(scores1), len(scores2)

        # Welch's t-statistic
        se = np.sqrt(var1 / n1 + var2 / n2)
        t_stat = (mean1 - mean2) / se

        # Welch-Satterthwaite degrees of freedom
        num = (var1 / n1 + var2 / n2) ** 2
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = num / denom if denom > 0 else 0

        # p-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        # Cohen's d effect size
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        # Significance at α = 0.05
        significant = p_value < 0.05

        return TTestResult(
            t_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            mean_difference=mean1 - mean2,
            significant=significant,
            effect_size_cohens_d=cohens_d,
        )

    def holm_bonferroni_correction(
        self, p_values: List[float], alpha: float = 0.05
    ) -> List[bool]:
        """Apply Holm-Bonferroni correction for multiple comparisons."""
        n = len(p_values)
        if n == 0:
            return []

        # Sort p-values and keep track of original indices
        sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])

        corrected = []
        for rank, (idx, p) in enumerate(sorted_p, 1):
            threshold = alpha / (n - rank + 1)
            corrected.append(p <= threshold)

        # Reorder to original order
        result = [False] * n
        for new_idx, (orig_idx, _) in enumerate(sorted_p):
            result[orig_idx] = corrected[new_idx]

        return result

    def one_way_anova(self, groups: List[List[float]]) -> ANOVAResult:
        """Perform one-way ANOVA with effect size η²."""
        if len(groups) < 2:
            return ANOVAResult(0, 1.0, 0, False)

        # Flatten all scores
        all_scores = [s for g in groups for s in g]
        grand_mean = np.mean(all_scores)

        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

        # Within-group sum of squares
        ss_within = sum(sum((s - np.mean(g)) ** 2 for s in g) for g in groups)

        # Total sum of squares
        ss_total = ss_between + ss_within

        # Degrees of freedom
        k = len(groups)
        df_between = k - 1
        df_within = len(all_scores) - k

        # Mean squares
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        # F-statistic
        f_stat = ms_between / ms_within if ms_within > 0 else 0

        # p-value
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)

        # Effect size η²
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Significance
        significant = p_value < 0.05

        return ANOVAResult(
            f_statistic=f_stat,
            p_value=p_value,
            effect_size_eta_squared=eta_squared,
            significant=significant,
        )

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete statistical analysis."""
        logger.info("=" * 60)
        logger.info("Starting ABC Statistical Analysis v3.0")
        logger.info("Significance level: α = 0.05")
        logger.info("=" * 60)

        metadata = self.results.get("metadata", {})
        benchmarks = metadata.get("benchmarks", {})
        models = ["A", "B", "C"]

        analysis = {
            "summary_statistics": {},
            "pairwise_ttests": {},
            "anova_results": {},
            "metadata": {
                "alpha": 0.05,
                "correction": "Holm-Bonferroni",
                "benchmarks": benchmarks,
                "models": models,
            },
        }

        # 1. Summary statistics for each model/benchmark
        logger.info("Computing summary statistics...")
        for model in models:
            analysis["summary_statistics"][model] = {}
            for bench_key, bench_name in benchmarks.items():
                scores = self.get_benchmark_scores(model, bench_key)
                stats_obj = self.compute_summary_stats(scores)
                analysis["summary_statistics"][model][bench_key] = {
                    "mean": stats_obj.mean,
                    "std": stats_obj.std,
                    "n": stats_obj.n,
                    "ci_95": [stats_obj.ci_95_lower, stats_obj.ci_95_upper],
                    "sem": stats_obj.sem,
                }
                logger.info(
                    f"  {model}/{bench_key}: {stats_obj.mean:.3f} ± {stats_obj.std:.3f}"
                )

        # 2. Pairwise t-tests (A vs B, A vs C, B vs C)
        logger.info("Performing pairwise Welch's t-tests...")
        comparisons = [("A", "B"), ("A", "C"), ("B", "C")]
        all_p_values = []

        for model1, model2 in comparisons:
            key = f"{model1}_vs_{model2}"
            analysis["pairwise_ttests"][key] = {}

            for bench_key in benchmarks.keys():
                scores1 = self.get_benchmark_scores(model1, bench_key)
                scores2 = self.get_benchmark_scores(model2, bench_key)

                if scores1 and scores2:
                    tt_result = self.welch_ttest(scores1, scores2)
                    analysis["pairwise_ttests"][key][bench_key] = {
                        "t_statistic": tt_result.t_statistic,
                        "p_value": tt_result.p_value,
                        "df": tt_result.degrees_of_freedom,
                        "mean_diff": tt_result.mean_difference,
                        "cohens_d": tt_result.effect_size_cohens_d,
                        "significant": tt_result.significant,
                    }
                    all_p_values.append(tt_result.p_value)
                    logger.info(
                        f"  {key}/{bench_key}: p={tt_result.p_value:.4f}, "
                        f"d={tt_result.effect_size_cohens_d:.3f}"
                    )

        # 3. Apply Holm-Bonferroni correction
        logger.info("Applying Holm-Bonferroni correction...")
        if all_p_values:
            corrected = self.holm_bonferroni_correction(all_p_values)
            analysis["holm_bonferroni"] = {
                "original_p_values": all_p_values,
                "corrected_significant": corrected,
            }

        # 4. One-way ANOVA for each benchmark
        logger.info("Performing one-way ANOVA...")
        for bench_key in benchmarks.keys():
            groups = [self.get_benchmark_scores(m, bench_key) for m in models]
            groups = [g for g in groups if g]  # Remove empty groups

            if len(groups) >= 2:
                anova_result = self.one_way_anova(groups)
                analysis["anova_results"][bench_key] = {
                    "f_statistic": anova_result.f_statistic,
                    "p_value": anova_result.p_value,
                    "eta_squared": anova_result.effect_size_eta_squared,
                    "significant": anova_result.significant,
                }
                logger.info(
                    f"  {bench_key}: F={anova_result.f_statistic:.3f}, "
                    f"p={anova_result.p_value:.4f}, η²={anova_result.effect_size_eta_squared:.3f}"
                )

        # 5. Save results
        output_path = (
            self.project_root / "results/abc_testing/statistical_analysis_v3.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis saved to: {output_path}")
        return analysis

    def generate_summary_table(self) -> str:
        """Generate markdown summary table."""
        summary = self.results.get("summary_statistics", {})
        benchmarks = self.results.get("metadata", {}).get("benchmarks", {})
        models = {"A": "Phi-3.5-instinct", "B": "Borea-phi3.5", "C": "AEGIS-v3.0"}

        table = "| Benchmark | Model A | Model B | Model C | ANOVA p | Significant |\n"
        table += "|-----------|---------|---------|---------|---------|-------------|\n"

        for bench_key, bench_name in benchmarks.items():
            row = f"| **{bench_name}** "

            model_means = []
            for model in ["A", "B", "C"]:
                stats_data = summary.get(model, {}).get(bench_key, {})
                mean = stats_data.get("mean", 0)
                ci = stats_data.get("ci_95", [0, 0])
                model_means.append(mean)
                row += f"| {mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] "

            # Add ANOVA result
            anova = self.results.get("anova_results", {}).get(bench_key, {})
            p_val = anova.get("p_value", 1.0)
            sig = "Yes" if anova.get("significant") else "No"
            row += f"| {p_val:.4f} | {sig} |\n"

        return table


def main():
    parser = argparse.ArgumentParser(description="ABC Statistics Analysis v3.0")
    parser.add_argument(
        "--results", type=str, default="results/abc_testing/abc_results_v3.json"
    )
    parser.add_argument(
        "--output", type=str, default="results/abc_testing/statistical_analysis_v3.json"
    )

    args = parser.parse_args()

    analysis = ABCStatisticsV3(results_path=args.results)
    results = analysis.run_full_analysis()

    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(analysis.generate_summary_table())


if __name__ == "__main__":
    main()
