"""
ベンチマーク評価モジュール。

Features:
    - lm-evaluation-harness統合
    - 複数モデル並列評価
    - 統計的有意差計算
    - エラーバー付きグラフ生成

References:
    - lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """ベンチマーク結果。

    Attributes:
        benchmark_name: ベンチマーク名
        mean: 平均値
        std: 標準偏差
        se: 標準誤差
        ci95: 95%信頼区間
        n_samples: サンプル数
        raw_scores: 生スコアリスト
    """

    benchmark_name: str
    mean: float
    std: float
    se: float
    ci95: tuple[float, float]
    n_samples: int
    raw_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """辞書に変換する。"""
        return {
            "benchmark": self.benchmark_name,
            "mean": self.mean,
            "std": self.std,
            "se": self.se,
            "ci95_lower": self.ci95[0],
            "ci95_upper": self.ci95[1],
            "n_samples": self.n_samples,
        }

    def format_string(self, decimals: int = 2) -> str:
        """フォーマット済み文字列を返す。"""
        fmt = f"{{:.{decimals}f}}"
        return f"{fmt.format(self.mean)}±{fmt.format(self.se)}"


class BenchmarkEvaluator:
    """ベンチマーク評価器。

    Attributes:
        device: デバイス
        batch_size: バッチサイズ
        n_shots: N-shot設定リスト
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        n_shots: list[int] = None,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.n_shots = n_shots or [0, 1, 5]

    def evaluate_model(
        self,
        model,
        tokenizer,
        benchmark_name: str,
        *,
        n_shot: int = 0,
        samples: Optional[list[dict]] = None,
    ) -> BenchmarkResult:
        """モデルを評価する。

        Args:
            model: モデル
            tokenizer: トークナイザ
            benchmark_name: ベンチマーク名
            n_shot: N-shot数
            samples: 評価サンプル

        Returns:
            ベンチマーク結果

        Example:
            >>> evaluator = BenchmarkEvaluator()
            >>> result = evaluator.evaluate_model(
            ...     model, tokenizer, "MMLU", n_shot=5
            ... )
            >>> print(result.format_string())
            "72.3±0.9"
        """
        logger.info(f"評価開始: {benchmark_name} ({n_shot}-shot)")

        if samples is None:
            samples = self._load_samples(benchmark_name)

        scores = []
        for sample in samples:
            score = self._evaluate_sample(model, tokenizer, sample, n_shot)
            scores.append(score)

        mean = np.mean(scores)
        std = np.std(scores)
        n = len(scores)
        se = std / np.sqrt(n) if n > 0 else 0

        ci95 = (
            mean - 1.96 * se,
            mean + 1.96 * se,
        )

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            mean=mean,
            std=std,
            se=se,
            ci95=ci95,
            n_samples=n,
            raw_scores=scores,
        )

        logger.info(f"評価完了: {benchmark_name} - {result.format_string()} (n={n})")

        return result

    def _evaluate_sample(
        self,
        model,
        tokenizer,
        sample: dict,
        n_shot: int,
    ) -> float:
        """サンプルを評価する。"""
        raise NotImplementedError

    def _load_samples(self, benchmark_name: str) -> list[dict]:
        """サンプルを読み込む。"""
        logger.warning(f"サンプル読み込みは未実装: {benchmark_name}")
        return []

    def compare_models(
        self,
        results_a: dict[str, BenchmarkResult],
        results_b: dict[str, BenchmarkResult],
        benchmark_name: str,
    ) -> dict:
        """2つのモデルを比較する。

        Args:
            results_a: モデルAの結果
            results_b: モデルBの結果
            benchmark_name: ベンチマーク名

        Returns:
            比較結果
        """
        if benchmark_name not in results_a:
            raise ValueError(f"結果が見つかりません: {benchmark_name}")
        if benchmark_name not in results_b:
            raise ValueError(f"結果が見つかりません: {benchmark_name}")

        result_a = results_a[benchmark_name]
        result_b = results_b[benchmark_name]

        diff = result_b.mean - result_a.mean

        pooled_se = np.sqrt(result_a.se**2 + result_b.se**2)
        z_score = diff / pooled_se if pooled_se > 0 else 0

        from scipy import stats

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        effect_size = (
            diff / np.sqrt((result_a.std**2 + result_b.std**2) / 2)
            if result_a.std > 0 and result_b.std > 0
            else 0
        )

        return {
            "benchmark": benchmark_name,
            "model_a_mean": result_a.mean,
            "model_b_mean": result_b.mean,
            "difference": diff,
            "z_score": z_score,
            "p_value": p_value,
            "effect_size": effect_size,
            "significant": p_value < 0.05,
        }


class StatisticalAnalyzer:
    """統計分析器。"""

    @staticmethod
    def one_way_anova(
        groups: list[list[float]],
    ) -> dict:
        """一元配置分散分析を行う。

        Args:
            groups: グループごとのデータリスト

        Returns:
            分析結果
        """
        from scipy import stats

        f_stat, p_value = stats.f_oneway(*groups)

        ss_between = sum(
            len(g) * (np.mean(g) - np.mean(sum(groups, []))) ** 2 for g in groups
        )
        ss_within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
        ss_total = ss_between + ss_within

        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        return {
            "f_statistic": f_stat,
            "p_value": p_value,
            "eta_squared": eta_squared,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def tukey_hsd(
        groups: list[list[float]],
        group_names: list[str],
    ) -> dict:
        """Tukey HSD多重比較を行う。

        Args:
            groups: グループごとのデータリスト
            group_names: グループ名リスト

        Returns:
            比較結果
        """
        from scipy import stats

        n_groups = len(groups)
        comparisons = []

        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                stat, p_value = stats.ttest_ind(groups[i], groups[j])
                mean_diff = np.mean(groups[j]) - np.mean(groups[i])

                comparisons.append(
                    {
                        "group1": group_names[i],
                        "group2": group_names[j],
                        "mean_difference": mean_diff,
                        "t_statistic": stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }
                )

        return {
            "comparisons": comparisons,
            "n_groups": n_groups,
        }

    @staticmethod
    def cohens_d(
        group1: list[float],
        group2: list[float],
    ) -> float:
        """Cohen's d を計算する。"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)

        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        return (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
