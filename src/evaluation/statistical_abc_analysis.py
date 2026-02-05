#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計的ABC分析スクリプト
多重比較補正、エラーバー計算、有意性検定を実装
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalABCAnalyzer:
    """ABCテストの統計的分析クラス"""

    def __init__(self, results_file: str = None, results_data: Dict = None):
        """
        初期化
        
        Args:
            results_file: ABCテスト結果JSONファイルのパス
            results_data: ABCテスト結果データ（直接指定）
        """
        if results_data:
            self.results = results_data
        elif results_file:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        else:
            raise ValueError("Either results_file or results_data must be provided")

        self.models = ['A', 'B', 'C']
        self.benchmarks = list(self.results.get('A', {}).keys())
        self.num_seeds = 10  # デフォルト10ランダムシード

    def calculate_error_bars(self, scores: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """
        エラーバーを計算（標準誤差、信頼区間）
        
        Args:
            scores: スコアのリスト
            confidence: 信頼水準（デフォルト0.95）
        
        Returns:
            エラーバー情報の辞書
        """
        scores_array = np.array(scores)
        n = len(scores_array)
        
        if n == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'sem': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'n': 0
            }
        
        mean = np.mean(scores_array)
        std = np.std(scores_array, ddof=1)  # 標本標準偏差
        
        # 標準誤差（SEM）
        sem = std / np.sqrt(n) if n > 1 else 0.0
        
        # 95%信頼区間（t分布を使用）
        if n > 1:
            df = n - 1
            t_critical = stats.t.ppf((1 + confidence) / 2, df)
            ci_lower = mean - t_critical * sem
            ci_upper = mean + t_critical * sem
        else:
            ci_lower = mean
            ci_upper = mean
        
        return {
            'mean': float(mean),
            'std': float(std),
            'sem': float(sem),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n': n
        }

    def bootstrap_confidence_interval(self, scores: List[float], n_bootstrap: int = 1000, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        ブートストラップ信頼区間を計算
        
        Args:
            scores: スコアのリスト
            n_bootstrap: ブートストラップ回数
            confidence: 信頼水準
        
        Returns:
            (下限, 上限)のタプル
        """
        scores_array = np.array(scores)
        n = len(scores_array)
        
        if n == 0:
            return (0.0, 0.0)
        
        bootstrap_means = []
        np.random.seed(42)  # 再現性のため
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores_array, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return (float(ci_lower), float(ci_upper))

    def perform_statistical_test(self, scores_a: List[float], scores_b: List[float]) -> Dict[str, Any]:
        """
        統計的有意性検定を実行
        
        Args:
            scores_a: モデルAのスコアリスト
            scores_b: モデルBのスコアリスト
        
        Returns:
            検定結果の辞書
        """
        scores_a_array = np.array(scores_a)
        scores_b_array = np.array(scores_b)
        
        if len(scores_a_array) == 0 or len(scores_b_array) == 0:
            return {
                'p_value': 1.0,
                'effect_size': 0.0,
                'method': 'insufficient_data',
                'statistically_significant': False
            }
        
        # 正規性検定（Shapiro-Wilk検定）
        if len(scores_a_array) >= 3 and len(scores_b_array) >= 3:
            _, p_a = stats.shapiro(scores_a_array)
            _, p_b = stats.shapiro(scores_b_array)
            normal_a = p_a > 0.05
            normal_b = p_b > 0.05
        else:
            normal_a = False
            normal_b = False
        
        # 等分散性検定（Levene検定）
        if len(scores_a_array) >= 2 and len(scores_b_array) >= 2:
            _, p_levene = stats.levene(scores_a_array, scores_b_array)
            equal_var = p_levene > 0.05
        else:
            equal_var = False
        
        # 適切な検定方法を選択
        if normal_a and normal_b and len(scores_a_array) >= 2 and len(scores_b_array) >= 2:
            # t検定（対応なし）
            t_stat, p_value = stats.ttest_ind(scores_a_array, scores_b_array, equal_var=equal_var)
            method = 't_test_independent'
        elif len(scores_a_array) >= 3 and len(scores_b_array) >= 3:
            # Mann-Whitney U検定（非正規分布の場合）
            u_stat, p_value = stats.mannwhitneyu(scores_a_array, scores_b_array, alternative='two-sided')
            method = 'mann_whitney_u'
        else:
            # サンプルサイズが小さい場合はブートストラップ
            p_value = self._bootstrap_p_value(scores_a_array, scores_b_array)
            method = 'bootstrap'
        
        # 効果量（Cohen's d）の計算
        pooled_std = np.sqrt(
            ((len(scores_a_array) - 1) * np.var(scores_a_array, ddof=1) +
             (len(scores_b_array) - 1) * np.var(scores_b_array, ddof=1)) /
            (len(scores_a_array) + len(scores_b_array) - 2)
        ) if len(scores_a_array) + len(scores_b_array) > 2 else np.std(np.concatenate([scores_a_array, scores_b_array]))
        
        mean_diff = np.mean(scores_b_array) - np.mean(scores_a_array)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        
        # 効果量の解釈
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        return {
            'p_value': float(p_value),
            'effect_size': float(cohens_d),
            'effect_interpretation': effect_interpretation,
            'method': method,
            'mean_a': float(np.mean(scores_a_array)),
            'mean_b': float(np.mean(scores_b_array)),
            'mean_diff': float(mean_diff),
            'normal_a': normal_a if 'normal_a' in locals() else None,
            'normal_b': normal_b if 'normal_b' in locals() else None,
            'equal_variance': equal_var if 'equal_var' in locals() else None
        }

    def _bootstrap_p_value(self, scores_a: np.ndarray, scores_b: np.ndarray, n_bootstrap: int = 1000) -> float:
        """ブートストラップ法によるp値計算"""
        original_diff = np.mean(scores_b) - np.mean(scores_a)
        combined = np.concatenate([scores_a, scores_b])
        
        bootstrap_diffs = []
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            sample_a = np.random.choice(combined, size=len(scores_a), replace=True)
            sample_b = np.random.choice(combined, size=len(scores_b), replace=True)
            bootstrap_diffs.append(np.mean(sample_b) - np.mean(sample_a))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        if original_diff >= 0:
            p_value = np.mean(bootstrap_diffs >= original_diff)
        else:
            p_value = np.mean(bootstrap_diffs <= original_diff)
        
        return float(2 * min(p_value, 1 - p_value))  # 両側検定

    def apply_multiple_comparison_correction(self, p_values: Dict[str, float], 
                                            method: str = 'bonferroni') -> Dict[str, float]:
        """
        多重比較補正を適用
        
        Args:
            p_values: ベンチマーク名をキー、p値を値とする辞書
            method: 補正方法（'bonferroni' または 'fdr'）
        
        Returns:
            補正後のp値の辞書
        """
        if method == 'bonferroni':
            # Bonferroni補正
            n_comparisons = len(p_values)
            alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
            corrected_p_values = {k: min(v * n_comparisons, 1.0) for k, v in p_values.items()}
            return {
                'corrected_p_values': corrected_p_values,
                'alpha': alpha,
                'method': 'bonferroni',
                'n_comparisons': n_comparisons
            }
        elif method == 'fdr':
            # FDR補正（Benjamini-Hochberg法）
            sorted_p = sorted(p_values.items(), key=lambda x: x[1])
            n = len(sorted_p)
            corrected_p_values = {}
            
            for i, (benchmark, p_value) in enumerate(sorted_p):
                rank = i + 1
                corrected_p = min(p_value * n / rank, 1.0)
                corrected_p_values[benchmark] = corrected_p
            
            return {
                'corrected_p_values': corrected_p_values,
                'alpha': 0.05,
                'method': 'fdr_benjamini_hochberg',
                'n_comparisons': n
            }
        else:
            raise ValueError(f"Unknown correction method: {method}")

    def analyze_abc_results(self, apply_correction: bool = True, 
                           correction_method: str = 'bonferroni') -> Dict[str, Any]:
        """
        ABCテスト結果の統計的分析を実行
        
        Args:
            apply_correction: 多重比較補正を適用するか
            correction_method: 補正方法（'bonferroni' または 'fdr'）
        
        Returns:
            統計的分析結果の辞書
        """
        logger.info("[STAT] Starting statistical analysis of ABC test results...")
        
        analysis_results = {
            'error_bars': {},
            'pairwise_comparisons': {},
            'multiple_comparison_correction': {},
            'summary_statistics': {}
        }
        
        # 各ベンチマークでのエラーバー計算
        for benchmark in self.benchmarks:
            benchmark_error_bars = {}
            
            for model in self.models:
                if benchmark in self.results.get(model, {}):
                    model_data = self.results[model][benchmark]
                    
                    # スコアリストを取得（シードごとの結果がある場合）
                    if isinstance(model_data, dict) and 'scores' in model_data:
                        scores = model_data['scores']
                    elif isinstance(model_data, dict) and 'accuracy' in model_data:
                        # 単一のaccuracy値の場合、リストに変換
                        scores = [model_data['accuracy']]
                    else:
                        # その他の形式
                        scores = [model_data] if isinstance(model_data, (int, float)) else []
                    
                    # エラーバー計算
                    error_bars = self.calculate_error_bars(scores)
                    benchmark_error_bars[model] = error_bars
            
            analysis_results['error_bars'][benchmark] = benchmark_error_bars
        
        # ペアワイズ比較（A vs B, A vs C, B vs C）
        for benchmark in self.benchmarks:
            pairwise = {}
            
            # 各モデルのスコアを取得
            model_scores = {}
            for model in self.models:
                if benchmark in self.results.get(model, {}):
                    model_data = self.results[model][benchmark]
                    if isinstance(model_data, dict) and 'scores' in model_data:
                        model_scores[model] = model_data['scores']
                    elif isinstance(model_data, dict) and 'accuracy' in model_data:
                        model_scores[model] = [model_data['accuracy']]
                    else:
                        model_scores[model] = [model_data] if isinstance(model_data, (int, float)) else []
            
            # A vs B
            if 'A' in model_scores and 'B' in model_scores:
                pairwise['A_vs_B'] = self.perform_statistical_test(model_scores['A'], model_scores['B'])
            
            # A vs C
            if 'A' in model_scores and 'C' in model_scores:
                pairwise['A_vs_C'] = self.perform_statistical_test(model_scores['A'], model_scores['C'])
            
            # B vs C
            if 'B' in model_scores and 'C' in model_scores:
                pairwise['B_vs_C'] = self.perform_statistical_test(model_scores['B'], model_scores['C'])
            
            analysis_results['pairwise_comparisons'][benchmark] = pairwise
        
        # 多重比較補正
        if apply_correction:
            # 各ベンチマークでのA vs Cのp値を収集（AEGIS vs Qwen-base）
            p_values = {}
            for benchmark in self.benchmarks:
                if benchmark in analysis_results['pairwise_comparisons']:
                    if 'A_vs_C' in analysis_results['pairwise_comparisons'][benchmark]:
                        p_values[benchmark] = analysis_results['pairwise_comparisons'][benchmark]['A_vs_C']['p_value']
            
            if p_values:
                correction_result = self.apply_multiple_comparison_correction(p_values, method=correction_method)
                analysis_results['multiple_comparison_correction'] = correction_result
        
        # サマリー統計
        summary = {}
        for benchmark in self.benchmarks:
            if benchmark in analysis_results['error_bars']:
                benchmark_summary = {}
                for model in self.models:
                    if model in analysis_results['error_bars'][benchmark]:
                        error_bars = analysis_results['error_bars'][benchmark][model]
                        benchmark_summary[model] = {
                            'mean': error_bars['mean'],
                            'std': error_bars['std'],
                            'ci_95_lower': error_bars['ci_lower'],
                            'ci_95_upper': error_bars['ci_upper'],
                            'n': error_bars['n']
                        }
                summary[benchmark] = benchmark_summary
        
        analysis_results['summary_statistics'] = summary
        
        logger.info("[STAT] Statistical analysis completed")
        return analysis_results

    def save_analysis_results(self, analysis_results: Dict[str, Any], output_file: str):
        """分析結果をJSONファイルに保存"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[STAT] Analysis results saved to {output_path}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Statistical ABC Analysis')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to ABC test results JSON file')
    parser.add_argument('--output_file', type=str,
                       default='results/abc_testing/statistical_analysis.json',
                       help='Output file path for statistical analysis results')
    parser.add_argument('--correction_method', type=str, default='bonferroni',
                       choices=['bonferroni', 'fdr'],
                       help='Multiple comparison correction method')
    
    args = parser.parse_args()
    
    # 統計的分析実行
    analyzer = StatisticalABCAnalyzer(results_file=args.results_file)
    analysis_results = analyzer.analyze_abc_results(
        apply_correction=True,
        correction_method=args.correction_method
    )
    
    # 結果保存
    analyzer.save_analysis_results(analysis_results, args.output_file)
    
    logger.info("[STAT] Statistical analysis completed successfully")


if __name__ == "__main__":
    main()
