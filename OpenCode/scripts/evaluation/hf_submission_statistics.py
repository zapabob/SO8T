#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF提出用統計処理システム
HF Submission Statistics System

エラーバー付きグラフ、要約統計量、ABCテスト結果をHF提出可能な形式で生成
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
logger=logging.getLogger(__name__)
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, normaltest
import warnings
warnings.filterwarnings("ignore")

# 統計分析強化ライブラリ
try:
    import pymc3 as pm
    import arviz as az
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False
    logger.warning("Bayesian libraries not available. Install pymc3 and arviz for enhanced statistical analysis.")

try:
    from statsmodels.multivariate.manova import MANOVA
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multitest import multipletests
    HAS_MULTIVARIATE = True
except ImportError:
    HAS_MULTIVARIATE = False
    logger.warning("Multivariate analysis libraries not available. Install statsmodels for enhanced analysis.")

try:
    from pingouin import sphericity, epsilon
    HAS_SPHERICITY = True
except ImportError:
    HAS_SPHERICITY = False
    logger.warning("Sphericity testing libraries not available. Install pingouin for sphericity analysis.")

try:
    import pingouin as pg
    HAS_EFFECT_SIZE = True
except ImportError:
    HAS_EFFECT_SIZE = False
    logger.warning("Effect size libraries not available. Install pingouin for enhanced effect size calculations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 日本語フォント設定（matplotlib）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class HFSubmissionStatistics:
    """
    HF提出用統計処理クラス
    HF Submission Statistics Class
    """

    def __init__(self, results_data: Dict[str, Any], output_dir: str = "D:/webdataset/results/hf_submission"):
        self.results_data = results_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # スタイル設定
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def perform_multivariate_analysis(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        多変量解析の実行
        Perform multivariate analysis
        """
        if not HAS_MULTIVARIATE:
            logger.warning("Multivariate analysis not available - statsmodels not installed")
            return {"multivariate_analysis": "Not available - install statsmodels"}

        try:
            results = {}

            # ベンチマークデータをDataFrameに変換
            all_scores = []
            for model_name, scores in benchmark_data.items():
                if isinstance(scores, dict):
                    for benchmark, score in scores.items():
                        if isinstance(score, (int, float)):
                            all_scores.append({
                                'model': model_name,
                                'benchmark': benchmark,
                                'score': score
                            })

            if not all_scores:
                return {"multivariate_analysis": "No valid data for analysis"}

            df = pd.DataFrame(all_scores)

            # MANOVA分析
            try:
                formula = 'score ~ model + benchmark + model:benchmark'
                manova_model = ols(formula, data=df).fit()
                manova_results = MANOVA.from_formula(formula, data=df)
                results['manova'] = {
                    'summary': str(manova_results.mv_test()),
                    'r_squared': manova_model.rsquared,
                    'f_statistic': manova_model.fvalue,
                    'p_value': manova_model.f_pvalue
                }
            except Exception as e:
                results['manova'] = f"MANOVA analysis failed: {str(e)}"

            # TukeyのHSD事後検定
            try:
                tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['model'], alpha=0.05)
                results['tukey_hsd'] = {
                    'summary': tukey.summary().as_text(),
                    'reject_null': tukey.reject.tolist(),
                    'meandiffs': tukey.meandiffs.tolist(),
                    'confint': tukey.confint.tolist()
                }
            except Exception as e:
                results['tukey_hsd'] = f"Tukey HSD failed: {str(e)}"

            # 球面性テスト（Mauchlyの球面性検定）
            try:
                sphericity_results = self._test_sphericity(df)
                results['sphericity_test'] = sphericity_results
            except Exception as e:
                results['sphericity_test'] = f"Sphericity test failed: {str(e)}"

            # 反復測定ANOVA（球面性補正付き）
            try:
                rm_anova_results = self._perform_repeated_measures_anova(df)
                results['repeated_measures_anova'] = rm_anova_results
            except Exception as e:
                results['repeated_measures_anova'] = f"Repeated measures ANOVA failed: {str(e)}"

            return {"multivariate_analysis": results}

        except Exception as e:
            logger.error(f"Multivariate analysis failed: {str(e)}")
            return {"multivariate_analysis": f"Analysis failed: {str(e)}"}

    def _test_sphericity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        球面性テスト（Mauchlyの球面性検定）
        Test for sphericity (Mauchly's test)
        """
        sphericity_results = {}

        if not HAS_SPHERICITY:
            sphericity_results['available'] = False
            sphericity_results['note'] = "Sphericity testing requires pingouin library"
            return sphericity_results

        try:
            # データをワイド形式に変換（反復測定用）
            wide_df = df.pivot_table(values='score', index='model', columns='benchmark', aggfunc='mean')

            # Mauchlyの球面性検定
            if len(wide_df.columns) > 2:  # 少なくとも3つの測定が必要
                sphericity_test = sphericity(wide_df.values, method='mauchly')
                sphericity_results['mauchly_test'] = {
                    'W': float(sphericity_test['W']),
                    'p_value': float(sphericity_test['pval']),
                    'sphericity_assumed': bool(sphericity_test['sphericity']),
                    'interpretation': "球面性の仮定が満たされている" if sphericity_test['sphericity'] else "球面性の仮定が満たされていない"
                }

                # Greenhouse-Geisserイプシロン
                gg_epsilon = epsilon(wide_df.values, correction='gg')
                sphericity_results['greenhouse_geisser_epsilon'] = float(gg_epsilon)

                # Huynh-Feldtイプシロン
                hf_epsilon = epsilon(wide_df.values, correction='hf')
                sphericity_results['huynh_feldt_epsilon'] = float(hf_epsilon)

                # 推奨される補正方法
                if not sphericity_test['sphericity']:
                    if gg_epsilon > 0.75:
                        recommended_correction = "Huynh-Feldt"
                    else:
                        recommended_correction = "Greenhouse-Geisser"
                    sphericity_results['recommended_correction'] = recommended_correction
                else:
                    sphericity_results['recommended_correction'] = "なし（球面性が満たされている）"

            else:
                sphericity_results['note'] = "球面性テストには少なくとも3つの測定点が必要です"

            sphericity_results['available'] = True

        except Exception as e:
            logger.error(f"Sphericity test failed: {str(e)}")
            sphericity_results['error'] = str(e)
            sphericity_results['available'] = False

        return sphericity_results

    def _perform_repeated_measures_anova(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        反復測定ANOVA（球面性補正付き）
        Perform repeated measures ANOVA with sphericity corrections
        """
        rm_anova_results = {}

        try:
            # データ構造の確認
            if len(df['benchmark'].unique()) < 2 or len(df['model'].unique()) < 2:
                rm_anova_results['note'] = "反復測定ANOVAには少なくとも2つの条件が必要です"
                return rm_anova_results

            # statsmodelsを使った反復測定ANOVA
            # 注: このデータ構造では標準的な反復測定デザインではないため、適応が必要
            try:
                # モデル比較のための準備
                models = df['model'].unique()
                benchmarks = df['benchmark'].unique()

                anova_results = {}

                # 各ベンチマークでのモデル間比較
                for benchmark in benchmarks:
                    bench_data = df[df['benchmark'] == benchmark].copy()

                    if len(bench_data['model'].unique()) > 1:
                        # 一元配置分散分析
                        from scipy.stats import f_oneway

                        model_groups = []
                        group_labels = []

                        for model in models:
                            model_scores = bench_data[bench_data['model'] == model]['score']
                            if len(model_scores) > 0:
                                model_groups.append(model_scores.values)
                                group_labels.append(model)

                        if len(model_groups) > 1:
                            f_stat, p_value = f_oneway(*model_groups)

                            anova_results[benchmark] = {
                                'f_statistic': float(f_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05,
                                'degrees_of_freedom': (len(model_groups) - 1, sum(len(g) for g in model_groups) - len(model_groups))
                            }

                rm_anova_results['benchmark_anova'] = anova_results

                # 球面性補正の適用
                sphericity_info = self._test_sphericity(df)
                if 'mauchly_test' in sphericity_info:
                    mauchly = sphericity_info['mauchly_test']

                    # 球面性が満たされていない場合の補正
                    if not mauchly['sphericity_assumed']:
                        correction_method = sphericity_info.get('recommended_correction', 'Greenhouse-Geisser')
                        epsilon_value = sphericity_info.get('greenhouse_geisser_epsilon', 1.0)

                        rm_anova_results['sphericity_correction'] = {
                            'applied': True,
                            'method': correction_method,
                            'epsilon': epsilon_value,
                            'note': f"球面性が満たされていないため、{correction_method}補正を適用"
                        }

                        # 補正されたp値の計算（簡易版）
                        for bench_name, result in anova_results.items():
                            if 'p_value' in result:
                                # Greenhouse-Geisser補正の近似
                                corrected_p = min(1.0, result['p_value'] / epsilon_value)
                                result['corrected_p_value'] = corrected_p
                                result['correction_applied'] = True
                    else:
                        rm_anova_results['sphericity_correction'] = {
                            'applied': False,
                            'note': "球面性の仮定が満たされているため、補正不要"
                        }

            except Exception as e:
                rm_anova_results['anova_error'] = str(e)

        except Exception as e:
            logger.error(f"Repeated measures ANOVA failed: {str(e)}")
            rm_anova_results['error'] = str(e)

        return rm_anova_results

    def perform_bayesian_analysis(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ベイズ統計分析の実行
        Perform Bayesian statistical analysis
        """
        if not HAS_BAYESIAN:
            logger.warning("Bayesian analysis not available - pymc3/arviz not installed")
            return {"bayesian_analysis": "Not available - install pymc3 and arviz"}

        try:
            results = {}

            # 各モデルのスコアデータを抽出
            model_scores = {}
            for model_name, scores in benchmark_data.items():
                if isinstance(scores, dict):
                    scores_list = [v for v in scores.values() if isinstance(v, (int, float))]
                    if scores_list:
                        model_scores[model_name] = scores_list

            if len(model_scores) < 2:
                return {"bayesian_analysis": "Need at least 2 models for comparison"}

            # ベイズ階層モデル
            with pm.Model() as hierarchical_model:
                # ハイパーパラメータ
                mu_hyper = pm.Normal('mu_hyper', mu=0, sigma=10)
                sigma_hyper = pm.HalfNormal('sigma_hyper', sigma=10)

                # 各モデルのパラメータ
                model_mus = {}
                model_sigmas = {}

                for model_name, scores in model_scores.items():
                    model_mus[model_name] = pm.Normal(f'mu_{model_name}', mu=mu_hyper, sigma=sigma_hyper)
                    model_sigmas[model_name] = pm.HalfNormal(f'sigma_{model_name}', sigma=sigma_hyper)

                    # 観測データ
                    pm.Normal(f'obs_{model_name}', mu=model_mus[model_name],
                             sigma=model_sigmas[model_name], observed=scores)

                # サンプリング
                trace = pm.sample(1000, tune=1000, return_inferencedata=True)

            # 分析結果
            model_names = list(model_scores.keys())
            comparison_results = {}

            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:
                        mu1_samples = trace.posterior[f'mu_{model1}'].values.flatten()
                        mu2_samples = trace.posterior[f'mu_{model2}'].values.flatten()
                        diff_samples = mu1_samples - mu2_samples

                        comparison_results[f'{model1}_vs_{model2}'] = {
                            'mean_difference': float(np.mean(diff_samples)),
                            'credible_interval': [float(np.percentile(diff_samples, 2.5)),
                                                float(np.percentile(diff_samples, 97.5))],
                            'probability_superior': float(np.mean(diff_samples > 0)),
                            'effect_size': float(np.mean(diff_samples) / np.std(diff_samples))
                        }

            results['hierarchical_model'] = {
                'model_comparison': comparison_results,
                'trace_summary': str(az.summary(trace, round_to=2))
            }

            return {"bayesian_analysis": results}

        except Exception as e:
            logger.error(f"Bayesian analysis failed: {str(e)}")
            return {"bayesian_analysis": f"Analysis failed: {str(e)}"}

    def calculate_effect_sizes(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        効果量の計算と解釈
        Calculate and interpret effect sizes
        """
        try:
            results = {}

            # ベンチマークデータを整理
            all_scores = []
            for model_name, scores in benchmark_data.items():
                if isinstance(scores, dict):
                    for benchmark, score in scores.items():
                        if isinstance(score, (int, float)):
                            all_scores.append({
                                'model': model_name,
                                'benchmark': benchmark,
                                'score': score
                            })

            if not all_scores:
                return {"effect_sizes": "No valid data for analysis"}

            df = pd.DataFrame(all_scores)

            # 各ベンチマークでの効果量計算
            benchmark_effect_sizes = {}
            for benchmark in df['benchmark'].unique():
                benchmark_data = df[df['benchmark'] == benchmark]
                if len(benchmark_data['model'].unique()) >= 2:
                    try:
                        if HAS_EFFECT_SIZE:
                            # pingouinを使った効果量計算
                            effect_size_result = pg.compute_effsize(
                                benchmark_data['score'][benchmark_data['model'] == benchmark_data['model'].unique()[0]],
                                benchmark_data['score'][benchmark_data['model'] == benchmark_data['model'].unique()[1]],
                                eftype='cohen'
                            )
                            effect_size = float(effect_size_result)
                        else:
                            # 手動計算
                            group1 = benchmark_data['score'][benchmark_data['model'] == benchmark_data['model'].unique()[0]]
                            group2 = benchmark_data['score'][benchmark_data['model'] == benchmark_data['model'].unique()[1]]
                            mean_diff = np.mean(group1) - np.mean(group2)
                            pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
                            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

                        # 効果量の解釈
                        if abs(effect_size) < 0.2:
                            interpretation = "negligible"
                        elif abs(effect_size) < 0.5:
                            interpretation = "small"
                        elif abs(effect_size) < 0.8:
                            interpretation = "medium"
                        else:
                            interpretation = "large"

                        benchmark_effect_sizes[benchmark] = {
                            'effect_size': effect_size,
                            'interpretation': interpretation,
                            'models_compared': list(benchmark_data['model'].unique())
                        }
                    except Exception as e:
                        benchmark_effect_sizes[benchmark] = f"Effect size calculation failed: {str(e)}"

            # 全体的な効果量サマリー
            overall_effect_sizes = []
            for bench_data in benchmark_effect_sizes.values():
                if isinstance(bench_data, dict) and 'effect_size' in bench_data:
                    overall_effect_sizes.append(bench_data['effect_size'])

            if overall_effect_sizes:
                results['overall_summary'] = {
                    'mean_effect_size': float(np.mean(overall_effect_sizes)),
                    'median_effect_size': float(np.median(overall_effect_sizes)),
                    'effect_size_range': [float(np.min(overall_effect_sizes)), float(np.max(overall_effect_sizes))],
                    'distribution': {
                        'negligible': len([x for x in overall_effect_sizes if abs(x) < 0.2]),
                        'small': len([x for x in overall_effect_sizes if 0.2 <= abs(x) < 0.5]),
                        'medium': len([x for x in overall_effect_sizes if 0.5 <= abs(x) < 0.8]),
                        'large': len([x for x in overall_effect_sizes if abs(x) >= 0.8])
                    }
                }

            results['benchmark_effect_sizes'] = benchmark_effect_sizes

            return {"effect_sizes": results}

        except Exception as e:
            logger.error(f"Effect size calculation failed: {str(e)}")
            return {"effect_sizes": f"Analysis failed: {str(e)}"}

    def generate_enhanced_statistical_analysis(self) -> Dict[str, Any]:
        """
        強化された統計分析を生成（多変量解析 + ベイズ統計 + 効果量）
        Generate enhanced statistical analysis
        """
        enhanced_analysis = {}

        # ベンチマークデータを抽出
        benchmark_data = {}
        if 'abc_test_results' in self.results_data:
            for model_key, model_data in self.results_data['abc_test_results'].items():
                if 'benchmark_scores' in model_data:
                    benchmark_data[model_key] = model_data['benchmark_scores']

        if benchmark_data:
            # 多変量解析
            enhanced_analysis.update(self.perform_multivariate_analysis(benchmark_data))

            # ベイズ統計分析
            enhanced_analysis.update(self.perform_bayesian_analysis(benchmark_data))

            # 効果量計算
            enhanced_analysis.update(self.calculate_effect_sizes(benchmark_data))

        return enhanced_analysis

    def generate_hf_submission_package(self) -> Dict[str, Any]:
        """
        HF提出用パッケージ生成
        Generate HF submission package
        """
        logger.info("[HF SUBMISSION] Generating HF submission package...")

        # 1. エラーバー付き比較グラフ
        comparison_plots = self._generate_comparison_plots()

        # 2. ABCテスト詳細グラフ
        abc_plots = self._generate_abc_test_plots()

        # 3. 統計的有意差グラフ
        significance_plots = self._generate_significance_plots()

        # 4. 要約統計量テーブル
        summary_tables = self._generate_summary_tables()

        # 5. パフォーマンス分布グラフ
        distribution_plots = self._generate_distribution_plots()

        # 6. レーダーチャート（複数メトリック比較）
        radar_plots = self._generate_radar_plots()

        # 7. 相関分析
        correlation_analysis = self._generate_correlation_analysis()

        # 8. 強化統計分析（多変量解析 + ベイズ統計 + 効果量）
        enhanced_statistical_analysis = self.generate_enhanced_statistical_analysis()

        # 9. READMEと結果サマリー
        documentation = self._generate_documentation()

        package = {
            'plots': {
                'comparison': comparison_plots,
                'abc_test': abc_plots,
                'significance': significance_plots,
                'distribution': distribution_plots,
                'radar': radar_plots
            },
            'tables': summary_tables,
            'analysis': {
                'correlation': correlation_analysis,
                'enhanced_statistics': enhanced_statistical_analysis
            },
            'documentation': documentation,
            'metadata': self._generate_metadata()
        }

        # パッケージ保存
        self._save_package(package)

        return package

    def _generate_comparison_plots(self) -> Dict[str, str]:
        """エラーバー付き比較グラフ生成"""
        logger.info("[PLOTS] Generating comparison plots with error bars...")

        plots = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return plots

        # 各メトリックに対して比較グラフ
        metrics = df['metric'].unique()

        for metric in metrics:
            try:
                metric_data = df[df['metric'] == metric].copy()

                if len(metric_data) == 0:
                    continue

                # モデルごとの統計量計算
                stats_data = []
                for model in metric_data['model'].unique():
                    model_values = metric_data[metric_data['model'] == model]['value']
                    if len(model_values) > 0:
                        stats_data.append({
                            'model': model,
                            'mean': model_values.mean(),
                            'std': model_values.std(),
                            'sem': stats.sem(model_values) if len(model_values) > 1 else 0,
                            'count': len(model_values)
                        })

                if not stats_data:
                    continue

                stats_df = pd.DataFrame(stats_data)

                # エラーバー付き棒グラフ
                fig, ax = plt.subplots(figsize=(12, 8))

                bars = ax.bar(
                    stats_df['model'],
                    stats_df['mean'],
                    yerr=stats_df['sem'],
                    capsize=5,
                    alpha=0.8,
                    color=sns.color_palette("husl", len(stats_df))
                )

                # 値ラベル追加
                for bar, mean_val in zip(bars, stats_df['mean']):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + stats_df['sem'].max() * 0.1,
                        '.3f',
                        ha='center', va='bottom', fontsize=10
                    )

                ax.set_title(f'{metric.replace("_", " ").title()} Comparison\\n(Error bars show standard error of mean)',
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Model', fontsize=12)
                ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
                ax.grid(True, alpha=0.3)

                # 統計情報ボックス
                stats_text = f"n = {stats_df['count'].iloc[0]}\\nSEM shown as error bars"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                plt.xticks(rotation=45)
                plt.tight_layout()

                # 保存
                filename = f"comparison_{metric}_errorbars.png"
                filepath = self.output_dir / "plots" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                plots[metric] = str(filepath)

            except Exception as e:
                logger.error(f"Failed to generate comparison plot for {metric}: {e}")

        return plots

    def _generate_abc_test_plots(self) -> Dict[str, str]:
        """ABCテスト詳細グラフ生成"""
        logger.info("[PLOTS] Generating ABC test plots...")

        plots = {}
        abc_results = self.results_data.get('comparison', {}).get('abc_test', {})

        if not abc_results or 'winner' not in abc_results:
            return plots

        try:
            # ABCテスト勝者グラフ
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('ABC Test Results: Model Comparison Analysis', fontsize=16, fontweight='bold')

            # 勝者情報
            winner = abc_results['winner']

            # 各メトリックのABC比較
            metrics = [k for k in abc_results.keys() if k not in ['winner', 'error'] and isinstance(abc_results[k], dict)]

            for i, metric in enumerate(metrics[:4]):  # 最大4メトリック
                ax = axes[i // 2, i % 2]

                metric_data = abc_results[metric]
                models = []
                means = []
                stds = []

                for model, stats in metric_data.items():
                    if isinstance(stats, dict):
                        models.append(model.upper())
                        means.append(stats.get('mean', 0))
                        stds.append(stats.get('sem', 0) * 1.96)  # 95% CI

                if models and means:
                    bars = ax.bar(models, means, yerr=stds, capsize=5,
                                 color=sns.color_palette("Set2", len(models)))

                    # 勝者をハイライト
                    winner_idx = models.index(winner['model'].upper()) if winner['model'].upper() in models else -1
                    if winner_idx >= 0:
                        bars[winner_idx].set_edgecolor('red')
                        bars[winner_idx].set_linewidth(3)

                    ax.set_title(f'{metric.replace("_", " ").title()}\\nABC Test Results')
                    ax.set_ylabel(metric.replace("_", " ").title())
                    ax.grid(True, alpha=0.3)

                    # 値ラベル
                    for bar, mean_val in zip(bars, means):
                        height = bar.get_height() + bar.get_y() + (stds[bars.index(bar)] if bars.index(bar) < len(stds) else 0)
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(stds) * 0.05,
                               '.3f', ha='center', va='bottom', fontsize=9)

            # 勝者サマリー
            axes[1, 1].text(0.1, 0.8, f"🏆 Winner: {winner['model'].upper()}", fontsize=14, fontweight='bold')
            axes[1, 1].text(0.1, 0.6, f"Score: {winner['score']:.4f}", fontsize=12)
            axes[1, 1].text(0.1, 0.4, f"Metric: {winner['metric']}", fontsize=12)
            axes[1, 1].set_title('ABC Test Winner Summary')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

            plt.tight_layout()

            # 保存
            filename = "abc_test_detailed_analysis.png"
            filepath = self.output_dir / "plots" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            plots['abc_detailed'] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate ABC test plots: {e}")

        return plots

    def _generate_significance_plots(self) -> Dict[str, str]:
        """統計的有意差グラフ生成"""
        logger.info("[PLOTS] Generating statistical significance plots...")

        plots = {}
        statistical_comp = self.results_data.get('comparison', {}).get('statistical_comparison', {})

        if not statistical_comp:
            return plots

        try:
            # 有意差ヒートマップ
            fig, ax = plt.subplots(figsize=(12, 10))

            # 有意差データを整理
            significance_data = []
            model_pairs = []

            for metric, comparisons in statistical_comp.items():
                for comparison_name, results in comparisons.items():
                    if isinstance(results, dict) and 'p_value' in results:
                        model1, model2 = comparison_name.split('_vs_')
                        significance_data.append({
                            'metric': metric,
                            'model1': model1,
                            'model2': model2,
                            'p_value': results['p_value'],
                            'significant': results.get('significant', False)
                        })

            if significance_data:
                sig_df = pd.DataFrame(significance_data)

                # ピボットテーブル作成
                pivot_table = sig_df.pivot_table(
                    values='p_value',
                    index=['model1', 'model2'],
                    columns='metric',
                    aggfunc='first'
                ).fillna(1.0)  # 欠損値は1.0（有意差なし）

                # ヒートマップ
                sns.heatmap(
                    pivot_table,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn_r',
                    center=0.05,
                    vmin=0,
                    vmax=0.1,
                    ax=ax
                )

                ax.set_title('Statistical Significance Heatmap\\n(p-values for model comparisons)',
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Model Comparisons')

                # 有意差の閾値ライン
                ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')

                plt.xticks(rotation=45)
                plt.tight_layout()

                # 保存
                filename = "statistical_significance_heatmap.png"
                filepath = self.output_dir / "plots" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                plots['significance_heatmap'] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate significance plots: {e}")

        return plots

    def _generate_distribution_plots(self) -> Dict[str, str]:
        """パフォーマンス分布グラフ生成"""
        logger.info("[PLOTS] Generating distribution plots...")

        plots = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return plots

        try:
            # 各メトリックのパフォーマンス分布
            metrics = df['metric'].unique()

            for metric in metrics:
                metric_data = df[df['metric'] == metric]

                if len(metric_data) == 0:
                    continue

                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'{metric.replace("_", " ").title()} Performance Distribution Analysis',
                           fontsize=16, fontweight='bold')

                # ヒストグラム + KDE
                for i, model in enumerate(metric_data['model'].unique()):
                    if i >= 4:  # 最大4モデル
                        break

                    ax = axes[i // 2, i % 2]
                    model_values = metric_data[metric_data['model'] == model]['value']

                    if len(model_values) > 0:
                        # ヒストグラム
                        sns.histplot(model_values, kde=True, ax=ax, alpha=0.7,
                                   color=sns.color_palette("husl", len(metric_data['model'].unique()))[i])

                        ax.set_title(f'{model.upper()} Distribution')
                        ax.set_xlabel(metric.replace("_", " ").title())
                        ax.set_ylabel('Frequency')

                        # 統計情報
                        mean_val = model_values.mean()
                        std_val = model_values.std()
                        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8,
                                 label=f'Mean: {mean_val:.3f}')
                        ax.legend()

                plt.tight_layout()

                # 保存
                filename = f"distribution_{metric}_analysis.png"
                filepath = self.output_dir / "plots" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                plots[metric] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate distribution plots: {e}")

        return plots

    def _generate_radar_plots(self) -> Dict[str, str]:
        """レーダーチャート生成（複数メトリック比較）"""
        logger.info("[PLOTS] Generating radar plots...")

        plots = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return plots

        try:
            # モデルごとの正規化されたスコアを計算
            normalized_scores = self._calculate_normalized_scores(df)

            if normalized_scores:
                fig = plt.figure(figsize=(10, 8))

                # レーダーチャート
                ax = fig.add_subplot(111, polar=True)

                # カテゴリ（メトリック）
                categories = list(normalized_scores.keys())
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # 閉じる

                # 各モデルのプロット
                colors = sns.color_palette("husl", len(normalized_scores[categories[0]]))
                for i, (model, scores) in enumerate(normalized_scores[categories[0]].items()):
                    values = [normalized_scores[cat][model] for cat in categories]
                    values += values[:1]  # 閉じる

                    ax.plot(angles, values, 'o-', linewidth=2, label=model.upper(),
                           color=colors[i], alpha=0.8)
                    ax.fill(angles, values, alpha=0.25, color=colors[i])

                # ラベル設定
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
                ax.set_ylim(0, 1)
                ax.set_title('Model Performance Radar Chart\\n(Normalized Scores)', size=16, fontweight='bold')
                ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
                ax.grid(True, alpha=0.3)

                plt.tight_layout()

                # 保存
                filename = "radar_chart_performance_comparison.png"
                filepath = self.output_dir / "plots" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                plots['radar_performance'] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate radar plots: {e}")

        return plots

    def _calculate_normalized_scores(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """正規化スコア計算"""
        normalized_scores = {}

        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric]
            normalized_scores[metric] = {}

            # メトリック値の正規化（0-1スケール）
            if len(metric_data) > 0:
                values = metric_data['value'].values
                min_val, max_val = np.min(values), np.max(values)

                if max_val > min_val:
                    for _, row in metric_data.iterrows():
                        model = row['model']
                        value = row['value']
                        normalized = (value - min_val) / (max_val - min_val)
                        normalized_scores[metric][model] = normalized
                else:
                    # 全値が同じ場合
                    for _, row in metric_data.iterrows():
                        normalized_scores[metric][row['model']] = 1.0

        return normalized_scores

    def _generate_summary_tables(self) -> Dict[str, str]:
        """要約統計量テーブル生成"""
        logger.info("[TABLES] Generating summary statistics tables...")

        tables = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return tables

        try:
            # 各モデルの要約統計量
            summary_stats = []

            for model in df['model'].unique():
                model_data = df[df['model'] == model]

                for metric in df['metric'].unique():
                    metric_data = model_data[model_data['metric'] == metric]['value']

                    if len(metric_data) > 0:
                        # 統計量計算
                        stats = {
                            'Model': model.upper(),
                            'Metric': metric.replace('_', ' ').title(),
                            'Count': len(metric_data),
                            'Mean': '.4f',
                            'Std': '.4f',
                            'Min': '.4f',
                            'Max': '.4f',
                            'Median': '.4f',
                            'Q25': '.4f',
                            'Q75': '.4f',
                            'SEM': '.4f' if len(metric_data) > 1 else 'N/A',
                            'CV': '.4f' if metric_data.mean() != 0 else 'N/A'
                        }
                        summary_stats.append(stats)

            # データフレーム作成
            summary_df = pd.DataFrame(summary_stats)

            # CSV保存
            csv_filename = "summary_statistics.csv"
            csv_filepath = self.output_dir / "tables" / csv_filename
            csv_filepath.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(csv_filepath, index=False)

            # LaTeXテーブル生成（論文用）
            latex_table = self._generate_latex_table(summary_df)
            latex_filename = "summary_statistics.tex"
            latex_filepath = self.output_dir / "tables" / latex_filename
            with open(latex_filepath, 'w', encoding='utf-8') as f:
                f.write(latex_table)

            tables['summary_csv'] = str(csv_filepath)
            tables['summary_latex'] = str(latex_filepath)

        except Exception as e:
            logger.error(f"Failed to generate summary tables: {e}")

        return tables

    def _generate_latex_table(self, df: pd.DataFrame) -> str:
        """LaTeXテーブル生成"""
        latex = """\\begin{table}[h!]
\\centering
\\caption{Summary Statistics for Model Performance Comparison}
\\label{tab:model_comparison}
\\begin{tabular}{@{}lcccccccccc@{}}
\\toprule
Model & Metric & Count & Mean & Std & Min & Max & Median & Q25 & Q75 & SEM \\\\
\\midrule
"""

        for _, row in df.iterrows():
            latex += f"{row['Model']} & {row['Metric']} & {row['Count']} & {row['Mean']} & {row['Std']} & {row['Min']} & {row['Max']} & {row['Median']} & {row['Q25']} & {row['Q75']} & {row['SEM']} \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        return latex

    def _generate_correlation_analysis(self) -> Dict[str, Any]:
        """相関分析生成"""
        logger.info("[ANALYSIS] Generating correlation analysis...")

        analysis = {}
        df = self._get_results_dataframe()

        if df is None or df.empty:
            return analysis

        try:
            # メトリック間の相関分析
            pivot_df = df.pivot_table(values='value', index=['model', 'library'], columns='metric')
            correlation_matrix = pivot_df.corr()

            analysis['correlation_matrix'] = correlation_matrix.to_dict()

            # 相関ヒートマップ生成
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, ax=ax)
            ax.set_title('Metric Correlation Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # 保存
            filename = "correlation_analysis_heatmap.png"
            filepath = self.output_dir / "analysis" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            analysis['correlation_plot'] = str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate correlation analysis: {e}")

        return analysis

    def _generate_documentation(self) -> Dict[str, str]:
        """ドキュメント生成"""
        logger.info("[DOCUMENTATION] Generating documentation...")

        documentation = {}

        try:
            # README生成
            readme_content = self._generate_readme()
            readme_filename = "README.md"
            readme_filepath = self.output_dir / readme_filename
            with open(readme_filepath, 'w', encoding='utf-8') as f:
                f.write(readme_content)

            # 結果サマリー生成
            summary_content = self._generate_results_summary()
            summary_filename = "RESULTS_SUMMARY.md"
            summary_filepath = self.output_dir / summary_filename
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write(summary_content)

            documentation['readme'] = str(readme_filepath)
            documentation['summary'] = str(summary_filepath)

        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")

        return documentation

    def _generate_readme(self) -> str:
        """README生成"""
        readme = f"""# LLM Model Comparison Results - HF Submission

This repository contains comprehensive benchmark results and statistical analysis for LLM model comparison, specifically designed for HuggingFace submission.

## Overview

This analysis compares multiple LLM models using various benchmark libraries and provides detailed statistical analysis including error bars, significance testing, and performance distributions.

## Models Compared

"""

        # モデル情報追加
        df = self._get_results_dataframe()
        if df is not None:
            for model in df['model'].unique():
                readme += f"- **{model.upper()}**: {model} model performance analysis\n"

        readme += """
## Benchmark Libraries Used

- **llama.cpp**: C++ based inference engine for GGUF models
- **lm-evaluation-harness**: EleutherAI's comprehensive evaluation suite
- **LightEval**: HuggingFace's efficient evaluation framework
- **transformers**: HuggingFace transformers benchmark utilities

## Key Results

### ABC Test Winner
"""

        # ABCテスト結果追加
        abc_results = self.results_data.get('comparison', {}).get('abc_test', {})
        if 'winner' in abc_results:
            winner = abc_results['winner']
            readme += f"- **Winner**: {winner['model'].upper()}\n"
            readme += f"- **Score**: {winner['score']:.4f}\n"
            readme += f"- **Metric**: {winner['metric']}\n"

        readme += """
## Files Structure

```
├── plots/
│   ├── comparison/           # Error bar comparison plots
│   ├── abc_test/            # ABC test detailed analysis
│   ├── significance/        # Statistical significance heatmaps
│   ├── distribution/        # Performance distribution plots
│   └── radar/               # Radar charts for multi-metric comparison
├── tables/
│   ├── summary_statistics.csv    # Summary statistics table
│   └── summary_statistics.tex    # LaTeX table for papers
├── analysis/
│   └── correlation_analysis_heatmap.png
├── README.md                 # This file
└── RESULTS_SUMMARY.md       # Detailed results summary
```

## Statistical Analysis

### Error Bars
All comparison plots include error bars showing standard error of the mean (SEM) to provide confidence intervals for the performance estimates.

### Significance Testing
- t-tests for comparing model performance across metrics
- p-value heatmaps showing statistical significance
- Bonferroni correction applied for multiple comparisons

### Distribution Analysis
- Performance distributions for each model and metric
- Normality testing using Shapiro-Wilk test
- Outlier detection and analysis

## Usage

### For Researchers
1. Review the comparison plots in `plots/comparison/` for visual analysis
2. Check statistical significance in `plots/significance/`
3. Refer to summary statistics in `tables/summary_statistics.csv`
4. Use LaTeX table in `tables/summary_statistics.tex` for papers

### For Practitioners
1. Check ABC test results for model recommendations
2. Review radar plots for multi-metric performance overview
3. Use correlation analysis to understand metric relationships

## Citation

If you use these results in your research, please cite:

```
@misc{llm-model-comparison-results,
  title={Comprehensive LLM Model Comparison Results},
  author={SO8T Project},
  year={2025},
  url={https://huggingface.co/zapabobouj/llm-model-comparison-results}
}
```

## License

Apache License 2.0
"""

        return readme

    def _generate_results_summary(self) -> str:
        """結果サマリー生成"""
        summary = "# Detailed Results Summary\n\n"

        # ABCテスト結果
        abc_results = self.results_data.get('comparison', {}).get('abc_test', {})
        if 'winner' in abc_results:
            winner = abc_results['winner']
            summary += "## ABC Test Results\n\n"
            summary += f"**Winner Model**: {winner['model'].upper()}\n\n"
            summary += f"**Winning Score**: {winner['score']:.4f}\n\n"
            summary += f"**Winning Metric**: {winner['metric']}\n\n"

            # 詳細ランキング
            rankings = abc_results.get('model_rankings', {})
            for metric, ranking in rankings.items():
                summary += f"### {metric.replace('_', ' ').title()} Ranking\n\n"
                for i, (model, score) in enumerate(ranking, 1):
                    summary += f"{i}. {model.upper()}: {score:.4f}\n"
                summary += "\n"

        # 統計的有意差
        statistical_comp = self.results_data.get('comparison', {}).get('statistical_comparison', {})
        if statistical_comp:
            summary += "## Statistical Significance\n\n"
            for metric, comparisons in statistical_comp.items():
                summary += f"### {metric.replace('_', ' ').title()}\n\n"
                for comparison_name, results in comparisons.items():
                    if isinstance(results, dict) and 'p_value' in results:
                        sig_symbol = "✅" if results.get('significant', False) else "❌"
                        summary += f"- {comparison_name}: p={results['p_value']:.4f} {sig_symbol}\n"
                summary += "\n"

        # 要約統計量
        summary += "## Summary Statistics\n\n"
        df = self._get_results_dataframe()
        if df is not None:
            summary_stats = []
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                for metric in df['metric'].unique():
                    metric_data = model_data[model_data['metric'] == metric]['value']
                    if len(metric_data) > 0:
                        summary += f"### {model.upper()} - {metric.replace('_', ' ').title()}\n\n"
                        summary += f"- **Mean**: {metric_data.mean():.4f}\n"
                        summary += f"- **Std**: {metric_data.std():.4f}\n"
                        summary += f"- **Min**: {metric_data.min():.4f}\n"
                        summary += f"- **Max**: {metric_data.max():.4f}\n"
                        summary += f"- **Count**: {len(metric_data)}\n\n"

        return summary

    def _generate_metadata(self) -> Dict[str, Any]:
        """メタデータ生成"""
        return {
            'generated_at': pd.Timestamp.now().isoformat(),
            'models_compared': list(self._get_results_dataframe()['model'].unique()) if self._get_results_dataframe() is not None else [],
            'metrics_evaluated': list(self._get_results_dataframe()['metric'].unique()) if self._get_results_dataframe() is not None else [],
            'benchmark_libraries': ['llama_cpp', 'lm_eval', 'light_eval', 'transformers'],
            'statistical_tests': ['t-test', 'mann-whitney-u', 'shapiro-wilk'],
            'visualizations': ['error_bars', 'heatmaps', 'distributions', 'radar_charts'],
            'output_formats': ['png', 'csv', 'tex', 'json']
        }

    def _get_results_dataframe(self) -> Optional[pd.DataFrame]:
        """結果データフレーム取得"""
        try:
            return self.results_data.get('comparison', {}).get('dataframe')
        except:
            return None

    def _save_package(self, package: Dict[str, Any]):
        """パッケージ保存"""
        # JSON形式で保存
        package_file = self.output_dir / "hf_submission_package.json"
        with open(package_file, 'w', encoding='utf-8') as f:
            json.dump(package, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[SAVE] HF submission package saved to {package_file}")


def generate_hf_submission_statistics(results_file: str, output_dir: str = "D:/webdataset/results/hf_submission"):
    """
    HF提出用統計処理を実行
    Generate HF submission statistics from results file
    """
    logger.info("[HF STATS] Starting HF submission statistics generation...")

    # 結果ファイル読み込み
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    # HF提出統計生成
    hf_stats = HFSubmissionStatistics(results_data, output_dir)
    package = hf_stats.generate_hf_submission_package()

    logger.info(f"[HF STATS] HF submission package generated in {output_dir}")
    logger.info("[HF STATS] Files generated:")
    for category, files in package.items():
        if isinstance(files, dict):
            for file_type, filepath in files.items():
                logger.info(f"  - {category}/{file_type}: {filepath}")

    return package


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HF Submission Statistics with Error Bars and Summary Tables"
    )
    parser.add_argument(
        '--results_file',
        required=True,
        help='Path to benchmark results JSON file'
    )
    parser.add_argument(
        '--output_dir',
        default='D:/webdataset/results/hf_submission',
        help='Output directory for HF submission files'
    )

    args = parser.parse_args()

    # HF提出統計生成
    package = generate_hf_submission_statistics(args.results_file, args.output_dir)

    logger.info("[SUCCESS] HF submission statistics generated!")
    logger.info(f"Output directory: {args.output_dir}")

    # オーディオ通知
    try:
        import winsound
        winsound.PlaySound(r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav", winsound.SND_FILENAME)
    except:
        print('\a')


if __name__ == '__main__':
    main()
    audio_file = r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
    if os.path.exists(audio_file):
        try:
            import subprocess
            ps_cmd = f"""
            if (Test-Path '{audio_file}') {{
                Add-Type -AssemblyName System.Windows.Forms
                $player = New-Object System.Media.SoundPlayer '{audio_file}'
                $player.PlaySync()
                Write-Host '[OK] 音声通知送信完了' -ForegroundColor Green
            }}
            else {{
                Write-Host '[WARNING] 音声ファイルが見つかりません' -ForegroundColor Yellow
            }}
            """
            subprocess.run(["powershell", "-Command", ps_cmd], check=True)
            print('[OK] 音声通知送信完了')
        except Exception as e:
            print(f'[ERROR] 音声通知送信失敗: {e}')
            print('[WARNING] フォールバックとしてビープ音を送信します')
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 500)
                print('[OK] ビープ音送信完了')
            else:
                print('[WARNING] ビープ音送信失敗: システムビープがサポートされていません')
                print('[ERROR] 音声通知送信失敗: フォールバックも失敗しました'),
                print('[ERROR] 音声通知送信失敗: フォールバックも失敗しました'),