#!/usr/bin/env python3
"""
SO(8) PPO学習済みPhi3 vs 元Phi3 A/Bテスト - lm-evaluation-harness使用
業界標準ベンチマーク + ELYZA-100 フル評価 + 統計処理 + 可視化

Model A: Borea-Phi-3.5-mini-Instruct-Jp (オリジナルHFモデル)
Model B: SO(8) PPO学習済みモデル (GGUF)
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# tqdm for progress bars
from tqdm import tqdm

# Statistics
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class LMEvalABTester:
    """lm-evaluation-harnessを使用した包括的なA/Bテスト"""

    def __init__(self, output_dir: str = "lm_eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model configurations
        self.project_root = Path(__file__).parent.parent.parent
        self.models = {
            'model_a': {
                'name': 'Borea-Phi-3.5-mini-Instruct-Jp',
                'path': self.project_root / 'models' / 'Borea-Phi-3.5-mini-Instruct-Jp',
                'type': 'hf',
                'description': 'オリジナルPhi3.5モデル'
            },
            'model_b': {
                'name': 'SO8T-Phi3.5-v2.0-BF16',
                'path': self.project_root / 'gguf_models' / 'so8t_phi35_v2' / 'SO8T-Phi3.5-v2.0-BF16.gguf',
                'tokenizer_path': self.project_root / 'models' / 'Borea-Phi-3.5-mini-Instruct-Jp',
                'type': 'hf',
                'description': 'SO(8) PPO学習済みHFモデル'
            }
        }

        # Benchmark configurations
        self.industry_benchmarks = [
            'hellaswag', 'mmlu', 'truthfulqa_mc', 'arc_challenge', 'arc_easy',
            'winogrande', 'piqa', 'sciq', 'logiqa', 'wsc'
        ]

        self.japanese_benchmarks = [
            'elyza-tasks-100'  # ELYZA-100
        ]

        # All benchmarks combined
        self.all_benchmarks = self.industry_benchmarks + self.japanese_benchmarks

    def run_single_benchmark(self, model_key: str, benchmark: str, num_runs: int = 3) -> dict:
        """単一ベンチマークを実行"""
        import subprocess
        import json
        import numpy as np

        model = self.models[model_key]
        results = []

        print(f"\n🧪 Running {benchmark} on {model['name']} ({model_key})")

        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")

            try:
                if model['type'] == 'hf':
                    # HFモデル評価
                    cmd = [
                        'python', '-m', 'lm_eval',
                        '--model', 'hf',
                        '--model_args', f'pretrained={model["path"]}',
                        '--tasks', benchmark,
                        '--device', 'cuda:0',
                        '--batch_size', 'auto',
                        '--output_path', str(self.output_dir / f'{model_key}_{benchmark}_run_{run}'),
                        '--log_samples'
                    ]

                elif model['type'] == 'gguf':
                    # GGUFモデル評価 (HF backend使用)
                    cmd = [
                        'python', '-m', 'lm_eval',
                        '--model', 'hf',
                        '--model_args', f'pretrained={model["path"].parent},gguf_file={model["path"].name},tokenizer={model["tokenizer_path"]}',
                        '--tasks', benchmark,
                        '--device', 'cuda:0',
                        '--batch_size', 'auto',
                        '--output_path', str(self.output_dir / f'{model_key}_{benchmark}_run_{run}'),
                        '--log_samples'
                    ]
                else:
                    print(f"    ❌ Unknown model type: {model['type']}")
                    results.append(None)
                    continue

                result = subprocess.run(
                    cmd,
                    cwd=self.project_root / 'lm-evaluation-harness',
                    capture_output=True, text=True, timeout=3600
                )

                if result.returncode == 0:
                    # 結果ファイルからスコアを抽出
                    output_file = self.output_dir / f'{model_key}_{benchmark}_run_{run}' / 'results.json'
                    if output_file.exists():
                        with open(output_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            score = self._extract_score(data, benchmark)
                            results.append(score)
                            print(f"    ✅ Score: {score:.3f}")
                    else:
                        print(f"    ❌ No results file found for run {run + 1}")
                        results.append(None)
                else:
                    print(f"    ❌ Run {run + 1} failed: {result.stderr[:200]}...")
                    results.append(None)

            except subprocess.TimeoutExpired:
                print(f"    ⏰ Run {run + 1} timed out")
                results.append(None)
            except Exception as e:
                print(f"    ❌ Run {run + 1} error: {e}")
                results.append(None)

        # 統計処理
        valid_results = [r for r in results if r is not None]
        if valid_results:
            stats_result = {
                'mean': float(np.mean(valid_results)),
                'std': float(np.std(valid_results)),
                'min': float(np.min(valid_results)),
                'max': float(np.max(valid_results)),
                'runs': len(valid_results),
                'total_runs': num_runs
            }
        else:
            stats_result = {'error': 'All runs failed'}

        return {
            'model': model_key,
            'benchmark': benchmark,
            'results': results,
            'statistics': stats_result
        }

    def _extract_score(self, data: Dict, benchmark: str) -> float:
        """結果データからスコアを抽出"""
        # lm-evalの結果構造に基づいてスコア抽出
        if 'results' in data:
            results = data['results']
            # ベンチマーク名に基づいて適切なスコアを取得
            if benchmark in results:
                task_results = results[benchmark]
                # acc, acc_normなどのスコアを取得
                if 'acc' in task_results:
                    return task_results['acc']
                elif 'acc_norm' in task_results:
                    return task_results['acc_norm']
                elif 'exact_match' in task_results:
                    return task_results['exact_match']
                else:
                    # 最初の数値スコアを取得
                    for key, value in task_results.items():
                        if isinstance(value, (int, float)) and key not in ['alias', 'samples']:
                            return value

        # デフォルト値
        return 0.0

    def run_all_benchmarks(self, num_runs: int = 3) -> Dict:
        """全ベンチマークを実行"""
        print("[START] Starting comprehensive A/B test with lm-evaluation-harness")
        print("=" * 80)

        all_results = {}

        # 進捗バーで実行
        total_tasks = len(self.all_benchmarks) * len(self.models) * num_runs
        with tqdm(total=total_tasks, desc="Running benchmarks") as pbar:

            for benchmark in self.all_benchmarks:
                benchmark_results = {}

                for model_key in ['model_a', 'model_b']:
                    result = self.run_single_benchmark(model_key, benchmark, num_runs)
                    benchmark_results[model_key] = result

                    # 進捗更新
                    pbar.update(num_runs)
                    pbar.set_description(f"Completed {benchmark} on {model_key}")

                all_results[benchmark] = benchmark_results

        return all_results

    def analyze_results(self, results: Dict) -> Dict:
        """統計分析を実行"""
        print("\n📊 Performing statistical analysis...")

        analysis = {
            'summary': {},
            'comparisons': {},
            'significance_tests': {}
        }

        # ベンチマークごとの分析
        for benchmark, benchmark_data in results.items():
            model_a_data = benchmark_data['model_a']
            model_b_data = benchmark_data['model_b']

            analysis['summary'][benchmark] = {
                'model_a': model_a_data['statistics'],
                'model_b': model_b_data['statistics']
            }

            # 有効な結果がある場合のみ比較
            if ('mean' in model_a_data['statistics'] and
                'mean' in model_b_data['statistics']):

                a_scores = [s for s in model_a_data['results'] if s is not None]
                b_scores = [s for s in model_b_data['results'] if s is not None]

                if len(a_scores) >= 2 and len(b_scores) >= 2:
                    # t-test
                    try:
                        t_stat, p_value = stats.ttest_ind(a_scores, b_scores, equal_var=False)

                        analysis['comparisons'][benchmark] = {
                            'mean_diff': model_b_data['statistics']['mean'] - model_a_data['statistics']['mean'],
                            'effect_size': self._calculate_effect_size(a_scores, b_scores),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except Exception as e:
                        analysis['comparisons'][benchmark] = {'error': str(e)}

        # 全体の集計
        analysis['overall'] = self._calculate_overall_stats(results)

        return analysis

    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Cohen's dを計算"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0

        return (np.mean(group2) - np.mean(group1)) / pooled_std

    def _calculate_overall_stats(self, results: Dict) -> Dict:
        """全体の統計を計算"""
        all_a_scores = []
        all_b_scores = []

        for benchmark_data in results.values():
            a_stats = benchmark_data['model_a']['statistics']
            b_stats = benchmark_data['model_b']['statistics']

            if 'mean' in a_stats:
                all_a_scores.append(a_stats['mean'])
            if 'mean' in b_stats:
                all_b_scores.append(b_stats['mean'])

        if all_a_scores and all_b_scores:
            return {
                'model_a_overall_mean': np.mean(all_a_scores),
                'model_b_overall_mean': np.mean(all_b_scores),
                'overall_improvement': np.mean(all_b_scores) - np.mean(all_a_scores),
                'benchmarks_count': len(all_a_scores)
            }
        else:
            return {'error': 'Insufficient data for overall analysis'}

    def create_visualizations(self, results: Dict, analysis: Dict):
        """結果の可視化を作成"""
        print("\n📈 Creating visualizations...")

        # スタイル設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # ベンチマーク比較グラフ
        self._create_benchmark_comparison_plot(results, analysis)

        # 統計的有意差グラフ
        self._create_significance_plot(analysis)

        # 分布比較グラフ
        self._create_distribution_plot(results)

        # 全体比較グラフ
        self._create_overall_comparison_plot(analysis)

        print(f"✅ Visualizations saved to {self.output_dir}")

    def _create_benchmark_comparison_plot(self, results: Dict, analysis: Dict):
        """ベンチマークごとの比較グラフ"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SO(8) PPO学習済みPhi3 vs オリジナルPhi3 A/Bテスト結果', fontsize=16)

        # データ準備
        benchmarks = list(results.keys())
        a_means = []
        b_means = []
        a_stds = []
        b_stds = []

        for benchmark in benchmarks:
            a_stats = results[benchmark]['model_a']['statistics']
            b_stats = results[benchmark]['model_b']['statistics']

            a_means.append(a_stats.get('mean', 0))
            b_means.append(b_stats.get('mean', 0))
            a_stds.append(a_stats.get('std', 0))
            b_stds.append(b_stats.get('std', 0))

        x = np.arange(len(benchmarks))

        # 個別ベンチマーク比較
        axes[0, 0].bar(x - 0.2, a_means, 0.4, label='Model A (オリジナル)', alpha=0.8,
                      yerr=a_stds, capsize=5, color='skyblue')
        axes[0, 0].bar(x + 0.2, b_means, 0.4, label='Model B (SO(8)学習済み)', alpha=0.8,
                      yerr=b_stds, capsize=5, color='lightcoral')
        axes[0, 0].set_title('各ベンチマークの性能比較')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(benchmarks, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 差分グラフ
        diffs = np.array(b_means) - np.array(a_means)
        colors = ['red' if d < 0 else 'green' for d in diffs]
        axes[0, 1].bar(benchmarks, diffs, color=colors, alpha=0.7)
        axes[0, 1].set_title('Model B - Model A (差分)')
        axes[0, 1].set_xticklabels(benchmarks, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Accuracy差分')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)

        # 効果量グラフ
        effect_sizes = []
        for benchmark in benchmarks:
            if benchmark in analysis.get('comparisons', {}):
                comp = analysis['comparisons'][benchmark]
                effect_sizes.append(comp.get('effect_size', 0))
            else:
                effect_sizes.append(0)

        axes[1, 0].bar(benchmarks, effect_sizes, color='purple', alpha=0.7)
        axes[1, 0].set_title('効果量 (Cohen\'s d)')
        axes[1, 0].set_xticklabels(benchmarks, rotation=45, ha='right')
        axes[1, 0].set_ylabel('効果量')
        axes[1, 0].axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='小効果')
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='中効果')
        axes[1, 0].axhline(y=0.8, color='darkred', linestyle='--', alpha=0.7, label='大効果')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 有意差のヒートマップ
        significance_matrix = np.zeros((len(benchmarks), 2))
        for i, benchmark in enumerate(benchmarks):
            if benchmark in analysis.get('comparisons', {}):
                comp = analysis['comparisons'][benchmark]
                significance_matrix[i, 0] = comp.get('p_value', 1.0)
                significance_matrix[i, 1] = comp.get('significant', False)

        sns.heatmap(significance_matrix[:, [1]], ax=axes[1, 1], cmap='RdYlGn_r',
                   cbar_kws={'label': '有意差 (p < 0.05)'})
        axes[1, 1].set_title('統計的有意差')
        axes[1, 1].set_yticklabels(benchmarks)
        axes[1, 1].set_xticklabels(['有意差'])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_significance_plot(self, analysis: Dict):
        """統計的有意差の可視化"""
        if 'comparisons' not in analysis:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        benchmarks = []
        p_values = []
        effect_sizes = []

        for benchmark, comp in analysis['comparisons'].items():
            if 'p_value' in comp:
                benchmarks.append(benchmark)
                p_values.append(comp['p_value'])
                effect_sizes.append(comp.get('effect_size', 0))

        x = np.arange(len(benchmarks))
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]

        bars = ax.bar(x, [-np.log10(p) if p > 0 else 50 for p in p_values],
                     color=colors, alpha=0.7)

        ax.set_title('統計的有意差 (-log10(p-value))', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.set_ylabel('-log10(p-value)')
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7,
                  label='p = 0.05 有意水準')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # p値の注釈
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   '.2e', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'significance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_distribution_plot(self, results: Dict):
        """スコア分布の比較"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('スコア分布比較', fontsize=16)

        subplot_idx = 0
        for benchmark, benchmark_data in list(results.items())[:4]:  # 最初の4つだけ
            ax = axes[subplot_idx // 2, subplot_idx % 2]

            a_results = [s for s in benchmark_data['model_a']['results'] if s is not None]
            b_results = [s for s in benchmark_data['model_b']['results'] if s is not None]

            if a_results and b_results:
                ax.hist(a_results, alpha=0.7, label='Model A', bins=10, density=True, color='skyblue')
                ax.hist(b_results, alpha=0.7, label='Model B', bins=10, density=True, color='lightcoral')

                # KDEプロット
                if len(a_results) > 1:
                    try:
                        sns.kdeplot(a_results, ax=ax, color='blue', alpha=0.5)
                    except:
                        pass
                if len(b_results) > 1:
                    try:
                        sns.kdeplot(b_results, ax=ax, color='red', alpha=0.5)
                    except:
                        pass

            ax.set_title(f'{benchmark}')
            ax.set_xlabel('Accuracy')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

            subplot_idx += 1

        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_overall_comparison_plot(self, analysis: Dict):
        """全体比較グラフ"""
        if 'overall' not in analysis or 'model_a_overall_mean' not in analysis['overall']:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        overall = analysis['overall']
        models = ['Model A\n(オリジナル)', 'Model B\n(SO(8)学習済み)']
        scores = [overall['model_a_overall_mean'], overall['model_b_overall_mean']]
        improvement = overall.get('overall_improvement', 0)

        bars = ax.bar(models, scores, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax.set_title('全体性能比較', fontsize=14)
        ax.set_ylabel('平均Accuracy')
        ax.set_ylim(0, max(scores) * 1.2)
        ax.grid(True, alpha=0.3)

        # 値の注釈
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   '.4f', ha='center', va='bottom', fontweight='bold')

        # 改善率の注釈
        if improvement != 0:
            ax.text(1, max(scores) * 1.1,
                   f'改善率: {improvement:+.4f} ({improvement*100:+.1f}%)',
                   ha='center', va='bottom', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, results: Dict, analysis: Dict):
        """結果を保存"""
        print("💾 Saving results...")

        # JSON形式で保存
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models': self.models,
                'benchmarks': self.all_benchmarks,
                'description': 'SO(8) PPO学習済みPhi3 vs オリジナルPhi3 A/Bテスト'
            },
            'results': results,
            'analysis': analysis
        }

        with open(self.output_dir / 'ab_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Markdownレポート生成
        self._generate_markdown_report(results, analysis)

        print(f"✅ Results saved to {self.output_dir}")

    def _generate_markdown_report(self, results: Dict, analysis: Dict):
        """Markdownレポート生成"""
        report_path = self.output_dir / 'ab_test_report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# SO(8) PPO学習済みPhi3 vs オリジナルPhi3 A/Bテストレポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 概要
            f.write("## テスト概要\n\n")
            f.write("- **Model A**: Borea-Phi-3.5-mini-Instruct-Jp (オリジナル)\n")
            f.write("- **Model B**: SO8T-Phi3.5-v2.0-BF16 (SO(8) PPO学習済み)\n")
            f.write(f"- **ベンチマーク数**: {len(self.all_benchmarks)}\n")
            f.write("- **評価ツール**: lm-evaluation-harness\n\n")

            # 全体結果
            if 'overall' in analysis:
                overall = analysis['overall']
                f.write("## 全体結果\n\n")
                f.write(f"- **Model A平均スコア**: {overall.get('model_a_overall_mean', 'N/A'):.4f}\n")
                f.write(f"- **Model B平均スコア**: {overall.get('model_b_overall_mean', 'N/A'):.4f}\n")
                f.write(f"- **改善率**: {overall.get('overall_improvement', 0):+.4f} ({overall.get('overall_improvement', 0)*100:+.1f}%)\n\n")

            # 個別ベンチマーク結果
            f.write("## 個別ベンチマーク結果\n\n")
            f.write("| ベンチマーク | Model A | Model B | 差分 | 有意差 |\n")
            f.write("|-------------|---------|---------|------|--------|\n")

            for benchmark in self.all_benchmarks:
                if benchmark in results:
                    a_stats = results[benchmark]['model_a']['statistics']
                    b_stats = results[benchmark]['model_b']['statistics']

                    a_mean = a_stats.get('mean', 'N/A')
                    b_mean = b_stats.get('mean', 'N/A')

                    if isinstance(a_mean, (int, float)) and isinstance(b_mean, (int, float)):
                        diff = b_mean - a_mean
                        diff_str = f"+{b_mean - a_mean:.4f}"                        
                        diff_str = "N/A"

                    # 有意差チェック
                    if benchmark in analysis.get('comparisons', {}):
                        comp = analysis['comparisons'][benchmark]
                        sig = "✅" if comp.get('significant', False) else "❌"
                    else:
                        sig = "N/A"

                    f.write(f"| {benchmark} | {a_mean} | {b_mean} | {diff_str} | {sig} |\n")

            f.write("\n## 統計的有意差詳細\n\n")
            if 'comparisons' in analysis:
                for benchmark, comp in analysis['comparisons'].items():
                    if 'p_value' in comp:
                        f.write(f"### {benchmark}\n")
                        f.write(f"- p値: {comp['p_value']:.4f}\n")
                        f.write(f"- t統計量: {comp.get('t_statistic', 'N/A')}\n")
                        f.write(f"- 効果量: {comp.get('effect_size', 'N/A'):.3f}\n")
                        f.write(f"- 有意差: {'あり' if comp.get('significant', False) else 'なし'}\n\n")

            # 結論
            f.write("## 結論\n\n")
            if 'overall' in analysis and 'overall_improvement' in analysis['overall']:
                improvement = analysis['overall']['overall_improvement']
                if improvement > 0:
                    f.write("🎉 **SO(8)学習により全体的な性能向上が確認されました！**\n\n")
                elif improvement < 0:
                    f.write("📊 **SO(8)学習による性能改善は確認されませんでした。**\n\n")
                else:
                    f.write("⚖️ **SO(8)学習による顕著な性能変化は確認されませんでした。**\n\n")
            else:
                f.write("❓ **結果の分析に十分なデータが得られませんでした。**\n\n")

            f.write("詳細な結果とグラフは同ディレクトリのファイルをご参照ください。\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SO(8) PPO学習済みPhi3 vs 元Phi3 A/Bテスト")
    parser.add_argument("--output_dir", type=str, default="lm_eval_ab_test_results",
                       help="出力ディレクトリ")
    parser.add_argument("--num_runs", type=int, default=3,
                       help="各ベンチマークの実行回数")
    parser.add_argument("--benchmarks", type=str, nargs='*',
                       help="実行するベンチマーク（指定なしで全ベンチマーク）")

    args = parser.parse_args()

    # A/Bテスター初期化
    tester = LMEvalABTester(args.output_dir)

    # ベンチマーク選択
    if args.benchmarks:
        tester.all_benchmarks = args.benchmarks

    print(f"🐾 Starting SO(8) A/B test with {len(tester.all_benchmarks)} benchmarks")
    print(f"📁 Output directory: {tester.output_dir}")
    print(f"🔄 Runs per benchmark: {args.num_runs}")

    # 全ベンチマーク実行
    results = tester.run_all_benchmarks(args.num_runs)

    # 統計分析
    analysis = tester.analyze_results(results)

    # 可視化作成
    tester.create_visualizations(results, analysis)

    # 結果保存
    tester.save_results(results, analysis)

    print("\n🎯 A/Bテスト完了！")
    print(f"📊 結果は {tester.output_dir} に保存されました")

    # 全体結果表示
    if 'overall' in analysis:
        overall = analysis['overall']
        print("\n🏆 最終結果:")
        if 'overall_a_mean' in overall:
            print(f"Model A (mean): {overall['overall_a_mean']:.4f}")
        if 'overall_b_mean' in overall:
            print(f"Model B (mean): {overall['overall_b_mean']:.4f}")
        if 'overall_improvement' in overall:
            imp = overall['overall_improvement']
            print(f"改善値: {imp:.1f}")

if __name__ == "__main__":
    main()
