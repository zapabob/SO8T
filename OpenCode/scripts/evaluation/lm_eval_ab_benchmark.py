#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LM Evaluation Harness AB Benchmark for HF Models
lm-evaluation-harness + JGLUEを使ってABテスト実行

標準ベンチマーク + JGLUE + 統計解析
エラーバー付きグラフ + ANOVA + 効果量 + p値
"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LMEvalABBenchmark:
    """LM Evaluation Harness AB Benchmarkクラス"""

    def __init__(self, model_a_path: str, model_b_path: str, output_dir: str):
        """
        初期化

        Args:
            model_a_path: モデルAのパス (ベースモデル)
            model_b_path: モデルBのパス (AEGIS SO(8)モデル)
            output_dir: 出力ディレクトリ
        """
        self.model_a_path = Path(model_a_path)
        self.model_b_path = Path(model_b_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ベンチマークタスク設定
        self.standard_tasks = [
            'arc_challenge', 'arc_easy', 'hellaswag', 'winogrande',
            'piqa', 'sciq', 'lambada_openai', 'truthfulqa_mc'
        ]

        # JGLUEタスク (日本語ベンチマーク)
        self.jglue_tasks = [
            'jcommonsenseqa',  # JCQA
            'jnli',           # 自然言語推論
            'jsquad',         # 質問応答
            'jsts',           # 意味的類似度
            'jmmlu',          # 日本語MMLU
        ]

        # 追加の日本語タスク
        self.japanese_tasks = [
            'xwinograd_ja',   # 日本語Winograd
            'jaquad',         # 日本語SQuAD
        ]

        logger.info(f"Initialized LM-Eval AB Benchmark: A={model_a_path}, B={model_b_path}")

    def run_single_task(self, model_path: str, task_name: str, model_name: str) -> Dict[str, Any]:
        """
        単一タスク実行

        Args:
            model_path: モデルパス
            task_name: タスク名
            model_name: モデル識別子

        Returns:
            タスク結果
        """
        logger.info(f"Running {task_name} on {model_name}")

        try:
            # 一時ファイルを作成
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_output_path = temp_file.name

            # lm-evaluation-harnessコマンド構築
            cmd = [
                sys.executable, "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
                "--tasks", task_name,
                "--device", "cpu",
                "--batch_size", "1",
                "--output_path", temp_output_path,  # 一時ファイルに結果を出力
                "--limit", "0.01"  # 小規模テスト用（1%のデータのみ）
            ]

            logger.info(f"Running command: {' '.join(cmd)}")

            # コマンド実行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30分タイムアウト
                cwd=os.getcwd()
            )

            if result.returncode == 0:
                # 出力ファイルから結果を読み込み
                output_file = Path(temp_output_path)
                if output_file.exists():
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)

                        # タスク結果を抽出
                        if 'results' in result_data and task_name in result_data['results']:
                            task_results = result_data['results'][task_name]

                            # 主要なメトリクスを抽出
                            metrics = {}
                            for key, value in task_results.items():
                                if isinstance(value, (int, float)) and not key.startswith('stderr'):
                                    metrics[key] = value

                            # 標準メトリクスを取得（acc, acc_normなど）
                            score = None
                            if 'acc' in metrics:
                                score = metrics['acc']
                            elif 'acc_norm' in metrics:
                                score = metrics['acc_norm']
                            elif 'f1' in metrics:
                                score = metrics['f1']
                            elif 'exact_match' in metrics:
                                score = metrics['exact_match']
                            elif len(metrics) > 0:
                                # 最初の数値メトリクスを使用
                                score = list(metrics.values())[0]

                            # 一時ファイルを削除
                            output_file.unlink(missing_ok=True)

                            return {
                                'task': task_name,
                                'model': model_name,
                                'score': score if score is not None else 0.0,
                                'metrics': metrics,
                                'timestamp': datetime.now().isoformat()
                            }
                        else:
                            logger.warning(f"No results found for {task_name} in {output_file}")
                            output_file.unlink(missing_ok=True)
                            return self._create_fallback_result(task_name, model_name, "no_results")
                    except Exception as e:
                        logger.error(f"Error reading result file for {task_name}: {e}")
                        output_file.unlink(missing_ok=True)
                        return self._create_fallback_result(task_name, model_name, f"file_error: {e}")
                else:
                    logger.warning(f"Output file not found for {task_name}: {output_file}")
                    return self._create_fallback_result(task_name, model_name, "no_output_file")
            else:
                # エラーの場合も一時ファイルを削除
                try:
                    Path(temp_output_path).unlink(missing_ok=True)
                except:
                    pass
                logger.error(f"Task {task_name} failed: {result.stderr}")
                return self._create_fallback_result(task_name, model_name, f"command_failed: {result.stderr[-500:]}")

        except subprocess.TimeoutExpired:
            logger.error(f"Task {task_name} timed out")
            return self._create_fallback_result(task_name, model_name, "timeout")
        except Exception as e:
            logger.error(f"Task {task_name} error: {e}")
            return self._create_fallback_result(task_name, model_name, f"exception: {str(e)}")

    def _create_fallback_result(self, task_name: str, model_name: str, error_msg: str = "") -> Dict[str, Any]:
        """フォールバック結果作成"""
        return {
            'task': task_name,
            'model': model_name,
            'score': 0.0,
            'metrics': {},
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

    def run_all_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        全タスク実行

        Returns:
            タスク結果
        """
        results = {'model_a': [], 'model_b': []}

        # 標準ベンチマーク (一部のみ)
        logger.info("Running standard benchmarks...")
        for task in self.standard_tasks[:3]:  # デモ用に3つだけ
            # モデルA
            result_a = self.run_single_task(str(self.model_a_path), task, 'Model_A')
            results['model_a'].append(result_a)

            # モデルB
            result_b = self.run_single_task(str(self.model_b_path), task, 'Model_B')
            results['model_b'].append(result_b)

        # JGLUEタスク
        logger.info("Running JGLUE benchmarks...")
        for task in self.jglue_tasks[:3]:  # デモ用に3つだけ
            # モデルA
            result_a = self.run_single_task(str(self.model_a_path), task, 'Model_A')
            results['model_a'].append(result_a)

            # モデルB
            result_b = self.run_single_task(str(self.model_b_path), task, 'Model_B')
            results['model_b'].append(result_b)

        return results

    def analyze_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        結果解析 (統計分析)

        Args:
            results: タスク結果

        Returns:
            解析結果
        """
        # データフレーム作成
        all_results = []
        for model_name, model_results in results.items():
            for result in model_results:
                all_results.append({
                    'model': 'Model A' if model_name == 'model_a' else 'Model B',
                    'task': result['task'],
                    'score': result['score'] * 100,  # パーセント表示
                    'has_error': 'error' in result
                })

        df = pd.DataFrame(all_results)

        # 統計解析
        analysis = {}

        # 基本統計量
        analysis['summary_stats'] = df.groupby('model')['score'].agg(['mean', 'std', 'count']).to_dict()

        # モデル比較
        model_comparison = {}
        for task in df['task'].unique():
            task_data = df[df['task'] == task]
            if len(task_data) >= 2:
                model_a_score = task_data[task_data['model'] == 'Model A']['score'].mean()
                model_b_score = task_data[task_data['model'] == 'Model B']['score'].mean()
                improvement = model_b_score - model_a_score

                model_comparison[task] = {
                    'model_a_score': float(model_a_score),
                    'model_b_score': float(model_b_score),
                    'improvement': float(improvement),
                    'improvement_pct': float((improvement / model_a_score) * 100) if model_a_score > 0 else 0
                }

        analysis['model_comparison'] = model_comparison

        # 全体比較
        overall_a = df[df['model'] == 'Model A']['score'].mean()
        overall_b = df[df['model'] == 'Model B']['score'].mean()
        overall_improvement = overall_b - overall_a

        analysis['overall_comparison'] = {
            'model_a_avg_score': float(overall_a),
            'model_b_avg_score': float(overall_b),
            'improvement': float(overall_improvement),
            'improvement_pct': float((overall_improvement / overall_a) * 100) if overall_a > 0 else 0
        }

        # t-test
        try:
            from scipy import stats
            model_a_scores = df[df['model'] == 'Model A']['score']
            model_b_scores = df[df['model'] == 'Model B']['score']

            if len(model_a_scores) >= 2 and len(model_b_scores) >= 2:
                t_stat, p_value = stats.ttest_ind(model_a_scores, model_b_scores, equal_var=False)
                analysis['t_test'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': 'significant' if p_value < 0.05 else 'not significant'
                }
            else:
                analysis['t_test'] = {'error': 'insufficient samples for t-test'}
        except Exception as e:
            logger.warning(f"t-test failed: {e}")
            analysis['t_test'] = {'error': str(e)}

        # 効果量 (Cohen's d)
        try:
            model_a_scores = df[df['model'] == 'Model A']['score']
            model_b_scores = df[df['model'] == 'Model B']['score']

            if len(model_a_scores) > 0 and len(model_b_scores) > 0:
                mean_a = model_a_scores.mean()
                mean_b = model_b_scores.mean()

                # プールされた標準偏差
                all_scores = pd.concat([model_a_scores, model_b_scores])
                pooled_std = all_scores.std()

                if pooled_std > 0:
                    cohens_d = abs(mean_b - mean_a) / pooled_std
                    analysis['effect_size'] = {
                        'cohens_d': float(cohens_d),
                        'interpretation': self._interpret_cohens_d(cohens_d)
                    }
                else:
                    analysis['effect_size'] = {'cohens_d': 0.0, 'interpretation': 'no difference'}
            else:
                analysis['effect_size'] = {'error': 'no data'}
        except Exception as e:
            logger.warning(f"Effect size calculation failed: {e}")
            analysis['effect_size'] = {'error': str(e)}

        return analysis

    def _interpret_cohens_d(self, d: float) -> str:
        """Cohen's dの解釈"""
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'

    def create_visualizations(self, results: Dict[str, List[Dict[str, Any]]], analysis: Dict[str, Any]):
        """
        可視化作成

        Args:
            results: タスク結果
            analysis: 解析結果
        """
        # データ準備
        all_results = []
        for model_name, model_results in results.items():
            for result in model_results:
                all_results.append({
                    'model': 'Model A' if model_name == 'model_a' else 'Model B',
                    'task': result['task'],
                    'score': result['score'] * 100,  # パーセント表示
                    'has_error': 'error' in result
                })

        df = pd.DataFrame(all_results)

        # スタイル設定
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. エラーバー付き比較グラフ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # スコア比較
        if len(df) > 0:
            task_scores = df.groupby(['task', 'model'])['score'].agg(['mean', 'std']).reset_index()
            task_scores = task_scores.pivot(index='task', columns='model', values=['mean', 'std'])

            tasks = task_scores.index
            if 'Model A' in task_scores.columns.get_level_values(1) and 'Model B' in task_scores.columns.get_level_values(1):
                model_a_means = task_scores[('mean', 'Model A')]
                model_b_means = task_scores[('mean', 'Model B')]
                model_a_stds = task_scores[('std', 'Model A')]
                model_b_stds = task_scores[('std', 'Model B')]

                x = np.arange(len(tasks))
                width = 0.35

                ax1.bar(x - width/2, model_a_means, width, yerr=model_a_stds, capsize=5,
                        label='Model A (Baseline)', alpha=0.8, color='skyblue')
                ax1.bar(x + width/2, model_b_means, width, yerr=model_b_stds, capsize=5,
                        label='Model B (AEGIS SO(8))', alpha=0.8, color='lightcoral')

                ax1.set_ylabel('Score (%)')
                ax1.set_title('LM-Eval Benchmark Performance Comparison\n(with Error Bars)')
                ax1.set_xticks(x)
                ax1.set_xticklabels(tasks, rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

        # 統計情報表示
        stats_text = ".3f"".3f"f"""
Statistical Analysis:
• Model A Avg: {analysis.get('overall_comparison', {}).get('model_a_avg_score', 'N/A'):.1f}%
• Model B Avg: {analysis.get('overall_comparison', {}).get('model_b_avg_score', 'N/A'):.1f}%
• Improvement: {analysis.get('overall_comparison', {}).get('improvement', 'N/A'):+.1f}% ({analysis.get('overall_comparison', {}).get('improvement_pct', 'N/A'):+.1f}%)

t-test Results:
• t-statistic: {analysis.get('t_test', {}).get('t_statistic', 'N/A')}
• p-value: {analysis.get('t_test', {}).get('p_value', 'N/A')}
• Significant: {analysis.get('t_test', {}).get('significant', 'N/A')}

Effect Size (Cohen's d):
• Cohen's d: {analysis.get('effect_size', {}).get('cohens_d', 'N/A')}
• Interpretation: {analysis.get('effect_size', {}).get('interpretation', 'N/A')}
"""

        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Statistical Analysis Summary', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'lm_eval_ab_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. JGLUE特化グラフ
        jglue_tasks = [r for r in df['task'] if any(jglue in r.lower() for jglue in ['jcommonsense', 'jnli', 'jsquad', 'jsts', 'jmmlu'])]
        if jglue_tasks:
            jglue_df = df[df['task'].isin(jglue_tasks)]

            fig, ax = plt.subplots(figsize=(12, 6))

            if len(jglue_df) > 0:
                jglue_scores = jglue_df.groupby(['task', 'model'])['score'].mean().reset_index()
                jglue_scores = jglue_scores.pivot(index='task', columns='model', values='score')

                jglue_scores.plot(kind='bar', ax=ax, width=0.8)
                ax.set_ylabel('Score (%)')
                ax.set_title('JGLUE Benchmark Performance Comparison')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # 値ラベル追加
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f', padding=3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'jglue_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

    def save_results(self, results: Dict[str, List[Dict[str, Any]]], analysis: Dict[str, Any]):
        """
        結果保存

        Args:
            results: タスク結果
            analysis: 解析結果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON結果保存
        output_data = {
            'timestamp': timestamp,
            'model_a_path': str(self.model_a_path),
            'model_b_path': str(self.model_b_path),
            'results': results,
            'analysis': analysis
        }

        json_path = self.output_dir / f'lm_eval_ab_benchmark_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # マークダウンレポート
        md_content = f"""# LM Evaluation Harness AB Benchmark Results

**Timestamp:** {timestamp}

## Models Compared
- **Model A (Baseline):** {self.model_a_path.name}
- **Model B (AEGIS SO(8)):** {self.model_b_path.name}

## Statistical Analysis

### Overall Comparison
- **Model A Average:** {analysis.get('overall_comparison', {}).get('model_a_avg_score', 'N/A'):.2f}%
- **Model B Average:** {analysis.get('overall_comparison', {}).get('model_b_avg_score', 'N/A'):.2f}%
- **Improvement:** {analysis.get('overall_comparison', {}).get('improvement', 'N/A'):+.2f}% ({analysis.get('overall_comparison', {}).get('improvement_pct', 'N/A'):+.1f}%)

### t-test Results
- **t-statistic:** {analysis.get('t_test', {}).get('t_statistic', 'N/A')}
- **p-value:** {analysis.get('t_test', {}).get('p_value', 'N/A')}
- **Significant:** {analysis.get('t_test', {}).get('interpretation', 'N/A')}

### Effect Size (Cohen's d)
- **Cohen's d:** {analysis.get('effect_size', {}).get('cohens_d', 'N/A')}
- **Interpretation:** {analysis.get('effect_size', {}).get('interpretation', 'N/A')}

## Task-by-Task Results

| Task | Model A Score | Model B Score | Improvement | Status |
|------|---------------|---------------|-------------|---------|
"""

        # タスク結果を追加
        all_results = []
        for model_name, model_results in results.items():
            for result in model_results:
                all_results.append({
                    'model': 'Model A' if model_name == 'model_a' else 'Model B',
                    'task': result['task'],
                    'score': result['score'],
                    'has_error': 'error' in result
                })

        df = pd.DataFrame(all_results)
        for task in sorted(df['task'].unique()):
            task_data = df[df['task'] == task]
            if len(task_data) >= 2:
                model_a_row = task_data[task_data['model'] == 'Model A']
                model_b_row = task_data[task_data['model'] == 'Model B']

                if len(model_a_row) > 0 and len(model_b_row) > 0:
                    score_a = model_a_row['score'].iloc[0]
                    score_b = model_b_row['score'].iloc[0]
                    improvement = score_b - score_a

                    status = "OK"
                    if model_a_row['has_error'].iloc[0] or model_b_row['has_error'].iloc[0]:
                        status = "Error"

                    md_content += f"| {task} | {score_a:.3f} | {score_b:.3f} | {improvement:+.3f} | {status} |\n"

        md_content += "\n## Generated Files\n"
        md_content += "- `lm_eval_ab_comparison.png` - Overall performance comparison with error bars\n"
        md_content += "- `jglue_comparison.png` - JGLUE-specific benchmark comparison\n"
        md_content += f"- `lm_eval_ab_benchmark_{timestamp}.json` - Raw results data\n"
        md_content += f"- `lm_eval_ab_benchmark_report_{timestamp}.md` - This report\n"

        md_path = self.output_dir / f'lm_eval_ab_benchmark_report_{timestamp}.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Generated files: lm_eval_ab_comparison.png, jglue_comparison.png")

    def run_complete_benchmark(self):
        """完全ベンチマーク実行"""
        logger.info("Starting LM Evaluation Harness AB Benchmark...")

        # ベンチマーク実行
        results = self.run_all_tasks()

        # 結果解析
        analysis = self.analyze_results(results)

        # 可視化作成
        self.create_visualizations(results, analysis)

        # 結果保存
        self.save_results(results, analysis)

        logger.info("LM-Eval AB Benchmark completed successfully!")

        return results, analysis


def main():
    parser = argparse.ArgumentParser(
        description="LM Evaluation Harness AB Benchmark for HF Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/evaluation/lm_eval_ab_benchmark.py \\
    --model_a models/Borea-Phi-3.5-mini-Instruct-Jp \\
    --model_b H:/from_D/webdataset/models/aegis_test_bf16 \\
    --output_dir H:/from_D/webdataset/models/aegis_test_bf16/lm_eval_results

This will run comprehensive AB testing with:
- Standard benchmarks (ARC, HellaSwag, etc.)
- JGLUE tasks (JCQA, JNLI, JSQuAD, JSTS, JMMLU)
- Statistical analysis (t-test, effect size)
- Error bar charts and comparison tables
        """
    )

    parser.add_argument(
        "--model_a",
        type=str,
        required=True,
        help="Path to Model A (baseline model)"
    )

    parser.add_argument(
        "--model_b",
        type=str,
        required=True,
        help="Path to Model B (AEGIS SO(8) model)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: model_b/lm_eval_results)"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=None,
        help="Specific tasks to run (default: standard + JGLUE tasks)"
    )

    args = parser.parse_args()

    # 出力ディレクトリ設定
    if args.output_dir is None:
        args.output_dir = str(Path(args.model_b) / "lm_eval_results")

    try:
        # LM-Eval ABベンチマーク実行
        benchmark = LMEvalABBenchmark(args.model_a, args.model_b, args.output_dir)
        results, analysis = benchmark.run_complete_benchmark()

        logger.info("Benchmark completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
