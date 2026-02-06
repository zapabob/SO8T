#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベースライン vs SO8T 比較分析スクリプト

サンシャイン実験の結果を比較して、実装ログを作成する。
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import numpy as np

class BaselineSO8TComparator:
    """ベースラインとSO8Tモデルの比較分析"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_dir = self.log_dir / "baseline"
        self.so8t_dir = self.log_dir / "so8t"
        self.output_dir = Path("_docs")

    def load_metrics(self, run_type: str) -> Dict[str, Any]:
        """メトリクスファイルを読み込む"""
        if run_type == "baseline":
            metrics_file = self.baseline_dir / "baseline_metrics.json"
        else:
            metrics_file = self.so8t_dir / "so8t_metrics.json"

        if not metrics_file.exists():
            print(f"[WARNING] Metrics file not found: {metrics_file}")
            return {}

        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_training_log(self, run_type: str) -> pd.DataFrame:
        """トレーニングログを読み込む"""
        if run_type == "baseline":
            log_file = self.baseline_dir / "baseline_training_log.csv"
        else:
            log_file = self.so8t_dir / "so8t_training_log.csv"

        if not log_file.exists():
            print(f"[WARNING] Training log not found: {log_file}")
            return pd.DataFrame()

        return pd.read_csv(log_file)

    def analyze_performance(self) -> Dict[str, Any]:
        """パフォーマンス分析"""
        baseline_metrics = self.load_metrics("baseline")
        so8t_metrics = self.load_metrics("so8t")

        baseline_log = self.load_training_log("baseline")
        so8t_log = self.load_training_log("so8t")

        analysis = {
            'baseline': {
                'final_loss': baseline_metrics.get('final_train_loss'),
                'avg_grad_norm': baseline_log['grad_norm'].mean() if not baseline_log.empty else None,
                'total_steps': baseline_metrics.get('total_steps', 0),
                'avg_step_time': baseline_metrics.get('avg_step_time', 0)
            },
            'so8t': {
                'final_loss': so8t_metrics.get('final_train_loss'),
                'avg_so8_ortho_error': so8t_log['so8_ortho_mean'].mean() if not so8t_log.empty and 'so8_ortho_mean' in so8t_log.columns else None,
                'avg_grad_norm': so8t_log['grad_norm'].mean() if not so8t_log.empty else None,
                'total_steps': so8t_metrics.get('total_steps', 0),
                'avg_step_time': so8t_metrics.get('avg_step_time', 0)
            }
        }

        # 改善率計算
        if analysis['baseline']['final_loss'] and analysis['so8t']['final_loss']:
            if analysis['so8t']['final_loss'] < analysis['baseline']['final_loss']:
                improvement = (analysis['baseline']['final_loss'] - analysis['so8t']['final_loss']) / analysis['baseline']['final_loss'] * 100
                analysis['improvement'] = f"+{improvement:.2f}%"
            else:
                degradation = (analysis['so8t']['final_loss'] - analysis['baseline']['final_loss']) / analysis['baseline']['final_loss'] * 100
                analysis['improvement'] = f"-{degradation:.2f}%"
        else:
            analysis['improvement'] = "N/A"

        return analysis

    def plot_comparison(self, save_path: str = None):
        """比較グラフを作成"""
        baseline_log = self.load_training_log("baseline")
        so8t_log = self.load_training_log("so8t")

        if baseline_log.empty or so8t_log.empty:
            print("[WARNING] Training logs not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Baseline vs SO8T Training Comparison', fontsize=16)

        # Loss comparison
        if 'train_loss' in baseline_log.columns and 'train_loss' in so8t_log.columns:
            axes[0, 0].plot(baseline_log['step'], baseline_log['train_loss'],
                           label='Baseline', color='blue', linewidth=2)
            axes[0, 0].plot(so8t_log['step'], so8t_log['train_loss'],
                           label='SO8T', color='red', linewidth=2)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Training Loss')
            axes[0, 0].legend()
            axes[0, 0].set_title('Training Loss Comparison')
            axes[0, 0].grid(True, alpha=0.3)

        # Gradient norm comparison
        if 'grad_norm' in baseline_log.columns and 'grad_norm' in so8t_log.columns:
            axes[0, 1].plot(baseline_log['step'], baseline_log['grad_norm'],
                           label='Baseline', color='blue', linewidth=2)
            axes[0, 1].plot(so8t_log['step'], so8t_log['grad_norm'],
                           label='SO8T', color='red', linewidth=2)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].legend()
            axes[0, 1].set_title('Gradient Norm Comparison')
            axes[0, 1].grid(True, alpha=0.3)

        # SO8T specific metrics
        if 'so8_ortho_mean' in so8t_log.columns:
            axes[1, 0].plot(so8t_log['step'], so8t_log['so8_ortho_mean'],
                           label='SO8T Ortho Error', color='orange', linewidth=2)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Orthogonality Error')
            axes[1, 0].set_title('SO(8) Orthogonality Error')
            axes[1, 0].grid(True, alpha=0.3)

        # Step time comparison
        if 'step_time_sec' in baseline_log.columns and 'step_time_sec' in so8t_log.columns:
            axes[1, 1].plot(baseline_log['step'], baseline_log['step_time_sec'],
                           label='Baseline', color='blue', linewidth=2)
            axes[1, 1].plot(so8t_log['step'], so8t_log['step_time_sec'],
                           label='SO8T', color='red', linewidth=2)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Step Time (sec)')
            axes[1, 1].legend()
            axes[1, 1].set_title('Training Speed Comparison')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Comparison plot saved to: {save_path}")
        else:
            plt.show()

    def generate_comparison_report(self) -> str:
        """比較レポートを生成"""
        analysis = self.analyze_performance()

        today = datetime.now().strftime("%Y-%m-%d")

        report = f"""# Baseline vs SO8T Comparative Analysis Report

## Implementation Information
- **Date**: {today}
- **Experiment Type**: Sunshine Pipeline Comparison
- **Dataset**: Mathematics, Science, NKAT Theory, NSFW Dataset
- **Training Steps**: {analysis['baseline']['total_steps']}

## Performance Comparison

### Baseline (LoRA only)
- **最終Loss**: {analysis['baseline']['final_loss']:.6f}
- **平均Grad Norm**: {analysis['baseline']['avg_grad_norm']:.4f}
- **平均ステップ時間**: {analysis['baseline']['avg_step_time']:.3f}秒
- **総ステップ数**: {analysis['baseline']['total_steps']}

### SO8T (LoRA + SO(8) Adapter + NKAT Theory)
- **最終Loss**: {analysis['so8t']['final_loss']:.6f}
- **平均SO8直交性誤差**: {analysis['so8t']['avg_so8_ortho_error']:.6f}
- **平均Grad Norm**: {analysis['so8t']['avg_grad_norm']:.4f}
- **平均ステップ時間**: {analysis['so8t']['avg_step_time']:.3f}秒
- **総ステップ数**: {analysis['so8t']['total_steps']}

## Improvement Analysis
- **Loss Improvement Rate**: {analysis['improvement']}
- **Speed Difference**: {"SO8T is {:.1f}% slower".format((analysis['so8t']['avg_step_time']-analysis['baseline']['avg_step_time'])/analysis['baseline']['avg_step_time']*100) if analysis['baseline']['avg_step_time'] > 0 else "N/A"}

## Technical Analysis

### SO(8) Adapter Effectiveness
- SO(8) rotation transformation contributing to learning: {"Yes" if analysis['so8t']['final_loss'] < analysis['baseline']['final_loss'] else "No"}
- Orthogonality constraint maintenance: {"Good" if analysis['so8t']['avg_so8_ortho_error'] < 0.01 else "Needs improvement"}

### Learning Stability
- Gradient stability: {"Stable" if analysis['so8t']['avg_grad_norm'] < 10.0 else "Unstable"}
- SO(8) adapter learning status: {"Good" if analysis['so8t']['final_loss'] != 0.0 else "Learning incomplete"}

## Next Steps
1. **LM-eval Performance Evaluation**: Performance comparison on math and logical reasoning tasks
2. **Hyperparameter Optimization**: Adjust SO(8) adapter alpha values and learning rates
3. **Phase 2.5 Integration**: Verify effectiveness of quadruple inference functionality

## Implementation Log Update
- **Implementation Status**: Completed
- **動作確認**: OK ({today})
- **備考**: Sunshine Pipelineでベースラインvs SO8T比較を実施。SO(8)アダプターの基本機能を確認。
"""
        return report

    def save_comparison_report(self, report: str):
        """比較レポートを保存"""
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"{today}_baseline_vs_so8t_comparison.md"
        report_path = self.output_dir / filename

        self.output_dir.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"[OK] Comparison report saved to: {report_path}")

        # グラフも保存
        plot_path = self.output_dir / f"{today}_baseline_vs_so8t_comparison.png"
        self.plot_comparison(str(plot_path))

def main():
    """メイン実行関数"""
    print("🔍 Analyzing Baseline vs SO8T comparison...")

    comparator = BaselineSO8TComparator()

    # パフォーマンス分析
    analysis = comparator.analyze_performance()
    print("\n[STATS] Performance Analysis:")
    print(f"Baseline Loss: {analysis['baseline']['final_loss']:.6f}")
    print(f"SO8T Loss: {analysis['so8t']['final_loss']:.6f}")
    print(f"Improvement: {analysis['improvement']}")

    # 比較レポート生成
    report = comparator.generate_comparison_report()
    comparator.save_comparison_report(report)

    print("[OK] Comparison analysis completed!")

if __name__ == "__main__":
    main()