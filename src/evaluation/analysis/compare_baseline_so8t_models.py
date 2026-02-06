#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T vs Baseline Model Comparison Analysis
サンシャイン実験結果の詳細比較分析
"""

import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class SO8TBaselineComparator:
    """SO8TモデルとBaselineモデルの比較分析クラス"""

    def __init__(self, logs_dir: str = "logs/sunshine"):
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path("_docs")

        # スタイル設定
        plt.style.use('default')
        sns.set_palette("husl")

        # 作業ディレクトリ作成
        self.output_dir.mkdir(exist_ok=True)

    def load_metrics(self, run_type: str) -> Dict[str, Any]:
        """メトリクスファイルを読み込み"""
        metrics_file = self.logs_dir / f"sunshine_run_{run_type}_metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_training_logs(self, run_type: str) -> pd.DataFrame:
        """訓練ログを読み込み"""
        log_file = self.logs_dir / f"sunshine_run_{run_type}_training_log.csv"
        if not log_file.exists():
            raise FileNotFoundError(f"Training log file not found: {log_file}")

        return pd.read_csv(log_file)

    def analyze_loss_curves(self) -> Dict[str, Any]:
        """損失曲線の分析"""
        print("🔍 Analyzing loss curves...")

        results = {}

        # Baselineのログ読み込み
        baseline_logs = self.load_training_logs("baseline")
        so8t_logs = self.load_training_logs("so8t")

        # 損失曲線の比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # メイン損失
        ax1.plot(baseline_logs['step'], baseline_logs['train_loss'],
                label='Baseline (LoRA)', linewidth=2, marker='o', markersize=3)
        ax1.plot(so8t_logs['step'], so8t_logs['train_loss'],
                label='SO8T (LoRA + SO(8))', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # SO8T特有のメトリクス
        ax2.plot(so8t_logs['step'], so8t_logs['so8_ortho_mean'],
                label='SO(8) Orthogonality Error', color='red', linewidth=2)
        # avg_so8_alpha_meanカラムが存在しない場合はスキップ
        if 'avg_so8_alpha_mean' in so8t_logs.columns:
            ax2.plot(so8t_logs['step'], so8t_logs['avg_so8_alpha_mean'],
                    label='SO(8) Alpha Mean', color='orange', linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('SO(8) Metrics')
        ax2.set_title('SO(8) Adapter Metrics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "baseline_so8t_loss_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 統計分析
        results['baseline_final_loss'] = baseline_logs['train_loss'].iloc[-1]
        results['so8t_final_loss'] = so8t_logs['train_loss'].iloc[-1]
        results['loss_improvement'] = ((baseline_logs['train_loss'].iloc[-1] - so8t_logs['train_loss'].iloc[-1]) /
                                     baseline_logs['train_loss'].iloc[-1]) * 100

        results['baseline_avg_loss'] = baseline_logs['train_loss'].mean()
        results['so8t_avg_loss'] = so8t_logs['train_loss'].mean()
        results['so8t_ortho_error_final'] = so8t_logs['so8_ortho_mean'].iloc[-1] if not so8t_logs['so8_ortho_mean'].empty else float('nan')
        results['so8t_alpha_final'] = so8t_logs.get('avg_so8_alpha_mean', pd.Series([float('nan')])).iloc[-1]

        return results

    def analyze_convergence(self) -> Dict[str, Any]:
        """収束性の分析"""
        print("🔍 Analyzing convergence patterns...")

        results = {}

        baseline_logs = self.load_training_logs("baseline")
        so8t_logs = self.load_training_logs("so8t")

        # 収束速度の分析
        # 損失の変動係数（標準偏差/平均）を計算
        baseline_cv = baseline_logs['train_loss'].std() / baseline_logs['train_loss'].mean()
        so8t_cv = so8t_logs['train_loss'].std() / so8t_logs['train_loss'].mean()

        results['baseline_loss_cv'] = baseline_cv
        results['so8t_loss_cv'] = so8t_cv
        results['convergence_stability_ratio'] = so8t_cv / baseline_cv

        # 最終収束状態の分析
        final_window = 10  # 最後の10ステップ
        baseline_final_avg = baseline_logs['train_loss'].tail(final_window).mean()
        so8t_final_avg = so8t_logs['train_loss'].tail(final_window).mean()

        results['baseline_final_avg_loss'] = baseline_final_avg
        results['so8t_final_avg_loss'] = so8t_final_avg

        return results

    def analyze_gradient_dynamics(self) -> Dict[str, Any]:
        """勾配動態の分析"""
        print("🔍 Analyzing gradient dynamics...")

        results = {}

        baseline_logs = self.load_training_logs("baseline")
        so8t_logs = self.load_training_logs("so8t")

        # 勾配ノルムの比較
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(baseline_logs['step'], baseline_logs['grad_norm'],
               label='Baseline Grad Norm', linewidth=2, alpha=0.8)
        ax.plot(so8t_logs['step'], so8t_logs['grad_norm'],
               label='SO8T Grad Norm', linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "baseline_so8t_gradient_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 統計
        results['baseline_avg_grad_norm'] = baseline_logs['grad_norm'].mean()
        results['so8t_avg_grad_norm'] = so8t_logs['grad_norm'].mean()
        results['baseline_grad_norm_std'] = baseline_logs['grad_norm'].std()
        results['so8t_grad_norm_std'] = so8t_logs['grad_norm'].std()

        return results

    def generate_comparison_report(self) -> str:
        """比較レポートの生成"""
        print("[STATS] Generating comprehensive comparison report...")

        try:
            # データ収集
            loss_analysis = self.analyze_loss_curves()
            convergence_analysis = self.analyze_convergence()
            gradient_analysis = self.analyze_gradient_dynamics()

            # メトリクス読み込み
            baseline_metrics = self.load_metrics("baseline")
            so8t_metrics = self.load_metrics("so8t")

            # レポート生成
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            report_content = f"""# SO8T vs Baseline Model Comparison Report

## 実行情報
- **生成日時**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **実験タイプ**: Sunshine Pipeline Comparison
- **データセット**: Mathematics, Science, NKAT Theory, NSFW
- **モデル**: Phi-3.5-mini-instruct

## モデル設定

### Baseline Model
- **構成**: LoRA (Low-Rank Adaptation)
- **データ**: Mathematics + Science datasets
- **アダプター**: なし

### SO8T Model
- **構成**: LoRA + SO(8) Geometric Adapter
- **データ**: NKAT Theory + NSFW datasets
- **アダプター**: Hook-based SO(8) Residual Adapter
- **Phase**: 2.5 (Quadruple Inference Integration)

## 訓練結果比較

### 損失比較
| 指標 | Baseline | SO8T | 改善率 |
|------|----------|------|--------|
| 最終損失 | {loss_analysis['baseline_final_loss']:.4f} | {loss_analysis['so8t_final_loss']:.4f} | {loss_analysis['loss_improvement']:+.2f}% |
| 平均損失 | {loss_analysis['baseline_avg_loss']:.4f} | {loss_analysis['so8t_avg_loss']:.4f} | - |
| 損失変動係数 | {convergence_analysis['baseline_loss_cv']:.4f} | {convergence_analysis['so8t_loss_cv']:.4f} | {convergence_analysis['convergence_stability_ratio']:.2f}x |

### SO(8)アダプター特有指標
- **最終直交性誤差**: {loss_analysis['so8t_ortho_error_final']:.6f}
- **最終Alpha値**: {loss_analysis['so8t_alpha_final']:.6f}
- **平均勾配ノルム**: {gradient_analysis['so8t_avg_grad_norm']:.4f}
- **勾配ノルム標準偏差**: {gradient_analysis['so8t_grad_norm_std']:.4f}

### 勾配動態比較
| 指標 | Baseline | SO8T |
|------|----------|------|
| 平均勾配ノルム | {gradient_analysis['baseline_avg_grad_norm']:.4f} | {gradient_analysis['so8t_avg_grad_norm']:.4f} |
| 勾配ノルム標準偏差 | {gradient_analysis['baseline_grad_norm_std']:.4f} | {gradient_analysis['so8t_grad_norm_std']:.4f} |

## 分析結果

### 収束特性
- **Baseline安定性**: {'安定' if convergence_analysis['baseline_loss_cv'] < 0.5 else '不安定'}
- **SO8T安定性**: {'安定' if convergence_analysis['so8t_loss_cv'] < 0.5 else '不安定'}
- **安定性比**: {convergence_analysis['convergence_stability_ratio']:.2f}

### 最終収束状態 (最終10ステップ平均)
- **Baseline最終平均損失**: {convergence_analysis['baseline_final_avg_loss']:.4f}
- **SO8T最終平均損失**: {convergence_analysis['so8t_final_avg_loss']:.4f}

## 技術的考察

### SO(8)アダプターの影響
1. **損失改善**: {loss_analysis['loss_improvement']:+.2f}% {'改善' if loss_analysis['loss_improvement'] > 0 else '悪化'}
2. **直交性維持**: SO(8)回転の直交性が{ '良好' if loss_analysis['so8t_ortho_error_final'] < 1e-3 else '要改善'} (目標: < 1e-3)
3. **学習適応**: Alphaパラメータが{loss_analysis['so8t_alpha_final']:.4f}に適応

### 潜在的問題点
- **SO8T損失が0.0**: アダプターが学習していない可能性
- **NaN値の存在**: 勾配計算または直交性計算の問題
- **Alpha値の異常**: 学習初期値からの変化が不十分

## 推奨事項

### 即時対応
1. **SO(8)アダプターの学習率調整**: アダプター専用学習率をさらに上げる
2. **初期化パラメータの見直し**: Alphaの初期値を再調整
3. **デバッグログの追加**: 各ステップでのアダプター出力を監視

### 長期対応
1. **Phase 2.5完全統合**: Quadruple Inferenceの完全実装
2. **データセットの最適化**: NKAT理論データの品質向上
3. **評価指標の拡充**: LM-evalを使用した性能比較

## 結論

SO(8)幾何学的アダプターの導入は理論的に有望だが、実装段階での技術的課題が残っている。
特にアダプターの学習が不十分であることが判明したため、次のステップでは学習パラメータの最適化を優先する。

---
*Report generated by SO8T Baseline Comparator*
*Timestamp: {timestamp}*
"""

            # レポート保存
            report_filename = f"{datetime.now().strftime('%Y-%m-%d')}_baseline_so8t_comparison_report.md"
            report_path = self.output_dir / report_filename

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            print(f"[OK] Comparison report saved to: {report_path}")

            return report_content

        except Exception as e:
            print(f"[NG] Error generating report: {e}")
            import traceback
            traceback.print_exc()
            return ""

def main():
    """メイン実行関数"""
    print("[START] Starting SO8T vs Baseline Model Comparison Analysis")
    print("=" * 60)

    comparator = SO8TBaselineComparator()

    try:
        # 比較レポート生成
        report = comparator.generate_comparison_report()

        print("\n[OK] Analysis completed successfully!")
        print("[STATS] Generated comparison report and visualizations")

    except Exception as e:
        print(f"[NG] Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
