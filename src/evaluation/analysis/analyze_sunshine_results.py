#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()

"""
Sunshine Results Analyzer

サンシャイン実験結果を分析・可視化するスクリプト
Baseline vs SO8T の比較分析
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class SunshineAnalyzer:
    """サンシャイン実験結果分析器"""

    def __init__(self, log_dir: str = "logs/sunshine"):
        self.log_dir = Path(log_dir)
        self.baseline_log = self.log_dir / "sunshine_run_baseline_training_log.csv"
        self.so8t_log = self.log_dir / "sunshine_run_so8t_training_log.csv"
        self.baseline_metrics = self.log_dir / "sunshine_run_baseline_metrics.json"
        self.so8t_metrics = self.log_dir / "sunshine_run_so8t_metrics.json"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ログデータを読み込み"""
        baseline_df = pd.read_csv(self.baseline_log) if self.baseline_log.exists() else pd.DataFrame()
        so8t_df = pd.read_csv(self.so8t_log) if self.so8t_log.exists() else pd.DataFrame()

        print(f"[STATS] Loaded data:")
        print(f"  Baseline: {len(baseline_df)} steps")
        print(f"  SO8T: {len(so8t_df)} steps")

        return baseline_df, so8t_df

    def load_metrics(self) -> Tuple[Dict, Dict]:
        """メトリクスデータを読み込み"""
        baseline_metrics = {}
        so8t_metrics = {}

        if self.baseline_metrics.exists():
            with open(self.baseline_metrics, 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)

        if self.so8t_metrics.exists():
            with open(self.so8t_metrics, 'r', encoding='utf-8') as f:
                so8t_metrics = json.load(f)

        return baseline_metrics, so8t_metrics

    def plot_loss_curves(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """学習曲線をプロット"""
        plt.figure(figsize=(12, 8))

        # Loss曲線
        plt.subplot(2, 2, 1)
        if not baseline_df.empty and 'train_loss' in baseline_df.columns:
            plt.plot(baseline_df['step'], baseline_df['train_loss'],
                    label='Baseline (LoRA)', color='blue', linewidth=2)
        if not so8t_df.empty and 'train_loss' in so8t_df.columns:
            plt.plot(so8t_df['step'], so8t_df['train_loss'],
                    label='SO8T (LoRA + SO(8))', color='red', linewidth=2)

        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('学習曲線比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SO(8)直交誤差（SO8Tのみ）
        plt.subplot(2, 2, 2)
        if not so8t_df.empty and 'so8_ortho_mean' in so8t_df.columns:
            ortho_data = so8t_df['so8_ortho_mean'].dropna()
            if not ortho_data.empty:
                plt.plot(so8t_df.loc[ortho_data.index, 'step'], ortho_data,
                        label='SO(8) Orthogonality Error', color='orange', linewidth=2)
                plt.axhline(y=ortho_data.mean(), color='orange', linestyle='--', alpha=0.7,
                           label='.2f')

        plt.xlabel('Step')
        plt.ylabel('Orthogonality Error')
        plt.title('SO(8)直交性誤差')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ステップ時間比較
        plt.subplot(2, 2, 3)
        if not baseline_df.empty and 'step_time_sec' in baseline_df.columns:
            baseline_times = baseline_df['step_time_sec'].dropna()
            if not baseline_times.empty:
                plt.plot(baseline_df.loc[baseline_times.index, 'step'],
                        baseline_times.rolling(10).mean(),
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'step_time_sec' in so8t_df.columns:
            so8t_times = so8t_df['step_time_sec'].dropna()
            if not so8t_times.empty:
                plt.plot(so8t_df.loc[so8t_times.index, 'step'],
                        so8t_times.rolling(10).mean(),
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Step Time (sec)')
        plt.title('ステップ時間比較 (移動平均10)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 勾配ノルム比較
        plt.subplot(2, 2, 4)
        if not baseline_df.empty and 'grad_norm' in baseline_df.columns:
            baseline_grads = baseline_df['grad_norm'].dropna()
            if not baseline_grads.empty:
                plt.plot(baseline_df.loc[baseline_grads.index, 'step'], baseline_grads,
                        label='Baseline', color='blue', alpha=0.7)

        if not so8t_df.empty and 'grad_norm' in so8t_df.columns:
            so8t_grads = so8t_df['grad_norm'].dropna()
            if not so8t_grads.empty:
                plt.plot(so8t_df.loc[so8t_grads.index, 'step'], so8t_grads,
                        label='SO8T', color='red', alpha=0.7)

        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルム比較')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'sunshine_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary_stats(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame,
                          baseline_metrics: Dict, so8t_metrics: Dict):
        """統計サマリーを表示"""
        print("\n📈 EXPERIMENT STATISTICS SUMMARY")
        print("=" * 60)

        def get_stats(name: str, df: pd.DataFrame, metrics: Dict):
            print(f"\n{name}:")
            if df.empty:
                print("  No data available")
                return

            # Loss統計
            if 'train_loss' in df.columns:
                losses = df['train_loss'].dropna()
                if not losses.empty:
                    print("  Loss Statistics:")
                    print(f"    Initial: {losses.iloc[0]:.4f}")
                    print(f"    Final: {losses.iloc[-1]:.4f}")
                    print(f"    Mean: {losses.mean():.4f}")
            # SO(8)直交誤差（SO8Tのみ）
            if 'so8_ortho_mean' in df.columns and name == "SO8T":
                ortho = df['so8_ortho_mean'].dropna()
                if not ortho.empty:
                    print("  SO(8) Orthogonality:")
                    print(f"    Mean: {ortho.mean():.6f}")
                    print(f"    Max: {ortho.max():.6f}")
                    print(f"    Min: {ortho.min():.6f}")
            # 時間統計
            if 'step_time_sec' in df.columns:
                times = df['step_time_sec'].dropna()
                if not times.empty:
                    print("  Performance:")
                    print(f"    Mean step time: {times.mean():.3f}s")
                    print(f"    Total time: {times.sum():.3f}s")
            # 最終メトリクス
            final_loss = metrics.get('final_train_loss')
            if final_loss is not None:
                print(f"  Final Loss: {final_loss:.4f}")

        get_stats("BASELINE (LoRA only)", baseline_df, baseline_metrics)
        get_stats("SO8T (LoRA + SO(8))", so8t_df, so8t_metrics)

        # 比較分析
        print("\n🔍 COMPARISON ANALYSIS:")
        if not baseline_df.empty and not so8t_df.empty:
            baseline_final = baseline_df['train_loss'].dropna().iloc[-1] if 'train_loss' in baseline_df.columns else None
            so8t_final = so8t_df['train_loss'].dropna().iloc[-1] if 'train_loss' in so8t_df.columns else None

            if baseline_final and so8t_final:
                loss_diff = so8t_final - baseline_final
                improvement = "better" if loss_diff < 0 else "worse"
                print(f"  Loss difference: {loss_diff:.4f}")
                print(f"  SO8T is {improvement} than baseline")
                # Loss減少率
                if baseline_final > 0:
                    reduction_rate = abs(loss_diff) / baseline_final * 100
                    print(f"  Loss reduction rate: {reduction_rate:.1f}%")

    def analyze_convergence(self, baseline_df: pd.DataFrame, so8t_df: pd.DataFrame):
        """収束性分析"""
        print("\n[TARGET] CONVERGENCE ANALYSIS")
        print("=" * 60)

        def analyze_run(name: str, df: pd.DataFrame):
            if df.empty or 'train_loss' not in df.columns:
                print(f"{name}: No loss data")
                return

            losses = df['train_loss'].dropna()
            if len(losses) < 10:
                print(f"{name}: Insufficient data for convergence analysis")
                return

            # 初期と最終の比較
            initial_loss = losses.iloc[:10].mean()
            final_loss = losses.iloc[-10:].mean()

            loss_reduction = initial_loss - final_loss
            reduction_rate = loss_reduction / initial_loss * 100

            print(f"{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Loss reduction: {loss_reduction:.4f} ({reduction_rate:.1f}%)")
            # 収束安定性（lossの変動係数）
            if len(losses) > 20:
                recent_losses = losses.iloc[-20:]
                cv = recent_losses.std() / recent_losses.mean() * 100
                stability = "Stable" if cv < 5 else "Unstable" if cv > 15 else "Moderate"
                print(f"    Stability: {stability} (CV: {cv:.1f}%)")
        analyze_run("BASELINE", baseline_df)
        analyze_run("SO8T", so8t_df)


def main():
    """メイン分析実行"""
    print("[RESEARCH] SO8T SUNSHINE RESULTS ANALYZER")
    print("=" * 50)

    analyzer = SunshineAnalyzer()

    # データ読み込み
    baseline_df, so8t_df = analyzer.load_data()
    baseline_metrics, so8t_metrics = analyzer.load_metrics()

    if baseline_df.empty and so8t_df.empty:
        print("[NG] No experiment data found in logs/sunshine/")
        print("Run sunshine pipeline first:")
        print("  py scripts/pipeline/sunshine_pipeline.py")
        return

    # 統計サマリー
    analyzer.print_summary_stats(baseline_df, so8t_df, baseline_metrics, so8t_metrics)

    # 収束分析
    analyzer.analyze_convergence(baseline_df, so8t_df)

    # 可視化
    try:
        analyzer.plot_loss_curves(baseline_df, so8t_df)
        print("\n[STATS] Charts saved to: logs/sunshine/sunshine_comparison.png")
    except ImportError:
        print("\n[WARN]  Matplotlib not available, skipping charts")
    except Exception as e:
        print(f"\n[WARN]  Chart generation failed: {e}")

    print("\n[OK] Analysis completed!")


if __name__ == "__main__":
    main()
