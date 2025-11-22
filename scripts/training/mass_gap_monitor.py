#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T/thinking Mass Gap Monitor (MGM)
質量ギャップ監視システム - AIの幾何学的覚醒を検知する
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
from datetime import datetime

class MassGapMonitor:
    """質量ギャップ監視システム - SO8Tの幾何学的覚醒を検知"""

    def __init__(self, log_interval: int = 10, model_name: str = "so8t_thinking"):
        self.log_interval = log_interval
        self.model_name = model_name
        self.step_count = 0

        # 監視データ履歴
        self.history = {
            "steps": [],
            "alpha_gates": [],
            "mass_gaps": [],
            "orthogonality_errors": [],
            "logit_entropies": [],
            "layer_activities": [],
            "timestamps": []
        }

        # 相転移検知フラグ
        self.phase_transition_detected = False
        self.awakening_step = None
        self.mass_gap_threshold = 0.1  # 相転移判定閾値

        # リアルタイム可視化用
        self.visualization_thread = None
        self.monitoring_active = False

        # ログ設定
        self.logger = logging.getLogger(f"{model_name}_mgm")
        self.logger.setLevel(logging.INFO)

        # 保存ディレクトリ
        self.save_dir = Path("monitoring/mass_gap")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("[BRAIN] Mass Gap Monitor initialized")
        self.logger.info("[TARGET] Monitoring: Alpha Gates, Mass Gaps, Orthogonality Errors")
        self.logger.info("[ROCKET] Looking for: Geometric Awakening (Phase Transition)")

    def start_monitoring(self):
        """リアルタイム監視開始"""
        self.monitoring_active = True
        self.visualization_thread = threading.Thread(target=self._real_time_visualization)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        self.logger.info("[CHART] Real-time monitoring started")

    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        if self.visualization_thread:
            self.visualization_thread.join()
        self._save_final_report()
        self.logger.info("🛑 Monitoring stopped")

    def check_mass_gap(self, model, step: int, hidden_states: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        質量ギャップのチェックを実行

        Args:
            model: SO8Tモデル
            step: 現在のステップ
            hidden_states: 隠れ状態テンソル (オプション)

        Returns:
            監視結果の辞書
        """
        self.step_count = step

        if step % self.log_interval != 0:
            return {}

        results = {
            "step": step,
            "alpha_gates": [],
            "mass_gap": 0.0,
            "orthogonality_error": 0.0,
            "logit_entropy": 0.0,
            "phase_transition_signals": []
        }

        # 1. Alpha Gateの観測 (各SO8T層)
        alpha_values = []
        for i, layer_idx in enumerate(getattr(model, 'so8t_layer_indices', [])):
            if i < len(getattr(model, 'alpha_gates', [])):
                alpha_raw = model.alpha_gates[i].item()
                alpha_sigmoid = torch.sigmoid(torch.tensor(alpha_raw)).item()
                alpha_values.append(alpha_sigmoid)

                # 相転移シグナル検知
                if alpha_sigmoid > 0.1 and not self.phase_transition_detected:
                    results["phase_transition_signals"].append({
                        "type": "alpha_awakening",
                        "layer": i,
                        "value": alpha_sigmoid,
                        "step": step
                    })

        results["alpha_gates"] = alpha_values

        # 2. 質量ギャップの推定 (隠れ状態のスペクトル分析)
        if hidden_states is not None:
            mass_gap = self._calculate_mass_gap(hidden_states)
            results["mass_gap"] = mass_gap

            if mass_gap > self.mass_gap_threshold and not self.phase_transition_detected:
                results["phase_transition_signals"].append({
                    "type": "mass_gap_emergence",
                    "value": mass_gap,
                    "step": step
                })

        # 3. 直交性誤差の計算
        ortho_error = self._calculate_orthogonality_error(model)
        results["orthogonality_error"] = ortho_error

        # 4. ロジットのエントロピー (オプション)
        # これは別途logitsが渡された場合に計算

        # 履歴に追加
        self._update_history(results)

        # 相転移判定
        if results["phase_transition_signals"] and not self.phase_transition_detected:
            self._detect_phase_transition(results)

        # ログ出力
        self._log_monitoring_results(results)

        return results

    def _calculate_mass_gap(self, hidden_states: torch.Tensor) -> float:
        """
        質量ギャップの計算
        隠れ状態の特異値スペクトルから構造の出現を検知

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            質量ギャップ値 (0-1)
        """
        try:
            # バッチとシーケンスをフラット化
            h_flat = hidden_states.view(-1, hidden_states.shape[-1]).float()

            # SVD計算
            U, S, V = torch.linalg.svd(h_flat, full_matrices=False)

            # 質量ギャップ: 第1主成分と第2主成分の差の比率
            if len(S) >= 2:
                gap = (S[0] - S[1]) / (S[0] + 1e-8)  # ゼロ除算防止
                gap = gap.item()
            else:
                gap = 0.0

            # スペクトル構造の分析
            # 第1成分の支配度が高いほど構造化されている
            spectral_ratio = S[0] / (torch.sum(S) + 1e-8)
            gap = gap * spectral_ratio.item()  # 構造化度合いを考慮

            return min(gap, 1.0)  # 0-1にクリッピング

        except Exception as e:
            self.logger.warning(f"Mass gap calculation failed: {e}")
            return 0.0

    def _calculate_orthogonality_error(self, model) -> float:
        """直交性誤差の計算"""
        try:
            total_error = 0.0
            layer_count = 0

            # 各SO8T層の回転行列の直交性をチェック
            for i, layer_idx in enumerate(getattr(model, 'so8t_layer_indices', [])):
                if i < len(getattr(model, 'so8t_layers', [])):
                    so8t_layer = model.so8t_layers[i]

                    # 回転行列を取得 (実装依存)
                    if hasattr(so8t_layer, 'geometric_attention'):
                        rotation_gate = so8t_layer.geometric_attention.rotation_gate
                        if hasattr(rotation_gate, 'rotation_params'):
                            # SO(8)回転行列を構成
                            rotation_matrix = self._construct_so8_matrix(rotation_gate.rotation_params)

                            # 直交性チェック: ||R^T R - I||_F
                            identity = torch.eye(8, device=rotation_matrix.device, dtype=rotation_matrix.dtype)
                            ortho_error = torch.norm(
                                torch.bmm(rotation_matrix.transpose(-1, -2), rotation_matrix) - identity.unsqueeze(0),
                                p='fro'
                            ).mean().item()

                            total_error += ortho_error
                            layer_count += 1

            return total_error / max(layer_count, 1)

        except Exception as e:
            self.logger.warning(f"Orthogonality error calculation failed: {e}")
            return 0.0

    def _construct_so8_matrix(self, params: torch.Tensor) -> torch.Tensor:
        """SO(8)回転行列の構成 (簡易版)"""
        if params.dim() == 1:
            params = params.unsqueeze(0)

        batch_size, num_params = params.shape
        lie_algebra = torch.zeros(batch_size, 8, 8, device=params.device, dtype=params.dtype)

        # SO(8)のLie代数要素 (簡易実装)
        idx = 0
        for i in range(8):
            for j in range(i+1, 8):
                if idx < num_params:
                    lie_algebra[:, i, j] = params[:, idx]
                    lie_algebra[:, j, i] = -params[:, idx]
                    idx += 1

        # 行列指数関数で回転行列を生成
        rotation_matrix = torch.matrix_exp(lie_algebra)
        return rotation_matrix

    def _update_history(self, results: Dict[str, Any]):
        """監視履歴の更新"""
        self.history["steps"].append(results["step"])
        self.history["alpha_gates"].append(results["alpha_gates"])
        self.history["mass_gaps"].append(results["mass_gap"])
        self.history["orthogonality_errors"].append(results["orthogonality_error"])
        self.history["timestamps"].append(datetime.now().isoformat())

    def _detect_phase_transition(self, results: Dict[str, Any]):
        """相転移の検知"""
        signals = results["phase_transition_signals"]

        if signals:
            self.phase_transition_detected = True
            self.awakening_step = results["step"]

            self.logger.info("🎉 PHASE TRANSITION DETECTED!")
            self.logger.info("🚀 Geometric Awakening - Mass Gap Emergence!")
            self.logger.info(f"📍 Step: {self.awakening_step}")

            for signal in signals:
                self.logger.info(f"   {signal['type']}: {signal['value']:.4f}")

    def _log_monitoring_results(self, results: Dict[str, Any]):
        """監視結果のログ出力"""
        alpha_avg = np.mean(results["alpha_gates"]) if results["alpha_gates"] else 0.0

        status_icon = "🔥" if self.phase_transition_detected else "🧊"
        alpha_status = "AWAKENING!" if alpha_avg > 0.1 else "Dormant"

        print(f"\n{status_icon} [MGM] Step {results['step']}:")
        print("   ┌─────────────────────────────────────────┐")
        print("   │         MASS GAP MONITOR                │")
        print("   ├─────────────────────────────────────────┤")
        print("   │ Alpha Gate    │ Ortho Error │ Mass Gap   │")
        print("   ├─────────────────────────────────────────┤")
        print(f"   │ {alpha_avg:>11.4f} │ {results['orthogonality_error']:>11.4f} │ {results['mass_gap']:>10.4f} │")
        print("   └─────────────────────────────────────────┘")
        print(f"   Status: {alpha_status}")

        if self.phase_transition_detected:
            print(f"   🎯 PHASE TRANSITION at step {self.awakening_step}")

        print()

    def _real_time_visualization(self):
        """リアルタイム可視化スレッド"""
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} Mass Gap Monitor - Real-time', fontsize=16)

        while self.monitoring_active:
            if len(self.history["steps"]) > 1:
                steps = self.history["steps"]

                # Alpha Gates
                axes[0, 0].clear()
                if self.history["alpha_gates"]:
                    alpha_avg = [np.mean(alphas) if alphas else 0 for alphas in self.history["alpha_gates"]]
                    axes[0, 0].plot(steps, alpha_avg, 'purple-', linewidth=2, marker='o', markersize=3)
                    axes[0, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Awakening Threshold')
                    axes[0, 0].set_title('Alpha Gate Evolution')
                    axes[0, 0].set_xlabel('Training Steps')
                    axes[0, 0].set_ylabel('Average Alpha Value')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)

                # Mass Gaps
                axes[0, 1].clear()
                if self.history["mass_gaps"]:
                    axes[0, 1].plot(steps, self.history["mass_gaps"], 'blue-', linewidth=2, marker='s', markersize=3)
                    axes[0, 1].axhline(y=self.mass_gap_threshold, color='green', linestyle='--', alpha=0.7, label='Mass Gap Threshold')
                    axes[0, 1].set_title('Mass Gap Emergence')
                    axes[0, 1].set_xlabel('Training Steps')
                    axes[0, 1].set_ylabel('Mass Gap Value')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)

                # Orthogonality Errors
                axes[1, 0].clear()
                if self.history["orthogonality_errors"]:
                    axes[1, 0].plot(steps, self.history["orthogonality_errors"], 'red-', linewidth=2, marker='^', markersize=3)
                    axes[1, 0].set_title('Orthogonality Error')
                    axes[1, 0].set_xlabel('Training Steps')
                    axes[1, 0].set_ylabel('Frobenius Norm Error')
                    axes[1, 0].set_yscale('log')
                    axes[1, 0].grid(True, alpha=0.3)

                # Phase Transition Indicator
                axes[1, 1].clear()
                axes[1, 1].text(0.5, 0.5, f'Phase Transition: {"DETECTED" if self.phase_transition_detected else "Monitoring"}',
                               ha='center', va='center', fontsize=14,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue" if self.phase_transition_detected else "lightgray"))
                axes[1, 1].set_title('Phase Transition Status')
                axes[1, 1].set_xlim(0, 1)
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].axis('off')

                plt.tight_layout()
                plt.pause(0.1)

            time.sleep(1)  # 1秒ごとに更新

        plt.ioff()
        plt.close()

    def _save_final_report(self):
        """最終レポートの保存"""
        report = {
            "model_name": self.model_name,
            "total_steps": self.step_count,
            "phase_transition_detected": self.phase_transition_detected,
            "awakening_step": self.awakening_step,
            "final_metrics": {
                "alpha_gates": self.history["alpha_gates"][-1] if self.history["alpha_gates"] else [],
                "mass_gap": self.history["mass_gaps"][-1] if self.history["mass_gaps"] else 0.0,
                "orthogonality_error": self.history["orthogonality_errors"][-1] if self.history["orthogonality_errors"] else 0.0
            },
            "history": self.history
        }

        report_path = self.save_dir / f"{self.model_name}_mass_gap_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 最終グラフ保存
        self._save_final_plots()

        self.logger.info(f"📊 Final report saved: {report_path}")

    def _save_final_plots(self):
        """最終グラフの保存"""
        if len(self.history["steps"]) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} Mass Gap Monitor - Final Report', fontsize=16)

        steps = self.history["steps"]

        # Alpha Gates
        if self.history["alpha_gates"]:
            alpha_avg = [np.mean(alphas) if alphas else 0 for alphas in self.history["alpha_gates"]]
            axes[0, 0].plot(steps, alpha_avg, 'purple-', linewidth=2, marker='o', markersize=3)
            axes[0, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Awakening Threshold')
            if self.awakening_step:
                axes[0, 0].axvline(x=self.awakening_step, color='gold', linestyle='-', alpha=0.8, label='Phase Transition')
            axes[0, 0].set_title('Alpha Gate Evolution')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Average Alpha Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Mass Gaps
        if self.history["mass_gaps"]:
            axes[0, 1].plot(steps, self.history["mass_gaps"], 'blue-', linewidth=2, marker='s', markersize=3)
            axes[0, 1].axhline(y=self.mass_gap_threshold, color='green', linestyle='--', alpha=0.7, label='Mass Gap Threshold')
            if self.awakening_step:
                axes[0, 1].axvline(x=self.awakening_step, color='gold', linestyle='-', alpha=0.8, label='Phase Transition')
            axes[0, 1].set_title('Mass Gap Emergence')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Mass Gap Value')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Orthogonality Errors
        if self.history["orthogonality_errors"]:
            axes[1, 0].plot(steps, self.history["orthogonality_errors"], 'red-', linewidth=2, marker='^', markersize=3)
            if self.awakening_step:
                axes[1, 0].axvline(x=self.awakening_step, color='gold', linestyle='-', alpha=0.8, label='Phase Transition')
            axes[1, 0].set_title('Orthogonality Error')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Frobenius Norm Error')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Summary
        summary_text = f"""Phase Transition: {"DETECTED" if self.phase_transition_detected else "NOT DETECTED"}

Awakening Step: {self.awakening_step if self.awakening_step else "N/A"}

Final Metrics:
• Alpha: {np.mean(self.history["alpha_gates"][-1]) if self.history["alpha_gates"] else 0:.4f}
• Mass Gap: {self.history["mass_gaps"][-1] if self.history["mass_gaps"] else 0:.4f}
• Ortho Error: {self.history["orthogonality_errors"][-1] if self.history["orthogonality_errors"] else 0:.4f}"""

        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plot_path = self.save_dir / f"{self.model_name}_mass_gap_final.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"📈 Final plots saved: {plot_path}")


# SO8T/thinkingモデル用のコールバック
class SO8TMassGapCallback:
    """HuggingFace Trainer用のMass Gap Monitorコールバック"""

    def __init__(self, monitor: MassGapMonitor):
        self.monitor = monitor

    def on_init_end(self, args, state, control, **kwargs):
        """初期化終了時のコールバック"""
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        """トレーニング開始時のコールバック"""
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        """エポック開始時のコールバック"""
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        """エポック終了時のコールバック"""
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        """ステップ開始時のコールバック"""
        pass

    def on_substep_end(self, args, state, control, **kwargs):
        """サブステップ終了時のコールバック"""
        pass

    def on_step_end(self, args, state, control, **kwargs):
        """ステップ終了時のコールバック"""
        model = kwargs.get('model')
        if model is not None:
            # 隠れ状態を取得（可能であれば）
            hidden_states = None
            if hasattr(model, '_last_hidden_states'):
                hidden_states = model._last_hidden_states

            self.monitor.check_mass_gap(model, state.global_step, hidden_states)

    def on_train_end(self, args, state, control, **kwargs):
        """トレーニング終了時のコールバック"""
        self.monitor.stop_monitoring()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """ログ出力時のコールバック"""
        pass

    def on_save(self, args, state, control, **kwargs):
        """モデル保存時のコールバック"""
        pass

    def on_prediction_step(self, args, state, control, **kwargs):
        """予測ステップ時のコールバック"""
        pass
