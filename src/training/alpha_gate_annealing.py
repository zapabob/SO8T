#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アルファゲートアニーリングシステム
Alpha Gate Annealing System with Golden Ratio Bayesian Optimization

黄金比の平方の逆数とベイズ最適化によるloss相転移監視
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class GoldenRatioBayesianOptimizer:
    """
    黄金比の平方の逆数を使用したベイズ最適化
    Bayesian optimization using inverse square of golden ratio
    """

    def __init__(self, alpha_range: Tuple[float, float] = (0.0, 1.0)):
        self.alpha_range = alpha_range

        # 黄金比 φ = (1 + √5) / 2
        self.phi = (1 + math.sqrt(5)) / 2
        self.phi_inv_square = 1 / (self.phi ** 2)  # ≈ 0.382

        # ベイズ最適化パラメータ
        self.gp_kernel = self._create_gp_kernel()
        self.observations = []
        self.alpha_history = []
        self.loss_history = []

    def _create_gp_kernel(self):
        """ガウス過程カーネル作成"""
        # RBFカーネル（簡易実装）
        def rbf_kernel(x1, x2, length_scale=0.1):
            return torch.exp(-0.5 * ((x1 - x2) / length_scale) ** 2)
        return rbf_kernel

    def suggest_alpha(self, current_step: int, max_steps: int) -> float:
        """
        次ステップのα値を提案
        Suggest next alpha value using golden ratio and Bayesian optimization
        """
        if len(self.observations) < 2:
            # 初期サンプル：黄金比の平方の逆数を使用
            if current_step == 0:
                return self.phi_inv_square
            else:
                # ランダム初期化
                return np.random.uniform(*self.alpha_range)

        # ベイズ最適化による次のα値提案
        alpha_candidates = np.linspace(self.alpha_range[0], self.alpha_range[1], 100)

        # 各候補の期待改善度計算
        ei_values = []
        for alpha in alpha_candidates:
            ei = self._expected_improvement(alpha)
            ei_values.append(ei)

        # 最適なαを選択
        best_idx = np.argmax(ei_values)
        suggested_alpha = alpha_candidates[best_idx]

        # 黄金比による調整
        golden_adjustment = suggested_alpha * (1 + self.phi_inv_square) / 2
        final_alpha = np.clip(golden_adjustment, *self.alpha_range)

        return float(final_alpha)

    def _expected_improvement(self, alpha: float) -> float:
        """期待改善度計算"""
        if not self.observations:
            return 1.0

        # 現在の最適値
        best_loss = min(self.loss_history)

        # 予測平均と分散
        mean, std = self._gp_predict(alpha)

        # 改善度
        improvement = best_loss - mean

        # 期待改善度
        if std == 0:
            return max(0, improvement)

        z = improvement / std
        ei = improvement * stats.norm.cdf(z) + std * stats.norm.pdf(z)

        return max(0, ei)

    def _gp_predict(self, alpha: float) -> Tuple[float, float]:
        """ガウス過程による予測"""
        if not self.observations:
            return 0.0, 1.0

        alphas = torch.tensor([obs['alpha'] for obs in self.observations])
        losses = torch.tensor([obs['loss'] for obs in self.observations])

        # カーネル行列
        K = torch.zeros(len(alphas), len(alphas))
        for i in range(len(alphas)):
            for j in range(len(alphas)):
                K[i, j] = self.gp_kernel(alphas[i], alphas[j])

        # 予測
        k_star = torch.tensor([self.gp_kernel(alpha, a) for a in alphas])
        K_inv = torch.inverse(K + 0.1 * torch.eye(len(alphas)))

        mean = torch.mv(k_star, torch.mv(K_inv, losses))
        var = self.gp_kernel(alpha, alpha) - torch.mv(k_star, torch.mv(K_inv, k_star))

        return float(mean), max(0.001, float(var.sqrt()))

    def observe(self, alpha: float, loss: float):
        """観測結果を記録"""
        observation = {
            'alpha': alpha,
            'loss': loss,
            'step': len(self.observations)
        }
        self.observations.append(observation)
        self.alpha_history.append(alpha)
        self.loss_history.append(loss)


class SigmoidAlphaGateAnnealing(nn.Module):
    """
    シグモイド関数によるアルファゲートアニーリング
    Sigmoid-based alpha gate annealing with phase transition monitoring
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # アニーリングパラメータ
        self.initial_alpha = config.get('initial_alpha', -0.5)  # 初期値 -0.5
        self.target_alpha = config.get('target_alpha', 1.0)
        self.k = config.get('sigmoid_k', 0.1)  # シグモイドの steepness

        # ベイズ最適化
        self.bayesian_optimizer = GoldenRatioBayesianOptimizer(
            alpha_range=(0.0, 1.0)
        )

        # 相転移監視
        self.phase_transitions = []
        self.loss_history = []
        self.alpha_history = []

        # 学習パラメータ
        self.current_step = 0
        self.max_steps = config.get('max_steps', 10000)

    def forward(self, current_loss: float) -> Tuple[float, Dict[str, Any]]:
        """
        アルファゲートアニーリング実行
        Args:
            current_loss: 現在のloss値
        Returns:
            alpha: 計算されたα値
            aux_info: 補助情報（相転移など）
        """
        # 現在のステップでのα値計算
        alpha = self._compute_alpha(self.current_step, self.max_steps)

        # ベイズ最適化による調整
        bayesian_alpha = self.bayesian_optimizer.suggest_alpha(self.current_step, self.max_steps)
        alpha = 0.7 * alpha + 0.3 * bayesian_alpha  # 重み付き平均

        # 相転移検出
        phase_info = self._detect_phase_transition(current_loss, alpha)

        # 履歴記録
        self.loss_history.append(current_loss)
        self.alpha_history.append(alpha)
        self.phase_transitions.append(phase_info)

        # ベイズ最適化に観測結果を記録
        self.bayesian_optimizer.observe(alpha, current_loss)

        # ステップ更新
        self.current_step += 1

        aux_info = {
            'phase_transition': phase_info,
            'bayesian_alpha': bayesian_alpha,
            'sigmoid_alpha': self._compute_alpha(self.current_step-1, self.max_steps),
            'step': self.current_step
        }

        return alpha, aux_info

    def _compute_alpha(self, step: int, max_steps: int) -> float:
        """シグモイド関数によるα値計算"""
        # ステップを[0,1]に正規化
        t = step / max_steps

        # シグモイド関数
        sigmoid_value = 1.0 / (1.0 + math.exp(-self.k * (t - 0.5)))

        # 初期値から目標値への遷移
        alpha = self.initial_alpha + (self.target_alpha - self.initial_alpha) * sigmoid_value

        # 範囲制限
        alpha = max(0.0, min(1.0, alpha))

        return alpha

    def _detect_phase_transition(self, current_loss: float, current_alpha: float) -> Dict[str, Any]:
        """
        lossの相転移を検出
        Detect phase transition in loss landscape
        """
        phase_info = {
            'transition_detected': False,
            'transition_type': None,
            'loss_gradient': 0.0,
            'alpha_gradient': 0.0,
            'phase_entropy': 0.0
        }

        if len(self.loss_history) < 10:
            return phase_info

        # 最近のloss勾配計算
        recent_losses = self.loss_history[-10:]
        loss_gradient = np.gradient(recent_losses).mean()

        # 最近のα勾配計算
        recent_alphas = self.alpha_history[-10:]
        alpha_gradient = np.gradient(recent_alphas).mean()

        # エントロピー計算（loss分布の複雑さ）
        loss_std = np.std(recent_losses)
        phase_entropy = -np.sum(np.histogram(recent_losses, bins=5, density=True)[0] *
                               np.log(np.histogram(recent_losses, bins=5, density=True)[0] + 1e-10))

        # 相転移判定
        # 1. loss勾配の急激な変化
        if abs(loss_gradient) > np.std(np.gradient(self.loss_history[-50:])) * 2:
            phase_info['transition_detected'] = True
            phase_info['transition_type'] = 'loss_gradient'

        # 2. α勾配の急激な変化
        elif abs(alpha_gradient) > np.std(np.gradient(self.alpha_history[-50:])) * 2:
            phase_info['transition_detected'] = True
            phase_info['transition_type'] = 'alpha_gradient'

        # 3. エントロピーの急激な変化
        elif phase_entropy > np.mean([self._calculate_entropy(self.loss_history[-i-10:-i])
                                    for i in range(1, min(5, len(self.loss_history)-10))]) * 1.5:
            phase_info['transition_detected'] = True
            phase_info['transition_type'] = 'entropy_surge'

        phase_info.update({
            'loss_gradient': float(loss_gradient),
            'alpha_gradient': float(alpha_gradient),
            'phase_entropy': float(phase_entropy)
        })

        return phase_info

    def _calculate_entropy(self, values: List[float]) -> float:
        """値の分布エントロピー計算"""
        if not values:
            return 0.0

        hist, _ = np.histogram(values, bins=5, density=True)
        hist = hist[hist > 0]  # ゼロを除去
        return -np.sum(hist * np.log(hist + 1e-10))

    def plot_annealing_history(self, save_path: Optional[str] = None):
        """
        アニーリング履歴をプロット
        """
        if len(self.alpha_history) < 10:
            logger.warning("Not enough data for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Alpha Gate Annealing Analysis', fontsize=16, fontweight='bold')

        steps = range(len(self.alpha_history))

        # 1. Alpha値の推移
        axes[0, 0].plot(steps, self.alpha_history, 'b-', linewidth=2, label='Alpha')
        axes[0, 0].axhline(y=self.initial_alpha, color='r', linestyle='--',
                          label=f'Initial α = {self.initial_alpha:.3f}')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Alpha Value')
        axes[0, 0].set_title('Alpha Annealing Schedule')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Lossの推移
        axes[0, 1].plot(steps, self.loss_history, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Alpha vs Lossの相関
        valid_indices = ~np.isnan(self.alpha_history) & ~np.isnan(self.loss_history)
        if np.sum(valid_indices) > 10:
            alpha_vals = np.array(self.alpha_history)[valid_indices]
            loss_vals = np.array(self.loss_history)[valid_indices]

            # 散布図
            axes[1, 0].scatter(alpha_vals, loss_vals, alpha=0.6, c=steps, cmap='viridis')
            axes[1, 0].set_xlabel('Alpha')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Alpha vs Loss Correlation')

            # トレンドライン
            if len(alpha_vals) > 2:
                z = np.polyfit(alpha_vals, loss_vals, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(alpha_vals.min(), alpha_vals.max(), 100)
                axes[1, 0].plot(x_trend, p(x_trend), 'r--', linewidth=2)

        # 4. 相転移イベント
        transition_steps = []
        transition_types = []

        for i, phase in enumerate(self.phase_transitions):
            if phase.get('transition_detected', False):
                transition_steps.append(i)
                transition_types.append(phase.get('transition_type', 'unknown'))

        if transition_steps:
            colors = {'loss_gradient': 'red', 'alpha_gradient': 'blue', 'entropy_surge': 'green'}
            for step, t_type in zip(transition_steps, transition_types):
                color = colors.get(t_type, 'black')
                axes[1, 1].axvline(x=step, color=color, linestyle='--', alpha=0.7,
                                  label=f'{t_type} at step {step}')

        axes[1, 1].plot(steps, [p['loss_gradient'] for p in self.phase_transitions],
                       'r-', label='Loss Gradient', alpha=0.7)
        axes[1, 1].plot(steps, [p['alpha_gradient'] for p in self.phase_transitions],
                       'b-', label='Alpha Gradient', alpha=0.7)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Gradient')
        axes[1, 1].set_title('Phase Transition Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Annealing history plot saved to {save_path}")

        plt.show()

    def save_annealing_state(self, save_path: str):
        """アニーリング状態を保存"""
        state = {
            'current_step': self.current_step,
            'alpha_history': self.alpha_history,
            'loss_history': self.loss_history,
            'phase_transitions': self.phase_transitions,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Annealing state saved to {save_path}")

    def load_annealing_state(self, load_path: str):
        """アニーリング状態を読み込み"""
        with open(load_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.current_step = state.get('current_step', 0)
        self.alpha_history = state.get('alpha_history', [])
        self.loss_history = state.get('loss_history', [])
        self.phase_transitions = state.get('phase_transitions', [])
        self.config = state.get('config', self.config)

        # ベイズ最適化に履歴を再登録
        for alpha, loss in zip(self.alpha_history, self.loss_history):
            self.bayesian_optimizer.observe(alpha, loss)

        logger.info(f"Annealing state loaded from {load_path}")


class MetaInferenceController:
    """
    メタ推論制御システム
    Meta inference control system for entropy management
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # エントロピー閾値
        self.high_entropy_threshold = config.get('high_entropy_threshold', 2.0)
        self.low_entropy_threshold = config.get('low_entropy_threshold', 0.5)

        # 制御パラメータ
        self.cooling_rate = config.get('cooling_rate', 0.9)
        self.heating_rate = config.get('heating_rate', 1.1)

        # 状態履歴
        self.entropy_history = []
        self.control_actions = []

    def control_inference(self, current_entropy: float, current_temperature: float) -> Tuple[float, str]:
        """
        推論制御
        Args:
            current_entropy: 現在のエントロピー
            current_temperature: 現在の温度パラメータ
        Returns:
            new_temperature: 新しい温度パラメータ
            action: 実行されたアクション
        """
        self.entropy_history.append(current_entropy)

        if current_entropy > self.high_entropy_threshold:
            # 高エントロピー状態：冷却
            new_temperature = current_temperature * self.cooling_rate
            action = "cooling_high_entropy"
        elif current_entropy < self.low_entropy_threshold:
            # 低エントロピー状態：加熱
            new_temperature = current_temperature * self.heating_rate
            action = "heating_low_entropy"
        else:
            # 最適範囲：維持
            new_temperature = current_temperature
            action = "maintaining_optimal"

        # 温度範囲制限
        new_temperature = max(0.1, min(10.0, new_temperature))

        control_info = {
            'entropy': current_entropy,
            'old_temperature': current_temperature,
            'new_temperature': new_temperature,
            'action': action,
            'timestamp': datetime.now().isoformat()
        }

        self.control_actions.append(control_info)

        return new_temperature, action


if __name__ == '__main__':
    # Test alpha gate annealing
    config = {
        'initial_alpha': (1 + math.sqrt(5)) / 2 * -2,  # φ^(-2)
        'target_alpha': 1.0,
        'sigmoid_k': 0.1,
        'max_steps': 1000
    }

    annealing = SigmoidAlphaGateAnnealing(config)

    # Simulate training
    for step in range(100):
        # ダミーのloss（徐々に減少）
        loss = 10.0 * math.exp(-step / 50) + np.random.normal(0, 0.1)
        alpha, aux = annealing.forward(loss)

        if step % 20 == 0:
            print(f"Step {step}: Alpha={alpha:.4f}, Loss={loss:.4f}, Phase={aux['phase_transition']['transition_type']}")

    # Plot results
    annealing.plot_annealing_history('alpha_gate_annealing_analysis.png')

    # Save state
    annealing.save_annealing_state('alpha_gate_state.json')
