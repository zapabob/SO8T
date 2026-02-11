"""
成形報酬計算モジュール。

GRPOの報酬関数を実装する：
- ツール使用せずに正解 → 最強報酬
- ツール使用して正解 → 正の報酬
- 不正解 → 負の報酬
- エラー → 強い負の報酬

References:
    Dai et al. (2026) arXiv:2601.20614
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from ...utils.logging import get_logger

logger = get_logger(__name__)


class AnswerStatus(Enum):
    """回答状態列挙型。"""

    CORRECT_NO_TOOL = "correct_no_tool"
    CORRECT_WITH_TOOL = "correct_with_tool"
    INCORRECT = "incorrect"
    ERROR = "error"


@dataclass
class RewardConfig:
    """報酬設定データクラス。

    Attributes:
        reward_correct_no_tool: 正解・ツール不使用の報酬
        reward_correct_with_tool: 正解・ツール使用の報酬
        reward_incorrect: 不正解の報酬
        reward_error: エラーの報酬
        entropy_coef: エントロピー係数
        tool_usage_penalty: ツール使用ペナルティ
        reasoning_bonus_coef: 推論ボーナス係数
    """

    reward_correct_no_tool: float = 3.0
    reward_correct_with_tool: float = 1.0
    reward_incorrect: float = -1.0
    reward_error: float = -2.0
    entropy_coef: float = 0.01
    tool_usage_penalty: float = 0.1
    reasoning_bonus_coef: float = 0.5


class ShapedGRPOReward(nn.Module):
    """成形されたGRPO報酬計算器。

    設計理念:
    - ツール使用せずに正解 → 最も高い報酬（推論能力を強化）
    - ツール使用して正解 → 正の報酬（実用性を維持）
    - 不正解/エラー → 負の報酬（学習の方向性を維持）

    Attributes:
        config: 報酬設定
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or RewardConfig()

    def forward(
        self,
        prediction: str,
        ground_truth: str,
        tool_calls: list,
        reasoning_steps: list,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> tuple[float, dict]:
        """報酬を計算する。

        Args:
            prediction: モデルの予測
            ground_truth: 正解
            tool_calls: ツール呼び出しリスト
            reasoning_steps: 推論ステップリスト
            execution_time: 実行時間
            error_message: エラーメッセージ

        Returns:
            (報酬スコア, 詳細辞書)

        Example:
            >>> reward = ShapedGRPOReward()
            >>> score, details = reward(
            ...     prediction="42",
            ...     ground_truth="42",
            ...     tool_calls=[],
            ...     reasoning_steps=["step1", "step2"]
            ... )
            >>> print(score)
            3.0
        """
        reward_details = {
            "base_reward": 0.0,
            "tool_bonus": 0.0,
            "reasoning_bonus": 0.0,
            "entropy_bonus": 0.0,
            "status": AnswerStatus.INCORRECT.value,
            "total": 0.0,
        }

        if error_message is not None:
            reward_details["status"] = AnswerStatus.ERROR.value
            reward_details["base_reward"] = self.config.reward_error
            reward_details["total"] = self.config.reward_error
            return self.config.reward_error, reward_details

        is_correct = self._check_correct(prediction, ground_truth)
        has_tool_calls = len(tool_calls) > 0
        has_reasoning = len(reasoning_steps) > 0

        if is_correct and not has_tool_calls:
            reward_details["status"] = AnswerStatus.CORRECT_NO_TOOL.value
            reward_details["base_reward"] = self.config.reward_correct_no_tool

            if has_reasoning:
                reasoning_bonus = min(
                    len(reasoning_steps) * self.config.reasoning_bonus_coef, 1.0
                )
                reward_details["reasoning_bonus"] = reasoning_bonus
                reward_details["total"] = (
                    self.config.reward_correct_no_tool + reasoning_bonus
                )
            else:
                reward_details["total"] = self.config.reward_correct_no_tool

        elif is_correct and has_tool_calls:
            reward_details["status"] = AnswerStatus.CORRECT_WITH_TOOL.value
            base_reward = self.config.reward_correct_with_tool

            tool_penalty = len(tool_calls) * self.config.tool_usage_penalty
            final_reward = max(base_reward - tool_penalty, 0.1)

            reward_details["base_reward"] = base_reward
            reward_details["tool_penalty"] = -tool_penalty
            reward_details["total"] = final_reward

        else:
            reward_details["status"] = AnswerStatus.INCORRECT.value
            reward_details["base_reward"] = self.config.reward_incorrect
            reward_details["total"] = self.config.reward_incorrect

        return reward_details["total"], reward_details

    def _check_correct(
        self,
        prediction: str,
        ground_truth: str,
    ) -> bool:
        """正解判定（ロバストな比較）。"""
        pred_clean = prediction.strip().lower()
        gt_clean = ground_truth.strip().lower()

        if pred_clean == gt_clean:
            return True

        try:
            pred_num = float(pred_clean)
            gt_num = float(gt_clean)
            if abs(pred_num - gt_num) < 1e-6:
                return True
        except ValueError:
            pass

        return False

    def compute_group_advantage(
        self,
        rewards: torch.Tensor,
        difficulties: torch.Tensor,
        tool_usage_mask: torch.Tensor,
        group_size: int = 8,
    ) -> torch.Tensor:
        """グループ相対アドバンテージを計算する。

        Args:
            rewards: 報酬 tensor [batch_size, group_size]
            difficulties: 難易度 tensor [batch_size, group_size]
            tool_usage_mask: ツール使用有無 mask [batch_size, group_size]
            group_size: グループサイズ

        Returns:
            アドバンテー tensor
        """
        difficulty_weight = torch.sigmoid(difficulties - 0.5)

        group_mean = rewards.mean(dim=-1, keepdim=True)
        group_std = rewards.std(dim=-1, keepdim=True) + 1e-8
        normalized_rewards = (rewards - group_mean) / group_std

        tool_penalty = tool_usage_mask.float() * 0.5

        advantage = normalized_rewards * difficulty_weight * (1.0 - tool_penalty)

        return advantage

    def compute_entropy_bonus(
        self,
        policy_logits: torch.Tensor,
    ) -> torch.Tensor:
        """方策のエントロピーボーナスを計算する。"""
        probs = torch.softmax(policy_logits, dim=-1)
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return self.config.entropy_coef * entropy


class CrossValidator:
    """交差検証器。

    複数のモデル/手法による回答を比較検証する。

    Attributes:
        threshold: 正解判定の信頼度閾値
    """

    def __init__(
        self,
        threshold: float = 0.8,
    ) -> None:
        self.threshold = threshold

    def validate(
        self,
        prediction: str,
        ground_truth: str,
        *,
        alternative_predictions: Optional[list[str]] = None,
    ) -> tuple[bool, dict]:
        """回答を検証する。

        Args:
            prediction: メインの予測
            ground_truth: 正解
            alternative_predictions: 代替予測のリスト

        Returns:
            (妥当性, 詳細辞書)
        """
        details = {
            "main_correct": False,
            "consensus": False,
            "confidence": 0.0,
            "alternatives": [],
        }

        main_correct = self._check_correct(prediction, ground_truth)
        details["main_correct"] = main_correct

        if alternative_predictions:
            alt_correct = [
                self._check_correct(alt, ground_truth)
                for alt in alternative_predictions
            ]
            details["alternatives"] = alt_correct

            consensus_count = sum(alt_correct) + (1 if main_correct else 0)
            total = len(alt_correct) + 1
            details["consensus"] = consensus_count / total >= self.threshold
            details["confidence"] = consensus_count / total
        else:
            details["consensus"] = main_correct
            details["confidence"] = 1.0 if main_correct else 0.0

        is_valid = main_correct and (
            not alternative_predictions or details["consensus"]
        )

        return is_valid, details

    def _check_correct(
        self,
        prediction: str,
        ground_truth: str,
    ) -> bool:
        """正解判定。"""
        pred_clean = prediction.strip().lower()
        gt_clean = ground_truth.strip().lower()

        if pred_clean == gt_clean:
            return True

        try:
            pred_num = float(pred_clean)
            gt_num = float(gt_clean)
            if abs(pred_num - gt_num) < 1e-6:
                return True
        except ValueError:
            pass

        return False
