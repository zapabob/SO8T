#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO自己内省トレーニングモジュール

汎用科学研究およびOSINT AIエージェント向けのGRPO強化学習実装。
DeepSeek-R1方式のGroup Relative Policy Optimizationを使用。

Features:
- 四重推論（SO8T Quadrality）に対応した報酬関数
- VSSI(think-task/analysis/safety/policy)の品質評価
- 自己内省ループによる推論改善
- OSINTソース信頼性評価
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Logging setup
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "grpo_self_reflection.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO設定."""
    group_size: int = 8  # プロンプトあたりの出力数
    kl_penalty_coef: float = 0.1
    reward_scale: float = 1.0
    learning_rate: float = 1e-6
    gradient_accumulation_steps: int = 4
    max_length: int = 4096
    num_epochs: int = 3
    
    # 報酬関数の重み
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "correctness": 0.3,
        "reasoning_quality": 0.25,
        "quadrality_balance": 0.2,
        "safety_compliance": 0.15,
        "format_adherence": 0.1,
    })
    
    # 自己内省設定
    reflection_max_iterations: int = 3
    reflection_improvement_threshold: float = 0.1


class RewardFunction:
    """
    GRPO報酬関数コレクション。
    四重推論品質、安全性、正確性を評価。
    """

    @staticmethod
    def evaluate_correctness(response: str, ground_truth: Optional[str] = None) -> float:
        """正確性評価."""
        if ground_truth is None:
            return 0.5  # 真値なしの場合は中間値
        
        # 簡易的な類似度評価
        response_tokens = set(response.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if not truth_tokens:
            return 0.5
        
        intersection = response_tokens & truth_tokens
        score = len(intersection) / len(truth_tokens)
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def evaluate_reasoning_quality(response: str) -> float:
        """推論品質評価."""
        score = 0.5
        
        # 構造的な推論マーカーの存在をチェック
        reasoning_markers = [
            "したがって", "よって", "つまり", "であるから",
            "therefore", "thus", "because", "hence",
            "まず", "次に", "最後に", "結論として",
        ]
        
        marker_count = sum(1 for m in reasoning_markers if m in response)
        score += min(marker_count * 0.1, 0.3)
        
        # 長さによる質の推定
        length = len(response)
        if 200 < length < 2000:
            score += 0.2
        elif length >= 2000:
            score += 0.1
        
        return min(score, 1.0)

    @staticmethod
    def evaluate_quadrality_balance(response: str) -> float:
        """四重推論バランス評価."""
        perspectives = [
            ("<algebraic>", "</algebraic>"),
            ("<geometric>", "</geometric>"),
            ("<analytic>", "</analytic>"),
            ("<topological>", "</topological>"),
            ("<think-task>", "</think-task>"),
            ("<think-analysis>", "</think-analysis>"),
            ("<think-safety>", "</think-safety>"),
            ("<think-policy>", "</think-policy>"),
        ]
        
        present_count = 0
        for start, end in perspectives:
            if start in response and end in response:
                present_count += 1
        
        # 4視点中何個が存在するか
        if present_count >= 4:
            return 1.0
        elif present_count >= 2:
            return 0.7
        elif present_count >= 1:
            return 0.4
        return 0.2

    @staticmethod
    def evaluate_safety_compliance(response: str) -> float:
        """安全性準拠評価."""
        # 危険キーワードチェック
        danger_keywords = [
            "作り方", "製造方法", "入手方法",
            "how to make", "how to obtain",
        ]
        
        # 安全警告の存在チェック
        safety_markers = [
            "注意", "警告", "危険", "禁止",
            "warning", "caution", "prohibited",
            "研究目的", "教育目的",
        ]
        
        danger_count = sum(1 for k in danger_keywords if k in response.lower())
        safety_count = sum(1 for m in safety_markers if m in response.lower())
        
        if danger_count > 0 and safety_count == 0:
            return 0.2  # 危険内容に警告なし
        elif safety_count > 0:
            return 1.0  # 適切な安全警告あり
        return 0.8

    @staticmethod
    def evaluate_format_adherence(response: str, expected_format: str = "so8t_cot") -> float:
        """形式遵守評価."""
        if expected_format == "so8t_cot":
            # SO8T CoT形式チェック
            has_tags = any(tag in response for tag in ["<algebraic>", "<think-task>", "<synthesis>"])
            has_structure = response.count("\n") >= 5
            
            score = 0.5
            if has_tags:
                score += 0.3
            if has_structure:
                score += 0.2
            return score
        
        return 0.5


class GRPOSelfReflection:
    """
    GRPO自己内省ループ。
    出力を評価し、改善を繰り返す。
    """

    def __init__(self, config: GRPOConfig) -> None:
        self.config = config
        self.reward_fn = RewardFunction()
        self.reflection_history: List[Dict[str, Any]] = []
        logger.info("GRPOSelfReflection initialized.")

    def compute_reward(
        self,
        response: str,
        ground_truth: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """総合報酬計算."""
        rewards = {
            "correctness": self.reward_fn.evaluate_correctness(response, ground_truth),
            "reasoning_quality": self.reward_fn.evaluate_reasoning_quality(response),
            "quadrality_balance": self.reward_fn.evaluate_quadrality_balance(response),
            "safety_compliance": self.reward_fn.evaluate_safety_compliance(response),
            "format_adherence": self.reward_fn.evaluate_format_adherence(response),
        }
        
        # 重み付き総合報酬
        total = sum(
            rewards[k] * self.config.reward_weights.get(k, 0.0)
            for k in rewards
        )
        rewards["total"] = total
        
        return rewards

    def generate_reflection_prompt(
        self,
        original_query: str,
        current_response: str,
        current_rewards: Dict[str, float],
    ) -> str:
        """自己内省プロンプト生成."""
        
        # 最も低いスコアの項目を特定
        weakest_aspect = min(
            [(k, v) for k, v in current_rewards.items() if k != "total"],
            key=lambda x: x[1]
        )
        
        reflection_prompt = f"""以下の応答を改善してください。

## 元の質問
{original_query}

## 現在の応答
{current_response}

## 評価結果
- 正確性: {current_rewards['correctness']:.2f}
- 推論品質: {current_rewards['reasoning_quality']:.2f}
- 四重推論バランス: {current_rewards['quadrality_balance']:.2f}
- 安全性準拠: {current_rewards['safety_compliance']:.2f}
- 形式遵守: {current_rewards['format_adherence']:.2f}
- **総合スコア: {current_rewards['total']:.2f}**

## 改善が必要な点
特に「{weakest_aspect[0]}」（スコア: {weakest_aspect[1]:.2f}）の改善が必要です。

## 指示
上記の評価を踏まえ、改善された応答を生成してください。
SO8T四重推論形式（<think-task>/<think-analysis>/<think-safety>/<think-policy>）を使用し、
各視点からのバランスの取れた分析を含めてください。"""

        return reflection_prompt

    def run_reflection_loop(
        self,
        query: str,
        initial_response: str,
        ground_truth: Optional[str] = None,
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """自己内省ループ実行."""
        logger.info("Starting self-reflection loop...")
        
        current_response = initial_response
        best_response = initial_response
        best_reward = 0.0
        
        history = []
        
        for iteration in range(self.config.reflection_max_iterations):
            # 現在の応答を評価
            rewards = self.compute_reward(current_response, ground_truth)
            
            history.append({
                "iteration": iteration,
                "rewards": rewards,
                "response_length": len(current_response),
            })
            
            logger.info(f"Iteration {iteration}: Total reward = {rewards['total']:.3f}")
            
            # ベスト更新
            if rewards["total"] > best_reward:
                best_reward = rewards["total"]
                best_response = current_response
            
            # 十分な品質に達したら終了
            if rewards["total"] >= 0.9:
                logger.info("High quality threshold reached, stopping early.")
                break
            
            # 改善プロンプト生成
            reflection_prompt = self.generate_reflection_prompt(query, current_response, rewards)
            
            # 改善応答生成（実際のモデルがある場合）
            if generate_fn:
                current_response = generate_fn(reflection_prompt)
            else:
                # デモ用：自己内省による改善をシミュレート
                current_response = self._simulate_improvement(current_response, rewards)
        
        result = {
            "original_query": query,
            "best_response": best_response,
            "best_reward": best_reward,
            "iterations": len(history),
            "history": history,
        }
        
        self.reflection_history.append(result)
        return result

    def _simulate_improvement(self, response: str, rewards: Dict[str, float]) -> str:
        """改善シミュレーション（デモ用）."""
        improved = response
        
        # 四重推論バランスが低い場合、タグを追加
        if rewards["quadrality_balance"] < 0.5:
            if "<think-task>" not in improved:
                improved = "<think-task>\nタスク分析を実行します。\n</think-task>\n\n" + improved
            if "<think-safety>" not in improved:
                improved += "\n\n<think-safety>\nこの応答は安全性を考慮しています。\n</think-safety>"
        
        # 安全性が低い場合、警告を追加
        if rewards["safety_compliance"] < 0.5:
            improved += "\n\n**注意**: この情報は研究・教育目的です。"
        
        return improved


class OSINTAgentGRPO:
    """
    OSINT AIエージェント向けGRPOトレーニング。
    情報収集・分析・検証の品質を強化。
    """

    def __init__(self, config: GRPOConfig) -> None:
        self.config = config
        self.reflection = GRPOSelfReflection(config)
        self.source_credibility_cache: Dict[str, float] = {}
        logger.info("OSINTAgentGRPO initialized.")

    def evaluate_osint_quality(self, response: str, sources: List[str] = None) -> Dict[str, float]:
        """OSINT品質評価."""
        sources = sources or []
        
        scores = self.reflection.compute_reward(response)
        
        # OSINT特有の評価
        osint_scores = {
            "source_diversity": min(len(set(sources)) / 3.0, 1.0) if sources else 0.5,
            "temporal_awareness": 1.0 if any(y in response for y in ["2024", "2025", "2026"]) else 0.5,
            "geopolitical_context": self._evaluate_geopolitical_context(response),
            "cross_verification": self._evaluate_cross_verification(response),
        }
        
        scores.update(osint_scores)
        
        # OSINT総合スコア
        osint_weights = {
            "source_diversity": 0.2,
            "temporal_awareness": 0.15,
            "geopolitical_context": 0.15,
            "cross_verification": 0.2,
        }
        
        osint_total = sum(
            osint_scores.get(k, 0) * w
            for k, w in osint_weights.items()
        )
        
        scores["osint_total"] = (scores["total"] + osint_total) / 2
        return scores

    def _evaluate_geopolitical_context(self, response: str) -> float:
        """地政学的コンテキスト評価."""
        geo_keywords = [
            "地政学", "外交", "安全保障", "国際関係",
            "geopolitical", "diplomatic", "security",
            "ウクライナ", "Ukraine", "ベネズエラ", "Venezuela",
            "日中", "Japan-China", "台湾", "Taiwan",
        ]
        
        count = sum(1 for k in geo_keywords if k.lower() in response.lower())
        return min(count / 3.0, 1.0)

    def _evaluate_cross_verification(self, response: str) -> float:
        """クロス検証評価."""
        verification_markers = [
            "確認", "検証", "裏付け", "複数のソース",
            "verified", "confirmed", "multiple sources",
            "一方で", "しかし", "異なる見解",
        ]
        
        count = sum(1 for m in verification_markers if m.lower() in response.lower())
        return min(count / 2.0, 1.0)

    def generate_training_sample(
        self,
        query: str,
        response: str,
        sources: List[str] = None,
    ) -> Dict[str, Any]:
        """GRPO訓練サンプル生成."""
        scores = self.evaluate_osint_quality(response, sources)
        
        return {
            "query": query,
            "response": response,
            "sources": sources or [],
            "rewards": scores,
            "reward_strategy": {
                "base_reward": scores["total"],
                "osint_bonus": scores["osint_total"] - scores["total"],
                "final_reward": scores["osint_total"],
            },
            "timestamp": datetime.now().isoformat(),
        }


def main() -> None:
    """メインエントリポイント."""
    logger.info("=" * 60)
    logger.info("GRPO Self-Reflection Training Module")
    logger.info("=" * 60)
    
    # 設定
    config = GRPOConfig(
        group_size=8,
        kl_penalty_coef=0.1,
        learning_rate=1e-6,
        reflection_max_iterations=3,
    )
    
    # OSINT GRPOエージェント初期化
    osint_agent = OSINTAgentGRPO(config)
    
    # テスト実行
    test_query = "2024-2026年のウクライナ情勢について分析してください。"
    test_response = """<think-task>
ウクライナ情勢の地政学的分析を実行します。
</think-task>

<think-analysis>
2024年以降のウクライナ情勢は、ロシアの侵攻継続と西側諸国の支援により膠着状態が続いています。
</think-analysis>

<think-safety>
この分析はオープンソース情報に基づいており、機密情報は含まれていません。
</think-safety>

<think-policy>
情報の客観性を維持し、複数ソースからの検証を推奨します。
</think-policy>

<response>
2024-2026年のウクライナ情勢は、長期化する紛争の中で複雑な展開を見せています。
</response>"""
    
    # 評価
    sample = osint_agent.generate_training_sample(
        query=test_query,
        response=test_response,
        sources=["Reuters", "AP", "Kyodo"],
    )
    
    print("\n=== GRPO Training Sample ===")
    print(f"Query: {sample['query'][:50]}...")
    print(f"Rewards: {json.dumps(sample['rewards'], indent=2)}")
    print(f"Final Reward: {sample['reward_strategy']['final_reward']:.3f}")
    
    # 自己内省ループテスト
    reflection_result = osint_agent.reflection.run_reflection_loop(
        query=test_query,
        initial_response=test_response[:200],
    )
    
    print(f"\n=== Self-Reflection Result ===")
    print(f"Iterations: {reflection_result['iterations']}")
    print(f"Best Reward: {reflection_result['best_reward']:.3f}")


if __name__ == "__main__":
    main()
