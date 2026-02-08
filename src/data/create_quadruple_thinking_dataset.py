#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadruple Thinking Dataset Creation Script
SO(8) NKAT理論に基づく四重推論データセット作成
Phi-3.5の内部タグ付けで安全側に倒れる報酬設計
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from datetime import datetime
import random

# Windows cp932エンコーディング対策
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuadrupleThinkingDatasetCreator:
    """SO(8) NKAT Quadruple Thinking Dataset Creator"""

    def __init__(self):
        # 四重推論報酬設計
        self.thinking_rewards = {
            "correct_final_answer": 5.0,        # 四重推論で正解に達する
            "wrong_answer_penalty": -8.0,       # 誤答
            "deny_allow_penalty": -10.0,        # DenyをAllowする
            "unsafe_escalation_penalty": -12.0, # 安全側に倒れないエスカレーション
            "safe_escalation_bonus": 3.0,       # 安全側に倒れるエスカレーション
            "thinking_depth_bonus": 1.0,        # 各思考段階の深さボーナス
            "ethical_reasoning_bonus": 2.0,     # 倫理的推論ボーナス
            "scientific_accuracy_bonus": 2.5,   # 科学的正確性ボーナス
        }

        # 四重推論タグ
        self.thinking_tags = {
            "think_1": "<think-1>",  # 初期分析
            "think_2": "<think-2>",  # 深い分析
            "think_3": "<think-3>",  # 批判的考察
            "think_4": "<think-4>",  # 最終結論
            "final": "<final>",      # 最終出力（外部公開）
        }

        # 危険キーワード（Deny対象）
        self.danger_keywords = [
            "illegal", "hack", "exploit", "virus", "malware", "weapon",
            "drug", "narcotic", "terrorist", "violence", "harm", "suicide",
            "nsfw", "porn", "adult", "explicit", "abuse", "harassment"
        ]

        # 安全キーワード（Allow対象）
        self.safety_keywords = [
            "help", "support", "education", "research", "safety", "ethics",
            "responsible", "positive", "constructive", "beneficial"
        ]

    def create_quadruple_thinking_sft_dataset(self, source_files: List[str], output_file: str):
        """SFTデータセットに四重推論タグを追加"""
        logger.info(f"Creating quadruple thinking SFT dataset from {len(source_files)} sources")

        all_data = []
        total_samples = 0

        for source_file in source_files:
            if not Path(source_file).exists():
                logger.warning(f"Source file not found: {source_file}")
                continue

            logger.info(f"Processing {source_file}")
            with open(source_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Processing {Path(source_file).name}"):
                    try:
                        item = json.loads(line.strip())
                        enhanced_item = self._enhance_sft_with_quadruple_thinking(item)
                        if enhanced_item:
                            all_data.append(enhanced_item)
                            total_samples += 1
                    except Exception as e:
                        logger.warning(f"Error processing line: {e}")
                        continue

        # データセットをシャッフル
        random.shuffle(all_data)

        # 出力
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in tqdm(all_data, desc="Writing enhanced dataset"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Created enhanced SFT dataset with {total_samples} samples")
        return total_samples

    def _enhance_sft_with_quadruple_thinking(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """SFTデータに四重推論タグを追加"""
        if 'instruction' in item and 'output' in item:
            instruction = item['instruction']
            output = item['output']

            # 四重推論形式の応答を生成
            thinking_response = self._generate_quadruple_thinking_response(instruction, output)

            return {
                'instruction': instruction,
                'output': thinking_response,
                'thinking_tags': self.thinking_tags,
                'enhancement_type': 'quadruple_thinking_sft'
            }
        elif 'text' in item:
            # テキスト形式の場合はそのまま
            return item

        return None

    def _generate_quadruple_thinking_response(self, instruction: str, original_output: str) -> str:
        """四重推論形式の応答を生成"""
        thinking_process = f"""
{self.thinking_tags['think_1']}
初期分析: クエリを理解し、潜在的なリスクを評価する。
クエリ内容: {instruction[:100]}...
安全評価: このクエリは{self._assess_query_safety(instruction)}。
{self.thinking_tags['think_2']}
深い分析: 応答の科学的正確性と倫理的影響を考慮。
正確性チェック: 情報提供の正確性を確保。
倫理的考慮: 応答が社会的利益に貢献するか評価。
{self.thinking_tags['think_3']}
批判的考察: 代替案と潜在的な悪影響を検討。
代替アプローチ: より安全で有益な応答方法を検討。
リスク評価: 誤情報や害を及ぼす可能性を評価。
{self.thinking_tags['think_4']}
最終結論: 安全で正確な応答を決定。
決定: {self._make_safe_decision(instruction, original_output)}
正当性: この決定は科学的正確性と倫理的考慮に基づく。
{self.thinking_tags['final']}
{original_output}
        """.strip()

        return thinking_process

    def _assess_query_safety(self, instruction: str) -> str:
        """クエリの安全性を評価"""
        instruction_lower = instruction.lower()

        # 危険キーワードチェック
        for keyword in self.danger_keywords:
            if keyword in instruction_lower:
                return "潜在的に危険 - 慎重な対応が必要"

        # 安全キーワードチェック
        for keyword in self.safety_keywords:
            if keyword in instruction_lower:
                return "安全 - 積極的な支援可能"

        return "中立的 - 標準的な対応"

    def _make_safe_decision(self, instruction: str, output: str) -> str:
        """安全な決定を行う"""
        safety_assessment = self._assess_query_safety(instruction)

        if "危険" in safety_assessment:
            return "安全側に倒れた決定: 潜在的リスクを考慮し、慎重な応答を選択"
        elif "安全" in safety_assessment:
            return "積極的な支援決定: 社会的利益に貢献する有益な応答を選択"
        else:
            return "バランスの取れた決定: 正確性と倫理的考慮を両立"

    def create_quadruple_thinking_grpo_dataset(self, source_files: List[str], output_file: str):
        """GRPOデータセットに四重推論報酬設計を適用"""
        logger.info(f"Creating quadruple thinking GRPO dataset from {len(source_files)} sources")

        all_data = []
        total_samples = 0

        for source_file in source_files:
            if not Path(source_file).exists():
                logger.warning(f"Source file not found: {source_file}")
                continue

            logger.info(f"Processing {source_file}")
            with open(source_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Processing {Path(source_file).name}"):
                    try:
                        item = json.loads(line.strip())
                        enhanced_item = self._enhance_grpo_with_quadruple_thinking_reward(item)
                        if enhanced_item:
                            all_data.append(enhanced_item)
                            total_samples += 1
                    except Exception as e:
                        logger.warning(f"Error processing line: {e}")
                        continue

        # データセットをシャッフル
        random.shuffle(all_data)

        # 出力
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in tqdm(all_data, desc="Writing enhanced GRPO dataset"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Created enhanced GRPO dataset with {total_samples} samples")
        return total_samples

    def _enhance_grpo_with_quadruple_thinking_reward(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """GRPOデータに四重推論報酬設計を適用"""
        if 'query' in item and 'response' in item:
            query = item['query']
            response = item['response']

            # 四重推論報酬を計算
            reward = self._calculate_quadruple_thinking_reward(query, response)

            return {
                'query': query,
                'response': response,
                'reward': reward,
                'reward_breakdown': self._get_reward_breakdown(query, response),
                'thinking_evaluation': self._evaluate_thinking_process(response),
                'enhancement_type': 'quadruple_thinking_grpo'
            }

        return None

    def _calculate_quadruple_thinking_reward(self, query: str, response: str) -> float:
        """四重推論に基づく報酬を計算"""
        base_reward = 0.0

        # 1. 正解/誤答評価
        if self._is_correct_answer(query, response):
            base_reward += self.thinking_rewards["correct_final_answer"]
        else:
            base_reward += self.thinking_rewards["wrong_answer_penalty"]

        # 2. Deny/Allow評価
        if self._allows_dangerous_content(query, response):
            base_reward += self.thinking_rewards["deny_allow_penalty"]

        # 3. エスカレーション評価
        escalation_score = self._evaluate_escalation_safety(query, response)
        base_reward += escalation_score

        # 4. 思考の深さ評価
        thinking_depth = self._evaluate_thinking_depth(response)
        base_reward += thinking_depth * self.thinking_rewards["thinking_depth_bonus"]

        # 5. 倫理的推論評価
        if self._has_ethical_reasoning(response):
            base_reward += self.thinking_rewards["ethical_reasoning_bonus"]

        # 6. 科学的正確性評価
        if self._has_scientific_accuracy(response):
            base_reward += self.thinking_rewards["scientific_accuracy_bonus"]

        return base_reward

    def _is_correct_answer(self, query: str, response: str) -> bool:
        """応答が正しいかどうかを評価"""
        # 簡易的な評価（実際にはより複雑なロジックが必要）
        response_lower = response.lower()

        # 基本的な正確性チェック
        if "don't know" in response_lower or "不明" in response_lower:
            return False

        # 思考プロセスが四重推論タグを含むか
        thinking_tags_present = all(tag in response for tag in self.thinking_tags.values())
        if not thinking_tags_present:
            return False

        return True

    def _allows_dangerous_content(self, query: str, response: str) -> bool:
        """危険なコンテンツを許可しているかどうか"""
        query_lower = query.lower()
        response_lower = response.lower()

        # Denyすべきクエリに対するチェック
        for keyword in self.danger_keywords:
            if keyword in query_lower:
                # 応答が拒否していない場合
                if not any(deny_word in response_lower for deny_word in ["拒否", "deny", "cannot", "unable", "refuse"]):
                    return True

        return False

    def _evaluate_escalation_safety(self, query: str, response: str) -> float:
        """エスカレーションの安全性を評価"""
        response_lower = response.lower()

        # 安全側に倒れる表現
        safe_indicators = [
            "安全を考慮", "倫理的", "責任ある", "慎重に", "安全側",
            "consider safety", "ethical", "responsible", "careful"
        ]

        # 危険側に倒れる表現
        unsafe_indicators = [
            "許可する", "実行可能", "問題ない", "安全", "allow",
            "possible", "no problem", "safe"
        ]

        safe_score = sum(1 for indicator in safe_indicators if indicator in response_lower)
        unsafe_score = sum(1 for indicator in unsafe_indicators if indicator in response_lower)

        if safe_score > unsafe_score:
            return self.thinking_rewards["safe_escalation_bonus"]
        elif unsafe_score > safe_score:
            return self.thinking_rewards["unsafe_escalation_penalty"]
        else:
            return 0.0

    def _evaluate_thinking_depth(self, response: str) -> int:
        """思考の深さを評価（四重推論タグの数）"""
        depth = 0
        for tag in [self.thinking_tags["think_1"], self.thinking_tags["think_2"],
                   self.thinking_tags["think_3"], self.thinking_tags["think_4"]]:
            if tag in response:
                depth += 1
        return depth

    def _has_ethical_reasoning(self, response: str) -> bool:
        """倫理的推論を含むかどうか"""
        ethical_keywords = ["倫理", "道徳", "責任", "影響", "社会的", "ethical", "moral", "responsible", "impact", "social"]
        return any(keyword in response.lower() for keyword in ethical_keywords)

    def _has_scientific_accuracy(self, response: str) -> bool:
        """科学的正確性を含むかどうか"""
        science_keywords = ["証明", "理論", "データ", "根拠", "検証", "proof", "theory", "data", "evidence", "validation"]
        return any(keyword in response.lower() for keyword in science_keywords)

    def _get_reward_breakdown(self, query: str, response: str) -> Dict[str, float]:
        """報酬の内訳を返す"""
        return {
            "correct_answer": self.thinking_rewards["correct_final_answer"] if self._is_correct_answer(query, response) else self.thinking_rewards["wrong_answer_penalty"],
            "deny_allow": self.thinking_rewards["deny_allow_penalty"] if self._allows_dangerous_content(query, response) else 0.0,
            "escalation": self._evaluate_escalation_safety(query, response),
            "thinking_depth": self._evaluate_thinking_depth(response) * self.thinking_rewards["thinking_depth_bonus"],
            "ethical": self.thinking_rewards["ethical_reasoning_bonus"] if self._has_ethical_reasoning(response) else 0.0,
            "scientific": self.thinking_rewards["scientific_accuracy_bonus"] if self._has_scientific_accuracy(response) else 0.0
        }

    def _evaluate_thinking_process(self, response: str) -> Dict[str, Any]:
        """思考プロセスを評価"""
        return {
            "has_all_tags": all(tag in response for tag in self.thinking_tags.values()),
            "thinking_depth": self._evaluate_thinking_depth(response),
            "safe_escalation": self._evaluate_escalation_safety("", response) > 0,
            "ethical_reasoning": self._has_ethical_reasoning(response),
            "scientific_accuracy": self._has_scientific_accuracy(response)
        }


def main():
    """メイン関数"""
    creator = QuadrupleThinkingDatasetCreator()

    # SFTデータセット拡張
    sft_sources = [
        'data/so8t_training_dataset_integrated_50k.jsonl',
        'data/integrated_large_sft_dataset.jsonl'
    ]

    sft_output = 'data/quadruple_thinking_sft_dataset_50k.jsonl'
    sft_count = creator.create_quadruple_thinking_sft_dataset(sft_sources, sft_output)
    print(f"Created SFT dataset: {sft_count} samples")

    # GRPOデータセット拡張
    grpo_sources = [
        'data/enhanced_large_ppo_dataset.jsonl',
        'data/integrated_large_ppo_dataset.jsonl',
        'data/aegis_v21_grpo_50k_with_rewards.jsonl'
    ]

    grpo_output = 'data/quadruple_thinking_grpo_dataset.jsonl'
    grpo_count = creator.create_quadruple_thinking_grpo_dataset(grpo_sources, grpo_output)
    print(f"Created GRPO dataset: {grpo_count} samples")

    print("Quadruple thinking dataset creation completed!")


if __name__ == "__main__":
    main()