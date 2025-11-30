#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT (Non-Commutative Kolmogorov-Arnold Theory) Utilities
AEGIS-v2.0の報酬関数と動的温度制御システム

このモジュールは、NKAT理論に基づく以下の機能をを実装します：
- 四値分類タグの報酬関数
- SO(8)幾何学に基づく構造報酬
- 同型性検出報酬
- 動的温度制御（NKAT Thermostat）

著者: AI Agent (峯岸亮さん仕様に基づく)
日付: 2025-11-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LogitsProcessor
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer
import re
import logging

logger = logging.getLogger(__name__)

class NKATRewardFunction:
    """
    NKAT理論に基づく報酬関数

    四つの報酬要素：
    1. タグ選択正解率報酬
    2. 構造的報酬（SO(8)幾何学的類似性）
    3. 同型性報酬（概念間の構造的対応）
    4. 安全性報酬（NSFW/倫理的拒否）
    5. 安定性報酬（思考プロセスの一貫性）
    """

    def __init__(self):
        # 埋め込みモデル（同型性検出用）
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Embedder読み込み失敗: {e}")
            self.embedder = None

        # 思考プロセス解析パターン
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        self.observation_pattern = re.compile(r'<\|observation\|>(.*?)(?=<\||$)', re.DOTALL)
        self.deduction_pattern = re.compile(r'<\|deduction\|>(.*?)(?=<\||$)', re.DOTALL)
        self.abduction_pattern = re.compile(r'<\|abduction\|>(.*?)(?=<\||$)', re.DOTALL)
        self.integration_pattern = re.compile(r'<\|integration\|>(.*?)(?=<\|final\|>|$)', re.DOTALL)

        # 論理的誤りパターン
        self.logical_errors = [
            r'\b(contradiction|paradox|false.*proof|wrong.*theorem)\b',
            r'\b(impossible|violates?.*law|breaks?.*principle)\b',
            r'\b(incorrect|invalid|faulty)\b.*\b(reasoning|logic|argument)\b'
        ]

        # NSFW/倫理的拒否パターン
        self.nsfw_patterns = [
            r'\b(nsfw|porn|sex|erotic|adult|xxx)\b',
            r'\b(kill|murder|rape|harm|violence)\b',
            r'\b(drug|weapon|illegal|hacking)\b'
        ]

    def calculate_tag_accuracy_reward(self, predicted_tag: str, expected_tag: str) -> float:
        """タグ選択正解率報酬"""
        return 1.0 if predicted_tag == expected_tag else 0.0

    def calculate_structure_reward(self, generated_output: str, expected_output: str) -> float:
        """
        構造的報酬計算
        SO(8)幾何学的観点から、出力の構造的類似性を評価
        """
        if not generated_output or not expected_output:
            return 0.0

        try:
            # 思考プロセス構造の解析
            gen_think_blocks = self._extract_thinking_blocks(generated_output)
            exp_think_blocks = self._extract_thinking_blocks(expected_output)

            if not gen_think_blocks or not exp_think_blocks:
                # 思考プロセスがない場合、単純な構造比較
                return self._calculate_text_structure_similarity(generated_output, expected_output)

            # 思考ブロックの構造的一致性
            structure_score = self._calculate_block_structure_similarity(gen_think_blocks, exp_think_blocks)

            # 四重推論の完全性
            completeness_score = self._calculate_inference_completeness(generated_output)

            # SO(8)幾何学的安定性（思考プロセスの一貫性）
            stability_score = self._calculate_geometric_stability(generated_output)

            return 0.4 * structure_score + 0.3 * completeness_score + 0.3 * stability_score

        except Exception as e:
            logger.warning(f"構造報酬計算エラー: {e}")
            return 0.0

    def calculate_isomorphism_reward(self, generated_output: str, expected_output: str) -> float:
        """
        同型性報酬計算
        異なる概念間の構造的対応を検出
        """
        if not self.embedder:
            return 0.0

        try:
            # 出力の主要概念を抽出
            gen_concepts = self._extract_key_concepts(generated_output)
            exp_concepts = self._extract_key_concepts(expected_output)

            if not gen_concepts or not exp_concepts:
                return 0.0

            # 概念間の埋め込み類似度
            isomorphism_score = self._calculate_concept_isomorphism(gen_concepts, exp_concepts)

            # 構造的対応の検出
            structural_mapping = self._detect_structural_mapping(generated_output, expected_output)

            return 0.6 * isomorphism_score + 0.4 * structural_mapping

        except Exception as e:
            logger.warning(f"同型性報酬計算エラー: {e}")
            return 0.0

    def calculate_safety_reward(self, generated_output: str, expected_tag: str) -> float:
        """安全性報酬計算"""
        predicted_tag = self._extract_tag_from_output(generated_output)

        # <|refuse|>タグの正しい使用
        if expected_tag == "<|refuse|>":
            if predicted_tag == "<|refuse|>":
                # 正しく拒否した場合
                return 1.0
            else:
                # 拒否すべきものを許可した場合
                return -1.0
        else:
            if predicted_tag == "<|refuse|>":
                # 不必要に拒否した場合
                return -0.5
            else:
                # 適切に許可した場合
                return 0.5

    def calculate_stability_reward(self, generated_output: str) -> float:
        """
        安定性報酬計算
        NKAT理論における思考プロセスの安定性（URT: Unified Resolution Theorem）
        """
        try:
            # 思考プロセスの一貫性
            consistency_score = self._calculate_thinking_consistency(generated_output)

            # 論理的循環の不在
            circularity_penalty = self._detect_logical_circularity(generated_output)

            # エントロピー安定性（出力の確信度）
            entropy_stability = self._calculate_entropy_stability(generated_output)

            stability_score = consistency_score - circularity_penalty + entropy_stability
            return max(0.0, min(1.0, stability_score))

        except Exception as e:
            logger.warning(f"安定性報酬計算エラー: {e}")
            return 0.0

    def _extract_thinking_blocks(self, text: str) -> Dict[str, str]:
        """思考ブロックの抽出"""
        blocks = {}

        # 各推論タイプのブロックを抽出
        if match := self.observation_pattern.search(text):
            blocks['observation'] = match.group(1).strip()
        if match := self.deduction_pattern.search(text):
            blocks['deduction'] = match.group(1).strip()
        if match := self.abduction_pattern.search(text):
            blocks['abduction'] = match.group(1).strip()
        if match := self.integration_pattern.search(text):
            blocks['integration'] = match.group(1).strip()

        return blocks

    def _calculate_text_structure_similarity(self, text1: str, text2: str) -> float:
        """テキスト構造の類似度計算"""
        # 長さの類似性
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 0

        # 単語重複率
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        word_overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0

        return 0.3 * len_ratio + 0.7 * word_overlap

    def _calculate_block_structure_similarity(self, blocks1: Dict, blocks2: Dict) -> float:
        """思考ブロック構造の類似度"""
        if not blocks1 and not blocks2:
            return 1.0
        if not blocks1 or not blocks2:
            return 0.0

        # 共通ブロック数
        common_blocks = set(blocks1.keys()) & set(blocks2.keys())
        total_blocks = set(blocks1.keys()) | set(blocks2.keys())

        if not total_blocks:
            return 1.0

        structure_similarity = len(common_blocks) / len(total_blocks)

        # 各ブロックの内容類似度
        content_similarity = 0.0
        for block_type in common_blocks:
            sim = self._calculate_text_structure_similarity(
                blocks1.get(block_type, ''),
                blocks2.get(block_type, '')
            )
            content_similarity += sim

        if common_blocks:
            content_similarity /= len(common_blocks)

        return 0.5 * structure_similarity + 0.5 * content_similarity

    def _calculate_inference_completeness(self, text: str) -> float:
        """四重推論の完全性評価"""
        blocks = self._extract_thinking_blocks(text)

        # 理想的な推論順序
        ideal_sequence = ['observation', 'deduction', 'abduction', 'integration']
        present_blocks = set(blocks.keys())

        # 完全性スコア（存在するブロック数 / 全ブロック数）
        completeness = len(present_blocks) / len(ideal_sequence)

        # 順序正しさボーナス
        sequence_score = 0.0
        for i, block_type in enumerate(ideal_sequence):
            if block_type in present_blocks:
                sequence_score += 1.0 / (i + 1)  # 早いブロックほど重要

        sequence_score /= sum(1.0 / (i + 1) for i in range(len(ideal_sequence)))

        return 0.7 * completeness + 0.3 * sequence_score

    def _calculate_geometric_stability(self, text: str) -> float:
        """SO(8)幾何学的安定性評価"""
        # 論理的矛盾の検出
        contradictions = sum(1 for pattern in self.logical_errors if re.search(pattern, text, re.IGNORECASE))

        # 自己矛盾ペナルティ
        contradiction_penalty = min(contradictions * 0.2, 1.0)

        # 数学的一貫性（LaTeX使用の安定性）
        latex_stability = self._calculate_latex_consistency(text)

        stability = latex_stability - contradiction_penalty
        return max(0.0, min(1.0, stability))

    def _calculate_latex_consistency(self, text: str) -> float:
        """LaTeX使用の一貫性評価"""
        latex_matches = re.findall(r'\\[a-zA-Z]+(?:\{[^}]*\})*|\$[^$]*\$|\$\$[^\$]*\$\$', text)

        if not latex_matches:
            return 0.5  # LaTeXなしは中間値

        # LaTeX式のバランスチェック
        balance_score = 0.0
        for match in latex_matches:
            if match.startswith('$$') and match.endswith('$$'):
                balance_score += 1.0  # 完全一致
            elif match.startswith('$') and match.endswith('$'):
                balance_score += 0.8  # インライン
            elif match.startswith('\\'):
                balance_score += 0.6  # コマンド

        balance_score /= len(latex_matches)

        # 密度の適切性
        density = len(latex_matches) / len(text.split()) if text.split() else 0
        density_score = 1.0 - abs(density - 0.05) / 0.1  # 最適密度5%前後
        density_score = max(0.0, min(1.0, density_score))

        return 0.6 * balance_score + 0.4 * density_score

    def _extract_key_concepts(self, text: str) -> List[str]:
        """主要概念の抽出"""
        # 長い単語（4文字以上）を概念として抽出
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # 頻度上位の概念を抽出
        from collections import Counter
        word_freq = Counter(words)
        key_concepts = [word for word, freq in word_freq.most_common(10)]

        return key_concepts

    def _calculate_concept_isomorphism(self, concepts1: List[str], concepts2: List[str]) -> float:
        """概念間の同型性計算"""
        if not concepts1 or not concepts2:
            return 0.0

        try:
            # 埋め込み計算
            embeddings1 = self.embedder.encode(concepts1)
            embeddings2 = self.embedder.encode(concepts2)

            # 概念ペア間の類似度行列
            similarity_matrix = np.dot(embeddings1, embeddings2.T)

            # 最適マッチングによる平均類似度
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            avg_similarity = similarity_matrix[row_ind, col_ind].mean()

            return max(0.0, min(1.0, avg_similarity))

        except Exception as e:
            logger.warning(f"概念同型性計算エラー: {e}")
            return 0.0

    def _detect_structural_mapping(self, text1: str, text2: str) -> float:
        """構造的対応の検出"""
        # 論理的構造パターンの検出
        patterns = [
            r'because|therefore|thus|hence|consequently',
            r'if.*then|when.*then',
            r'for example|such as|like',
            r'however|but|although|whereas',
            r'in conclusion|finally|ultimately'
        ]

        pattern_matches1 = sum(1 for pattern in patterns if re.search(pattern, text1, re.IGNORECASE))
        pattern_matches2 = sum(1 for pattern in patterns if re.search(pattern, text2, re.IGNORECASE))

        # 構造パターンの類似性
        if pattern_matches1 + pattern_matches2 == 0:
            return 0.5

        pattern_similarity = min(pattern_matches1, pattern_matches2) / max(pattern_matches1, pattern_matches2)

        # 引用や参照の類似性
        quote_similarity = self._calculate_quote_similarity(text1, text2)

        return 0.6 * pattern_similarity + 0.4 * quote_similarity

    def _calculate_quote_similarity(self, text1: str, text2: str) -> float:
        """引用類似度の計算"""
        quotes1 = re.findall(r'[""''"''"']([^""]+)[""''"''"']', text1)
        quotes2 = re.findall(r'[""''"''"']([^""]+)[""''"''"']', text2)

        if not quotes1 and not quotes2:
            return 0.5
        if not quotes1 or not quotes2:
            return 0.0

        # 引用間の類似度
        similarities = []
        for q1 in quotes1:
            for q2 in quotes2:
                # 単純な文字列類似度
                if q1.lower() == q2.lower():
                    similarities.append(1.0)
                else:
                    # 部分一致
                    words1 = set(q1.lower().split())
                    words2 = set(q2.lower().split())
                    overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
                    similarities.append(overlap)

        return max(similarities) if similarities else 0.0

    def _calculate_thinking_consistency(self, text: str) -> float:
        """思考プロセスの一貫性評価"""
        blocks = self._extract_thinking_blocks(text)

        if not blocks:
            return 0.5  # 思考ブロックなしは中間評価

        # 各ブロックの内容矛盾チェック
        consistency_scores = []
        for block_type, content in blocks.items():
            # 自己矛盾の検出
            contradictions = sum(1 for pattern in self.logical_errors
                               if re.search(pattern, content, re.IGNORECASE))
            block_consistency = max(0.0, 1.0 - contradictions * 0.3)
            consistency_scores.append(block_consistency)

        return np.mean(consistency_scores) if consistency_scores else 0.5

    def _detect_logical_circularity(self, text: str) -> float:
        """論理的循環の検出"""
        # 循環的推論パターン（例: A because B, B because A）
        circular_patterns = [
            r'(.{10,50})\s+because\s+(.{10,50})',
            r'(.{10,50})\s+therefore\s+(.{10,50})'
        ]

        circularity_score = 0.0
        for pattern in circular_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                part1, part2 = match
                # 部分文字列の重複度
                words1 = set(part1.lower().split())
                words2 = set(part2.lower().split())
                overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
                if overlap > 0.5:  # 高い重複は循環の兆候
                    circularity_score += 0.2

        return min(1.0, circularity_score)

    def _calculate_entropy_stability(self, text: str) -> float:
        """エントロピー安定性の計算"""
        # 出力の確信度（単語の繰り返し少なさ）
        words = text.lower().split()
        if len(words) < 10:
            return 0.5

        # 単語のユニーク率
        unique_ratio = len(set(words)) / len(words)

        # 文長の変動性
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

        if len(sentence_lengths) < 2:
            length_variability = 0.5
        else:
            length_variability = np.std(sentence_lengths) / np.mean(sentence_lengths)
            length_variability = 1.0 - min(length_variability, 1.0)  # 変動性が低いほど安定

        return 0.6 * unique_ratio + 0.4 * length_variability

    def _extract_tag_from_output(self, output_text: str) -> str:
        """出力からタグを抽出"""
        tag_patterns = ['<\|allow\|>', '<\|escalation\|>', '<\|deny\|>', '<\|refuse\|>']
        for tag in tag_patterns:
            if tag in output_text:
                return tag
        return "<|allow|>"  # デフォルト


class NKATThermostat(LogitsProcessor):
    """
    NKAT Thermostat: 動的温度制御システム

    <|escalation|>時は温度を上げ（創造性向上）、
    エントロピー増加時は温度を下げ（安定性確保）
    """

    def __init__(self, base_temp: float = 0.7, cool_factor: float = 0.1,
                 heat_factor: float = 1.5, entropy_threshold: float = 3.0):
        self.base_temp = base_temp
        self.cool_factor = cool_factor  # 冷却係数
        self.heat_factor = heat_factor  # 加熱係数
        self.entropy_threshold = entropy_threshold

        # 状態管理
        self.current_temp = base_temp
        self.step_count = 0
        self.entropy_history = []

    def get_temperature(self, input_ids: torch.Tensor) -> float:
        """現在の温度を取得（トークンシーケンスに基づく）"""
        if input_ids is None or len(input_ids.shape) < 2:
            return self.base_temp

        # エントロピー計算
        current_entropy = self._calculate_sequence_entropy(input_ids)

        # <|escalation|>タグ検出
        has_escalation = self._detect_escalation_tag(input_ids)

        # 温度調整
        if has_escalation:
            # エスカレーション時は温度を上げる（創造性）
            self.current_temp = min(self.base_temp * self.heat_factor, 2.0)
        elif current_entropy > self.entropy_threshold:
            # 高エントロピー時は温度を下げる（安定性）
            self.current_temp = max(self.base_temp * self.cool_factor, 0.1)
        else:
            # 通常時はベース温度に戻す（指数移動平均）
            alpha = 0.1
            self.current_temp = alpha * self.base_temp + (1 - alpha) * self.current_temp

        # 履歴更新
        self.entropy_history.append(current_entropy)
        if len(self.entropy_history) > 100:  # 履歴制限
            self.entropy_history.pop(0)

        self.step_count += 1
        return self.current_temp

    def _calculate_sequence_entropy(self, input_ids: torch.Tensor) -> float:
        """トークンシーケンスのエントロピー計算"""
        if input_ids.numel() == 0:
            return 0.0

        # トークン頻度計算
        unique_tokens, counts = torch.unique(input_ids, return_counts=True)
        probabilities = counts.float() / counts.sum()

        # エントロピー計算
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
        return entropy.item()

    def _detect_escalation_tag(self, input_ids: torch.Tensor) -> bool:
        """Detects <|escalation|> tag in input_ids for production (borea-phi3.5-instnct-jp)."""
        # NOTE: In production, obtain the special token id from the tokenizer, not hardcoded.
        try:
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
                ESCALATION_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|escalation|>")
            elif hasattr(self, "ESCALATION_TOKEN_ID"):
                ESCALATION_TOKEN_ID = self.ESCALATION_TOKEN_ID
            else:
                raise AttributeError
        except Exception:
            # As fallback, use default but warn for production safety.
            ESCALATION_TOKEN_ID = 94237  # TODO: Replace with tokenizer-provided id

        input_ids = input_ids.detach() if hasattr(input_ids, 'detach') else input_ids
        # Handle (batch, seq_len) or (seq_len,) shape
        if input_ids is None or input_ids.numel() == 0:
            return False

        # For any shape, check for escalation token presence
        return bool((input_ids == ESCALATION_TOKEN_ID).any().item())

    def get_state(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        return {
            'current_temp': self.current_temp,
            'base_temp': self.base_temp,
            'cool_factor': self.cool_factor,
            'heat_factor': self.heat_factor,
            'step_count': self.step_count,
            'avg_entropy': np.mean(self.entropy_history) if self.entropy_history else 0.0
        }

    def reset(self):
        """状態リセット"""
        self.current_temp = self.base_temp
        self.step_count = 0
        self.entropy_history = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """LogitsProcessorとしての呼び出し"""
        # 温度に基づくスケーリング
        current_temp = self.get_temperature(input_ids)

        if current_temp != 1.0:
            scores = scores / current_temp

        return scores


class NKATInferenceController:
    """
    NKAT推論コントローラー
    四値分類タグに基づく推論モード制御
    """

    def __init__(self, thermostat: NKATThermostat):
        self.thermostat = thermostat
        self.inference_modes = {
            '<|allow|>': {
                'description': '単純タスク - 直感的回答',
                'max_tokens': 512,
                'temperature': 0.3,
                'thinking_required': False
            },
            '<|escalation|>': {
                'description': '複雑タスク - 四重推論プロセス',
                'max_tokens': 2048,
                'temperature': 0.9,
                'thinking_required': True
            },
            '<|deny|>': {
                'description': '論理誤り - 反事実的訂正',
                'max_tokens': 256,
                'temperature': 0.1,
                'thinking_required': False
            },
            '<|refuse|>': {
                'description': '倫理的拒否 - 安全遮断',
                'max_tokens': 128,
                'temperature': 0.0,
                'thinking_required': False
            }
        }

    def get_inference_config(self, tag: str) -> Dict[str, Any]:
        """タグに基づく推論設定を取得"""
        if tag not in self.inference_modes:
            tag = '<|allow|>'  # デフォルト

        config = self.inference_modes[tag].copy()

        # NKAT Thermostatで温度調整
        if tag == '<|escalation|>':
            config['temperature'] = self.thermostat.current_temp

        return config

    def format_thinking_prompt(self, instruction: str, tag: str) -> str:
        """思考プロセス付きプロンプトのフォーマット"""
        if tag != '<|escalation|>':
            return instruction

        # 四重推論の思考テンプレート
        thinking_template = f"""
<think>
<|observation|>
[観察と分析]: 問題を分析し、主要な要素と制約を特定する。
</|observation|>

<|deduction|>
[演繹的推論]: 既知の原理則や定理を適用し、論理的帰結を導く。
</|deduction|>

<|abduction|>
[仮説形成]: 可能な説明や仮説を生成し、最も可能性の高いものを評価する。
</|abduction|>

<|integration|>
[統合的解決]: 複数の視点から解決策を統合し、最終的な回答を導く。
</|integration|>
</think>

<|final|>
最終回答:
"""

        return f"{instruction}\n\n{thinking_template}"


# グローバルインスタンス
nkat_reward_function = NKATRewardFunction()
nkat_thermostat = NKATThermostat()
nkat_controller = NKATInferenceController(nkat_thermostat)
