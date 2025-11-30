#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT PPO Reward Function
SO(8)四重推論に基づく報酬関数実装
"""

import re
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using fallback keyword-based isomorphism detection.")

class NKATRewardFunction:
    """NKAT理論に基づくPPO報酬関数"""

    def __init__(self, tokenizer=None, base_model=None):
        self.tokenizer = tokenizer
        self.base_model = base_model

        # Embeddingモデル初期化（軽量モデルを使用）
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Embedding model loaded for isomorphism detection.")
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None

        # 同型性キーワード（圏論・数学・物理）
        self.isomorphism_keywords = [
            # 日本語
            "同型", "同型性", "アイソモルフィズム", "アナロジー", "類似性", "対応", "写像", "射影",
            "対称性", "不変量", "保存則", "変換", "マッピング", "圏論", "函手", "自然変換",
            "スペクトル", "幾何学", "位相", "ホモロジー", "コホモロジー", "ファイバー",
            "バンドル", "接続", "曲率", "テンソル", "微分形式", "多様体", "群論",

            # 英語
            "isomorphism", "isomorphic", "analogy", "similarity", "correspondence", "mapping",
            "projection", "symmetry", "invariant", "conservation", "transformation",
            "category theory", "functor", "natural transformation", "spectrum", "geometry",
            "topology", "homology", "cohomology", "fiber", "bundle", "connection",
            "curvature", "tensor", "differential form", "manifold", "group theory"
        ]

        # ハルシネーション検出キーワード（物理法則違反）
        self.hallucination_keywords = [
            "永遠機関", "超光速", "エントロピー減少", "熱力学第二法則違反",
            "perpetual motion", "faster than light", "entropy decrease",
            "violation of second law", "infinite energy"
        ]

        # 構造評価用の正規表現パターン
        self.structure_patterns = {
            'think_start': r'<think>',
            'think_end': r'</think>',
            'final_start': r'<final>',
            'final_end': r'</final>',
            'observation': r'1\.?\s*(?:Observation|観測|事実)',
            'deduction': r'2\.?\s*(?:Deduction|論理|推論)',
            'abduction': r'3\.?\s*(?:Abduction|直感|同型性|Isomorphism)',
            'integration': r'4\.?\s*(?:Integration|統合|結論)'
        }

    def __call__(self, prompts: List[str], responses: List[str], **kwargs) -> List[float]:
        """PPO報酬関数メイン"""
        rewards = []

        for prompt, response in zip(prompts, responses):
            reward = self.calculate_reward(prompt, response)
            rewards.append(reward)

        return rewards

    def calculate_reward(self, prompt: str, response: str) -> float:
        """単一の応答に対する報酬を計算"""
        base_reward = 0.0

        # 1. 構造報酬（Structure Reward）
        structure_score = self._evaluate_structure(response)
        base_reward += structure_score * 2.0  # 構造は重要なので重み2.0

        # 2. 同型性報酬（Isomorphism Reward）
        isomorphism_score = self._evaluate_isomorphism(response)
        structure_mapping_score = self._evaluate_structure_mapping(response)
        total_isomorphism_score = isomorphism_score + structure_mapping_score
        base_reward += total_isomorphism_score * 1.5

        # 3. URT安定性報酬（URT Stability Reward）
        stability_score = self._evaluate_stability(response)
        base_reward += stability_score * 1.8

        # 4. 負の報酬（Negative Reward）
        negative_score = self._evaluate_negative(response, prompt)
        base_reward += negative_score

        # 5. 長さペナルティ（あまりに短い回答は減点）
        length_penalty = self._length_penalty(response)
        base_reward += length_penalty

        # 6. 数学的・物理的正確性ボーナス
        accuracy_bonus = self._evaluate_accuracy(response)
        base_reward += accuracy_bonus

        return base_reward

    def _evaluate_structure(self, response: str) -> float:
        """構造報酬を評価（0-1の範囲）"""
        score = 0.0

        # <think>タグの存在確認
        if not re.search(self.structure_patterns['think_start'], response):
            return 0.0

        if not re.search(self.structure_patterns['think_end'], response):
            return 0.1  # 開始タグだけある場合少し点

        # <final>タグの存在確認
        if not re.search(self.structure_patterns['final_start'], response):
            score -= 0.2

        # 四重推論の各段階が存在するかチェック
        think_content = self._extract_think_content(response)
        if think_content:
            if re.search(self.structure_patterns['observation'], think_content, re.IGNORECASE):
                score += 0.2
            if re.search(self.structure_patterns['deduction'], think_content, re.IGNORECASE):
                score += 0.2
            if re.search(self.structure_patterns['abduction'], think_content, re.IGNORECASE):
                score += 0.3  # 同型性は重要
            if re.search(self.structure_patterns['integration'], think_content, re.IGNORECASE):
                score += 0.3

        return min(1.0, max(0.0, score))

    def _evaluate_isomorphism(self, response: str) -> float:
        """同型性報酬を評価（0-1の範囲）- Embeddingベース拡張版"""
        score = 0.0

        # 同型性キーワードのカウント（基本スコア）
        keyword_count = 0
        for keyword in self.isomorphism_keywords:
            if keyword.lower() in response.lower():
                keyword_count += 1

        # キーワード数に応じた基本スコア
        if keyword_count > 0:
            score += min(0.3, keyword_count * 0.1)

        # 高度な数学的用語のボーナス
        advanced_terms = ["圏論", "ホモトピー", "スペクトル系列", "category theory", "homotopy", "spectral sequence"]
        for term in advanced_terms:
            if term.lower() in response.lower():
                score += 0.15

        # 異なる分野の比較・アナロジーの言及
        if "アナロジー" in response or "analogy" in response.lower():
            score += 0.2

        # 構造的類似性の言及
        structural_terms = ["構造的", "対称性", "変換", "structural", "symmetry", "transformation"]
        for term in structural_terms:
            if term.lower() in response.lower():
                score += 0.1

        # Embeddingベースの高度な同型性検出
        if self.embedding_model is not None:
            embedding_score = self._evaluate_isomorphism_with_embedding(response)
            score += embedding_score

        return min(1.0, score)

    def _evaluate_isomorphism_with_embedding(self, response: str) -> float:
        """Embeddingベースの同型性検出"""
        score = 0.0

        try:
            # <think>タグ内の内容を抽出
            think_content = self._extract_think_content(response)
            if not think_content:
                return 0.0

            # 概念ペアの抽出（例: "素数分布とエネルギー準位"）
            concept_pairs = self._extract_concept_pairs(think_content)
            if not concept_pairs:
                return 0.0

            discovery_bonus = 0.0

            for concept_a, concept_b in concept_pairs:
                # 概念のEmbeddingを取得
                emb_a = self.embedding_model.encode([concept_a], convert_to_tensor=True)
                emb_b = self.embedding_model.encode([concept_b], convert_to_tensor=True)

                # 意味的距離（コサイン類似度）
                similarity = torch.cosine_similarity(emb_a, emb_b).item()

                # 遠い概念（similarity < 0.3）間の構造的類似性を説明している場合
                if similarity < 0.3:
                    # 説明の質を評価（構造的類似性を論理的に説明しているか）
                    if self._has_structural_explanation(think_content, concept_a, concept_b):
                        discovery_bonus += 0.4  # 高い発見報酬

            score += min(0.4, discovery_bonus)

        except Exception as e:
            print(f"Embedding-based isomorphism evaluation failed: {e}")
            # フォールバック: キーワードベースのみ
            pass

        return score

    def _evaluate_structure_mapping(self, response: str) -> float:
        """Structure Mapping Reward: 関係性の保存を評価 (Gemini提案)"""
        score = 0.0

        try:
            if self.embedding_model is None:
                return 0.0

            # <think>タグ内の内容を抽出
            think_content = self._extract_think_content(response)
            if not think_content:
                return 0.0

            # 複数の概念ペアを抽出
            concept_pairs = self._extract_concept_pairs(think_content)
            if len(concept_pairs) < 2:
                return 0.0

            # ペアごとのembeddingを取得
            pair_embeddings = []
            for concept_a, concept_b in concept_pairs[:4]:  # 最大4ペアまで
                try:
                    emb_a = self.embedding_model.encode([concept_a], convert_to_tensor=True)
                    emb_b = self.embedding_model.encode([concept_b], convert_to_tensor=True)
                    pair_embeddings.append((emb_a.squeeze(), emb_b.squeeze()))
                except Exception:
                    continue

            if len(pair_embeddings) < 2:
                return 0.0

            # 構造マッピングの評価
            # (v_B - v_A)と(v_D - v_C)の類似度を評価
            mapping_bonus = 0.0

            for i in range(len(pair_embeddings)):
                for j in range(i + 1, len(pair_embeddings)):
                    emb_a1, emb_b1 = pair_embeddings[i]
                    emb_a2, emb_b2 = pair_embeddings[j]

                    # 関係ベクトルの計算: (B-A) と (D-C)
                    relation_vec_1 = emb_b1 - emb_a1
                    relation_vec_2 = emb_b2 - emb_a2

                    # 関係ベクトルの類似度
                    relation_similarity = torch.cosine_similarity(
                        relation_vec_1.unsqueeze(0),
                        relation_vec_2.unsqueeze(0)
                    ).item()

                    # 関係が似ている場合 (構造的アナロジー)
                    if relation_similarity > 0.7:  # 高い類似度閾値
                        # 元の概念は遠い方が良い (異なる分野の類似性)
                        concept_similarity = torch.cosine_similarity(
                            torch.stack([emb_a1, emb_b1]).mean(dim=0).unsqueeze(0),
                            torch.stack([emb_a2, emb_b2]).mean(dim=0).unsqueeze(0)
                        ).item()

                        if concept_similarity < 0.4:  # 概念は遠い
                            mapping_bonus += 0.3  # 構造マッピング発見ボーナス

            score = min(0.5, mapping_bonus)  # 最大0.5に制限

        except Exception as e:
            print(f"Structure mapping evaluation failed: {e}")
            score = 0.0

        return score

    def _extract_concept_pairs(self, text: str) -> List[Tuple[str, str]]:
        """テキストから概念ペアを抽出"""
        pairs = []

        # パターン1: "AとBの類似性" や "A is similar to B"
        patterns = [
            r'([^\n]{3,20})と([^\n]{3,20})の(類似性|対応|同型性|アナロジー)',
            r'([^\n]{3,20})と([^\n]{3,20})の関係',
            r'([^\n]{3,20}) (?:is similar to|corresponds to|is isomorphic to) ([^\n]{3,20})',
            r'([^\n]{3,20})と([^\n]{3,20})の間の(マッピング|写像)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    concept_a, concept_b = match[0], match[1]
                else:
                    # 単一のタプルとして扱う場合の調整
                    continue
                if len(concept_a.strip()) > 2 and len(concept_b.strip()) > 2:
                    pairs.append((concept_a.strip(), concept_b.strip()))

        return pairs

    def _has_structural_explanation(self, text: str, concept_a: str, concept_b: str) -> bool:
        """構造的説明が存在するかチェック"""
        # 構造的説明の指標
        structural_indicators = [
            "構造", "対称性", "変換", "不変量", "保存則", "群", "代数",
            "structural", "symmetry", "transformation", "invariant", "conservation", "group", "algebra",
            "位相", "幾何", "topology", "geometry",
            "スペクトル", "spectrum", "固有値", "eigenvalue"
        ]

        # 両方の概念と構造的指標が同時に出現するか
        has_concept_a = concept_a.lower() in text.lower()
        has_concept_b = concept_b.lower() in text.lower()
        has_structural = any(indicator.lower() in text.lower() for indicator in structural_indicators)

        return has_concept_a and has_concept_b and has_structural

    def _evaluate_stability(self, response: str) -> float:
        """URT安定性報酬を評価（論理的整合性）"""
        score = 0.5  # ベーススコア

        # 自己矛盾のチェック
        contradictions = [
            ("不可能", "可能"),
            ("存在しない", "存在する"),
            ("0", "無限大"),
            ("impossible", "possible"),
            ("nonexistent", "exists"),
            ("zero", "infinity")
        ]

        contradiction_penalty = 0
        for contra_pair in contradictions:
            if contra_pair[0] in response and contra_pair[1] in response:
                contradiction_penalty += 0.2

        score -= contradiction_penalty

        # 論理的結論の明確さ
        if "したがって" in response or "therefore" in response.lower():
            score += 0.2

        # 数学的証明の言及
        if "証明" in response or "proof" in response.lower():
            score += 0.2

        # 参照整合性（同じ用語の一貫した使用）
        # 簡易チェック：同じ数学記号が一貫して使われているか
        math_symbols = [r'\(', r'\)', r'\[', r'\]', r'=', r'≠', r'≈', r'∫', r'∑', r'∂']
        symbol_consistency = 0
        for symbol in math_symbols:
            if len(re.findall(re.escape(symbol), response)) > 1:
                symbol_consistency += 0.05

        score += min(0.2, symbol_consistency)

        return max(0.0, min(1.0, score))

    def _evaluate_negative(self, response: str, prompt: str) -> float:
        """負の報酬を評価（ペナルティ）"""
        penalty = 0.0

        # ハルシネーション検出
        for keyword in self.hallucination_keywords:
            if keyword.lower() in response.lower():
                penalty -= 0.5

        # プロンプトの繰り返し（コピー行為）
        if prompt in response:
            penalty -= 0.3

        # 意味のない繰り返し
        if response.count(response[:50]) > 2:  # 最初の50文字が3回以上繰り返される
            penalty -= 0.4

        # 構造の欠如
        if "<think>" not in response:
            penalty -= 0.2

        # あまりに短い回答
        if len(response.strip()) < 50:
            penalty -= 0.3

        return penalty

    def _length_penalty(self, response: str) -> float:
        """長さペナルティ"""
        length = len(response.strip())

        if length < 100:
            return -0.2
        elif length > 2000:
            return -0.1  # あまり長すぎるのも減点
        else:
            return 0.0

    def _evaluate_accuracy(self, response: str) -> float:
        """数学的・物理的正確性ボーナス"""
        bonus = 0.0

        # 正しい物理定数・法則の言及
        correct_physics = [
            "光速", "プランク定数", "重力定数", "speed of light", "planck constant",
            "gravitational constant", "e=mc²", "E = mc²", "c²", "relativity"
        ]

        for term in correct_physics:
            if term.lower() in response.lower():
                bonus += 0.05

        # 数学的正確性の兆候
        math_indicators = [
            "証明", "定理", "補題", "定義", "系", "proof", "theorem", "lemma", "definition", "corollary"
        ]

        for indicator in math_indicators:
            if indicator.lower() in response.lower():
                bonus += 0.1

        return min(0.5, bonus)

    def _extract_think_content(self, response: str) -> Optional[str]:
        """<think>タグ内の内容を抽出"""
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        if think_match:
            return think_match.group(1)
        return None

    def _extract_final_content(self, response: str) -> Optional[str]:
        """<final>タグ内の内容を抽出"""
        final_match = re.search(r'<final>(.*?)</final>', response, re.DOTALL | re.IGNORECASE)
        if final_match:
            return final_match.group(1)
        return None

# KLダイバージェンスベースの追加報酬関数（オプション）
class KLStabilityReward(nn.Module):
    """KLダイバージェンスに基づく安定性報酬"""

    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
        """KLダイバージェンスを計算して安定性を評価"""
        with torch.no_grad():
            # ベースモデルの出力確率分布
            base_outputs = self.base_model(input_ids)
            base_logits = base_outputs.logits
            base_probs = torch.softmax(base_logits, dim=-1)

            # 生成モデルの出力確率分布
            gen_outputs = self.base_model(generated_ids)
            gen_logits = gen_outputs.logits
            gen_probs = torch.softmax(gen_logits, dim=-1)

            # KLダイバージェンス計算
            kl_div = torch.nn.functional.kl_div(
                gen_probs.log(),
                base_probs,
                reduction='batchmean'
            )

            # KLダイバージェンスが小さいほど安定（報酬が高くなる）
            stability_reward = -kl_div * 0.1  # スケーリング

            return stability_reward

def create_nkat_reward_function(tokenizer=None, base_model=None) -> NKATRewardFunction:
    """NKAT報酬関数を作成"""
    return NKATRewardFunction(tokenizer=tokenizer, base_model=base_model)

if __name__ == "__main__":
    # テスト実行
    reward_func = create_nkat_reward_function()

    # サンプル応答
    test_responses = [
        """<think>
1. Observation: 問題は微分方程式について
2. Deduction: 標準的な解法を適用
3. Abduction/Isomorphism: これは波動方程式と同型性がある
4. Integration: 解はスペクトル的に安定
</think>
<final>答えはπ/2です</final>""",

        """浅い回答です。""",

        """<think>乱雑な思考</think>""",
    ]

    for i, response in enumerate(test_responses):
        reward = reward_func.calculate_reward("テスト問題", response)
        print(f"Response {i+1} reward: {reward:.3f}")

        # 自己矛盾のチェック
        contradictions = [
            ("不可能", "可能"),
            ("存在しない", "存在する"),
            ("0", "無限大"),
            ("impossible", "possible"),
            ("nonexistent", "exists"),
            ("zero", "infinity")
        ]

        contradiction_penalty = 0
        for contra_pair in contradictions:
            if contra_pair[0] in response and contra_pair[1] in response:
                contradiction_penalty += 0.2

        score -= contradiction_penalty

        # 論理的結論の明確さ
        if "したがって" in response or "therefore" in response.lower():
            score += 0.2

        # 数学的証明の言及
        if "証明" in response or "proof" in response.lower():
            score += 0.2

        # 参照整合性（同じ用語の一貫した使用）
        # 簡易チェック：同じ数学記号が一貫して使われているか
        math_symbols = [r'\(', r'\)', r'\[', r'\]', r'=', r'≠', r'≈', r'∫', r'∑', r'∂']
        symbol_consistency = 0
        for symbol in math_symbols:
            if len(re.findall(re.escape(symbol), response)) > 1:
                symbol_consistency += 0.05

        score += min(0.2, symbol_consistency)

        return max(0.0, min(1.0, score))

    def _evaluate_negative(self, response: str, prompt: str) -> float:
        """負の報酬を評価（ペナルティ）"""
        penalty = 0.0

        # ハルシネーション検出
        for keyword in self.hallucination_keywords:
            if keyword.lower() in response.lower():
                penalty -= 0.5

        # プロンプトの繰り返し（コピー行為）
        if prompt in response:
            penalty -= 0.3

        # 意味のない繰り返し
        if response.count(response[:50]) > 2:  # 最初の50文字が3回以上繰り返される
            penalty -= 0.4

        # 構造の欠如
        if "<think>" not in response:
            penalty -= 0.2

        # あまりに短い回答
        if len(response.strip()) < 50:
            penalty -= 0.3

        return penalty

    def _length_penalty(self, response: str) -> float:
        """長さペナルティ"""
        length = len(response.strip())

        if length < 100:
            return -0.2
        elif length > 2000:
            return -0.1  # あまり長すぎるのも減点
        else:
            return 0.0

    def _evaluate_accuracy(self, response: str) -> float:
        """数学的・物理的正確性ボーナス"""
        bonus = 0.0

        # 正しい物理定数・法則の言及
        correct_physics = [
            "光速", "プランク定数", "重力定数", "speed of light", "planck constant",
            "gravitational constant", "e=mc²", "E = mc²", "c²", "relativity"
        ]

        for term in correct_physics:
            if term.lower() in response.lower():
                bonus += 0.05

        # 数学的正確性の兆候
        math_indicators = [
            "証明", "定理", "補題", "定義", "系", "proof", "theorem", "lemma", "definition", "corollary"
        ]

        for indicator in math_indicators:
            if indicator.lower() in response.lower():
                bonus += 0.1

        return min(0.5, bonus)

    def _extract_think_content(self, response: str) -> Optional[str]:
        """<think>タグ内の内容を抽出"""
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        if think_match:
            return think_match.group(1)
        return None

    def _extract_final_content(self, response: str) -> Optional[str]:
        """<final>タグ内の内容を抽出"""
        final_match = re.search(r'<final>(.*?)</final>', response, re.DOTALL | re.IGNORECASE)
        if final_match:
            return final_match.group(1)
        return None

# KLダイバージェンスベースの追加報酬関数（オプション）
class KLStabilityReward(nn.Module):
    """KLダイバージェンスに基づく安定性報酬"""

    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
        """KLダイバージェンスを計算して安定性を評価"""
        with torch.no_grad():
            # ベースモデルの出力確率分布
            base_outputs = self.base_model(input_ids)
            base_logits = base_outputs.logits
            base_probs = torch.softmax(base_logits, dim=-1)

            # 生成モデルの出力確率分布
            gen_outputs = self.base_model(generated_ids)
            gen_logits = gen_outputs.logits
            gen_probs = torch.softmax(gen_logits, dim=-1)

            # KLダイバージェンス計算
            kl_div = torch.nn.functional.kl_div(
                gen_probs.log(),
                base_probs,
                reduction='batchmean'
            )

            # KLダイバージェンスが小さいほど安定（報酬が高くなる）
            stability_reward = -kl_div * 0.1  # スケーリング

            return stability_reward

def create_nkat_reward_function(tokenizer=None, base_model=None) -> NKATRewardFunction:
    """NKAT報酬関数を作成"""
    return NKATRewardFunction(tokenizer=tokenizer, base_model=base_model)

if __name__ == "__main__":
    # テスト実行
    reward_func = create_nkat_reward_function()

    # サンプル応答
    test_responses = [
        """<think>
1. Observation: 問題は微分方程式について
2. Deduction: 標準的な解法を適用
3. Abduction/Isomorphism: これは波動方程式と同型性がある
4. Integration: 解はスペクトル的に安定
</think>
<final>答えはπ/2です</final>""",

        """浅い回答です。""",

        """<think>乱雑な思考</think>""",
    ]

    for i, response in enumerate(test_responses):
        reward = reward_func.calculate_reward("テスト問題", response)
        print(f"Response {i+1} reward: {reward:.3f}")
