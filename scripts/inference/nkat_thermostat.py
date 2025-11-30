#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Dynamic Temperature Control (Thermostat)
動的温度制御プロセッサ

NKAT理論に基づく推論時温度制御システム。
エントロピー監視とEscalationトークン判定により、
リアルタイムで推論温度を「冷却」または「加熱」する。

著者: 峯岸亮 (SO8Tプロジェクト)
"""

import torch
from transformers import LogitsProcessor
import torch.nn.functional as F
from typing import Optional, Union


class NKATDynamicTemperature(LogitsProcessor):
    """
    NKAT理論に基づく動的温度制御プロセッサ。

    エントロピーと特定のトークン（Escalation等）を監視し、
    推論温度を動的に「冷却」または「加熱」する。

    物理学的メカニズム:
    - 冷却 (Cooling): エントロピー過大時 → 結晶化 (Crystallization)
    - 加熱 (Heating): Escalation時 → 昇華 (Sublimation)

    NKAT的解釈:
    - SO(8)空間での回転制御
    - スペクトル収束と発散のダイナミック制御
    """

    def __init__(self,
                 base_temp: float = 0.7,
                 cool_factor: float = 0.1,  # 冷却時の倍率 (< 1.0) - Gemini推奨: 0.1-0.2
                 heat_factor: float = 1.5,  # 加熱時の倍率 (> 1.0) - Gemini推奨: 1.2-1.5
                 escalation_token_id: Optional[int] = None,
                 entropy_threshold: float = 4.5,  # エントロピー閾値
                 confidence_threshold: float = 0.9,  # 確信度閾値（低確信時加熱）
                 device: Optional[str] = None):
        """
        NKAT Thermostat初期化

        Args:
            base_temp: ベース温度 (通常時の温度)
            cool_factor: 冷却倍率 (0.1程度で鋭く尖らせる)
            heat_factor: 加熱倍率 (2.0程度で分布を広げる)
            escalation_token_id: EscalationトークンID
            entropy_threshold: エントロピー閾値 (これ以上で冷却)
            confidence_threshold: 確信度閾値 (これ以下で加熱)
            device: デバイス指定
        """
        self.base_temp = base_temp
        self.cool_factor = cool_factor
        self.heat_factor = heat_factor
        self.escalation_token_id = escalation_token_id
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        LogitsProcessorコールバック

        Args:
            input_ids: 現在の入力トークン列
            scores: Logitsスコア (未正規化確率)

        Returns:
            温度制御適用後のLogits
        """
        # 1. 現在の確率分布のエントロピーを計算
        probs = F.softmax(scores, dim=-1)
        log_probs = F.log_softmax(scores, dim=-1)

        # エントロピー H(p) = - sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

        # 最大確率 (確信度)
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]

        # 2. 直前のトークンを確認 (Escalation判定)
        last_token = input_ids[:, -1].unsqueeze(-1)
        is_escalation = (last_token == self.escalation_token_id) if self.escalation_token_id is not None else False

        # 3. 温度スケーリング係数の決定
        # 初期値: 1.0 (変化なし)
        temp_modifiers = torch.ones_like(entropy, device=self.device)

        # === Case A: Escalation (Heating) ===
        # Escalationタグが出た直後は、創造的な飛躍が必要 -> 加熱
        if self.escalation_token_id is not None:
            temp_modifiers = torch.where(
                is_escalation,
                torch.tensor(self.heat_factor, device=self.device),
                temp_modifiers
            )

        # === Case B: High Entropy / Hallucination Risk (Cooling) ===
        # エントロピーが高すぎる(迷っている)場合 -> 冷却して結晶化
        # ただし、Escalation時は加熱を優先
        high_entropy_mask = (entropy > self.entropy_threshold) & (~is_escalation)
        temp_modifiers = torch.where(
            high_entropy_mask,
            torch.tensor(self.cool_factor, device=self.device),
            temp_modifiers
        )

        # === Case C: Stuck / Low Confidence Loop (Heating) ===
        # 確信度が低すぎる場合も、局所解から脱出するために加熱
        # ただし、Escalation時や高エントロピー時は優先度を下げる
        low_confidence_mask = (max_prob < self.confidence_threshold) & (~is_escalation) & (~high_entropy_mask)
        temp_modifiers = torch.where(
            low_confidence_mask,
            torch.tensor(self.heat_factor * 0.8, device=self.device),  # 少し弱めの加熱
            temp_modifiers
        )

        # 4. Logitsへの適用
        # 温度制御: scores / (base_temp * modifier)
        # - 温度を上げる = 分布を平らにする = Logitsを小さくする
        # - 温度を下げる = 分布を尖らせる = Logitsを大きくする

        final_temp_scale = self.base_temp * temp_modifiers

        # ゼロ除算防止
        final_temp_scale = torch.clamp(final_temp_scale, min=0.01)

        return scores / final_temp_scale


class NKATThermostatController:
    """
    NKAT Thermostatの制御クラス
    推論中の統計情報収集と制御パラメータの動的調整を行う
    """

    def __init__(self, thermostat: NKATDynamicTemperature):
        self.thermostat = thermostat
        self.entropy_history = []
        self.temp_history = []
        self.escalation_count = 0

    def update_statistics(self, entropy: float, temp_modifier: float, was_escalation: bool):
        """統計情報の更新"""
        self.entropy_history.append(entropy)
        self.temp_history.append(temp_modifier)
        if was_escalation:
            self.escalation_count += 1

    def get_statistics(self) -> dict:
        """統計情報の取得"""
        return {
            'avg_entropy': sum(self.entropy_history) / len(self.entropy_history) if self.entropy_history else 0,
            'avg_temp_modifier': sum(self.temp_history) / len(self.temp_history) if self.temp_history else 0,
            'escalation_count': self.escalation_count,
            'total_steps': len(self.entropy_history)
        }

    def adapt_parameters(self):
        """統計に基づくパラメータ適応（学習）"""
        if len(self.entropy_history) < 10:
            return  # サンプルが少ない場合は適応しない

        stats = self.get_statistics()

        # エントロピーが平均的に高い場合、閾値を上げる
        if stats['avg_entropy'] > 5.0:
            self.thermostat.entropy_threshold = min(self.thermostat.entropy_threshold + 0.1, 6.0)

        # Escalationが多すぎる場合、heat_factorを下げる
        escalation_rate = stats['escalation_count'] / stats['total_steps']
        if escalation_rate > 0.1:  # 10%を超えると多すぎる
            self.thermostat.heat_factor = max(self.thermostat.heat_factor * 0.95, 1.1)


def create_nkat_thermostat(tokenizer=None,
                          base_temp: float = 0.7,
                          cool_factor: float = 0.1,
                          heat_factor: float = 2.0,
                          entropy_threshold: float = 4.5) -> NKATDynamicTemperature:
    """
    NKAT Thermostatのファクトリ関数

    Args:
        tokenizer: トークナイザー (escalation_token_id取得用)
        base_temp: ベース温度
        cool_factor: 冷却倍率
        heat_factor: 加熱倍率
        entropy_threshold: エントロピー閾値

    Returns:
        NKATDynamicTemperatureインスタンス
    """
    # EscalationトークンIDの取得
    escalation_token_id = None
    if tokenizer is not None:
        try:
            # 一般的なescalationトークンを試す
            for token in ['<|escalation|>', '<escalation>', 'escalation', '<ESCALATION>']:
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if token_id != tokenizer.unk_token_id:  # unknown tokenでない場合
                        escalation_token_id = token_id
                        break
        except:
            pass  # トークナイザーが対応していない場合はNoneのまま

    return NKATDynamicTemperature(
        base_temp=base_temp,
        cool_factor=cool_factor,
        heat_factor=heat_factor,
        escalation_token_id=escalation_token_id,
        entropy_threshold=entropy_threshold
    )


# 使用例とテスト関数
def test_nkat_thermostat():
    """NKAT Thermostatのテスト関数"""
    import numpy as np

    # テスト用のLogitsProcessor
    thermostat = NKATDynamicTemperature(
        base_temp=0.7,
        cool_factor=0.1,
        heat_factor=2.0,
        escalation_token_id=12345,  # テスト用ID
        entropy_threshold=4.0
    )

    # テスト用のinput_idsとscores
    batch_size = 2
    vocab_size = 1000

    input_ids = torch.randint(0, vocab_size, (batch_size, 10))

    # 高エントロピーのscores (迷っている状態)
    high_entropy_scores = torch.randn(batch_size, vocab_size) * 2.0

    # 低エントロピーのscores (確信している状態)
    low_entropy_scores = torch.zeros(batch_size, vocab_size)
    low_entropy_scores[:, 0] = 10.0  # 一つのトークンに集中

    # Escalation時のinput_ids
    escalation_input_ids = input_ids.clone()
    escalation_input_ids[:, -1] = 12345  # Escalationトークン

    print("=== NKAT Thermostat Test ===")

    # 通常時
    result_normal = thermostat(input_ids, high_entropy_scores)
    print(f"Normal: shape={result_normal.shape}, mean={result_normal.mean().item():.3f}")

    # 高エントロピー時 (冷却されるはず)
    result_high_ent = thermostat(input_ids, high_entropy_scores)
    print(f"High Entropy: mean={result_high_ent.mean().item():.3f}")

    # Escalation時 (加熱されるはず)
    result_escalation = thermostat(escalation_input_ids, low_entropy_scores)
    print(f"Escalation: mean={result_escalation.mean().item():.3f}")

    print("Test completed successfully!")


if __name__ == '__main__':
    test_nkat_thermostat()

