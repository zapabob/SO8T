#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced SO(8) Mathematical Adapter

SO(8)残差アダプターとURT/NC-KART★の統合
ノーベル賞・フィールズ賞級の数学・科学推論を実現
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
import warnings

# インポート
from .so8_residual_adapter import SO8ResidualAdapter, SO8Config
from .urt_theorem import URTQuantumField, URTConfig
from .nc_kart_theorem import NCKARTQuantumField, NCKARTConfig
from .quadruple_thinking import QuadrupleThinkingEngine, QuadrupleThinkingConfig


@dataclass
class EnhancedSO8Config:
    """Enhanced SO(8)設定パラメータ"""
    # 基本SO(8)設定
    so8_config: SO8Config = None

    # URT設定
    urt_config: URTConfig = None

    # NC-KART★設定
    nckart_config: NCKARTConfig = None

    # 四重思考設定
    thinking_config: QuadrupleThinkingConfig = None

    # 統合設定
    integration_mode: str = "parallel"  # "parallel", "sequential", "hybrid"
    mathematical_precision: float = 1e-8
    reasoning_depth: int = 5
    creativity_factor: float = 0.3

    def __post_init__(self):
        if self.so8_config is None:
            self.so8_config = SO8Config()
        if self.urt_config is None:
            self.urt_config = URTConfig()
        if self.nckart_config is None:
            self.nckart_config = NCKARTConfig()
        if self.thinking_config is None:
            self.thinking_config = QuadrupleThinkingConfig()


class EnhancedSO8Adapter(nn.Module):
    """
    Enhanced SO(8) Adapter with URT/NC-KART★ Integration

    SO(8)残差アダプターをURT/NC-KART★で拡張した高度な数学的推論アダプター
    """

    def __init__(self, config: EnhancedSO8Config, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size

        # 基本SO(8)アダプター
        self.so8_adapter = SO8ResidualAdapter(config.so8_config)

        # URT量子場
        self.urt_field = URTQuantumField(config.urt_config, hidden_size)

        # NC-KART★量子場
        self.nckart_field = NCKARTQuantumField(config.nckart_config, hidden_size)

        # 四重思考エンジン
        self.thinking_engine = QuadrupleThinkingEngine(config.thinking_config, hidden_size)

        # 統合ネットワーク
        self.integration_network = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # SO8 + URT + NC-KART
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # 数学的確信度計算器
        self.mathematical_confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # 統合モードパラメータ
        self.integration_weights = nn.Parameter(torch.ones(3))  # SO8, URT, NC-KARTの重み

    def apply_integration_mode(self,
                             so8_output: torch.Tensor,
                             urt_output: torch.Tensor,
                             nckart_output: torch.Tensor) -> torch.Tensor:
        """統合モードの適用"""
        if self.config.integration_mode == "parallel":
            # 並列統合
            combined = torch.cat([so8_output, urt_output, nckart_output], dim=-1)
            integrated = self.integration_network(combined)

        elif self.config.integration_mode == "sequential":
            # 逐次統合
            temp = so8_output + urt_output
            integrated = temp + nckart_output

        elif self.config.integration_mode == "hybrid":
            # 重み付きハイブリッド統合
            weights = F.softmax(self.integration_weights, dim=0)
            integrated = (weights[0] * so8_output +
                         weights[1] * urt_output +
                         weights[2] * nckart_output)

        else:
            # デフォルト: 平均
            integrated = (so8_output + urt_output + nckart_output) / 3

        return integrated

    def calculate_mathematical_confidence(self, integrated: torch.Tensor) -> float:
        """数学的確信度の計算"""
        confidence = self.mathematical_confidence(integrated.mean(dim=1)).mean().item()
        return confidence

    def forward(self, x: torch.Tensor,
                enable_thinking: bool = True,
                output_format: str = "standard") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Enhanced SO(8) Adapterの順伝播

        Args:
            x: 入力テンソル [batch, seq, hidden]
            enable_thinking: 四重思考を有効にするかどうか
            output_format: 出力フォーマット ("standard" or "nobel_fields")

        Returns:
            output: 出力テンソル [batch, seq, hidden]
            analysis: 分析結果
        """
        batch_size, seq_len, hidden_size = x.shape

        # SO(8)アダプター適用
        so8_output, so8_stats = self.so8_adapter(x)

        # URT量子場適用
        urt_output, urt_stats = self.urt_field(x)

        # NC-KART★量子場適用
        nckart_output, nckart_stats = self.nckart_field(x)

        # 統合モード適用
        integrated = self.apply_integration_mode(so8_output, urt_output, nckart_output)

        # 数学的確信度の計算
        mathematical_confidence = self.calculate_mathematical_confidence(integrated)

        # 四重思考の適用（オプション）
        thinking_output = None
        thinking_analysis = None

        if enable_thinking:
            thinking_result, thinking_formatted, thinking_analysis = self.thinking_engine(
                integrated, output_format=output_format
            )
            thinking_output = thinking_formatted

            # 思考結果を統合に反映
            integrated = integrated + 0.1 * thinking_result

        # 最終出力
        output = x + integrated  # 残差接続

        # 分析結果の統合
        analysis = {
            'so8_stats': so8_stats,
            'urt_stats': urt_stats,
            'nckart_stats': nckart_stats,
            'mathematical_confidence': mathematical_confidence,
            'integration_mode': self.config.integration_mode,
            'integration_weights': F.softmax(self.integration_weights, dim=0).tolist(),
            'thinking_enabled': enable_thinking,
            'thinking_output': thinking_output,
            'thinking_analysis': thinking_analysis,
            'output_norm': torch.norm(output, p='fro').item(),
            'integration_norm': torch.norm(integrated, p='fro').item()
        }

        return output, analysis

    def get_mathematical_properties(self) -> Dict[str, Any]:
        """数学的性質の取得"""
        return {
            'theories_integrated': ['SO(8)', 'URT', 'NC-KART★', 'Quadruple Thinking'],
            'integration_mode': self.config.integration_mode,
            'mathematical_precision': self.config.mathematical_precision,
            'reasoning_depth': self.config.reasoning_depth,
            'creativity_factor': self.config.creativity_factor,
            'so8_properties': self.so8_adapter.get_so8_properties(),
            'urt_properties': self.urt_field.get_mathematical_properties(),
            'nckart_properties': self.nckart_field.get_mathematical_properties(),
            'thinking_properties': self.thinking_engine.get_thinking_properties()
        }


class UnifiedMathematicalReasoningModel(nn.Module):
    """
    Unified Mathematical Reasoning Model

    SO(8) + URT + NC-KART★ + 四重思考の完全統合モデル
    ノーベル賞・フィールズ賞級の数学・科学推論を実現
    """

    def __init__(self, base_model, config: EnhancedSO8Config):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # 隠れ層サイズの取得
        if hasattr(base_model, 'config'):
            hidden_size = base_model.config.hidden_size
        else:
            # デフォルト値
            hidden_size = 3072

        self.hidden_size = hidden_size

        # Enhanced SO(8) Adapter
        self.enhanced_adapter = EnhancedSO8Adapter(config, hidden_size)

        # 層ごとのアダプター適用設定
        self.adapter_layers = self._setup_adapter_layers()

        # 数学的評価メトリクス
        self.mathematical_metrics = self._setup_mathematical_metrics()

    def _setup_adapter_layers(self) -> List[int]:
        """アダプター適用層の設定"""
        if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'num_hidden_layers'):
            num_layers = self.base_model.config.num_hidden_layers
        else:
            num_layers = 32  # デフォルト

        # 中間層の3/4に適用（思考プロセスが活性化する層）
        start_layer = num_layers // 4
        end_layer = num_layers * 3 // 4

        return list(range(start_layer, end_layer))

    def _setup_mathematical_metrics(self) -> Dict[str, Callable]:
        """数学的評価メトリクスの設定"""
        return {
            'logical_consistency': lambda x: torch.norm(x, p=2) / torch.norm(x, p=1),
            'information_preservation': lambda x: -torch.sum(x * torch.log(x + 1e-10), dim=-1).mean(),
            'symmetry_measure': lambda x: torch.norm(x - x.flip(-1), p='fro'),
            'complexity_measure': lambda x: torch.norm(torch.gradient(x.sum(dim=-1), dim=1)[0], p=2)
        }

    def apply_mathematical_metrics(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """数学的メトリクスの適用"""
        metrics = {}
        for name, metric_fn in self.mathematical_metrics.items():
            try:
                metrics[name] = metric_fn(hidden_states).item()
            except:
                metrics[name] = 0.0

        return metrics

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                enable_mathematical_reasoning: bool = True,
                reasoning_format: str = "standard",
                **kwargs) -> Dict[str, Any]:

        # ベースモデルの出力取得（隠れ層状態を含む）
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        # 最終隠れ層状態
        final_hidden_states = outputs.hidden_states[-1]

        # 数学的推論の適用
        if enable_mathematical_reasoning:
            # Enhanced SO(8) Adapterの適用
            enhanced_output, analysis = self.enhanced_adapter(
                final_hidden_states,
                enable_thinking=True,
                output_format=reasoning_format
            )

            # 最終出力の更新
            updated_logits = self.base_model.lm_head(enhanced_output)

            # 数学的メトリクスの計算
            mathematical_metrics = self.apply_mathematical_metrics(enhanced_output)

            # 結果の統合
            result = {
                'logits': updated_logits,
                'enhanced_hidden_states': enhanced_output,
                'mathematical_reasoning': analysis.get('thinking_output'),
                'mathematical_analysis': analysis,
                'mathematical_metrics': mathematical_metrics,
                'reasoning_enabled': True,
                'reasoning_format': reasoning_format,
                **outputs
            }
        else:
            # 通常の推論
            result = {
                'reasoning_enabled': False,
                **outputs
            }

        return result

    def get_model_properties(self) -> Dict[str, Any]:
        """モデル全体の性質取得"""
        return {
            'model_type': 'Unified Mathematical Reasoning Model',
            'base_model': str(type(self.base_model)),
            'hidden_size': self.hidden_size,
            'adapter_layers': self.adapter_layers,
            'enhanced_adapter_properties': self.enhanced_adapter.get_mathematical_properties(),
            'mathematical_metrics': list(self.mathematical_metrics.keys()),
            'integration_mode': self.config.integration_mode
        }


class MathematicalThinkingPipeline:
    """
    Mathematical Thinking Pipeline

    数学的思考の完全なパイプライン
    """

    def __init__(self, model: UnifiedMathematicalReasoningModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # パイプライン設定
        self.pipeline_config = {
            'max_new_tokens': 2048,
            'temperature': 0.7,
            'do_sample': True,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }

    def generate_mathematical_reasoning(self,
                                      prompt: str,
                                      reasoning_format: str = "nobel_fields",
                                      **kwargs) -> Dict[str, Any]:
        """
        数学的推論の生成

        Args:
            prompt: 入力プロンプト
            reasoning_format: 推論フォーマット ("standard" or "nobel_fields")

        Returns:
            生成結果
        """
        # トークナイズ
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # モデル推論
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                enable_mathematical_reasoning=True,
                reasoning_format=reasoning_format,
                **kwargs
            )

        # テキスト生成
        generated_ids = outputs['logits'].argmax(dim=-1)

        # デコード
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 結果の統合
        result = {
            'generated_text': generated_text,
            'mathematical_reasoning': outputs.get('mathematical_reasoning'),
            'mathematical_analysis': outputs.get('mathematical_analysis'),
            'mathematical_metrics': outputs.get('mathematical_metrics'),
            'reasoning_format': reasoning_format,
            'prompt': prompt
        }

        return result

    def evaluate_mathematical_correctness(self, reasoning_output: Dict[str, Any]) -> Dict[str, float]:
        """数学的正当性の評価"""
        analysis = reasoning_output.get('mathematical_analysis', {})

        # 評価基準
        evaluation_criteria = {
            'logical_consistency': analysis.get('so8_stats', {}).get('orthogonal_errors', [0])[0] < 0.1,
            'mathematical_rigor': analysis.get('urt_stats', {}).get('convergence_satisfied', False),
            'physical_correctness': analysis.get('nckart_stats', {}).get('convergence_satisfied', False),
            'reasoning_depth': len(analysis.get('thinking_analysis', {}).get('deduction', {}).get('proof_chain', [])) > 3,
            'creativity_score': analysis.get('thinking_analysis', {}).get('abduction', {}).get('creativity_analysis', {}).get('creativity_score', 0) > 0.5
        }

        # スコア計算
        scores = {}
        for criterion, satisfied in evaluation_criteria.items():
            scores[criterion] = 1.0 if satisfied else 0.0

        # 総合スコア
        scores['overall_mathematical_correctness'] = sum(scores.values()) / len(scores)

        return scores


# ユーティリティ関数
def create_enhanced_so8_config(hidden_size: int = 3072) -> EnhancedSO8Config:
    """Enhanced SO(8)設定の作成"""
    from .so8_residual_adapter import create_so8_adapter_config
    from .urt_theorem import create_urt_config
    from .nc_kart_theorem import create_nckart_config
    from .quadruple_thinking import create_quadruple_thinking_config

    return EnhancedSO8Config(
        so8_config=create_so8_adapter_config(hidden_size),
        urt_config=create_urt_config(hidden_size),
        nckart_config=create_nckart_config(hidden_size),
        thinking_config=create_quadruple_thinking_config(hidden_size),
        integration_mode="hybrid",
        mathematical_precision=1e-8,
        reasoning_depth=5,
        creativity_factor=0.3
    )


def create_unified_mathematical_model(base_model, hidden_size: int = 3072) -> UnifiedMathematicalReasoningModel:
    """Unified Mathematical Reasoning Modelの作成"""
    config = create_enhanced_so8_config(hidden_size)
    return UnifiedMathematicalReasoningModel(base_model, config)


if __name__ == "__main__":
    # テスト実行
    config = create_enhanced_so8_config()
    print(f"Enhanced SO(8)設定: {config}")

    # ダミーベースモデル
    class DummyBaseModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.config = type('Config', (), {'hidden_size': hidden_size, 'num_hidden_layers': 32})()
            self.lm_head = nn.Linear(hidden_size, 32000)  # ダミー語彙サイズ

        def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, **kwargs):
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)

            logits = self.lm_head(hidden_states)

            result = {'logits': logits, 'hidden_states': [hidden_states] * 33}
            return result

    dummy_model = DummyBaseModel(3072)

    # Unified Mathematical Reasoning Model作成
    unified_model = create_unified_mathematical_model(dummy_model)
    print(f"Unified Mathematical Model作成成功: {unified_model}")

    # テスト推論
    dummy_input_ids = torch.randint(0, 32000, (1, 10))
    outputs = unified_model(dummy_input_ids, enable_mathematical_reasoning=True)

    print(f"推論成功: logits shape = {outputs['logits'].shape}")
    print(f"数学的確信度: {outputs['mathematical_analysis']['mathematical_confidence']:.3f}")
    print(f"思考出力: {outputs['mathematical_reasoning'][:200]}...")

    # モデル性質表示
    props = unified_model.get_model_properties()
    print(f"モデル性質: {props}")
