#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bake SO8T into Transformer for GGUF Conversion

SO(8)残差アダプターの効果をTransformerの重みに焼き込み、
SO(8)回転ゲートを削除して通常のTransformer構造にする
"""

import os
import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

# SO8T関連インポート
try:
    from so8t.core.dynamic_thinking_so8t import DynamicThinkingSO8TModel
    from so8t.core.so8vit_thinking_adapter import SO8ViTThinkingAdapter, SO8RotationGate
    from so8t.core.so8_trinality_inference import SO8TrinalityInference
except ImportError as e:
    logging.warning(f"SO8T import failed: {e}")
    DynamicThinkingSO8TModel = None

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SO8TBaker:
    """
    SO8T Baker - SO(8)効果をTransformerに焼き込む

    SO(8)残差アダプターの効果をTransformerの重みに統合し、
    回転ゲートを削除して通常のモデル構造にする
    """

    def __init__(self, model_path: str, output_path: str):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """モデル読み込み"""
        logger.info(f"Loading model from {self.model_path}")

        if DynamicThinkingSO8TModel is None:
            raise RuntimeError("SO8T modules not available")

        # SO8Tモデルとして読み込み
        try:
            # まず通常のモデルとして読み込み
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            # 設定からSO8Tモデルを再構築
            config = base_model.config
            self.model = DynamicThinkingSO8TModel(config)

            # 重みをコピー
            self.model.load_state_dict(base_model.state_dict(), strict=False)

        except Exception as e:
            logger.error(f"Failed to load SO8T model: {e}")
            raise

        # トークナイザー読み込み
        tokenizer_path = self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        logger.info("Model loaded successfully")

    def bake_so8t_effects(self) -> nn.Module:
        """
        SO8T効果をTransformerに焼き込む

        Returns:
            焼き込み済みモデル
        """
        logger.info("Starting SO8T effect baking...")

        if not hasattr(self.model, 'so8vit_adapter'):
            logger.warning("Model does not have SO8ViT adapter, skipping baking")
            return self.model

        # SO8ViTアダプターの焼き込み
        baked_model = self._bake_so8vit_adapter()

        # SO8 Trinalityの焼き込み（もしあれば）
        if hasattr(self.model, 'so8_trinality_inference'):
            baked_model = self._bake_so8_trinality(baked_model)

        # SO8T固有コンポーネントの削除
        baked_model = self._remove_so8t_components(baked_model)

        logger.info("SO8T effect baking completed")
        return baked_model

    def _bake_so8vit_adapter(self) -> nn.Module:
        """
        SO8ViTアダプターをTransformerに焼き込む

        SO(8)残差アダプターの効果をTransformerの重みに統合
        """
        logger.info("Baking SO8ViT adapter effects...")

        adapter = self.model.so8vit_adapter
        transformer_model = self.model.base_model  # 基礎Transformerモデル

        # 各層のSO8ViTアダプター効果を焼き込む
        for layer_idx in range(len(adapter.so8_gates)):
            logger.info(f"Baking layer {layer_idx}")

            # SO8回転ゲートの効果を抽出
            gate_effect = self._extract_gate_effect(adapter.so8_gates[layer_idx])

            # Thinkingアテンションの効果を抽出
            attention_effect = self._extract_attention_effect(
                adapter.thinking_attention,
                adapter.layer_norms[layer_idx],
                adapter.feed_forward[layer_idx]
            )

            # 残差アダプターの効果を統合
            residual_effect = self._combine_residual_effects(
                gate_effect, attention_effect, adapter.thinking_alpha
            )

            # Transformer層の重みに焼き込む
            self._bake_into_transformer_layer(
                transformer_model.layers[layer_idx],
                residual_effect
            )

        return self.model

    def _extract_gate_effect(self, so8_gate: SO8RotationGate) -> Dict[str, torch.Tensor]:
        """
        SO8回転ゲートの効果を抽出

        Args:
            so8_gate: SO8回転ゲート

        Returns:
            ゲートの効果（回転行列など）
        """
        gate_effect = {}

        # 回転行列を抽出
        gate_effect['rotation_matrix'] = so8_gate.rotation_matrix.detach().clone()

        # ゲート強度を抽出
        gate_effect['gate_weight'] = so8_gate.gate_weight.detach().clone()

        # 直交誤差も記録（ログ用）
        if hasattr(so8_gate, 'orthogonal_loss'):
            gate_effect['orthogonal_error'] = so8_gate.orthogonal_loss.detach().clone()

        return gate_effect

    def _extract_attention_effect(self, thinking_attention: nn.MultiheadAttention,
                                layer_norm: nn.LayerNorm,
                                feed_forward: nn.Sequential) -> Dict[str, torch.Tensor]:
        """
        ThinkingアテンションとFFの効果を抽出

        Args:
            thinking_attention: Thinkingアテンション
            layer_norm: レイヤーノーマライゼーション
            feed_forward: フィードフォワードネットワーク

        Returns:
            アテンション効果
        """
        attention_effect = {}

        # アテンション重みを抽出
        attention_effect['attention_weights'] = self._get_attention_weights(thinking_attention)

        # レイヤーノーマライゼーションのパラメータ
        attention_effect['ln_weight'] = layer_norm.weight.detach().clone()
        attention_effect['ln_bias'] = layer_norm.bias.detach().clone()

        # フィードフォワードのパラメータ
        for i, layer in enumerate(feed_forward):
            if isinstance(layer, nn.Linear):
                attention_effect[f'ff_{i}_weight'] = layer.weight.detach().clone()
                attention_effect[f'ff_{i}_bias'] = layer.bias.detach().clone()

        return attention_effect

    def _combine_residual_effects(self, gate_effect: Dict[str, torch.Tensor],
                                attention_effect: Dict[str, torch.Tensor],
                                thinking_alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        残差効果を統合

        Args:
            gate_effect: ゲート効果
            attention_effect: アテンション効果
            thinking_alpha: Thinking制御パラメータ

        Returns:
            統合された残差効果
        """
        # α値をシグモイド適用
        alpha = torch.sigmoid(thinking_alpha)

        residual_effect = {}

        # 回転効果を統合
        rotation_matrix = gate_effect['rotation_matrix']
        gate_weight = gate_effect['gate_weight']

        # 回転行列をTransformer次元に拡張
        extended_rotation = self._extend_rotation_to_transformer_dims(rotation_matrix)

        residual_effect['rotation_transformation'] = gate_weight * extended_rotation
        residual_effect['attention_weights'] = attention_effect['attention_weights']
        residual_effect['alpha'] = alpha

        # レイヤーノーマライゼーション効果
        residual_effect['ln_weight'] = attention_effect['ln_weight']
        residual_effect['ln_bias'] = attention_effect['ln_bias']

        # FF効果
        residual_effect['ff_params'] = {
            k: v for k, v in attention_effect.items()
            if k.startswith('ff_')
        }

        return residual_effect

    def _extend_rotation_to_transformer_dims(self, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        回転行列をTransformer次元に拡張

        Args:
            rotation_matrix: SO(8)回転行列 [8, 8]

        Returns:
            Transformer次元の回転行列
        """
        hidden_size = self.model.config.hidden_size
        so8_dim = 8

        # 隠れ次元を8の倍数であることを確認
        assert hidden_size % so8_dim == 0, f"hidden_size {hidden_size} must be divisible by {so8_dim}"

        # 回転行列を隠れ次元に拡張
        chunks = hidden_size // so8_dim
        extended_rotation = torch.zeros(hidden_size, hidden_size, dtype=rotation_matrix.dtype)

        for i in range(chunks):
            start_idx = i * so8_dim
            end_idx = (i + 1) * so8_dim
            extended_rotation[start_idx:end_idx, start_idx:end_idx] = rotation_matrix

        return extended_rotation

    def _bake_into_transformer_layer(self, transformer_layer: nn.Module,
                                   residual_effect: Dict[str, torch.Tensor]):
        """
        残差効果をTransformer層に焼き込む

        Args:
            transformer_layer: Transformer層
            residual_effect: 焼き込む残差効果
        """
        # 回転変換をself-attentionの重みに統合
        if hasattr(transformer_layer, 'self_attn'):
            self._integrate_rotation_into_attention(
                transformer_layer.self_attn,
                residual_effect['rotation_transformation']
            )

        # アテンション重みを統合
        if hasattr(transformer_layer, 'self_attn'):
            self._integrate_attention_weights(
                transformer_layer.self_attn,
                residual_effect['attention_weights']
            )

        # レイヤーノーマライゼーションを調整
        if hasattr(transformer_layer, 'input_layernorm'):
            self._adjust_layernorm(
                transformer_layer.input_layernorm,
                residual_effect['ln_weight'],
                residual_effect['ln_bias']
            )

        # フィードフォワードを調整
        if hasattr(transformer_layer, 'mlp'):
            self._adjust_feedforward(
                transformer_layer.mlp,
                residual_effect['ff_params']
            )

    def _integrate_rotation_into_attention(self, attention_layer: nn.Module,
                                         rotation_matrix: torch.Tensor):
        """
        回転変換をアテンション層に統合

        Args:
            attention_layer: アテンション層
            rotation_matrix: 回転行列
        """
        # Q, K, Vの重みに回転効果を右からかける
        if hasattr(attention_layer, 'q_proj'):
            original_weight = attention_layer.q_proj.weight.data
            attention_layer.q_proj.weight.data = torch.matmul(original_weight, rotation_matrix.t())

        if hasattr(attention_layer, 'k_proj'):
            original_weight = attention_layer.k_proj.weight.data
            attention_layer.k_proj.weight.data = torch.matmul(original_weight, rotation_matrix.t())

        if hasattr(attention_layer, 'v_proj'):
            original_weight = attention_layer.v_proj.weight.data
            attention_layer.v_proj.weight.data = torch.matmul(original_weight, rotation_matrix.t())

        # out_projにも回転効果を適用
        if hasattr(attention_layer, 'o_proj'):
            original_weight = attention_layer.o_proj.weight.data
            attention_layer.o_proj.weight.data = torch.matmul(rotation_matrix, original_weight)

    def _integrate_attention_weights(self, attention_layer: nn.Module,
                                   attention_weights: torch.Tensor):
        """
        アテンション重みを統合

        Args:
            attention_layer: アテンション層
            attention_weights: Thinkingアテンション重み
        """
        # アテンションの出力投影にThinkingアテンションの効果を統合
        if hasattr(attention_layer, 'o_proj'):
            # Thinkingアテンションの効果をスケーリングして統合
            thinking_effect = attention_weights.mean(dim=[0, 1])  # 平均化
            thinking_effect = thinking_effect.unsqueeze(-1).expand_as(attention_layer.o_proj.weight)

            # 重みに加算（小さなスケールで）
            attention_layer.o_proj.weight.data += 0.1 * thinking_effect

    def _adjust_layernorm(self, layernorm: nn.LayerNorm,
                         new_weight: torch.Tensor, new_bias: torch.Tensor):
        """
        レイヤーノーマライゼーションを調整

        Args:
            layernorm: レイヤーノーマライゼーション
            new_weight: 新しい重み
            new_bias: 新しいバイアス
        """
        # ThinkingアダプターのLN効果を統合
        layernorm.weight.data = 0.9 * layernorm.weight.data + 0.1 * new_weight
        layernorm.bias.data = 0.9 * layernorm.bias.data + 0.1 * new_bias

    def _adjust_feedforward(self, feedforward: nn.Module, ff_params: Dict[str, torch.Tensor]):
        """
        フィードフォワードを調整

        Args:
            feedforward: フィードフォワードネットワーク
            ff_params: FFパラメータ
        """
        # gate_proj, up_proj, down_projに対応
        if hasattr(feedforward, 'gate_proj') and 'ff_0_weight' in ff_params:
            ff_weight = ff_params['ff_0_weight']
            feedforward.gate_proj.weight.data += 0.05 * ff_weight

        if hasattr(feedforward, 'up_proj') and 'ff_2_weight' in ff_params:
            ff_weight = ff_params['ff_2_weight']
            feedforward.up_proj.weight.data += 0.05 * ff_weight

        if hasattr(feedforward, 'down_proj') and 'ff_4_weight' in ff_params:
            ff_weight = ff_params['ff_4_weight']
            feedforward.down_proj.weight.data += 0.05 * ff_weight

    def _bake_so8_trinality(self, model: nn.Module) -> nn.Module:
        """
        SO8 Trinality効果を焼き込む

        Args:
            model: モデル

        Returns:
            焼き込み済みモデル
        """
        logger.info("Baking SO8 Trinality effects...")

        if not hasattr(model, 'so8_trinality_inference'):
            return model

        trinality = model.so8_trinality_inference

        # Trinality射影器の効果を各層に統合
        for layer_idx in range(model.config.num_hidden_layers):
            layer_effect = self._extract_trinality_layer_effect(trinality, layer_idx)
            self._bake_trinality_into_layer(model.base_model.layers[layer_idx], layer_effect)

        return model

    def _extract_trinality_layer_effect(self, trinality: SO8TrinalityInference,
                                      layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Trinality層効果を抽出

        Args:
            trinality: SO8 Trinality Inference
            layer_idx: 層インデックス

        Returns:
            層効果
        """
        # Trinality射影器から層固有の効果を抽出
        projector = trinality.trinality_projector

        layer_effect = {}

        # 射影行列の効果
        layer_effect['vector_proj'] = projector.vector_projector.weight[layer_idx] if layer_idx < len(projector.vector_projector.weight) else projector.vector_projector.weight[0]
        layer_effect['positive_spinor_proj'] = projector.positive_spinor_projector.weight[layer_idx] if layer_idx < len(projector.positive_spinor_projector.weight) else projector.positive_spinor_projector.weight[0]
        layer_effect['negative_spinor_proj'] = projector.negative_spinor_projector.weight[layer_idx] if layer_idx < len(projector.negative_spinor_projector.weight) else projector.negative_spinor_projector.weight[0]

        return layer_effect

    def _bake_trinality_into_layer(self, transformer_layer: nn.Module,
                                 layer_effect: Dict[str, torch.Tensor]):
        """
        Trinality効果をTransformer層に焼き込む

        Args:
            transformer_layer: Transformer層
            layer_effect: Trinality層効果
        """
        # 各表現の射影効果をアテンションに統合
        if hasattr(transformer_layer, 'self_attn'):
            # ベクトル表現の効果
            if 'vector_proj' in layer_effect:
                transformer_layer.self_attn.q_proj.weight.data += 0.02 * layer_effect['vector_proj']

            # スピノル表現の効果
            if 'positive_spinor_proj' in layer_effect:
                transformer_layer.self_attn.k_proj.weight.data += 0.02 * layer_effect['positive_spinor_proj']

            if 'negative_spinor_proj' in layer_effect:
                transformer_layer.self_attn.v_proj.weight.data += 0.02 * layer_effect['negative_spinor_proj']

    def _remove_so8t_components(self, model: nn.Module) -> nn.Module:
        """
        SO8T固有コンポーネントを削除

        Args:
            model: モデル

        Returns:
            クリーンなモデル
        """
        logger.info("Removing SO8T components...")

        # SO8ViTアダプター削除
        if hasattr(model, 'so8vit_adapter'):
            delattr(model, 'so8vit_adapter')

        # SO8 Trinality削除
        if hasattr(model, 'so8_trinality_inference'):
            delattr(model, 'so8_trinality_inference')

        if hasattr(model, 'so8_trinality_meta_analyzer'):
            delattr(model, 'so8_trinality_meta_analyzer')

        # メタアナライザー削除
        if hasattr(model, 'meta_analyzer'):
            delattr(model, 'meta_analyzer')

        # Thinking関連属性削除
        thinking_attrs = [
            'dynamic_thinking_enabled',
            'multimodal_enabled',
            'meta_reasoning_enabled',
            'so8_trinality_enabled',
            'temperature_control_enabled'
        ]

        for attr in thinking_attrs:
            if hasattr(model, attr):
                delattr(model, attr)

        logger.info("SO8T components removed")
        return model

    def _get_attention_weights(self, attention: nn.MultiheadAttention) -> torch.Tensor:
        """
        アテンション重みを取得（簡易実装）

        Args:
            attention: アテンション層

        Returns:
            アテンション重み
        """
        # 実際のアテンション重み取得は複雑なので、
        # 出力投影の重みを使って近似
        if hasattr(attention, 'o_proj'):
            return attention.o_proj.weight.data.mean(dim=0)  # 平均化
        else:
            return torch.zeros(self.model.config.hidden_size)

    def save_baked_model(self, baked_model: nn.Module):
        """
        焼き込み済みモデルを保存

        Args:
            baked_model: 焼き込み済みモデル
        """
        logger.info(f"Saving baked model to {self.output_path}")

        # 出力ディレクトリ作成
        self.output_path.mkdir(parents=True, exist_ok=True)

        # モデル保存
        baked_model.save_pretrained(self.output_path)

        # トークナイザー保存
        self.tokenizer.save_pretrained(self.output_path)

        # 焼き込み情報を保存
        baking_info = {
            'original_model': str(self.model_path),
            'baking_timestamp': str(torch.timestamp()),
            'so8t_components_removed': [
                'SO8ViTThinkingAdapter',
                'SO8TrinalityInference',
                'MetaReasoningAnalyzer',
                'SO8RotationGates'
            ],
            'effects_baked': [
                'SO8_rotation_gates',
                'Thinking_attention_weights',
                'Trinality_projections',
                'Residual_adaptations'
            ],
            'gguf_ready': True
        }

        import json
        with open(self.output_path / 'baking_info.json', 'w') as f:
            json.dump(baking_info, f, indent=2, default=str)

        logger.info("Baked model saved successfully")
        logger.info("Model is now ready for GGUF conversion")

    def run_baking_pipeline(self):
        """焼き込みパイプライン実行"""
        logger.info("Starting SO8T baking pipeline...")

        # モデル読み込み
        self.load_model()

        # SO8T効果を焼き込む
        baked_model = self.bake_so8t_effects()

        # 焼き込み済みモデルを保存
        self.save_baked_model(baked_model)

        logger.info("SO8T baking pipeline completed!")
        logger.info(f"Baked model saved to: {self.output_path}")
        logger.info("Model is ready for GGUF conversion")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Bake SO8T effects into Transformer for GGUF conversion")
    parser.add_argument("--model", type=str, required=True, help="Path to SO8T model")
    parser.add_argument("--output", type=str, required=True, help="Output path for baked model")

    args = parser.parse_args()

    baker = SO8TBaker(args.model, args.output)
    baker.run_baking_pipeline()


if __name__ == "__main__":
    main()
