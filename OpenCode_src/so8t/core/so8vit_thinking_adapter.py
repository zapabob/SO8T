#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8ViT/Thinking Model Residual Adapter

Transformer部分をSO8ViT/thinkingモデル残差アダプターに置き換え、
中間レイヤーにSO8回転ゲートを導入し直交誤差をloggingするシステム
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Dict, Any, List
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer

logger = logging.getLogger(__name__)


class SO8RotationGate(nn.Module):
    """
    SO8回転ゲート - 中間レイヤーに導入される回転ゲート
    直交性を維持しながら幾何学的変換を実行
    """

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # SO(8)回転行列（8次元回転群）
        self.rotation_matrix = nn.Parameter(
            torch.eye(8, dtype=torch.float32)
        )

        # 回転強度制御
        self.gate_weight = nn.Parameter(torch.ones(1))

        # 直交性維持のための正規化
        self.register_buffer("orthogonal_loss", torch.tensor(0.0))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SO8回転ゲート適用

        Args:
            hidden_states: 入力テンソル [batch_size, seq_len, hidden_size]

        Returns:
            (transformed_states, orthogonal_error)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 8次元チャンクに分割（hidden_sizeが8の倍数であることを仮定）
        assert hidden_size % 8 == 0, f"hidden_size {hidden_size} must be divisible by 8 for SO(8)"

        # テンソルを8次元チャンクに再形成
        chunk_size = hidden_size // 8
        reshaped = hidden_states.view(batch_size * seq_len, 8, chunk_size)

        # SO(8)回転適用
        rotated = torch.einsum('bij,jk->bik', reshaped, self.rotation_matrix)
        rotated = rotated.view(batch_size, seq_len, hidden_size)

        # ゲート強度で制御
        gated_output = hidden_states + self.gate_weight * (rotated - hidden_states)

        # 直交誤差計算（回転行列の直交性チェック）
        identity = torch.eye(8, device=self.rotation_matrix.device, dtype=self.rotation_matrix.dtype)
        orthogonal_error = torch.norm(
            torch.mm(self.rotation_matrix.t(), self.rotation_matrix) - identity,
            p='fro'
        )

        # 直交誤差をバッファに保存（logging用）
        self.orthogonal_loss.copy_(orthogonal_error.detach())

        return gated_output, orthogonal_error


class SO8ViTThinkingAdapter(nn.Module):
    """
    SO8ViT/Thinking Model Residual Adapter

    Vision Transformerベースの思考モデルアダプター
    SO8回転ゲートを中間レイヤーに導入
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads

        # SO8回転ゲート層
        self.so8_gates = nn.ModuleList([
            SO8RotationGate(self.hidden_size, self.num_attention_heads)
            for _ in range(self.num_hidden_layers)
        ])

        # Thinkingアテンション（マルチヘッドアテンションの拡張）
        self.thinking_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # 残差接続のためのレイヤーノーマライゼーション
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
            for _ in range(self.num_hidden_layers)
        ])

        # フィードフォワードネットワーク
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.intermediate_size, self.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            ) for _ in range(self.num_hidden_layers)
        ])

        # 動的Thinking制御パラメータ
        self.thinking_alpha = nn.Parameter(torch.tensor(0.5))  # [0,1]の範囲でシグモイド適用

        # メタ推論用の統計蓄積
        self.register_buffer("thinking_stats", torch.zeros(self.num_hidden_layers, 4))
        # [layer, [attention_entropy, rotation_error, thinking_confidence, adaptation_score]]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        SO8ViT/Thinking Adapter forward

        Args:
            hidden_states: 入力テンソル [batch_size, seq_len, hidden_size]
            attention_mask: アテンションマスク
            query_type: クエリタイプ（動的Thinking用）

        Returns:
            (output_states, thinking_metadata)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Thinking alphaのシグモイド適用（[0,1]の範囲に制限）
        alpha = torch.sigmoid(self.thinking_alpha)

        # 動的Thinking適応
        thinking_config = self._adapt_thinking_structure(query_type)

        layer_outputs = []
        thinking_metadata = {
            'orthogonal_errors': [],
            'attention_weights': [],
            'thinking_confidence': [],
            'layer_adaptations': []
        }

        for layer_idx in range(self.num_hidden_layers):
            # 残差接続の開始
            residual = hidden_states

            # SO8回転ゲート適用
            rotated_states, orthogonal_error = self.so8_gates[layer_idx](hidden_states)
            thinking_metadata['orthogonal_errors'].append(orthogonal_error)

            # Thinkingアテンション
            attn_output, attn_weights = self.thinking_attention(
                rotated_states, rotated_states, rotated_states,
                key_padding_mask=attention_mask,
                need_weights=True
            )
            thinking_metadata['attention_weights'].append(attn_weights)

            # 動的Thinking制御（alphaベースのブレンド）
            if thinking_config['use_geometric']:
                # 幾何学的思考（α=1に近い）
                thinking_output = attn_output + rotated_states
            else:
                # 統計的思考（α=0に近い）
                thinking_output = attn_output

            # Thinking confidence計算（アテンションの確信度）
            attention_entropy = self._compute_attention_entropy(attn_weights)
            thinking_confidence = 1.0 - attention_entropy  # エントロピーが低いほど確信度が高い
            thinking_metadata['thinking_confidence'].append(thinking_confidence)

            # フィードフォワード
            ff_output = self.feed_forward[layer_idx](thinking_output)

            # 残差接続とレイヤーノーマライゼーション
            hidden_states = self.layer_norms[layer_idx](residual + ff_output)

            # 層適応スコア計算
            adaptation_score = self._compute_adaptation_score(
                residual, hidden_states, thinking_config
            )
            thinking_metadata['layer_adaptations'].append(adaptation_score)

            # 統計蓄積
            self.thinking_stats[layer_idx] = torch.stack([
                attention_entropy,
                orthogonal_error,
                thinking_confidence,
                adaptation_score
            ])

            layer_outputs.append(hidden_states)

        # メタ推論分析
        meta_analysis = self._analyze_meta_reasoning(thinking_metadata, query_type)

        return hidden_states, {
            'thinking_metadata': thinking_metadata,
            'meta_analysis': meta_analysis,
            'final_alpha': alpha,
            'thinking_config': thinking_config
        }

    def _adapt_thinking_structure(self, query_type: Optional[str]) -> Dict[str, Any]:
        """
        クエリタイプに応じた動的Thinking構造適応

        Args:
            query_type: クエリタイプ

        Returns:
            thinking_config: Thinking設定
        """
        base_config = {
            'use_geometric': False,
            'attention_focus': 'balanced',
            'rotation_strength': 0.5,
            'thinking_depth': 1.0
        }

        if not query_type:
            return base_config

        # クエリタイプ別の適応
        if query_type in ['math', 'logic', 'reasoning']:
            # 数学・論理クエリ：幾何学的思考を重視
            base_config.update({
                'use_geometric': True,
                'attention_focus': 'sequential',
                'rotation_strength': 0.8,
                'thinking_depth': 1.5
            })
        elif query_type in ['creative', 'generation']:
            # 創造的クエリ：統計的思考を重視
            base_config.update({
                'use_geometric': False,
                'attention_focus': 'diverse',
                'rotation_strength': 0.3,
                'thinking_depth': 0.8
            })
        elif query_type in ['factual', 'search']:
            # 事実ベースクエリ：バランス重視
            base_config.update({
                'use_geometric': True,
                'attention_focus': 'focused',
                'rotation_strength': 0.6,
                'thinking_depth': 1.2
            })

        return base_config

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """アテンション重みのエントロピー計算"""
        # アテンション重みを確率分布として扱いエントロピーを計算
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1)
        return entropy.mean()

    def _compute_adaptation_score(self, input_states: torch.Tensor,
                                output_states: torch.Tensor,
                                config: Dict[str, Any]) -> torch.Tensor:
        """層適応スコア計算"""
        # 入力と出力の変化量を測る
        change_magnitude = torch.norm(output_states - input_states, dim=-1).mean()

        # 設定に応じた適応スコア
        if config['use_geometric']:
            # 幾何学的適応：変化が大きいほど良い
            return torch.sigmoid(change_magnitude)
        else:
            # 統計的適応：変化が安定しているほど良い
            return torch.sigmoid(-change_magnitude + 1.0)

    def _analyze_meta_reasoning(self, thinking_metadata: Dict[str, List],
                              query_type: Optional[str]) -> Dict[str, Any]:
        """メタ推論分析"""
        meta_analysis = {
            'overall_confidence': torch.stack(thinking_metadata['thinking_confidence']).mean(),
            'geometric_consistency': self._compute_geometric_consistency(thinking_metadata),
            'thinking_efficiency': self._compute_thinking_efficiency(thinking_metadata),
            'adaptation_quality': torch.stack(thinking_metadata['layer_adaptations']).mean(),
            'orthogonal_stability': torch.stack(thinking_metadata['orthogonal_errors']).mean(),
            'query_type_match': self._evaluate_query_type_match(query_type, thinking_metadata)
        }

        return meta_analysis

    def _compute_geometric_consistency(self, metadata: Dict[str, List]) -> torch.Tensor:
        """幾何学的整合性計算"""
        orthogonal_errors = torch.stack(metadata['orthogonal_errors'])
        # 直交誤差が小さいほど整合性が高い
        return torch.exp(-orthogonal_errors.mean())

    def _compute_thinking_efficiency(self, metadata: Dict[str, List]) -> torch.Tensor:
        """思考効率計算"""
        confidences = torch.stack(metadata['thinking_confidence'])
        adaptations = torch.stack(metadata['layer_adaptations'])

        # 確信度が高く、適応が効率的なほど効率的
        return (confidences * adaptations).mean()

    def _evaluate_query_type_match(self, query_type: Optional[str],
                                 metadata: Dict[str, List]) -> torch.Tensor:
        """クエリタイプとの適合度評価"""
        if not query_type:
            return torch.tensor(0.5)

        # メタデータに基づく適合度計算
        confidence_trend = torch.stack(metadata['thinking_confidence'])
        adaptation_trend = torch.stack(metadata['layer_adaptations'])

        # 適合度の簡易計算（実際にはより複雑なロジックが必要）
        return (confidence_trend.mean() + adaptation_trend.mean()) / 2.0


class MultimodalThinkingIntegrator(nn.Module):
    """
    マルチモーダルThinking統合器

    画像/音声/テキストを統合した思考プロセス
    """

    def __init__(self, text_adapter: SO8ViTThinkingAdapter,
                 vision_config: Optional[Dict] = None,
                 audio_config: Optional[Dict] = None):
        super().__init__()
        self.text_adapter = text_adapter

        # ビジョンモダリティ
        if vision_config:
            self.vision_projector = nn.Linear(vision_config.get('vision_dim', 768),
                                            text_adapter.hidden_size)
            self.vision_so8_gate = SO8RotationGate(text_adapter.hidden_size)

        # オーディオモダリティ
        if audio_config:
            self.audio_projector = nn.Linear(audio_config.get('audio_dim', 768),
                                           text_adapter.hidden_size)
            self.audio_so8_gate = SO8RotationGate(text_adapter.hidden_size)

        # クロスモーダルアテンション
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=text_adapter.hidden_size,
            num_heads=text_adapter.num_attention_heads,
            batch_first=True
        )

        # モダリティ融合
        self.modality_fusion = nn.Sequential(
            nn.Linear(text_adapter.hidden_size * 3, text_adapter.hidden_size),
            nn.LayerNorm(text_adapter.hidden_size),
            nn.GELU(),
            nn.Linear(text_adapter.hidden_size, text_adapter.hidden_size)
        )

    def forward(self, text_states: torch.Tensor,
                vision_states: Optional[torch.Tensor] = None,
                audio_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                query_type: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        マルチモーダルThinking統合

        Args:
            text_states: テキスト状態
            vision_states: ビジョン状態（オプション）
            audio_states: オーディオ状態（オプション）
            attention_mask: アテンションマスク
            query_type: クエリタイプ

        Returns:
            (fused_output, multimodal_metadata)
        """
        modalities = [text_states]
        modality_types = ['text']

        # ビジョンモダリティ処理
        if vision_states is not None:
            vision_projected = self.vision_projector(vision_states)
            vision_gated, _ = self.vision_so8_gate(vision_projected)
            modalities.append(vision_gated)
            modality_types.append('vision')

        # オーディオモダリティ処理
        if audio_states is not None:
            audio_projected = self.audio_projector(audio_states)
            audio_gated, _ = self.audio_so8_gate(audio_projected)
            modalities.append(audio_gated)
            modality_types.append('audio')

        # テキストThinking適用
        text_output, thinking_metadata = self.text_adapter(
            text_states, attention_mask, query_type
        )

        # クロスモーダル統合（複数のモダリティがある場合）
        if len(modalities) > 1:
            # クロスモーダルアテンション
            fused_modalities = torch.stack(modalities, dim=1)  # [batch, num_modalities, seq, hidden]
            batch_size, num_modalities, seq_len, hidden_size = fused_modalities.shape

            # アテンション適用
            fused_flat = fused_modalities.view(batch_size * num_modalities, seq_len, hidden_size)
            attended, _ = self.cross_modal_attention(
                fused_flat, fused_flat, fused_flat
            )
            attended = attended.view(batch_size, num_modalities, seq_len, hidden_size)

            # モダリティ融合
            attended_mean = attended.mean(dim=1)  # モダリティ間で平均
            fusion_input = torch.cat([text_output, attended_mean, attended[:, 0]], dim=-1)
            fused_output = self.modality_fusion(fusion_input)
        else:
            fused_output = text_output

        # マルチモーダルメタデータ
        multimodal_metadata = {
            'thinking_metadata': thinking_metadata,
            'active_modalities': modality_types,
            'fusion_performed': len(modalities) > 1,
            'modality_count': len(modalities)
        }

        return fused_output, multimodal_metadata
