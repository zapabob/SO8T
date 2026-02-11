#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Thinking SO8T Model

動的Thinking、マルチモーダル、メタ推論機能を統合した高度なSO8Tモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer

from .so8vit_thinking_adapter import SO8ViTThinkingAdapter, MultimodalThinkingIntegrator
from .so8_trinality_inference import SO8TrinalityInference, SO8TrinalityMetaAnalyzer
from .safety_aware_so8t import SafetyAwareSO8TModel

logger = logging.getLogger(__name__)


class QueryTypeClassifier(nn.Module):
    """クエリタイプ分類器"""

    def __init__(self, hidden_size: int, num_types: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_types)
        )

        # クエリタイプ定義
        self.query_types = [
            'factual', 'reasoning', 'creative', 'math', 'code',
            'search', 'conversation', 'analysis', 'generation', 'other'
        ]

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        クエリタイプ分類

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            (logits, probabilities)
        """
        # CLSトークンまたは平均プーリングを使用
        if hidden_states.shape[1] > 1:
            # 平均プーリング
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states.squeeze(1)

        logits = self.classifier(pooled)
        probabilities = F.softmax(logits, dim=-1)

        return logits, probabilities

    def predict_type(self, hidden_states: torch.Tensor) -> str:
        """クエリタイプ予測"""
        _, probabilities = self.forward(hidden_states)
        predicted_idx = probabilities.argmax(dim=-1).item()
        return self.query_types[predicted_idx]


class MetaReasoningAnalyzer(nn.Module):
    """
    メタ推論アナライザー with 温度制御

    自身の推論プロセスを分析・評価し、エントロピーに基づいて温度制御
    - 高エントロピー（ハルシネーション）：冷却
    - 低エントロピー（確信不足）：加熱
    """

    def __init__(self, hidden_size: int, base_temperature: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_temperature = base_temperature

        # 推論品質評価器
        self.quality_evaluator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 不確実性推定器
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # エントロピー計算器（温度制御用）
        self.entropy_calculator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # [0,1]の範囲でエントロピー強度を出力
        )

        # 温度制御器
        self.temperature_controller = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size // 2),  # +1 for entropy
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # [0,1]の範囲で温度調整係数
        )

        # 推論一貫性チェッカー
        self.consistency_checker = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )

        # 温度制御パラメータ
        self.register_buffer("cooling_threshold", torch.tensor(0.7))  # 高エントロピー閾値
        self.register_buffer("heating_threshold", torch.tensor(0.3))  # 低エントロピー閾値
        self.register_buffer("max_temperature_factor", torch.tensor(2.0))  # 最大加熱倍率
        self.register_buffer("min_temperature_factor", torch.tensor(0.1))  # 最小冷却倍率

        # メタ推論統計蓄積
        self.register_buffer("meta_stats", torch.zeros(7))
        # [quality_score, uncertainty, consistency, confidence, adaptation, entropy, temperature_factor]

    def analyze_reasoning(self, thinking_metadata: Dict[str, Any],
                         final_output: torch.Tensor, current_temperature: float = 1.0) -> Dict[str, Any]:
        """
        推論プロセス分析 with 温度制御

        Args:
            thinking_metadata: Thinkingメタデータ
            final_output: 最終出力
            current_temperature: 現在の温度

        Returns:
            メタ分析結果 with 温度制御
        """
        # 推論品質評価
        quality_score = self.quality_evaluator(final_output.mean(dim=1))

        # 不確実性推定
        uncertainty = self.uncertainty_estimator(final_output.mean(dim=1))

        # エントロピー計算（温度制御の鍵）
        entropy_score = self.entropy_calculator(final_output.mean(dim=1))

        # 推論一貫性チェック
        consistency_score = self._check_consistency(thinking_metadata)

        # 全体確信度
        confidence = self._calculate_confidence(thinking_metadata)

        # 適応性評価
        adaptation = self._evaluate_adaptation(thinking_metadata)

        # 温度制御計算
        temperature_factor = self._calculate_temperature_factor(
            entropy_score, quality_score, consistency_score, current_temperature
        )

        # 新しい温度計算
        new_temperature = current_temperature * temperature_factor.item()
        new_temperature = torch.clamp(new_temperature,
                                    self.base_temperature * self.min_temperature_factor,
                                    self.base_temperature * self.max_temperature_factor)

        # 統計更新
        self.meta_stats = torch.stack([
            quality_score.mean(),
            uncertainty.mean(),
            consistency_score,
            confidence,
            adaptation,
            entropy_score.mean(),
            temperature_factor
        ])

        return {
            'quality_score': quality_score.item(),
            'uncertainty': uncertainty.item(),
            'consistency': consistency_score.item(),
            'confidence': confidence.item(),
            'adaptation': adaptation.item(),
            'entropy_score': entropy_score.item(),
            'temperature_factor': temperature_factor.item(),
            'current_temperature': current_temperature,
            'new_temperature': new_temperature.item(),
            'temperature_adjusted': not torch.isclose(temperature_factor, torch.tensor(1.0), atol=0.05),
            'meta_stats': self.meta_stats.clone()
        }

    def _check_consistency(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """推論一貫性チェック"""
        # 層ごとの思考出力を比較
        if 'thinking_metadata' not in metadata:
            return torch.tensor(0.5)

        thinking_meta = metadata['thinking_metadata']

        # 確信度の標準偏差（低いほど一貫性が高い）
        if 'thinking_confidence' in thinking_meta:
            confidences = torch.stack(thinking_meta['thinking_confidence'])
            consistency = 1.0 - confidences.std(dim=0).mean()
            return torch.clamp(consistency, 0.0, 1.0)

        return torch.tensor(0.5)

    def _calculate_confidence(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """全体確信度計算"""
        if 'meta_analysis' not in metadata:
            return torch.tensor(0.5)

        meta_analysis = metadata['meta_analysis']

        # 複数の指標を統合
        confidence_indicators = []
        if 'overall_confidence' in meta_analysis:
            confidence_indicators.append(meta_analysis['overall_confidence'])
        if 'thinking_efficiency' in meta_analysis:
            confidence_indicators.append(meta_analysis['thinking_efficiency'])

        if confidence_indicators:
            return torch.stack(confidence_indicators).mean()

        return torch.tensor(0.5)

    def _evaluate_adaptation(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """適応性評価"""
        if 'meta_analysis' not in metadata:
            return torch.tensor(0.5)

        meta_analysis = metadata['meta_analysis']

        # 適応品質と幾何学的整合性を統合
        adaptation_indicators = []
        if 'adaptation_quality' in meta_analysis:
            adaptation_indicators.append(meta_analysis['adaptation_quality'])
        if 'geometric_consistency' in meta_analysis:
            adaptation_indicators.append(meta_analysis['geometric_consistency'])

        if adaptation_indicators:
            return torch.stack(adaptation_indicators).mean()

        return torch.tensor(0.5)

    def _calculate_temperature_factor(self, entropy_score: torch.Tensor,
                                    quality_score: torch.Tensor,
                                    consistency_score: torch.Tensor,
                                    current_temperature: float) -> torch.Tensor:
        """
        温度制御係数計算

        高エントロピー（ハルシネーションなど）：冷却
        低エントロピー（確信不足など）：加熱

        Args:
            entropy_score: エントロピー強度 [0,1]
            quality_score: 品質スコア
            consistency_score: 一貫性スコア
            current_temperature: 現在の温度

        Returns:
            温度調整係数
        """
        # エントロピーに基づく基本制御
        if entropy_score > self.cooling_threshold:
            # 高エントロピー：冷却（温度を下げる）
            # エントロピーが高いほど強く冷却
            base_factor = 1.0 - (entropy_score - self.cooling_threshold) / (1.0 - self.cooling_threshold)
            base_factor = torch.clamp(base_factor, self.min_temperature_factor, 1.0)
            control_type = "cooling"
        elif entropy_score < self.heating_threshold:
            # 低エントロピー：加熱（温度を上げる）
            # エントロピーが低いほど強く加熱
            base_factor = 1.0 + (self.heating_threshold - entropy_score) / self.heating_threshold
            base_factor = torch.clamp(base_factor, 1.0, self.max_temperature_factor)
            control_type = "heating"
        else:
            # 中間エントロピー：温度維持
            base_factor = torch.tensor(1.0)
            control_type = "stable"

        # 品質と一貫性による調整
        # 高品質・高一貫性：温度を安定させる方向に調整
        # 低品質・低一貫性：温度調整を強化

        quality_consistency_factor = (quality_score + consistency_score) / 2.0

        if control_type == "cooling":
            # 冷却時は高品質でさらに冷却、低品質で冷却を弱める
            adjustment = 0.8 + 0.4 * quality_consistency_factor  # [0.8, 1.2]
            final_factor = base_factor * adjustment
        elif control_type == "heating":
            # 加熱時は高品質で加熱を弱める、低品質でさらに加熱
            adjustment = 1.2 - 0.4 * quality_consistency_factor  # [0.8, 1.2]
            final_factor = base_factor * adjustment
        else:
            # 安定時は品質による微調整
            adjustment = 0.9 + 0.2 * quality_consistency_factor  # [0.9, 1.1]
            final_factor = base_factor * adjustment

        # 最終クランプ
        final_factor = torch.clamp(final_factor,
                                 self.min_temperature_factor,
                                 self.max_temperature_factor)

        logger.debug(f"Temperature control: entropy={entropy_score:.3f}, "
                    f"quality={quality_score:.3f}, consistency={consistency_score:.3f}, "
                    f"type={control_type}, factor={final_factor:.3f}")

        return final_factor

    def get_temperature_control_stats(self) -> Dict[str, float]:
        """温度制御統計取得"""
        return {
            'cooling_threshold': self.cooling_threshold.item(),
            'heating_threshold': self.heating_threshold.item(),
            'max_temperature_factor': self.max_temperature_factor.item(),
            'min_temperature_factor': self.min_temperature_factor.item(),
            'current_meta_stats': self.meta_stats.tolist()
        }


class DynamicThinkingSO8TModel(SafetyAwareSO8TModel):
    """
    Dynamic Thinking SO8T Model

    動的Thinking、マルチモーダル、メタ推論機能を統合
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        # SO8ViT Thinking Adapter
        self.so8vit_adapter = SO8ViTThinkingAdapter(config)

        # マルチモーダル統合器
        self.multimodal_integrator = MultimodalThinkingIntegrator(
            self.so8vit_adapter,
            vision_config={'vision_dim': 768},  # CLIPまたは他のビジョンモデル
            audio_config={'audio_dim': 768}    # Whisperまたは他のオーディオモデル
        )

        # クエリタイプ分類器
        self.query_classifier = QueryTypeClassifier(config.hidden_size)

        # メタ推論アナライザー
        self.meta_analyzer = MetaReasoningAnalyzer(config.hidden_size)

        # SO8 Trinality Inference - SO(8)群の表現論に基づく四重推論
        self.so8_trinality_inference = SO8TrinalityInference(config)
        self.so8_trinality_meta_analyzer = SO8TrinalityMetaAnalyzer(config.hidden_size)

        # 動的Thinking制御
        self.dynamic_thinking_enabled = True
        self.multimodal_enabled = True
        self.meta_reasoning_enabled = True
        self.so8_trinality_enabled = True  # SO8 Trinality推論有効化

        logger.info("Dynamic Thinking SO8T Model initialized with:")
        logger.info("  - SO8ViT Thinking Adapter: ✓")
        logger.info("  - Multimodal Integration: ✓")
        logger.info("  - Meta Reasoning Analyzer: ✓")
        logger.info("  - Query Type Classification: ✓")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_inputs: Optional[torch.Tensor] = None,
        audio_inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        enable_so8_trinality: bool = True,
        temperature_control_temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Dynamic Thinking SO8T forward with SO8 Trinality Inference and 温度制御

        SO(8)群のTrinality（ベクトル表現、正/負スピノル表現）に基づく四重推論

        Args:
            input_ids: テキスト入力
            attention_mask: アテンションマスク
            vision_inputs: ビジョン入力（オプション）
            audio_inputs: オーディオ入力（オプション）
            labels: ラベル（学習用）
            enable_so8_trinality: SO8 Trinality推論有効化
            temperature_control_temperature: 温度制御の基準温度

        Returns:
            モデル出力 with SO8 Trinality Inference and 温度制御
        """
        # 基本のSO8T処理
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 隠れ状態を取得
        hidden_states = outputs.get('hidden_states', [outputs['last_hidden_state']])[-1]

        # クエリタイプ分類
        query_logits, query_probs = self.query_classifier(hidden_states)
        predicted_query_type = self.query_classifier.predict_type(hidden_states)

        # SO8 Trinality推論 or 通常Thinking処理
        if enable_so8_trinality and self.so8_trinality_enabled:
            thinking_output, thinking_metadata = self._perform_so8_trinality_inference(
                hidden_states, attention_mask, predicted_query_type,
                vision_inputs, audio_inputs, temperature_control_temperature
            )
        else:
            # 通常のThinking処理
            thinking_output = hidden_states
            thinking_metadata = {}

            if self.dynamic_thinking_enabled:
                # SO8ViTアダプター適用
                adapter_output, adapter_metadata = self.so8vit_adapter(
                    hidden_states,
                    attention_mask,
                    predicted_query_type
                )
                thinking_output = adapter_output
                thinking_metadata.update(adapter_metadata)

            # マルチモーダル統合
            if self.multimodal_enabled and (vision_inputs is not None or audio_inputs is not None):
                multimodal_output, multimodal_metadata = self.multimodal_integrator(
                    thinking_output,
                    vision_inputs,
                    audio_inputs,
                    attention_mask,
                    predicted_query_type
                )
                thinking_output = multimodal_output
                thinking_metadata.update(multimodal_metadata)

        # メタ推論分析
        if self.meta_reasoning_enabled:
            meta_analysis = self.meta_analyzer.analyze_reasoning(
                thinking_metadata,
                thinking_output
            )
            thinking_metadata['meta_analysis'] = meta_analysis

        # 最終出力を更新
        if 'last_hidden_state' in outputs:
            outputs['last_hidden_state'] = thinking_output

        # Thinkingメタデータを追加
        outputs['thinking_metadata'] = thinking_metadata
        outputs['query_type'] = predicted_query_type
        outputs['query_probabilities'] = query_probs
        outputs['so8_trinality_enabled'] = enable_so8_trinality and self.so8_trinality_enabled

        return outputs

    def generate_with_thinking(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_inputs: Optional[torch.Tensor] = None,
        audio_inputs: Optional[torch.Tensor] = None,
        max_length: int = 512,
        enable_temperature_control: bool = True,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Thinkingを伴う生成 with 温度制御

        Args:
            input_ids: 入力トークン
            attention_mask: アテンションマスク
            vision_inputs: ビジョン入力
            audio_inputs: オーディオ入力
            max_length: 最大生成長
            enable_temperature_control: 温度制御有効化
            **generation_kwargs: 生成パラメータ

        Returns:
            生成結果とThinking分析
        """
        # Thinking分析を有効化
        self.dynamic_thinking_enabled = True
        self.meta_reasoning_enabled = True
        self.so8_trinality_enabled = True

        # 温度制御設定
        temperature_control_enabled = enable_temperature_control and self.meta_reasoning_enabled
        current_temperature = generation_kwargs.get('temperature', 1.0)

        # 生成
        generated = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            return_dict_in_generate=True,
            output_hidden_states=True,
            **generation_kwargs
        )

        # 生成された各ステップのThinking分析 with 温度制御
        thinking_analyses = []
        temperature_history = [current_temperature]

        for step_idx, step_hidden_states in enumerate(generated.hidden_states):
            step_analysis = {}

            # クエリタイプ分類
            _, query_probs = self.query_classifier(step_hidden_states[-1])
            step_analysis['query_probabilities'] = query_probs

            # Thinking適用
            if self.dynamic_thinking_enabled:
                _, adapter_metadata = self.so8vit_adapter(
                    step_hidden_states[-1],
                    attention_mask,
                    None  # 動的判定
                )
                step_analysis['thinking_metadata'] = adapter_metadata

            # メタ分析 with 温度制御
            if self.meta_reasoning_enabled:
                meta_analysis = self.meta_analyzer.analyze_reasoning(
                    step_analysis,
                    step_hidden_states[-1],
                    current_temperature
                )
                step_analysis['meta_analysis'] = meta_analysis

                # 温度制御適用
                if temperature_control_enabled and 'new_temperature' in meta_analysis:
                    new_temperature = meta_analysis['new_temperature']
                    temperature_history.append(new_temperature)
                    current_temperature = new_temperature

                    # 温度制御ログ
                    if meta_analysis.get('temperature_adjusted', False):
                        entropy = meta_analysis.get('entropy_score', 0.5)
                        temp_factor = meta_analysis.get('temperature_factor', 1.0)
                        logger.debug(f"Step {step_idx}: Temperature adjusted "
                                   f"from {meta_analysis['current_temperature']:.3f} "
                                   f"to {new_temperature:.3f} "
                                   f"(entropy: {entropy:.3f}, factor: {temp_factor:.3f})")

            thinking_analyses.append(step_analysis)

        return {
            'generated_sequences': generated.sequences,
            'thinking_analyses': thinking_analyses,
            'final_analysis': thinking_analyses[-1] if thinking_analyses else None,
            'temperature_history': temperature_history,
            'temperature_control_enabled': temperature_control_enabled,
            'final_temperature': current_temperature
        }

    def enable_thinking_features(self, dynamic: bool = True,
                               multimodal: bool = True,
                               meta_reasoning: bool = True,
                               so8_trinality: bool = True,
                               temperature_control: bool = True):
        """Thinking機能の有効化設定"""
        self.dynamic_thinking_enabled = dynamic
        self.multimodal_enabled = multimodal
        self.meta_reasoning_enabled = meta_reasoning

        # 新機能の有効化設定
        self.so8_trinality_enabled = so8_trinality
        self.temperature_control_enabled = temperature_control

        logger.info("Thinking features updated:"        logger.info(f"  Dynamic Thinking: {'✓' if dynamic else '✗'}")
        logger.info(f"  Multimodal Integration: {'✓' if multimodal else '✗'}")
        logger.info(f"  Meta Reasoning: {'✓' if meta_reasoning else '✗'}")
        logger.info(f"  SO8 Trinality Inference: {'✓' if so8_trinality else '✗'}")
        logger.info(f"  Temperature Control: {'✓' if temperature_control else '✗'}")

    def get_thinking_stats(self) -> Dict[str, Any]:
        """Thinking統計取得"""
        return {
            'so8vit_adapter_stats': self.so8vit_adapter.thinking_stats,
            'meta_analyzer_stats': self.meta_analyzer.meta_stats,
            'orthogonal_errors': [
                gate.orthogonal_loss.item()
                for gate in self.so8vit_adapter.so8_gates
            ]
        }

    def _perform_so8_trinality_inference(self, hidden_states: torch.Tensor,
                                       attention_mask: Optional[torch.Tensor],
                                       query_type: str,
                                       vision_inputs: Optional[torch.Tensor] = None,
                                       audio_inputs: Optional[torch.Tensor] = None,
                                       base_temperature: float = 1.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        SO8 Trinality推論実行 with 温度制御

        SO(8)群のTrinalityに基づく四重推論:
        - Vector Stream (V): タスク指向思考
        - Positive Spinor Stream (S⁺): 安全/倫理指向思考
        - Negative Spinor Stream (S⁻): 論理/批判指向思考
        - Trinality Integration: SO(8)群の線形和表現

        メタ推論ステップで温度制御を適用

        Args:
            hidden_states: 入力隠れ状態
            attention_mask: アテンションマスク
            query_type: クエリタイプ
            vision_inputs: ビジョン入力
            audio_inputs: オーディオ入力
            base_temperature: 基準温度

        Returns:
            (final_output, thinking_metadata)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        # SO8 Trinality推論実行
        trinality_results = self.so8_trinality_inference(
            hidden_states, attention_mask
        )

        thinking_metadata = {
            'so8_trinality_inference': True,
            'trinality_results': trinality_results,
            'temperature_control': {
                'base_temperature': base_temperature,
                'adjustments': [],
                'final_temperature': base_temperature
            }
        }

        current_temperature = base_temperature

        # SO8 Trinalityの各ストリームにマルチモーダル統合を適用
        if self.multimodal_enabled and (vision_inputs is not None or audio_inputs is not None):
            # ストリーム固有のモダリティ重み付け
            modality_configs = self._get_so8_modality_configs(query_type)

            enhanced_streams = {}
            for stream_name, stream_data in trinality_results['streams'].items():
                if stream_name == 'trinality_integration':
                    continue

                config = modality_configs.get(stream_name, {})
                modality_weights = config.get('modality_weights', {'vision': 1.0, 'audio': 1.0})

                # モダリティ入力をストリーム用に調整
                adjusted_vision = vision_inputs * modality_weights.get('vision', 1.0) if vision_inputs is not None else None
                adjusted_audio = audio_inputs * modality_weights.get('audio', 1.0) if audio_inputs is not None else None

                multimodal_output, multimodal_metadata = self.multimodal_integrator(
                    stream_data['output'],
                    adjusted_vision,
                    adjusted_audio,
                    attention_mask,
                    config.get('query_override', query_type)
                )

                enhanced_streams[stream_name] = multimodal_output
                thinking_metadata.setdefault('multimodal_enhancements', {})[stream_name] = multimodal_metadata

            # マルチモーダル統合後のTrinality再実行
            if enhanced_streams:
                # 統合ストリームの作成
                enhanced_hidden_states = torch.stack(list(enhanced_streams.values()), dim=0).mean(dim=0)
                trinality_results = self.so8_trinality_inference(
                    enhanced_hidden_states, attention_mask
                )
                thinking_metadata['trinality_results'] = trinality_results

        # SO8 Trinalityメタ分析
        if self.meta_reasoning_enabled:
            trinality_meta_analysis = self.so8_trinality_meta_analyzer.analyze_trinality(
                trinality_results
            )
            thinking_metadata['trinality_meta_analysis'] = trinality_meta_analysis

            # 温度制御適用（Trinality品質に基づく）
            overall_quality = trinality_meta_analysis.get('overall_quality_score', 0.5)

            # 品質に基づく温度制御
            quality_based_temperature = self._compute_quality_based_temperature(
                overall_quality, base_temperature
            )

            if abs(quality_based_temperature - base_temperature) > 0.05:
                thinking_metadata['temperature_control']['final_temperature'] = quality_based_temperature
                thinking_metadata['temperature_control']['quality_based_adjustment'] = True
                current_temperature = quality_based_temperature

        # 最終出力取得
        final_output = trinality_results['final_output']

        # 最終温度制御適用（従来のエントロピー制御）
        final_output, final_temperature = self._apply_final_temperature_control(
            final_output, {'trinality_quality': trinality_meta_analysis}, current_temperature
        )

        # 温度制御メタデータを更新
        thinking_metadata['temperature_control']['final_temperature'] = final_temperature

        return final_output, thinking_metadata



    def _apply_final_temperature_control(self, integrated_output: torch.Tensor,
                                       integration_metadata: Dict[str, Any],
                                       current_temperature: float) -> Tuple[torch.Tensor, float]:
        """最終温度制御適用"""
        # 統合出力のメタ分析
        final_meta_analysis = self.meta_analyzer.analyze_reasoning(
            {'integration': integration_metadata},
            integrated_output,
            current_temperature
        )

        # 温度制御適用
        if 'new_temperature' in final_meta_analysis:
            final_temperature = final_meta_analysis['new_temperature']

            # 温度制御のログ
            if final_meta_analysis.get('temperature_adjusted', False):
                entropy = final_meta_analysis.get('entropy_score', 0.5)
                temp_factor = final_meta_analysis.get('temperature_factor', 1.0)
                logger.info(f"Final temperature control: "
                          f"{current_temperature:.3f} -> {final_temperature:.3f} "
                          f"(entropy: {entropy:.3f}, factor: {temp_factor:.3f})")

        else:
            final_temperature = current_temperature

        return integrated_output, final_temperature

    def _get_so8_modality_configs(self, query_type: str) -> Dict[str, Dict[str, Any]]:
        """SO8 Trinalityストリームのモダリティ設定取得"""
        base_configs = {
            'vector': {  # V: タスク指向
                'modality_weights': {'vision': 1.2, 'audio': 0.8},
                'query_override': 'task'
            },
            'positive_spinor': {  # S⁺: 安全/倫理指向
                'modality_weights': {'vision': 0.8, 'audio': 1.3},
                'query_override': 'safety'
            },
            'negative_spinor': {  # S⁻: 論理/批判指向
                'modality_weights': {'vision': 1.0, 'audio': 1.1},
                'query_override': 'logic'
            }
        }

        # クエリタイプによる調整
        if query_type in ['math', 'logic']:
            base_configs['negative_spinor']['modality_weights']['vision'] = 1.3  # 論理タスクではビジョン入力重視
        elif query_type in ['creative', 'generation']:
            base_configs['vector']['modality_weights']['audio'] = 1.2  # 創造タスクではオーディオ入力重視
        elif query_type in ['safety', 'dangerous']:
            base_configs['positive_spinor']['modality_weights']['audio'] = 1.5  # 安全タスクではオーディオ入力最重視

        return base_configs

    def _compute_quality_based_temperature(self, quality_score: float, base_temperature: float) -> float:
        """Trinality品質に基づく温度計算"""
        # 品質スコアを温度係数に変換
        # 高品質（>0.7）：温度を安定させる（若干下げる）
        # 中品質（0.4-0.7）：温度維持
        # 低品質（<0.4）：温度を上げる（多様性確保）

        if quality_score > 0.7:
            # 高品質：温度を0.9倍（安定化）
            temperature_factor = 0.9
        elif quality_score > 0.4:
            # 中品質：温度維持
            temperature_factor = 1.0
        else:
            # 低品質：温度を1.2倍（多様性確保）
            temperature_factor = 1.2

        new_temperature = base_temperature * temperature_factor

        # 温度範囲を制限
        new_temperature = torch.clamp(new_temperature,
                                    self.meta_analyzer.base_temperature * self.meta_analyzer.min_temperature_factor,
                                    self.meta_analyzer.base_temperature * self.meta_analyzer.max_temperature_factor)

        return new_temperature.item()


def create_dynamic_thinking_so8t(config_path: str) -> DynamicThinkingSO8TModel:
    """Dynamic Thinking SO8Tモデルの作成"""
    import yaml
    from transformers import AutoConfig

    # 設定読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # モデル設定
    model_config = AutoConfig.from_pretrained(
        config_dict.get('model', {}).get('name', 'microsoft/phi-3.5-mini-instruct'),
        **config_dict.get('model', {})
    )

    # Dynamic Thinking SO8Tモデル作成
    model = DynamicThinkingSO8TModel(model_config)

    logger.info("Dynamic Thinking SO8T Model created successfully!")
    return model
