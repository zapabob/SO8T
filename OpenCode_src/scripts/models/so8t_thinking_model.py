#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T/Thinking Model Architecture
SO8T理論に基づく思考モデル実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from typing import Dict, List, Any, Optional, Tuple, Union
import math
import logging

from scripts.models.so8_quad_inference import SO8RotationGate, QuadrupleInference
from scripts.inference.nkat_thermostat import NKATDynamicTemperature

logger = logging.getLogger(__name__)

class SO8TThinkingModel(nn.Module):
    """
    SO8T/Thinking Model: Phi-3.5ベースのSO8T思考アーキテクチャ

    特徴:
    - SO(8)幾何学的回転ゲート
    - 四重推論 (Observation/Deduction/Abduction/Integration)
    - NKAT Thermostat (動的温度制御)
    - 思考プロセス内部表現
    """

    def __init__(self,
                 base_model,
                 hidden_size: int = 3072,  # Phi-3.5 hidden size
                 so8_rotations: int = 8,
                 thermostat_enabled: bool = True):
        super().__init__()

        self.base_model = base_model
        self.hidden_size = hidden_size
        self.so8_rotations = so8_rotations
        self.thermostat_enabled = thermostat_enabled

        # SO(8)幾何学的層
        self.so8_geometry = SO8GeometryLayer(hidden_size, so8_rotations)

        # 四重推論層
        self.quadruple_inference = QuadrupleInference(hidden_size, so8_rotations)

        # 思考プロセス制御層
        self.thinking_controller = ThinkingProcessController(hidden_size)

        # NKAT Thermostat (オプション)
        self.thermostat = True

        # 出力投影
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        logger.info(f"SO8T/Thinking Model initialized with {so8_rotations} SO(8) rotations")

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                thinking_mode: bool = True,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        SO8T思考推論

        Args:
            input_ids: 入力トークン
            attention_mask: アテンションマスク
            thinking_mode: 思考モード有効化
            **kwargs: その他の引数

        Returns:
            思考結果を含む辞書
        """

        # 1. ベースモデルの出力を取得
        with torch.no_grad():  # ベースモデルは凍結
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )

        # 最後の隠れ層を取得
        hidden_states = base_outputs.hidden_states[-1]  # [batch, seq, hidden]

        # 2. SO(8)幾何学的変換
        geometric_states = self.so8_geometry(hidden_states)

        # 3. 思考モードの場合、四重推論を実行
        if thinking_mode:
            thinking_result = self._execute_quadruple_inference(
                geometric_states, input_ids, attention_mask
            )

            # 思考プロセス制御
            controlled_output = self.thinking_controller(
                thinking_result['final_output'],
                thinking_result['thinking_trace']
            )

            return {
                'logits': controlled_output,
                'thinking_trace': thinking_result['thinking_trace'],
                'inference_types': thinking_result['inference_types'],
                'stability_score': thinking_result['stability_score']
            }
        else:
            # 通常推論
            final_output = self.output_projection(geometric_states)
            return {
                'logits': final_output,
                'thinking_trace': None,
                'inference_types': None,
                'stability_score': None
            }

    def _execute_quadruple_inference(self,
                                   geometric_states: torch.Tensor,
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor) -> Dict[str, Any]:
        """
        四重推論実行

        Returns:
            思考結果
        """

        batch_size, seq_len, hidden_size = geometric_states.shape

        # 四重推論の各段階を実行
        observation_output = self.quadruple_inference.observation_layer(geometric_states)
        deduction_output = self.quadruple_inference.deduction_layer(observation_output)
        abduction_output = self.quadruple_inference.abduction_layer(deduction_output)
        integration_output = self.quadruple_inference.integration_layer(abduction_output)

        # 思考トレース記録
        thinking_trace = {
            'observation': observation_output,
            'deduction': deduction_output,
            'abduction': abduction_output,
            'integration': integration_output
        }

        # 推論タイプ分類（簡易版）
        inference_types = self._classify_inference_types(integration_output)

        # 安定性スコア計算
        stability_score = self._calculate_stability_score(thinking_trace)

        # 最終出力
        final_output = self.output_projection(integration_output)

        return {
            'final_output': final_output,
            'thinking_trace': thinking_trace,
            'inference_types': inference_types,
            'stability_score': stability_score
        }

    def _classify_inference_types(self, integration_output: torch.Tensor) -> List[str]:
        """推論タイプを分類"""
        # 簡易分類: 出力の統計的特性に基づく
        batch_size = integration_output.shape[0]

        # 出力の分散で分類
        variances = torch.var(integration_output, dim=-1)  # [batch, seq]

        inference_types = []
        for i in range(batch_size):
            var_mean = variances[i].mean().item()
            if var_mean < 0.5:
                inference_types.append('observation')  # 安定した観測
            elif var_mean < 1.0:
                inference_types.append('deduction')   # 論理的推論
            elif var_mean < 1.5:
                inference_types.append('abduction')   # 創造的飛躍
            else:
                inference_types.append('integration') # 統合的思考

        return inference_types

    def _calculate_stability_score(self, thinking_trace: Dict[str, torch.Tensor]) -> torch.Tensor:
        """思考プロセスの安定性スコアを計算 (URT安定性)"""
        # 各段階間の出力変化の滑らかさを評価
        stages = ['observation', 'deduction', 'abduction', 'integration']

        stability_scores = []
        for i in range(len(stages) - 1):
            current = thinking_trace[stages[i]]
            next_stage = thinking_trace[stages[i + 1]]

            # コサイン類似度で変化の滑らかさを評価
            similarity = F.cosine_similarity(
                current.flatten(start_dim=1),
                next_stage.flatten(start_dim=1),
                dim=-1
            )
            stability_scores.append(similarity)

        # 平均安定性スコア
        return torch.stack(stability_scores).mean(dim=0)

    def set_thermostat(self, thermostat: NKATDynamicTemperature):
        """NKAT Thermostatを設定"""
        self.thermostat = thermostat
        logger.info("NKAT Thermostat enabled")

    def generate_with_thinking(self,
                              tokenizer,
                              prompt: str,
                              max_new_tokens: int = 512,
                              temperature: float = 0.7,
                              **kwargs) -> Dict[str, Any]:
        """
        思考付き生成

        Returns:
            生成結果と思考トレース
        """

        # 入力をトークナイズ
        inputs = tokenizer(prompt, return_tensors="pt").to(self.base_model.device)

        # 思考モードで推論
        with torch.no_grad():
            thinking_result = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                thinking_mode=True,
                **kwargs
            )

        # NKAT Thermostat適用
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'temperature': temperature,
            'pad_token_id': tokenizer.eos_token_id,
        }

        if self.thermostat:
            generation_kwargs['logits_processor'] = [self.thermostat]

        # テキスト生成
        generated_ids = self.base_model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            **generation_kwargs
        )

        # 思考プロンプトを除去して生成テキストを取得
        generated_text = tokenizer.decode(
            generated_ids[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )

        return {
            'generated_text': generated_text,
            'thinking_trace': thinking_result['thinking_trace'],
            'inference_types': thinking_result['inference_types'],
            'stability_score': thinking_result['stability_score'].item() if thinking_result['stability_score'] is not None else None
        }

class SO8GeometryLayer(nn.Module):
    """SO(8)幾何学的変換層"""

    def __init__(self, hidden_size: int, num_rotations: int = 8):
        super().__init__()
        self.so8_gate = SO8RotationGate(hidden_size, num_rotations)
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SO(8)回転適用
        rotated = self.so8_gate(x)
        # 正規化
        return self.layer_norm(rotated)

class ThinkingProcessController(nn.Module):
    """思考プロセス制御層"""

    def __init__(self, hidden_size: int):
        super().__init__()

        self.controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            LayerNorm(hidden_size)
        )

        # 思考品質評価
        self.quality_evaluator = nn.Linear(hidden_size, 1)

    def forward(self, final_output: torch.Tensor, thinking_trace: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        思考プロセスに基づいて出力を制御
        """

        # 思考品質スコア計算
        quality_scores = []
        for stage_output in thinking_trace.values():
            quality = self.quality_evaluator(stage_output.mean(dim=1))  # [batch, 1]
            quality_scores.append(quality)

        # 品質スコアの平均
        avg_quality = torch.stack(quality_scores, dim=0).mean(dim=0)  # [batch, 1]

        # 品質に基づいて出力を調整
        quality_weight = torch.sigmoid(avg_quality)  # 0-1の範囲

        # 制御適用
        controlled = self.controller(final_output)
        output = quality_weight * controlled + (1 - quality_weight) * final_output

        return output

    def inject_so8_residual_adapters(self):
        """
        SO(8)回転レイヤーをtransformerの中間レイヤーに残差アダプター接続
        """
        # transformerのレイヤーを取得
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            # Llama-style model
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
            # GPT-style model
            layers = self.base_model.transformer.h
        else:
            logger.warning("Could not find transformer layers, skipping SO(8) adapter injection")
            return

        # 中間レイヤーにSO(8)アダプターを注入 (例: レイヤー12, 18, 24に注入)
        adapter_positions = [len(layers) // 4, len(layers) // 2, 3 * len(layers) // 4]

        for pos in adapter_positions:
            if pos < len(layers):
                original_layer = layers[pos]

                # SO(8)残差アダプターを作成
                so8_adapter = SO8ResidualAdapter(self.hidden_size, self.so8_rotations)

                # 元のレイヤーをラップ
                layers[pos] = SO8AdaptedLayer(original_layer, so8_adapter)

                logger.info(f"Injected SO(8) residual adapter at layer {pos}")

class SO8ResidualAdapter(nn.Module):
    """
    SO(8)回転レイヤーを使用した残差アダプター
    """

    def __init__(self, hidden_size: int, so8_rotations: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.so8_rotations = so8_rotations

        # SO(8)回転ゲート
        self.so8_gate = SO8RotationGate(hidden_size)

        # 残差接続用の線形層
        self.residual_proj = nn.Linear(hidden_size, hidden_size)

        # LayerNorm
        self.norm = LayerNorm(hidden_size)

    def forward(self, x):
        # SO(8)幾何学的変換
        so8_output = self.so8_gate(x)

        # 残差投影
        residual = self.residual_proj(so8_output)

        # 残差接続
        output = x + residual

        # 正規化
        output = self.norm(output)

        return output

class SO8AdaptedLayer(nn.Module):
    """
    SO(8)アダプターが注入されたtransformerレイヤー
    """

    def __init__(self, original_layer, so8_adapter):
        super().__init__()
        self.original_layer = original_layer
        self.so8_adapter = so8_adapter

    def forward(self, *args, **kwargs):
        # 元のレイヤーの出力を取得
        original_output = self.original_layer(*args, **kwargs)

        # SO(8)アダプターを適用（残差接続）
        adapted_output = self.so8_adapter(original_output)

        return adapted_output

def create_so8t_thinking_model(base_model=None,
                              tokenizer=None,
                              base_model_path: str = "microsoft/phi-3.5-mini-instruct",
                              thermostat_enabled: bool = True,
                              freeze_base_weights: bool = False,
                              inject_so8_adapters: bool = False) -> SO8TThinkingModel:
    """
    SO8T/Thinkingモデルを作成

    Args:
        base_model_path: ベースモデルのパス
        thermostat_enabled: NKAT Thermostat有効化

    Returns:
        SO8TThinkingModelインスタンス
    """

    # ベースモデルが提供されていない場合は読み込み
    if base_model is None:
        # ベースモデル読み込み (Unsloth使用を想定)
        try:
            from unsloth import FastLanguageModel
            base_model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            logger.info(f"Loaded base model from {base_model_path}")
        except ImportError:
            logger.warning("Unsloth not available, using transformers directly")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # SO8T/Thinkingモデル作成
    so8t_model = SO8TThinkingModel(
        base_model=base_model,
        thermostat_enabled=thermostat_enabled
    )

    # ベースモデルの重みを凍結（オプション）
    if freeze_base_weights:
        logger.info("Freezing base model weights...")
        for param in so8t_model.base_model.parameters():
            param.requires_grad = False

    # SO(8)アダプターをtransformerの中間レイヤーに注入（オプション）
    if inject_so8_adapters:
        logger.info("Injecting SO(8) residual adapters into transformer layers...")
        so8t_model.inject_so8_residual_adapters()

    return so8t_model

# 使用例
if __name__ == "__main__":
    # SO8T/Thinkingモデル作成
    model = create_so8t_thinking_model()

    # トークナイザー (仮定)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3.5-mini-instruct")

    # NKAT Thermostat設定
    from scripts.inference.nkat_thermostat import create_nkat_thermostat
    thermostat_controller = create_nkat_thermostat(tokenizer)
    model.set_thermostat(thermostat_controller.get_logits_processor())

    # 思考付き生成テスト
    test_prompt = "SO(8)群のトライアリティについて説明してください。"

    result = model.generate_with_thinking(
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_new_tokens=256
    )

    print("Generated Text:")
    print(result['generated_text'])
    print(f"\nStability Score: {result['stability_score']}")
    print(f"Inference Types: {result['inference_types']}")
