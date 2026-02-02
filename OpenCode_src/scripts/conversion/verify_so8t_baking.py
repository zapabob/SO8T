#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify SO8T Baking - 焼き込み処理の検証

SO(8)効果が正しくTransformerに焼き込まれ、
回転ゲートが削除されているかを検証
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class SO8TBakingVerifier:
    """
    SO8T Baking Verifier

    焼き込み処理の正しさを検証
    """

    def __init__(self, original_model_path: str, baked_model_path: str):
        self.original_path = Path(original_model_path)
        self.baked_path = Path(baked_model_path)
        self.original_model = None
        self.baked_model = None
        self.tokenizer = None

    def load_models(self):
        """モデル読み込み"""
        logger.info("Loading models for verification...")

        # オリジナルモデル
        self.original_model = AutoModelForCausalLM.from_pretrained(
            self.original_path, torch_dtype=torch.bfloat16
        )

        # 焼き込み済みモデル
        self.baked_model = AutoModelForCausalLM.from_pretrained(
            self.baked_path, torch_dtype=torch.bfloat16
        )

        # トークナイザー
        self.tokenizer = AutoTokenizer.from_pretrained(self.baked_path)

        logger.info("Models loaded successfully")

    def verify_so8t_removal(self) -> Dict[str, Any]:
        """SO8Tコンポーネントの削除を検証"""
        logger.info("Verifying SO8T component removal...")

        removal_results = {}

        # SO8ViTアダプターの削除確認
        has_so8vit_original = hasattr(self.original_model, 'so8vit_adapter')
        has_so8vit_baked = hasattr(self.baked_model, 'so8vit_adapter')

        removal_results['so8vit_adapter_removed'] = has_so8vit_original and not has_so8vit_baked
        removal_results['so8vit_adapter_original'] = has_so8vit_original
        removal_results['so8vit_adapter_baked'] = has_so8vit_baked

        # SO8Trinalityの削除確認
        has_trinality_original = hasattr(self.original_model, 'so8_trinality_inference')
        has_trinality_baked = hasattr(self.baked_model, 'so8_trinality_inference')

        removal_results['so8_trinality_removed'] = has_trinality_original and not has_trinality_baked
        removal_results['so8_trinality_original'] = has_trinality_original
        removal_results['so8_trinality_baked'] = has_trinality_baked

        # メタアナライザーの削除確認
        has_meta_original = hasattr(self.original_model, 'meta_analyzer')
        has_meta_baked = hasattr(self.baked_model, 'meta_analyzer')

        removal_results['meta_analyzer_removed'] = has_meta_original and not has_meta_baked
        removal_results['meta_analyzer_original'] = has_meta_original
        removal_results['meta_analyzer_baked'] = has_meta_baked

        # Thinking属性の削除確認
        thinking_attrs = [
            'dynamic_thinking_enabled',
            'multimodal_enabled',
            'meta_reasoning_enabled',
            'so8_trinality_enabled',
            'temperature_control_enabled'
        ]

        removal_results['thinking_attrs_removed'] = {}
        for attr in thinking_attrs:
            has_original = hasattr(self.original_model, attr)
            has_baked = hasattr(self.baked_model, attr)
            removed = has_original and not has_baked
            removal_results['thinking_attrs_removed'][attr] = {
                'removed': removed,
                'original': has_original,
                'baked': has_baked
            }

        return removal_results

    def verify_weight_baking(self) -> Dict[str, Any]:
        """重み焼き込みの検証"""
        logger.info("Verifying weight baking...")

        baking_results = {}

        # モデル構造の比較
        original_layers = len(self.original_model.base_model.layers)
        baked_layers = len(self.baked_model.base_model.layers)

        baking_results['layer_count_match'] = original_layers == baked_layers
        baking_results['original_layers'] = original_layers
        baking_results['baked_layers'] = baked_layers

        # 重みの変化分析
        weight_changes = []
        for i in range(min(original_layers, baked_layers)):
            original_layer = self.original_model.base_model.layers[i]
            baked_layer = self.baked_model.base_model.layers[i]

            # self-attentionの重み比較
            if hasattr(original_layer, 'self_attn') and hasattr(baked_layer, 'self_attn'):
                orig_attn = self._extract_attention_weights(original_layer.self_attn)
                baked_attn = self._extract_attention_weights(baked_layer.self_attn)

                change = torch.norm(baked_attn - orig_attn).item()
                weight_changes.append(change)

        baking_results['attention_weight_changes'] = weight_changes
        baking_results['avg_attention_change'] = np.mean(weight_changes) if weight_changes else 0
        baking_results['max_attention_change'] = np.max(weight_changes) if weight_changes else 0

        # 重みが変化していることを確認（焼き込み効果）
        baking_results['weights_modified'] = baking_results['avg_attention_change'] > 1e-6

        return baking_results

    def verify_inference_compatibility(self) -> Dict[str, Any]:
        """推論互換性の検証"""
        logger.info("Verifying inference compatibility...")

        compatibility_results = {}

        # テスト入力
        test_prompts = [
            "Hello, how are you?",
            "Explain quantum computing in simple terms.",
            "What is the capital of France?"
        ]

        compatibility_results['inference_tests'] = []

        for prompt in test_prompts:
            test_result = self._run_inference_test(prompt)
            compatibility_results['inference_tests'].append(test_result)

        # 全体の互換性評価
        successful_tests = sum(1 for test in compatibility_results['inference_tests'] if test['success'])
        total_tests = len(compatibility_results['inference_tests'])

        compatibility_results['overall_compatibility'] = successful_tests == total_tests
        compatibility_results['successful_tests'] = successful_tests
        compatibility_results['total_tests'] = total_tests

        return compatibility_results

    def _run_inference_test(self, prompt: str) -> Dict[str, Any]:
        """推論テスト実行"""
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)

            # オリジナルモデル推論
            with torch.no_grad():
                orig_outputs = self.original_model.generate(
                    **inputs, max_length=inputs['input_ids'].shape[1] + 20,
                    do_sample=False, num_return_sequences=1
                )

            # 焼き込み済みモデル推論
            with torch.no_grad():
                baked_outputs = self.baked_model.generate(
                    **inputs, max_length=inputs['input_ids'].shape[1] + 20,
                    do_sample=False, num_return_sequences=1
                )

            orig_text = self.tokenizer.decode(orig_outputs[0], skip_special_tokens=True)
            baked_text = self.tokenizer.decode(baked_outputs[0], skip_special_tokens=True)

            # 出力の比較（大まかな類似性）
            similarity = self._calculate_text_similarity(orig_text, baked_text)

            return {
                'success': True,
                'prompt': prompt,
                'original_output': orig_text,
                'baked_output': baked_text,
                'similarity': similarity,
                'acceptable_similarity': similarity > 0.5  # 50%以上類似
            }

        except Exception as e:
            logger.error(f"Inference test failed for prompt '{prompt}': {e}")
            return {
                'success': False,
                'prompt': prompt,
                'error': str(e)
            }

    def _extract_attention_weights(self, attention_layer: nn.Module) -> torch.Tensor:
        """アテンション重みを抽出"""
        weights = []

        # Q, K, V, Oの重みを収集
        if hasattr(attention_layer, 'q_proj'):
            weights.append(attention_layer.q_proj.weight.data.flatten())
        if hasattr(attention_layer, 'k_proj'):
            weights.append(attention_layer.k_proj.weight.data.flatten())
        if hasattr(attention_layer, 'v_proj'):
            weights.append(attention_layer.v_proj.weight.data.flatten())
        if hasattr(attention_layer, 'o_proj'):
            weights.append(attention_layer.o_proj.weight.data.flatten())

        if weights:
            return torch.cat(weights)
        else:
            return torch.tensor([])

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """テキスト類似度計算（簡易版）"""
        # 単純な文字ベースの類似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """包括的検証実行"""
        logger.info("Running comprehensive SO8T baking verification...")

        self.load_models()

        verification_results = {
            'so8t_removal': self.verify_so8t_removal(),
            'weight_baking': self.verify_weight_baking(),
            'inference_compatibility': self.verify_inference_compatibility(),
            'overall_assessment': {}
        }

        # 全体評価
        removal_ok = all([
            verification_results['so8t_removal'].get('so8vit_adapter_removed', False),
            verification_results['so8t_removal'].get('so8_trinality_removed', False),
            verification_results['so8t_removal'].get('meta_analyzer_removed', False)
        ])

        baking_ok = verification_results['weight_baking'].get('weights_modified', False)

        inference_ok = verification_results['inference_compatibility'].get('overall_compatibility', False)

        verification_results['overall_assessment'] = {
            'so8t_components_removed': removal_ok,
            'weights_properly_baked': baking_ok,
            'inference_compatible': inference_ok,
            'baking_successful': removal_ok and baking_ok and inference_ok,
            'ready_for_gguf': removal_ok and inference_ok
        }

        logger.info("Comprehensive verification completed")
        return verification_results

    def print_verification_report(self, results: Dict[str, Any]):
        """検証レポート出力"""
        print("="*80)
        print("SO8T BAKING VERIFICATION REPORT")
        print("="*80)

        # SO8T削除検証
        print("\n1. SO8T Component Removal:")
        removal = results['so8t_removal']
        print(f"   SO8ViT Adapter removed: {removal['so8vit_adapter_removed']}")
        print(f"   SO8 Trinality removed: {removal['so8_trinality_removed']}")
        print(f"   Meta Analyzer removed: {removal['meta_analyzer_removed']}")

        # 重み焼き込み検証
        print("\n2. Weight Baking:")
        baking = results['weight_baking']
        print(f"   Weights modified: {baking['weights_modified']}")
        print(".6f")
        print(".6f")

        # 推論互換性
        print("\n3. Inference Compatibility:")
        compat = results['inference_compatibility']
        print(f"   Overall compatibility: {compat['overall_compatibility']}")
        print(f"   Successful tests: {compat['successful_tests']}/{compat['total_tests']}")

        # 全体評価
        print("\n4. Overall Assessment:")
        assessment = results['overall_assessment']
        print(f"   SO8T components removed: {assessment['so8t_components_removed']}")
        print(f"   Weights properly baked: {assessment['weights_properly_baked']}")
        print(f"   Inference compatible: {assessment['inference_compatible']}")
        print(f"   Baking successful: {assessment['baking_successful']}")
        print(f"   Ready for GGUF: {assessment['ready_for_gguf']}")

        print("\n" + "="*80)

        if assessment['baking_successful']:
            print("✅ SUCCESS: Model is properly baked and ready for GGUF conversion!")
        else:
            print("❌ FAILURE: Baking process has issues that need to be addressed.")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Verify SO8T baking process")
    parser.add_argument("--original", type=str, required=True, help="Path to original SO8T model")
    parser.add_argument("--baked", type=str, required=True, help="Path to baked model")

    args = parser.parse_args()

    verifier = SO8TBakingVerifier(args.original, args.baked)
    results = verifier.run_comprehensive_verification()
    verifier.print_verification_report(results)


if __name__ == "__main__":
    main()
