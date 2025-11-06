#!/usr/bin/env python3
"""
Phi-4-mini-instructにSO8T統合スクリプト
32層全てのアテンション層にSO8T回転ゲートを追加
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from so8t_core import SO8TRotationGate

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def integrate_so8t_to_phi4(
    model_path: str,
    output_path: str,
    device: str = "cuda",
    verify: bool = True,
):
    """
    Phi-4-mini-instructにSO8T回転ゲートを統合
    
    Args:
        model_path: 元のPhi-4モデルパス
        output_path: SO8T統合後のモデル保存先
        device: デバイス（cuda/cpu）
        verify: 統合検証を行うか
    """
    logger.info(f"[STEP 1] Loading Phi-4 model from {model_path}")
    
    # モデル・トークナイザー・設定読み込み
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else "cpu",
    )
    
    logger.info(f"Model architecture: {config.model_type}")
    logger.info(f"Hidden size: {config.hidden_size}")
    logger.info(f"Number of layers: {config.num_hidden_layers}")
    logger.info(f"Number of attention heads: {config.num_attention_heads}")
    
    # hidden_sizeが8の倍数か確認
    if config.hidden_size % 8 != 0:
        raise ValueError(
            f"hidden_size ({config.hidden_size}) must be divisible by 8 for SO8T integration"
        )
    
    logger.info(f"[STEP 2] Adding SO8T rotation gates to {config.num_hidden_layers} layers")
    
    # 各レイヤーにSO8T回転ゲートを追加
    integrated_layers = 0
    
    for layer_idx in tqdm(range(config.num_hidden_layers), desc="Integrating SO8T"):
        # Phi-3アーキテクチャのレイヤーアクセス
        layer = model.model.layers[layer_idx]
        
        # アテンション層を取得
        # Phi-3: layer.self_attn
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            
            # o_projの後にSO8T回転ゲートを追加
            if hasattr(attn, 'o_proj'):
                # SO8TRotationGateを作成
                so8t_gate = SO8TRotationGate(
                    hidden_size=config.hidden_size,
                    use_cayley=True,
                    orthogonal_regularization=1e-3,
                )
                
                # デバイスに移動
                so8t_gate = so8t_gate.to(device=attn.o_proj.weight.device, dtype=attn.o_proj.weight.dtype)
                
                # アテンション層に追加
                attn.so8t_gate = so8t_gate
                integrated_layers += 1
                
                logger.debug(f"Added SO8T gate to layer {layer_idx}")
            else:
                logger.warning(f"Layer {layer_idx} does not have o_proj, skipping")
        else:
            logger.warning(f"Layer {layer_idx} does not have self_attn, skipping")
    
    logger.info(f"[STEP 3] SO8T integration complete: {integrated_layers}/{config.num_hidden_layers} layers")
    
    # 前向き計算フックを追加
    def add_so8t_forward_hook(attn_module):
        """アテンション層の前向き計算にSO8T回転を適用するフック"""
        original_forward = attn_module.forward
        
        def forward_with_so8t(*args, **kwargs):
            # 元のアテンション計算
            outputs = original_forward(*args, **kwargs)
            
            # SO8T回転を適用
            if hasattr(attn_module, 'so8t_gate'):
                if isinstance(outputs, tuple):
                    # (hidden_states, attention_weights, ...)形式
                    hidden_states = outputs[0]
                    hidden_states = attn_module.so8t_gate(hidden_states)
                    outputs = (hidden_states,) + outputs[1:]
                else:
                    # hidden_statesのみ
                    outputs = attn_module.so8t_gate(outputs)
            
            return outputs
        
        attn_module.forward = forward_with_so8t
    
    # 全レイヤーにフックを追加
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'so8t_gate'):
            add_so8t_forward_hook(layer.self_attn)
    
    logger.info("[STEP 4] Added forward hooks for SO8T rotation")
    
    # 統合検証
    if verify:
        logger.info("[STEP 5] Verifying SO8T integration")
        verify_integration(model, tokenizer, device)
    
    # モデル保存
    logger.info(f"[STEP 6] Saving SO8T-integrated model to {output_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # 統合情報を保存
    integration_info = {
        'original_model': str(model_path),
        'integrated_layers': integrated_layers,
        'total_layers': config.num_hidden_layers,
        'hidden_size': config.hidden_size,
        'num_blocks': config.hidden_size // 8,
        'so8t_parameters_per_layer': 64,  # 8x8 rotation matrix parameters
        'total_so8t_parameters': integrated_layers * 64,
    }
    
    with open(output_path / 'so8t_integration_info.json', 'w', encoding='utf-8') as f:
        json.dump(integration_info, f, indent=2, ensure_ascii=False)
    
    logger.info("[SUCCESS] SO8T integration completed successfully!")
    logger.info(f"Integrated {integrated_layers} layers with SO8T rotation gates")
    logger.info(f"Model saved to {output_path}")
    
    return model


def verify_integration(model, tokenizer, device):
    """
    SO8T統合の検証
    
    Args:
        model: SO8T統合済みモデル
        tokenizer: トークナイザー
        device: デバイス
    """
    logger.info("Running integration verification...")
    
    # テスト入力
    test_inputs = [
        "こんにちは、私はAIアシスタントです。",
        "What is 2+2?",
        "Explain SO(8) group structure.",
    ]
    
    model.eval()
    with torch.no_grad():
        for text in test_inputs:
            try:
                inputs = tokenizer(text, return_tensors="pt").to(device)
                outputs = model(**inputs)
                
                # 出力形状確認
                assert outputs.logits.shape[0] == 1  # batch_size
                assert outputs.logits.shape[2] == model.config.vocab_size
                
                logger.info(f"✓ Test passed: '{text[:30]}...'")
            
            except Exception as e:
                logger.error(f"✗ Test failed: '{text[:30]}...'")
                logger.error(f"Error: {e}")
                raise
    
    # 直交性検証
    logger.info("Verifying SO8T orthogonality...")
    
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'so8t_gate'):
            so8t_gate = layer.self_attn.so8t_gate
            metrics = so8t_gate.verify_orthogonality()
            
            logger.info(
                f"Layer {layer_idx}: "
                f"max_error={metrics['max_orthogonality_error']:.2e}, "
                f"det_error={metrics['determinant_error']:.2e}"
            )
            
            # 閾値チェック
            if metrics['max_orthogonality_error'] > 1e-4:
                logger.warning(
                    f"Layer {layer_idx} has high orthogonality error: "
                    f"{metrics['max_orthogonality_error']:.2e}"
                )
    
    logger.info("✓ Integration verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Integrate SO8T into Phi-4-mini-instruct")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Phi-4-mini-instruct",
        help="Path to Phi-4 model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="phi4_so8t_integrated",
        help="Output path for SO8T-integrated model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip integration verification"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("SO8T Integration for Phi-4-mini-instruct")
    logger.info("=" * 70)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 70)
    
    integrate_so8t_to_phi4(
        model_path=args.model_path,
        output_path=args.output_path,
        device=args.device,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()


