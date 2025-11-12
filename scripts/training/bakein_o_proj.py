#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Rotation Bake-in Tool

学習した各層のR_effをW_oに右掛けで吸収し、追加演算を除去して標準グラフ化する。

Usage:
    python scripts/training/bakein_o_proj.py \
        --model_path D:/webdataset/checkpoints/training/so8t_lora \
        --output_path D:/webdataset/models/so8t_baked \
        --base_model models/Borea-Phi-3.5-mini-Instruct-Jp
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# SO8T modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from so8t_core.so8t_layer import SO8TRotationGate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def get_block_diag_matrix(rotation_gate: SO8TRotationGate) -> torch.Tensor:
    """
    回転ゲートからブロック対角行列を取得
    
    Args:
        rotation_gate: SO8TRotationGateインスタンス
    
    Returns:
        R_eff: [D, D] ブロック対角回転行列
    """
    with torch.no_grad():
        # 回転行列を生成
        if rotation_gate.use_cayley:
            rot_blocks = rotation_gate._cayley_rotation(rotation_gate.theta)
        else:
            rot_blocks = rotation_gate._exp_rotation(rotation_gate.theta)
        
        # ブロック対角行列を構築
        R_eff = torch.block_diag(*[rot_blocks[i] for i in range(rot_blocks.size(0))])
        
        return R_eff


def bake_so8t_into_o_proj(
    model: nn.Module,
    rotation_gates: Dict[str, SO8TRotationGate]
) -> int:
    """
    SO8T回転をo_projに焼き込む
    
    Args:
        model: モデル
        rotation_gates: 回転ゲートの辞書（レイヤー名 -> SO8TRotationGate）
    
    Returns:
        焼き込み済みレイヤー数
    """
    baked_count = 0
    
    logger.info("Baking SO8T rotations into o_proj weights...")
    
    for layer_name, rotation_gate in rotation_gates.items():
        try:
            # レイヤーを取得
            parts = layer_name.split('.')
            layer = model
            for part in parts:
                if hasattr(layer, part):
                    layer = getattr(layer, part)
                else:
                    logger.warning(f"  {layer_name}: Layer not found, skipping")
                    break
            else:
                # o_projを取得
                if hasattr(layer, 'o_proj'):
                    o_proj = layer.o_proj
                    
                    # Sequentialの場合、最後の要素がo_proj
                    if isinstance(o_proj, nn.Sequential):
                        # 最初の要素がSO8T回転ゲート、最後がo_proj
                        if len(o_proj) >= 2:
                            so8t_gate = o_proj[0]
                            original_o_proj = o_proj[-1]
                            
                            # 回転行列を取得
                            if isinstance(so8t_gate, SO8TRotationGate):
                                R_eff = get_block_diag_matrix(so8t_gate)
                                
                                # W_oの重みを取得
                                W_o = original_o_proj.weight.data  # [out_features, in_features]
                                
                                # 次元チェック
                                if W_o.shape[1] != R_eff.shape[0]:
                                    logger.warning(
                                        f"  {layer_name}: Dimension mismatch "
                                        f"(W_o: {W_o.shape[1]}, R_eff: {R_eff.shape[0]}), skipping"
                                    )
                                    continue
                                
                                # 焼き込み: W' = W · R
                                with torch.no_grad():
                                    W_baked = W_o @ R_eff
                                    original_o_proj.weight.data.copy_(W_baked)
                                
                                # SequentialからSO8T回転ゲートを除去
                                # o_projを元のLinear層に置き換え
                                layer.o_proj = original_o_proj
                                
                                baked_count += 1
                                logger.info(f"  Baked rotation into {layer_name}.o_proj")
                            else:
                                logger.warning(f"  {layer_name}: First element is not SO8TRotationGate, skipping")
                        else:
                            logger.warning(f"  {layer_name}: Sequential has < 2 elements, skipping")
                    else:
                        logger.warning(f"  {layer_name}: o_proj is not Sequential, skipping")
                else:
                    logger.warning(f"  {layer_name}: No o_proj found, skipping")
        except Exception as e:
            logger.error(f"  {layer_name}: Error during baking: {e}")
            continue
    
    logger.info(f"Baking complete: {baked_count} layers processed")
    return baked_count


def collect_rotation_gates(model: nn.Module) -> Dict[str, SO8TRotationGate]:
    """
    モデルからSO8T回転ゲートを収集
    
    Args:
        model: モデル
    
    Returns:
        回転ゲートの辞書（レイヤー名 -> SO8TRotationGate）
    """
    rotation_gates = {}
    
    for name, module in model.named_modules():
        if isinstance(module, SO8TRotationGate):
            rotation_gates[name] = module
        elif isinstance(module, nn.Sequential):
            # Sequential内のSO8TRotationGateを検索
            for idx, submodule in enumerate(module):
                if isinstance(submodule, SO8TRotationGate):
                    rotation_gates[f"{name}[{idx}]"] = submodule
    
    return rotation_gates


def main():
    parser = argparse.ArgumentParser(description="SO8T Rotation Bake-in")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model with SO8T gates")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for baked model")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model path (for tokenizer)")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("SO8T Rotation Bake-in")
    logger.info("="*80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output path: {args.output_path}")
    
    # 出力ディレクトリ作成
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # トークナイザー読み込み
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデル読み込み
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # SO8T回転ゲートを収集
    logger.info("Collecting SO8T rotation gates...")
    rotation_gates = collect_rotation_gates(model)
    logger.info(f"Found {len(rotation_gates)} SO8T rotation gates")
    
    if len(rotation_gates) == 0:
        logger.warning("No SO8T rotation gates found. Model may already be baked.")
    else:
        # 焼き込み実行
        baked_count = bake_so8t_into_o_proj(model, rotation_gates)
        
        if baked_count == 0:
            logger.error("Failed to bake any rotations. Check model structure.")
            return
    
    # 焼き込み済みモデルを保存
    logger.info(f"Saving baked model to {output_path}...")
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    
    logger.info("="*80)
    logger.info("Bake-in completed!")
    logger.info("="*80)
    logger.info(f"Baked model saved to: {output_path}")
    logger.info("Model is now ready for GGUF conversion (standard graph only)")


if __name__ == "__main__":
    main()

