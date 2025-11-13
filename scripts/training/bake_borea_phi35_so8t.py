#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5 SO8T焼き込み処理スクリプト

学習済みSO8Tモデルの回転ゲートをo_projに焼き込み、標準Phi3アーキテクチャに戻す
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Phi3Config

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm" / "src"))

# SO8T統合モデルをインポート
try:
    from models.Borea_Phi_3_5_mini_Instruct_Jp.modeling_phi3_so8t import (
        SO8TPhi3Model,
        SO8TPhi3DecoderLayer
    )
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "Borea-Phi-3.5-mini-Instruct-Jp"))
    from modeling_phi3_so8t import (
        SO8TPhi3Model,
        SO8TPhi3DecoderLayer
    )

# SO8T回転ゲートをインポート
try:
    from so8t_mmllm.src.so8t_layer import SO8TRotationGate
except ImportError:
    try:
        from so8t_layer import SO8TRotationGate
    except ImportError:
        logger.warning("SO8TRotationGate not found, some functions may not work")
        SO8TRotationGate = None

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bake_borea_phi35_so8t.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_rotation_matrices_from_layer(layer: SO8TPhi3DecoderLayer) -> Optional[torch.Tensor]:
    """
    SO8Tレイヤーから回転行列を取得
    
    Args:
        layer: SO8TPhi3DecoderLayer
    
    Returns:
        回転行列 [num_blocks, 8, 8] または None
    """
    if not isinstance(layer, SO8TPhi3DecoderLayer):
        return None
    
    try:
        # SO8TPhi3Attentionから回転ゲートを取得
        if hasattr(layer.self_attn, 'so8t_rotation_gate'):
            rotation_gate = layer.self_attn.so8t_rotation_gate
            if hasattr(rotation_gate, 'get_rotation_matrices'):
                return rotation_gate.get_rotation_matrices()
            elif hasattr(rotation_gate, 'theta'):
                # thetaから回転行列を計算
                try:
                    from so8t_mmllm.src.so8t_layer import make_skew_symmetric, block_expm
                except ImportError:
                    try:
                        from so8t_layer import make_skew_symmetric, block_expm
                    except ImportError:
                        logger.warning("so8t_layer functions not found, cannot compute rotation matrices")
                        return None
                A = make_skew_symmetric(rotation_gate.theta)
                R = block_expm(A)
                return R
    except Exception as e:
        logger.warning(f"Failed to get rotation matrices: {e}")
        return None
    
    return None


def bake_rotation_into_linear(
    linear_layer: nn.Linear,
    rotation_matrices: torch.Tensor,
    layer_name: str = "unknown"
) -> nn.Linear:
    """
    線形層に回転行列を焼き込み（右掛け: W' = W @ R）
    
    Args:
        linear_layer: 焼き込み対象の線形層
        rotation_matrices: 回転行列 [num_blocks, 8, 8]
        layer_name: レイヤー名（ログ用）
    
    Returns:
        焼き込み済み線形層
    """
    with torch.no_grad():
        W = linear_layer.weight.data.clone()  # [out_features, in_features]
        b = linear_layer.bias.data.clone() if linear_layer.bias is not None else None
        
        out_features, in_features = W.shape
        
        if in_features % 8 != 0:
            logger.warning(f"  {layer_name}: in_features ({in_features}) not divisible by 8, skipping")
            return linear_layer
        
        num_blocks = in_features // 8
        
        if rotation_matrices.shape[0] != num_blocks:
            logger.warning(f"  {layer_name}: num_blocks mismatch "
                          f"(weight: {num_blocks}, rotation: {rotation_matrices.shape[0]}), skipping")
            return linear_layer
        
        # ブロック対角回転行列を構築
        R_full = torch.zeros(in_features, in_features, device=W.device, dtype=W.dtype)
        for i in range(num_blocks):
            start_idx = i * 8
            end_idx = start_idx + 8
            R_full[start_idx:end_idx, start_idx:end_idx] = rotation_matrices[i]
        
        # 右掛け焼き込み: W' = W @ R
        W_baked = torch.mm(W, R_full)
        
        # 新しい線形層を作成
        baked_layer = nn.Linear(in_features, out_features, bias=(b is not None))
        baked_layer.weight.data = W_baked
        if b is not None:
            baked_layer.bias.data = b
        
        logger.info(f"  [OK] Baked rotation into {layer_name}")
        return baked_layer


def bake_so8t_model(
    model_path: Path,
    output_path: Path,
    verify: bool = True
) -> bool:
    """
    SO8Tモデルを焼き込み処理
    
    Args:
        model_path: 学習済みSO8Tモデルパス
        output_path: 出力パス
        verify: 検証を実行するか
    
    Returns:
        成功したかどうか
    """
    logger.info("="*80)
    logger.info("SO8T Baking Process")
    logger.info("="*80)
    logger.info(f"Input model: {model_path}")
    logger.info(f"Output path: {output_path}")
    
    # モデル読み込み
    logger.info("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # 検証用のテスト入力（検証する場合）
    test_input = None
    test_output_before = None
    if verify:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        test_text = "Hello, how are you?"
        test_input = tokenizer(test_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            test_output_before = model(**test_input)
    
    # 各レイヤーを処理
    baked_count = 0
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        logger.error("Could not find decoder layers in model")
        return False
    
    for layer_idx, layer in enumerate(layers):
        if not isinstance(layer, SO8TPhi3DecoderLayer):
            continue
        
        # 回転行列を取得
        rotation_matrices = get_rotation_matrices_from_layer(layer)
        if rotation_matrices is None:
            logger.warning(f"Layer {layer_idx}: Could not get rotation matrices, skipping")
            continue
        
        # o_projに焼き込み
        if hasattr(layer.self_attn, 'o_proj'):
            o_proj = layer.self_attn.o_proj
            baked_o_proj = bake_rotation_into_linear(
                o_proj,
                rotation_matrices,
                layer_name=f"layer_{layer_idx}.self_attn.o_proj"
            )
            layer.self_attn.o_proj = baked_o_proj
            baked_count += 1
    
    logger.info(f"[OK] Baked {baked_count} layers")
    
    # 検証（焼き込み前後で出力が一致するか）
    if verify and test_input is not None:
        logger.info("Verifying baked model...")
        with torch.no_grad():
            test_output_after = model(**test_input)
        
        # 簡易検証: 最初の10トークンの確率分布を比較
        logits_before = test_output_before.logits[0, -1, :10]
        logits_after = test_output_after.logits[0, -1, :10]
        
        diff = torch.abs(logits_before - logits_after).mean()
        logger.info(f"Verification: Mean absolute difference = {diff.item():.6e}")
        
        if diff.item() > 0.1:
            logger.warning("Large difference detected! Baking may have issues.")
        else:
            logger.info("[OK] Verification passed")
    
    # モデル保存
    logger.info(f"Saving baked model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    if verify:
        tokenizer.save_pretrained(str(output_path))
    
    logger.info("[SUCCESS] Baking completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Bake SO8T rotations into Borea-Phi-3.5 model"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Trained SO8T model path"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output path for baked model"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification"
    )
    
    args = parser.parse_args()
    
    success = bake_so8t_model(
        model_path=args.model_path,
        output_path=args.output_path,
        verify=not args.no_verify
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()


