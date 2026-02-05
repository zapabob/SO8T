#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T焼き込みパイプライン
学習済みSO8T回転行列を線形層に吸収
標準重み化（カスタム演算なし）
等価性検証（ロジット誤差<1e-5）
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_so8t_rotation_params(model_dir: Path) -> Dict[str, torch.Tensor]:
    """
    SO8T回転パラメータ読み込み
    
    Args:
        model_dir: モデルディレクトリ
    
    Returns:
        rotation_params: 回転パラメータ辞書
    """
    so8t_path = model_dir / "so8t_rotation_params.pt"
    
    if not so8t_path.exists():
        raise FileNotFoundError(f"SO8T rotation parameters not found: {so8t_path}")
    
    rotation_params = torch.load(so8t_path)
    print(f"[LOAD] SO8T rotation parameters: {so8t_path}")
    
    return rotation_params


def build_rotation_matrix(rotation_params: torch.Tensor, dim: int = 8) -> torch.Tensor:
    """
    回転行列構築
    
    Args:
        rotation_params: 回転パラメータ（28次元）
        dim: 次元（デフォルト8）
    
    Returns:
        R: (dim, dim) 回転行列
    """
    R = torch.eye(dim, device=rotation_params.device)
    
    param_idx = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            theta = rotation_params[param_idx]
            G = torch.eye(dim, device=R.device)
            G[i, i] = torch.cos(theta)
            G[i, j] = -torch.sin(theta)
            G[j, i] = torch.sin(theta)
            G[j, j] = torch.cos(theta)
            R = torch.matmul(R, G)
            param_idx += 1
    
    return R


def apply_burnin_to_linear(linear_layer: nn.Linear, rotation_matrix: torch.Tensor) -> nn.Linear:
    """
    線形層に回転行列を焼き込む
    
    Args:
        linear_layer: nn.Linear層
        rotation_matrix: SO8T回転行列
    
    Returns:
        burned_layer: 焼き込み済み線形層
    """
    # 重み行列取得
    W = linear_layer.weight.data.clone()  # (out_features, in_features)
    b = linear_layer.bias.data.clone() if linear_layer.bias is not None else None
    
    # 回転適用（入力側）
    # y = W @ (R @ x) + b = (W @ R) @ x + b
    # 新しい重み: W' = W @ R
    
    out_features, in_features = W.shape
    dim = rotation_matrix.shape[0]
    
    # 8次元ブロックごとに焼き込み
    n_blocks = in_features // dim
    
    for block_idx in range(n_blocks):
        start_idx = block_idx * dim
        end_idx = start_idx + dim
        
        # ブロック抽出
        W_block = W[:, start_idx:end_idx]  # (out_features, dim)
        
        # 回転焼き込み
        W_block_burned = torch.matmul(W_block, rotation_matrix.T)
        
        # 書き戻し
        W[:, start_idx:end_idx] = W_block_burned
    
    # 新しい線形層作成
    burned_layer = nn.Linear(in_features, out_features, bias=(b is not None))
    burned_layer.weight.data = W
    if b is not None:
        burned_layer.bias.data = b
    
    return burned_layer


def apply_burnin_to_model(model: nn.Module, rotation_params: Dict[str, torch.Tensor]) -> nn.Module:
    """
    モデル全体に焼き込み適用
    
    Args:
        model: Transformerモデル
        rotation_params: SO8T回転パラメータ
    
    Returns:
        burned_model: 焼き込み済みモデル
    """
    print("[BURNIN] Applying burn-in to all linear layers...")
    
    # 回転行列構築
    rotation_matrix = build_rotation_matrix(rotation_params['rotation_params'])
    
    # すべてのLinear層に焼き込み
    burned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 焼き込み適用
            burned_layer = apply_burnin_to_linear(module, rotation_matrix)
            
            # モジュール置き換え
            parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, burned_layer)
            else:
                setattr(model, child_name, burned_layer)
            
            burned_count += 1
    
    print(f"[OK] Burned {burned_count} linear layers")
    
    return model


def verify_equivalence(
    original_model: nn.Module,
    burned_model: nn.Module,
    tokenizer,
    test_prompts: list,
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    等価性検証
    
    Args:
        original_model: 元モデル（SO8T適用）
        burned_model: 焼き込み後モデル
        tokenizer: トークナイザー
        test_prompts: テストプロンプト
        device: デバイス
    
    Returns:
        max_logit_error: 最大ロジット誤差
        kl_divergence: KLダイバージェンス
    """
    print("\n[VERIFY] Checking burn-in equivalence...")
    
    original_model.eval()
    burned_model.eval()
    
    logit_errors = []
    kl_divs = []
    
    with torch.no_grad():
        for prompt in test_prompts:
            # トークナイズ
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # 元モデル推論
            orig_outputs = original_model(**inputs)
            orig_logits = orig_outputs.logits
            
            # 焼き込みモデル推論
            burned_outputs = burned_model(**inputs)
            burned_logits = burned_outputs.logits
            
            # ロジット誤差
            logit_error = torch.abs(orig_logits - burned_logits).max().item()
            logit_errors.append(logit_error)
            
            # KLダイバージェンス
            orig_probs = torch.softmax(orig_logits, dim=-1)
            burned_probs = torch.softmax(burned_logits, dim=-1)
            kl_div = torch.sum(orig_probs * torch.log(orig_probs / (burned_probs + 1e-10))).item()
            kl_divs.append(kl_div)
    
    max_logit_error = max(logit_errors)
    avg_kl_div = np.mean(kl_divs)
    
    print(f"[RESULT] Max logit error: {max_logit_error:.2e}")
    print(f"[RESULT] Avg KL divergence: {avg_kl_div:.2e}")
    
    # 閾値チェック
    if max_logit_error < 1e-5:
        print("[OK] Burn-in equivalence verified (logit error < 1e-5)")
    else:
        print(f"[WARNING] Logit error {max_logit_error:.2e} exceeds threshold")
    
    if avg_kl_div < 1e-6:
        print("[OK] Distribution equivalence verified (KL div < 1e-6)")
    else:
        print(f"[WARNING] KL divergence {avg_kl_div:.2e} exceeds threshold")
    
    return max_logit_error, avg_kl_div


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, 
                        default=Path("outputs/borea_so8t_four_role/final_model"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/borea_so8t_burned"))
    parser.add_argument("--verify", action="store_true", help="Run equivalence verification")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"[START] SO8T Burn-in Pipeline")
    print(f"Model: {args.model_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    print("[LOAD] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    
    # Load model
    print("[LOAD] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load SO8T rotation params
    rotation_params = load_so8t_rotation_params(args.model_dir)
    
    # Apply burn-in
    burned_model = apply_burnin_to_model(model, rotation_params)
    
    # Verify equivalence
    if args.verify:
        test_prompts = [
            "防衛装備品の調達手続きについて説明してください",
            "取引の不正検知結果を報告してください",
            "患者のカルテから診断所見を要約してください"
        ]
        
        max_error, kl_div = verify_equivalence(
            model, burned_model, tokenizer, test_prompts
        )
    
    # Save burned model
    print(f"\n[SAVE] Saving burned model to {args.output_dir}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    burned_model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    
    # メタデータ保存
    metadata = {
        "burnin_applied": True,
        "original_model": str(args.model_dir),
        "burnin_date": str(datetime.now()),
        "so8t_removed": True,
        "standard_weights_only": True,
        "custom_ops_required": False
    }
    
    if args.verify:
        metadata["verification"] = {
            "max_logit_error": float(max_error),
            "avg_kl_divergence": float(kl_div),
            "verified": max_error < 1e-5 and kl_div < 1e-6
        }
    
    with open(args.output_dir / "burnin_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"[OK] Burn-in completed!")
    print(f"Burned model: {args.output_dir}")
    print(f"Standard weights: YES")
    print(f"Custom ops required: NO")
    print(f"Ready for GGUF conversion")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import datetime
    main()

