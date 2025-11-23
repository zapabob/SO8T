"""
Phi-4-mini-instruct へのSO8T統合スクリプト

このスクリプトは既存のPhi-4モデルにSO8T回転ゲートを追加する。
アテンション層の出力後にSO8T回転を挿入し、学習可能にする。

Usage:
    python integrate_phi4_so8t.py \\
        --model_path ../Phi-4-mini-instruct \\
        --output_path ./phi4_so8t \\
        --enable_so8t

Author: SO8T Project Team
Date: 2024-11-06
"""

import sys
import os
from pathlib import Path

# SO8T modulesをインポート可能にする
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from typing import Optional, Dict, Any
import logging
import json
from datetime import datetime

from so8t_layer import SO8TAttentionWrapper, SO8TRotationGate, collect_so8t_orthogonality_loss
from pet_regularization import PETRegularization, PETConfig, create_pet_regularization

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def integrate_so8t_into_phi4(
    model: nn.Module,
    config: Any,
    so8t_enabled: bool = True,
    init_scale: float = 0.05,
    orthogonal_reg: float = 1e-4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Phi-4モデルにSO8T回転ゲートを統合
    
    Args:
        model: Phi-4モデル
        config: モデルconfig
        so8t_enabled: SO8Tを有効にするか
        init_scale: 回転の初期化スケール
        orthogonal_reg: 直交性正則化の重み
        verbose: 詳細ログを出力するか
        
    Returns:
        integration_info: 統合情報
    """
    if not so8t_enabled:
        logger.info("[SO8T Integration] SO8T is disabled, skipping integration")
        return {'so8t_enabled': False, 'layers_modified': 0}
    
    logger.info("[SO8T Integration] Starting integration into Phi-4...")
    
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    
    # hidden_sizeが8の倍数であることを確認
    if hidden_size % 8 != 0:
        raise ValueError(f"hidden_size {hidden_size} must be divisible by 8 for SO8T")
    
    logger.info(f"[SO8T Integration] Model config:")
    logger.info(f"  hidden_size: {hidden_size}")
    logger.info(f"  num_layers: {num_layers}")
    logger.info(f"  num_blocks (SO8T): {hidden_size // 8}")
    
    # 各デコーダ層のアテンションをラップ
    layers_modified = 0
    
    # モデルのベース（model.model）から layers にアクセス
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        decoder_layers = model.model.layers
    elif hasattr(model, 'layers'):
        decoder_layers = model.layers
    else:
        raise AttributeError("Could not find decoder layers in model")
    
    for i, layer in enumerate(decoder_layers):
        if hasattr(layer, 'self_attn'):
            original_attn = layer.self_attn
            
            # SO8TAttentionWrapperでラップ
            wrapped_attn = SO8TAttentionWrapper(
                base_attention=original_attn,
                hidden_size=hidden_size,
                rotation_enabled=True,
                init_scale=init_scale,
                orthogonal_reg=orthogonal_reg
            )
            
            # 置き換え
            layer.self_attn = wrapped_attn
            layers_modified += 1
            
            if verbose and i % 8 == 0:
                logger.info(f"[SO8T Integration] Layer {i}/{num_layers} wrapped with SO8T")
    
    logger.info(f"[SO8T Integration] Complete!")
    logger.info(f"[SO8T Integration] Modified {layers_modified}/{num_layers} layers")
    
    # パラメータ数の計算
    so8t_params = sum(
        p.numel() for n, p in model.named_parameters() 
        if 'rotation_gate' in n
    )
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"[SO8T Integration] Parameters:")
    logger.info(f"  SO8T parameters: {so8t_params:,} ({so8t_params/1e6:.2f}M)")
    logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"  SO8T overhead: {100*so8t_params/total_params:.2f}%")
    
    integration_info = {
        'so8t_enabled': True,
        'layers_modified': layers_modified,
        'total_layers': num_layers,
        'hidden_size': hidden_size,
        'num_blocks': hidden_size // 8,
        'so8t_params': so8t_params,
        'total_params': total_params,
        'init_scale': init_scale,
        'orthogonal_reg': orthogonal_reg,
    }
    
    return integration_info


def save_so8t_model(
    model: nn.Module,
    tokenizer: Any,
    output_path: str,
    integration_info: Dict[str, Any],
    config_updates: Optional[Dict[str, Any]] = None
):
    """
    SO8T統合済みモデルを保存
    
    Args:
        model: SO8T統合済みモデル
        tokenizer: トークナイザー
        output_path: 保存先パス
        integration_info: 統合情報
        config_updates: config追加項目
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Save] Saving SO8T-integrated model to {output_path}")
    
    # モデルを保存
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # 統合情報を保存
    info_path = output_path / "so8t_integration_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(integration_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[Save] Integration info saved to {info_path}")
    
    # configに追加情報を記録
    if config_updates:
        config_path = output_path / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config_dict.update(config_updates)
        config_dict['so8t_integrated'] = True
        config_dict['so8t_integration_date'] = datetime.now().isoformat()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Save] Config updated with SO8T information")
    
    logger.info(f"[Save] Model saved successfully!")


def load_so8t_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16
) -> tuple:
    """
    SO8T統合済みモデルをロード
    
    Args:
        model_path: モデルパス
        device: デバイス
        torch_dtype: データ型
        
    Returns:
        model, tokenizer, integration_info
    """
    model_path = Path(model_path)
    
    logger.info(f"[Load] Loading SO8T-integrated model from {model_path}")
    
    # 統合情報を確認
    info_path = model_path / "so8t_integration_info.json"
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            integration_info = json.load(f)
        logger.info(f"[Load] Found integration info: {integration_info}")
    else:
        logger.warning(f"[Load] No integration info found, this may not be an SO8T model")
        integration_info = None
    
    # モデルをロード
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    logger.info(f"[Load] Model loaded successfully!")
    
    return model, tokenizer, integration_info


def test_so8t_forward(
    model: nn.Module,
    tokenizer: Any,
    device: str = "cuda"
):
    """
    SO8T統合後の前向き計算をテスト
    
    Args:
        model: SO8T統合済みモデル
        tokenizer: トークナイザー
        device: デバイス
    """
    logger.info("[Test] Testing SO8T forward pass...")
    
    model.eval()
    
    # テスト入力（日本語）
    test_inputs = [
        "防衛システムの安全性について説明してください。",
        "航空宇宙産業における最新技術の動向は？",
        "運輸・物流の効率化に関する提案をお願いします。"
    ]
    
    with torch.no_grad():
        for i, text in enumerate(test_inputs):
            logger.info(f"[Test {i+1}] Input: {text}")
            
            # トークナイズ
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            # 前向き計算
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 統計情報
            logger.info(f"[Test {i+1}] Output shape: {logits.shape}")
            logger.info(f"[Test {i+1}] Logits mean: {logits.mean().item():.4f}")
            logger.info(f"[Test {i+1}] Logits std: {logits.std().item():.4f}")
            logger.info(f"[Test {i+1}] Logits max: {logits.max().item():.4f}")
            logger.info(f"[Test {i+1}] Logits min: {logits.min().item():.4f}")
            
            # 生成テスト（短い）
            generated = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            logger.info(f"[Test {i+1}] Generated: {output_text}")
    
    logger.info("[Test] All forward tests passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate SO8T into Phi-4-mini-instruct")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Phi-4-mini-instruct model")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for SO8T-integrated model")
    parser.add_argument("--enable_so8t", action="store_true",
                       help="Enable SO8T rotation gates")
    parser.add_argument("--init_scale", type=float, default=0.05,
                       help="SO8T initialization scale")
    parser.add_argument("--orthogonal_reg", type=float, default=1e-4,
                       help="Orthogonal regularization weight")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                       help="Torch dtype (bfloat16/float16/float32)")
    parser.add_argument("--test", action="store_true",
                       help="Test forward pass after integration")
    
    args = parser.parse_args()
    
    # データ型の設定
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)
    
    print("=" * 80)
    print("Phi-4-mini-instruct SO8T Integration")
    print("=" * 80)
    print(f"\n[Config]")
    print(f"  Model path: {args.model_path}")
    print(f"  Output path: {args.output_path}")
    print(f"  SO8T enabled: {args.enable_so8t}")
    print(f"  Init scale: {args.init_scale}")
    print(f"  Orthogonal reg: {args.orthogonal_reg}")
    print(f"  Device: {args.device}")
    print(f"  Dtype: {args.torch_dtype}")
    
    # モデルとトークナイザーをロード
    print(f"\n[Step 1] Loading Phi-4-mini-instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    config = model.config
    print(f"[Step 1] Model loaded: {config.model_type}, {config.num_hidden_layers} layers")
    
    # SO8Tを統合
    print(f"\n[Step 2] Integrating SO8T...")
    integration_info = integrate_so8t_into_phi4(
        model=model,
        config=config,
        so8t_enabled=args.enable_so8t,
        init_scale=args.init_scale,
        orthogonal_reg=args.orthogonal_reg,
        verbose=True
    )
    
    # テスト（オプション）
    if args.test:
        print(f"\n[Step 3] Testing forward pass...")
        test_so8t_forward(model, tokenizer, args.device)
    
    # モデルを保存
    print(f"\n[Step 4] Saving SO8T-integrated model...")
    save_so8t_model(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output_path,
        integration_info=integration_info,
        config_updates={
            'so8t_init_scale': args.init_scale,
            'so8t_orthogonal_reg': args.orthogonal_reg,
        }
    )
    
    print("\n" + "=" * 80)
    print("[SO8T Integration] Complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Fine-tune with Japanese data: python train_so8t_ja.py")
    print(f"  2. Apply burn-in: python burn_in_so8t.py --model_path {args.output_path}")
    print(f"  3. Convert to GGUF: python convert_to_gguf.py")

