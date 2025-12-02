#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) アダプター焼き込みスクリプト

学習済み SO(8) 残差アダプター付きモデルを読み込み、
回転の効果をベースモデルの重みに焼き込んで、
アダプターを削除した純粋な HF モデルを生成します。

生成されたモデルは llama.cpp で GGUF 変換可能です。
"""

import torch
import argparse
from pathlib import Path
from typing import Optional
import json
import sys
import gc

# 自作モジュールをインポート
sys.path.append(str(Path(__file__).parent.parent))
from training.so8_compatible_adapter import (
    SO8CompatibleLoRA,
    bake_so8_adapter_into_base_model,
    save_baked_so8_model
)


def load_model_with_so8_adapters(
    model_path: str,
    adapter_config_path: Optional[str] = None,
    device: str = "auto"
):
    """
    SO(8) アダプター付きモデルをロード

    Args:
        model_path: モデルディレクトリパス
        adapter_config_path: アダプタ設定ファイルパス（Noneなら自動検出）
        device: デバイス指定

    Returns:
        (model, injected_adapters, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)

    # アダプタ設定ファイルの自動検出
    if adapter_config_path is None:
        adapter_config_path = model_path / "adapter_config.json"

    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")

    # モデルをロード
    print(f"[SO8] Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device
    )

    # トークナイザーをロード
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # アダプタ設定をロード
    with open(adapter_config_path, 'r', encoding='utf-8') as f:
        adapter_config = json.load(f)

    print(f"[SO8] Adapter config loaded: {adapter_config_path}")
    print(f"[SO8] PEFT type: {adapter_config.get('peft_type', 'UNKNOWN')}")

    # SO(8) アダプターを再構築
    injected_adapters = {}
    target_modules = adapter_config.get('target_modules', [])

    print(f"[SO8] Reconstructing SO(8) adapters for {len(target_modules)} modules...")

    # アダプタ重みをロード
    adapter_model_path = model_path / "adapter_model.safetensors"
    if adapter_model_path.exists():
        adapter_state = torch.load(adapter_model_path, map_location='cpu')
        print(f"[SO8] Loaded adapter weights from {adapter_model_path}")
    else:
        # binファイルの場合
        adapter_bin_path = model_path / "adapter_model.bin"
        if adapter_bin_path.exists():
            adapter_state = torch.load(adapter_bin_path, map_location='cpu')
            print(f"[SO8] Loaded adapter weights from {adapter_bin_path}")
        else:
            raise FileNotFoundError("No adapter weights found")

    # 各ターゲットモジュールに対してSO(8)アダプターを再構築
    for module_name in target_modules:
        # PEFT形式のキーから重みを取得
        peft_module_name = module_name.replace(".", ".lora_")
        lora_A_key = f"base_model.model.{peft_module_name}.lora_A.weight"
        lora_B_key = f"base_model.model.{peft_module_name}.lora_B.weight"

        if lora_A_key in adapter_state and lora_B_key in adapter_state:
            lora_A_weight = adapter_state[lora_A_key]
            lora_B_weight = adapter_state[lora_B_key]

            # SO(8) アダプターを再構築
            # 注意: 元のSO(8)パラメータは失われているので、標準LoRAとして扱う
            adapter = SO8CompatibleLoRA(
                hidden_size=lora_A_weight.size(1),  # lora_A: [rank, hidden]
                rank=lora_A_weight.size(0),
                alpha=adapter_config.get('lora_alpha', 1.0)
            )

            # 重みを設定（逆変換: 元の lora_A = R @ standard_lora_A）
            # ここでは標準LoRAとして扱うのでそのまま設定
            adapter.lora_A.data.copy_(lora_A_weight)
            adapter.lora_B.data.copy_(lora_B_weight)

            injected_adapters[module_name] = adapter
            print(f"[SO8] Reconstructed adapter for {module_name}")
        else:
            print(f"[WARNING] Adapter weights not found for {module_name}")

    print(f"[SO8] Successfully reconstructed {len(injected_adapters)} SO(8) adapters")

    return model, injected_adapters, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SO(8) アダプタ焼き込みスクリプト")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="SO(8) アダプター付きモデルディレクトリ"
    )
    parser.add_argument(
        "--adapter_config",
        type=str,
        default=None,
        help="アダプタ設定ファイルパス（自動検出の場合は指定不要）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="焼き込み済みモデルの出力ディレクトリ"
    )
    parser.add_argument(
        "--adapter_position",
        type=str,
        default="input",
        choices=["input", "output"],
        help="アダプターの位置（デフォルト: input）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="デバイス指定（auto/cpu/cuda）"
    )
    parser.add_argument(
        "--convert_gguf",
        action="store_true",
        help="GGUF変換も実行"
    )
    parser.add_argument(
        "--gguf_quantization",
        type=str,
        default="q4_k_m",
        choices=["f16", "q8_0", "q4_k_m", "q4_0", "q3_k_l", "q2_k"],
        help="GGUF量子化タイプ"
    )

    args = parser.parse_args()

    print("=== SO(8) アダプター焼き込みスクリプト ===")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Adapter position: {args.adapter_position}")
    print(f"Device: {args.device}")

    try:
        # 1. SO(8) アダプター付きモデルをロード
        model, injected_adapters, tokenizer = load_model_with_so8_adapters(
            args.model_path,
            args.adapter_config,
            args.device
        )

        # 2. SO(8) アダプターをベース重みに焼き込み
        print(f"\n[SO8] Baking {len(injected_adapters)} adapters into base model...")
        baked_model = bake_so8_adapter_into_base_model(
            model,
            injected_adapters,
            args.adapter_position
        )

        # メモリ解放
        del model, injected_adapters
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. 焼き込み済みモデルを保存
        save_baked_so8_model(baked_model, args.output_dir, tokenizer)

        print("
✅ SO(8) アダプターの焼き込みが完了しました！"        print(f"出力ディレクトリ: {args.output_dir}")
        print(f"アダプター位置: {args.adapter_position}")
        print(f"モデルサイズ: {sum(p.numel() for p in baked_model.parameters()):,}")

        # 4. GGUF変換（オプション）
        if args.convert_gguf:
            print(f"\n[SO8] GGUF変換を実行します...")
            from training.so8_compatible_adapter import convert_baked_so8_to_gguf

            gguf_output = f"{args.output_dir}/baked_so8_model_{args.gguf_quantization}.gguf"
            success = convert_baked_so8_to_gguf(
                args.output_dir,
                gguf_output,
                args.gguf_quantization
            )

            if success:
                print(f"✅ GGUF変換完了: {gguf_output}")
            else:
                print("❌ GGUF変換失敗")
                return 1

        # 完了通知
        print("
🎉 すべての処理が完了しました！"        print("llama.cpp で GGUF モデルを使用できます。")

        return 0

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
