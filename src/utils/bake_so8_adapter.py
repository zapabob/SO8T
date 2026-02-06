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
    SO(8) アダプター付きモデルをロード（既存のSO(8)構造に対応）

    Args:
        model_path: モデルディレクトリパス
        adapter_config_path: アダプタ設定ファイルパス（Noneなら自動検出）
        device: デバイス指定

    Returns:
        (model, injected_adapters, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)

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

    # 既存のSO(8)アダプター構造を検出
    injected_adapters = {}

    def find_so8_adapters(module, name=""):
        """モデル内のSO(8)アダプターを再帰的に検索"""
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # so8_adapterが見つかったら登録
            if child_name == "so8_adapter":
                injected_adapters[name] = child_module
                print(f"[SO8] Found existing SO(8) adapter: {name}")
                continue

            # 子モジュールを再帰的に検索
            find_so8_adapters(child_module, full_name)

    find_so8_adapters(model)

    print(f"[SO8] Found {len(injected_adapters)} existing SO(8) adapters")

    return model, injected_adapters, tokenizer


def get_so8_adapter_effective_matrix(adapter):
    """
    既存のSO(8)アダプターから有効行列を計算

    Args:
        adapter: SO(8) アダプターモジュール

    Returns:
        [hidden_size, hidden_size] の有効行列
    """
    # アダプターの隠れ層サイズを取得
    if hasattr(adapter, 'adapter_down') and hasattr(adapter.adapter_down, 'weight'):
        hidden_size = adapter.adapter_down.weight.size(0)  # [out, in] の out
    else:
        raise ValueError("Cannot determine hidden size from adapter structure")

    # 単位行列で初期化
    I = torch.eye(hidden_size, dtype=torch.float16, device='cpu')

    # SO(8)ゲートがある場合、回転行列を取得
    if hasattr(adapter, 'so8_gate'):
        so8_gate = adapter.so8_gate
        # 回転行列を計算（簡易バージョン）
        # 注意: 実際のSO(8)ゲート構造に依存
        try:
            if hasattr(so8_gate, 'compute_rotation_matrix'):
                R = so8_gate.compute_rotation_matrix()
            else:
                # デフォルトで単位行列
                R = I
        except:
            R = I
    else:
        R = I

    # アダプターダウン/アップの重みを取得
    if hasattr(adapter, 'adapter_down') and hasattr(adapter, 'adapter_up'):
        down_weight = adapter.adapter_down.weight  # [hidden, rank] or similar
        up_weight = adapter.adapter_up.weight      # [rank, hidden] or similar

        # アダプターのスケールを取得
        scale = getattr(adapter, 'adapter_scale', 1.0)
        if isinstance(scale, torch.Tensor):
            scale = scale.item()

        # 有効行列を計算: I + scale * (up @ R @ down)
        # 注意: 実際の構造に合わせて調整が必要
        try:
            adapter_matrix = torch.matmul(up_weight, torch.matmul(R, down_weight))
            effective_matrix = I + scale * adapter_matrix
        except:
            # 計算できない場合は単位行列
            effective_matrix = I
    else:
        effective_matrix = I

    return effective_matrix


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

        print("\n[OK] SO(8) アダプターの焼き込みが完了しました！")
        print(f"出力ディレクトリ: {args.output_dir}")
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
                print(f"[OK] GGUF変換完了: {gguf_output}")
            else:
                print("[NG] GGUF変換失敗")
                return 1

        # 完了通知
        print("\n[DONE] すべての処理が完了しました！")
        print("llama.cpp で GGUF モデルを使用できます。")

        return 0

    except Exception as e:
        print(f"\n[ERROR] エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
