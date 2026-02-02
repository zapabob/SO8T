#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) 焼き込み済みモデルを GGUF に変換

SO(8) アダプターを焼き込んだ純粋な HF モデルを
llama.cpp 形式の GGUF に変換します。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def convert_baked_so8_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
    llama_cpp_dir: str = None
):
    """
    SO(8) 焼き込み済みモデルを GGUF に変換

    Args:
        model_path: HFモデルディレクトリパス
        output_path: GGUF出力ファイルパス
        quantization: 量子化タイプ
        llama_cpp_dir: llama.cpp ディレクトリパス（Noneで自動検出）
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    # llama.cpp パスを自動検出
    if llama_cpp_dir is None:
        llama_cpp_dir = Path(__file__).parent.parent.parent / "external" / "llama.cpp-master"

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}")

    # 出力ディレクトリ作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== SO(8) Baked Model GGUF Conversion ===")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {quantization}")
    print(f"llama.cpp dir: {llama_cpp_dir}")

    # 変換コマンド構築
    cmd = [
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", quantization
    ]

    print(f"Running: {' '.join(cmd)}")

    # コマンド実行
    result = subprocess.run(cmd, cwd=str(llama_cpp_dir), capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ GGUF conversion completed successfully!")
        print(f"Output: {output_path}")

        # ファイルサイズ確認
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(".1f"
        return True
    else:
        print("❌ GGUF conversion failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="SO(8) 焼き込み済みモデル GGUF 変換")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="SO(8) 焼き込み済み HF モデルディレクトリ"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="GGUF 出力ファイルパス"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["f16", "q8_0", "q4_k_m", "q4_0", "q3_k_l", "q2_k"],
        help="量子化タイプ"
    )
    parser.add_argument(
        "--llama_cpp_dir",
        type=str,
        default=None,
        help="llama.cpp ディレクトリパス（自動検出の場合は指定不要）"
    )

    args = parser.parse_args()

    try:
        success = convert_baked_so8_to_gguf(
            args.model_path,
            args.output_path,
            args.quantization,
            args.llama_cpp_dir
        )

        return 0 if success else 1

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
