#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-3.5 to GGUF Converter
Phi-3.5モデルをGGUF形式に変換するスクリプト
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def convert_phi35_to_gguf(model_path: str, output_path: str, quantization: str = "bf16"):
    """Phi-3.5モデルをGGUFに変換"""

    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # llama.cppのconvert_hf_to_gguf.pyを使用
    llama_cpp_path = Path(__file__).parent.parent.parent / "external" / "llama.cpp-master"

    if not llama_cpp_path.exists():
        raise FileNotFoundError(f"llama.cpp not found at: {llama_cpp_path}")

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")

    # 量子化タイプ設定
    quant_types = {
        "bf16": "bf16",
        "f16": "f16",
        "q8_0": "q8_0",
        "q4_k_m": "q4_k_m"
    }

    if quantization not in quant_types:
        raise ValueError(f"Unsupported quantization: {quantization}")

    outtype = quant_types[quantization]

    # 変換コマンド実行
    cmd = [
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", outtype
    ]

    print(f"Converting {model_path} to {output_path} with quantization {quantization}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=llama_cpp_path)

    if result.returncode == 0:
        print(f"[SUCCESS] Conversion completed successfully: {output_path}")
        return True
    else:
        print(f"[ERROR] Conversion failed:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert Phi-3.5 model to GGUF format')
    parser.add_argument('--model_path', required=True, help='Path to Phi-3.5 model')
    parser.add_argument('--output_path', required=True, help='Output GGUF file path')
    parser.add_argument('--quantization', default='bf16', choices=['bf16', 'f16', 'q8_0', 'q4_k_m'],
                       help='Quantization type (default: bf16)')

    args = parser.parse_args()

    try:
        success = convert_phi35_to_gguf(
            args.model_path,
            args.output_path,
            args.quantization
        )

        if success:
            print(f"🎉 GGUF conversion completed: {args.output_path}")
        else:
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
