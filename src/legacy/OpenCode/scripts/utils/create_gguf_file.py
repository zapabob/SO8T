#!/usr/bin/env python3
"""
個別ファイルから統合GGUFファイルを作成するスクリプト
メモリ効率化のため個別保存されたファイルを統合
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_gguf_file(input_dir: str, output_file: str):
    """個別ファイルから統合GGUFファイルを作成"""
    
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    logger.info(f"入力ディレクトリ: {input_path}")
    logger.info(f"出力ファイル: {output_path}")
    
    # 個別ファイル読み込み
    logger.info("個別ファイル読み込み中...")
    
    # メタデータ読み込み
    metadata_path = input_path / "model_metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    logger.info(f"メタデータ読み込み完了: {metadata_path}")
    
    # テンソルデータ読み込み
    tensor_path = input_path / "model_tensors_8bit.npz"
    tensor_data = np.load(tensor_path)
    logger.info(f"テンソルデータ読み込み完了: {tensor_path}")
    
    # 量子化情報読み込み
    quant_path = input_path / "quantization_info.json"
    with open(quant_path, 'r', encoding='utf-8') as f:
        quantization_info = json.load(f)
    logger.info(f"量子化情報読み込み完了: {quant_path}")
    
    # トークナイザー情報読み込み
    tokenizer_path = input_path / "tokenizer_info.json"
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_info = json.load(f)
    logger.info(f"トークナイザー情報読み込み完了: {tokenizer_path}")
    
    # GGUFデータ構築
    logger.info("GGUFデータ構築中...")
    
    gguf_data = {
        "metadata": metadata,
        "tensors": {},
        "quantization_info": quantization_info,
        "tokenizer": tokenizer_info
    }
    
    # テンソルデータを辞書形式に変換
    for key in tensor_data.files:
        gguf_data["tensors"][key] = {
            "data": tensor_data[key],
            "shape": list(tensor_data[key].shape),
            "dtype": str(tensor_data[key].dtype)
        }
    
    logger.info(f"テンソル数: {len(gguf_data['tensors'])}")
    
    # 統合GGUFファイル保存
    logger.info("統合GGUFファイル保存中...")
    
    # 出力ディレクトリ作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # PyTorch形式で保存
    torch.save(gguf_data, output_path)
    
    # ファイルサイズ計算
    file_size_gb = output_path.stat().st_size / (1024**3)
    logger.info(f"統合GGUFファイル保存完了: {output_path}")
    logger.info(f"ファイルサイズ: {file_size_gb:.2f} GB")
    
    return output_path

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="個別ファイルから統合GGUFファイルを作成")
    parser.add_argument("--input_dir", type=str, default="models/qwen3_so8t_8bit_gguf",
                       help="個別ファイルのディレクトリ")
    parser.add_argument("--output_file", type=str, default="models/qwen3_so8t_8bit_gguf/qwen3_so8t_transformer_8bit.gguf",
                       help="出力GGUFファイル")
    
    args = parser.parse_args()
    
    # GGUFファイル作成
    gguf_path = create_gguf_file(args.input_dir, args.output_file)
    
    print(f"\n[SUCCESS] 統合GGUFファイル作成完了！")
    print(f"[OUTPUT] GGUFファイル: {gguf_path}")
    print(f"[SIZE] ファイルサイズ: {gguf_path.stat().st_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
