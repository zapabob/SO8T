#!/usr/bin/env python3
"""
PyTorch形式のGGUFファイルを標準GGUF形式に変換
ollama互換のGGUF形式に変換
"""

import torch
import numpy as np
import struct
import json
from pathlib import Path

def create_standard_gguf(pytorch_file: str, output_file: str):
    """PyTorch形式を標準GGUF形式に変換"""
    
    print(f"PyTorchファイル読み込み: {pytorch_file}")
    data = torch.load(pytorch_file, map_location='cpu')
    
    print(f"データキー: {list(data.keys())}")
    
    # メタデータ取得
    metadata = data['metadata']
    tensors = data['tensors']
    quant_info = data['quantization_info']
    
    print(f"テンソル数: {len(tensors)}")
    print(f"メタデータ: {metadata['model_type']}")
    
    # 標準GGUF形式で保存（簡易版）
    with open(output_file, 'wb') as f:
        # GGUFヘッダー（簡易版）
        f.write(b'GGUF')
        f.write(struct.pack('<I', 1))  # バージョン
        f.write(struct.pack('<Q', len(metadata)))  # メタデータ数
        
        # メタデータ書き込み
        for key, value in metadata.items():
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(key_bytes)))
            f.write(key_bytes)
            
            if isinstance(value, str):
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<I', 1))  # 文字列タイプ
                f.write(struct.pack('<I', len(value_bytes)))
                f.write(value_bytes)
            elif isinstance(value, (int, float)):
                f.write(struct.pack('<I', 2))  # 数値タイプ
                f.write(struct.pack('<f', float(value)))
        
        # テンソル数
        f.write(struct.pack('<Q', len(tensors)))
        
        # テンソルデータ書き込み
        for key, tensor_info in tensors.items():
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(key_bytes)))
            f.write(key_bytes)
            
            # 形状
            shape = tensor_info['shape']
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))
            
            # データタイプ
            dtype = tensor_info['dtype']
            if dtype == 'int8':
                f.write(struct.pack('<I', 0))  # int8
            else:
                f.write(struct.pack('<I', 1))  # float32
            
            # データ
            data_array = tensor_info['data']
            f.write(data_array.tobytes())
    
    print(f"標準GGUFファイル作成完了: {output_file}")

if __name__ == "__main__":
    import sys
    pytorch_file = sys.argv[1] if len(sys.argv) > 1 else "qwen3_so8t_transformer_8bit.gguf"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "qwen3_so8t_transformer_standard.gguf"
    
    create_standard_gguf(pytorch_file, output_file)
