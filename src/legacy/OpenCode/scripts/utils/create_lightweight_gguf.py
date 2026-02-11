#!/usr/bin/env python3
"""
軽量版GGUFファイル作成スクリプト
メモリ効率化のため重要なテンソルのみを抽出
"""

import torch
import numpy as np
import struct
import json
from pathlib import Path

def create_lightweight_gguf(pytorch_file: str, output_file: str, max_tensors: int = 50):
    """軽量版GGUFファイル作成"""
    
    print(f"PyTorchファイル読み込み: {pytorch_file}")
    data = torch.load(pytorch_file, map_location='cpu', weights_only=True)
    
    # メタデータ取得
    metadata = data['metadata']
    tensors = data['tensors']
    quant_info = data['quantization_info']
    
    print(f"元テンソル数: {len(tensors)}")
    
    # 重要なテンソルのみ抽出
    important_keys = []
    
    # 埋め込み層
    for key in tensors.keys():
        if 'embed_tokens' in key:
            important_keys.append(key)
    
    # 最初の数層のアテンション層
    for i in range(min(5, metadata['num_hidden_layers'])):
        for key in tensors.keys():
            if f'layers.{i}.' in key and ('attn' in key or 'mlp' in key):
                important_keys.append(key)
    
    # 出力層
    for key in tensors.keys():
        if any(head in key for head in ['lm_head', 'task_head', 'safety_head', 'authority_head']):
            important_keys.append(key)
    
    # レイヤーノーム
    for key in tensors.keys():
        if 'norm' in key:
            important_keys.append(key)
    
    # 重複除去
    important_keys = list(set(important_keys))
    
    # 最大テンソル数制限
    if len(important_keys) > max_tensors:
        important_keys = important_keys[:max_tensors]
    
    print(f"抽出テンソル数: {len(important_keys)}")
    
    # 軽量版テンソル作成
    lightweight_tensors = {}
    for key in important_keys:
        if key in tensors:
            lightweight_tensors[key] = tensors[key]
    
    # 軽量版メタデータ作成
    lightweight_metadata = metadata.copy()
    lightweight_metadata['model_type'] = 'qwen3_so8t_transformer_lightweight'
    lightweight_metadata['description'] = 'Lightweight SO8T Transformer for testing'
    
    # 標準GGUF形式で保存
    with open(output_file, 'wb') as f:
        # GGUFヘッダー（簡易版）
        f.write(b'GGUF')
        f.write(struct.pack('<I', 1))  # バージョン
        f.write(struct.pack('<Q', len(lightweight_metadata)))  # メタデータ数
        
        # メタデータ書き込み
        for key, value in lightweight_metadata.items():
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
        f.write(struct.pack('<Q', len(lightweight_tensors)))
        
        # テンソルデータ書き込み
        for key, tensor_info in lightweight_tensors.items():
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
    
    print(f"軽量版GGUFファイル作成完了: {output_file}")
    
    # ファイルサイズ確認
    file_size_mb = Path(output_file).stat().st_size / (1024**2)
    print(f"ファイルサイズ: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    import sys
    pytorch_file = sys.argv[1] if len(sys.argv) > 1 else "qwen3_so8t_transformer_8bit.gguf"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "qwen3_so8t_transformer_lightweight.gguf"
    max_tensors = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    create_lightweight_gguf(pytorch_file, output_file, max_tensors)
