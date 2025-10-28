#!/usr/bin/env python3
"""
デバッグ版軽量GGUF作成スクリプト
"""

import torch
import numpy as np
import struct
import json
from pathlib import Path

def main():
    pytorch_file = "./qwen3_so8t_transformer_standard.gguf"
    output_file = "./qwen3_so8t_transformer_lightweight.gguf"
    max_tensors = 30
    
    print(f"開始: {pytorch_file}")
    
    try:
        print("PyTorchファイル読み込み中...")
        data = torch.load(pytorch_file, map_location='cpu', weights_only=False)
        print("読み込み完了")
        
        print(f"データキー: {list(data.keys())}")
        
        # メタデータ取得
        metadata = data['metadata']
        tensors = data['tensors']
        
        print(f"テンソル数: {len(tensors)}")
        print(f"メタデータ: {metadata['model_type']}")
        
        # 重要なテンソルのみ抽出
        important_keys = []
        
        # 埋め込み層
        for key in tensors.keys():
            if 'embed_tokens' in key:
                important_keys.append(key)
                print(f"埋め込み層: {key}")
        
        # 最初の数層のアテンション層
        for i in range(min(5, metadata['num_hidden_layers'])):
            for key in tensors.keys():
                if f'layers.{i}.' in key and ('attn' in key or 'mlp' in key):
                    important_keys.append(key)
                    print(f"レイヤー{i}: {key}")
        
        # 出力層
        for key in tensors.keys():
            if any(head in key for head in ['lm_head', 'task_head', 'safety_head', 'authority_head']):
                important_keys.append(key)
                print(f"出力層: {key}")
        
        # レイヤーノーム
        for key in tensors.keys():
            if 'norm' in key:
                important_keys.append(key)
                print(f"ノーム: {key}")
        
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
        
        print(f"軽量版テンソル数: {len(lightweight_tensors)}")
        
        # 軽量版メタデータ作成
        lightweight_metadata = metadata.copy()
        lightweight_metadata['model_type'] = 'qwen3_so8t_transformer_lightweight'
        lightweight_metadata['description'] = 'Lightweight SO8T Transformer for testing'
        
        print("軽量版GGUFファイル作成中...")
        
        # 簡易版GGUF形式で保存
        with open(output_file, 'wb') as f:
            # 簡易ヘッダー
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
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
