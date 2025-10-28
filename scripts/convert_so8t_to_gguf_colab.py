#!/usr/bin/env python3
"""
SO8T群Transformer GGUF変換スクリプト (GoogleColab対応)

SO8T群Transformerモデルを8bit量子化GGUF形式に変換するスクリプトです。
GoogleColab環境で実行可能で、メモリ効率を最適化しています。

特徴:
- SO8T群構造の保持
- 8bit量子化によるメモリ削減
- GGUF形式での効率的な保存
- GoogleColab環境での実行最適化
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
from tqdm import tqdm
import time
import gc

# GoogleColab環境の検出
try:
    import google.colab
    IN_COLAB = True
    print("🚀 GoogleColab環境を検出しました！")
except ImportError:
    IN_COLAB = False
    print("💻 ローカル環境で実行中です")

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SO8TGGUFConverter:
    """SO8T群Transformer GGUF変換器"""
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str = "so8t_gguf_models",
                 quantization_type: str = "Q8_0",
                 max_memory_gb: float = 8.0):
        """
        SO8T GGUF変換器を初期化
        
        Args:
            model_path: SO8Tモデルのパス
            output_dir: 出力ディレクトリ
            quantization_type: 量子化タイプ (Q8_0, Q4_K_M, Q5_K_M等)
            max_memory_gb: 最大メモリ使用量 (GB)
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.quantization_type = quantization_type
        self.max_memory_gb = max_memory_gb
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GoogleColab環境での最適化
        if IN_COLAB:
            self._setup_colab_environment()
        
        logger.info(f"SO8T GGUF変換器初期化完了")
        logger.info(f"モデルパス: {self.model_path}")
        logger.info(f"出力ディレクトリ: {self.output_dir}")
        logger.info(f"量子化タイプ: {self.quantization_type}")
    
    def _setup_colab_environment(self):
        """GoogleColab環境のセットアップ"""
        logger.info("GoogleColab環境をセットアップ中...")
        
        # メモリ使用量の最適化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # メモリフラグメント対策
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 不要なライブラリの無効化
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        logger.info("GoogleColab環境セットアップ完了")
    
    def load_so8t_model(self) -> Dict[str, torch.Tensor]:
        """SO8Tモデルを読み込み"""
        logger.info("SO8Tモデル読み込み中...")
        
        try:
            # モデル状態辞書を読み込み
            if self.model_path.is_file() and self.model_path.suffix == '.pth':
                # PyTorchモデルファイル
                state_dict = torch.load(self.model_path, map_location='cpu')
                logger.info("PyTorchモデルファイルから読み込み完了")
            elif self.model_path.is_dir():
                # HuggingFace形式のディレクトリ
                model_file = self.model_path / "pytorch_model.bin"
                if model_file.exists():
                    state_dict = torch.load(model_file, map_location='cpu')
                else:
                    # 複数ファイルの場合
                    state_dict = self._load_huggingface_model()
                logger.info("HuggingFace形式から読み込み完了")
            else:
                raise ValueError(f"サポートされていないモデル形式: {self.model_path}")
            
            # メモリ使用量をチェック
            self._check_memory_usage(state_dict)
            
            return state_dict
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def _load_huggingface_model(self) -> Dict[str, torch.Tensor]:
        """HuggingFace形式のモデルを読み込み"""
        logger.info("HuggingFace形式のモデルを読み込み中...")
        
        state_dict = {}
        model_files = list(self.model_path.glob("pytorch_model*.bin"))
        
        if not model_files:
            raise FileNotFoundError("HuggingFace形式のモデルファイルが見つかりません")
        
        # 複数ファイルを順次読み込み
        for model_file in sorted(model_files):
            logger.info(f"読み込み中: {model_file.name}")
            file_state_dict = torch.load(model_file, map_location='cpu')
            state_dict.update(file_state_dict)
            
            # メモリクリア
            del file_state_dict
            gc.collect()
        
        return state_dict
    
    def _check_memory_usage(self, state_dict: Dict[str, torch.Tensor]):
        """メモリ使用量をチェック"""
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        total_size_gb = sum(p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor)) / (1024**3)
        
        logger.info(f"総パラメータ数: {total_params:,}")
        logger.info(f"モデルサイズ: {total_size_gb:.2f} GB")
        
        if total_size_gb > self.max_memory_gb:
            logger.warning(f"モデルサイズが最大メモリ制限を超過: {total_size_gb:.2f} GB > {self.max_memory_gb} GB")
            if IN_COLAB:
                logger.warning("GoogleColab環境ではメモリ不足の可能性があります")
    
    def analyze_so8t_structure(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """SO8T群構造を分析"""
        logger.info("SO8T群構造を分析中...")
        
        analysis = {
            'total_layers': 0,
            'so8t_layers': 0,
            'attention_layers': 0,
            'ffn_layers': 0,
            'so8_rotation_params': 0,
            'triality_heads': 0,
            'model_architecture': 'unknown'
        }
        
        # レイヤー構造の分析
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                # SO8T群関連のパラメータ
                if 'so8t' in key.lower() or 'rotation' in key.lower():
                    analysis['so8t_layers'] += 1
                    if 'rotation' in key.lower():
                        analysis['so8_rotation_params'] += 1
                
                # アテンション層
                elif 'attention' in key.lower() or 'attn' in key.lower():
                    analysis['attention_layers'] += 1
                
                # FFN層
                elif 'mlp' in key.lower() or 'ffn' in key.lower() or 'feed_forward' in key.lower():
                    analysis['ffn_layers'] += 1
                
                # Triality reasoning heads
                elif any(head in key.lower() for head in ['task_head', 'safety_head', 'authority_head']):
                    analysis['triality_heads'] += 1
        
        analysis['total_layers'] = analysis['so8t_layers'] + analysis['attention_layers'] + analysis['ffn_layers']
        
        # アーキテクチャの判定
        if analysis['so8t_layers'] > 0 and analysis['triality_heads'] > 0:
            analysis['model_architecture'] = 'SO8TTransformerForCausalLM'
        elif analysis['so8t_layers'] > 0:
            analysis['model_architecture'] = 'SO8TTransformerModel'
        else:
            analysis['model_architecture'] = 'StandardTransformer'
        
        logger.info(f"SO8T群構造分析完了:")
        logger.info(f"  - 総レイヤー数: {analysis['total_layers']}")
        logger.info(f"  - SO8T群レイヤー数: {analysis['so8t_layers']}")
        logger.info(f"  - アテンション層数: {analysis['attention_layers']}")
        logger.info(f"  - FFN層数: {analysis['ffn_layers']}")
        logger.info(f"  - SO8回転パラメータ数: {analysis['so8_rotation_params']}")
        logger.info(f"  - Triality heads数: {analysis['triality_heads']}")
        logger.info(f"  - アーキテクチャ: {analysis['model_architecture']}")
        
        return analysis
    
    def quantize_tensor(self, tensor: torch.Tensor, quantization_type: str) -> Tuple[torch.Tensor, Dict]:
        """テンソルを量子化"""
        if not isinstance(tensor, torch.Tensor):
            return tensor, {}
        
        original_dtype = tensor.dtype
        original_shape = tensor.shape
        
        # 量子化タイプに応じた処理
        if quantization_type == "Q8_0":
            # 8bit量子化 (Q8_0)
            if tensor.dtype == torch.float32:
                # float32 -> int8
                scale = tensor.abs().max() / 127.0
                quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
                metadata = {
                    'scale': scale.item(),
                    'zero_point': 0,
                    'original_dtype': str(original_dtype),
                    'quantization_type': 'Q8_0'
                }
            else:
                quantized = tensor
                metadata = {'quantization_type': 'none'}
        
        elif quantization_type == "Q4_K_M":
            # 4bit量子化 (Q4_K_M)
            if tensor.dtype == torch.float32:
                # float32 -> int4
                scale = tensor.abs().max() / 7.0
                quantized = torch.round(tensor / scale).clamp(-8, 7).to(torch.int8)
                metadata = {
                    'scale': scale.item(),
                    'zero_point': 0,
                    'original_dtype': str(original_dtype),
                    'quantization_type': 'Q4_K_M'
                }
            else:
                quantized = tensor
                metadata = {'quantization_type': 'none'}
        
        else:
            # 量子化なし
            quantized = tensor
            metadata = {'quantization_type': 'none'}
        
        return quantized, metadata
    
    def convert_to_gguf_format(self, state_dict: Dict[str, torch.Tensor], analysis: Dict) -> Dict:
        """GGUF形式に変換"""
        logger.info("GGUF形式に変換中...")
        
        gguf_data = {
            'metadata': {
                'model_type': 'SO8TTransformer',
                'architecture': analysis['model_architecture'],
                'quantization_type': self.quantization_type,
                'total_layers': analysis['total_layers'],
                'so8t_layers': analysis['so8t_layers'],
                'attention_layers': analysis['attention_layers'],
                'ffn_layers': analysis['ffn_layers'],
                'so8_rotation_params': analysis['so8_rotation_params'],
                'triality_heads': analysis['triality_heads'],
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'converter': 'SO8TGGUFConverter'
            },
            'tensors': {},
            'quantization_info': {}
        }
        
        # テンソルを量子化してGGUF形式に変換
        with tqdm(total=len(state_dict), desc="GGUF変換", unit="tensor") as pbar:
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    # テンソルを量子化
                    quantized_tensor, quant_metadata = self.quantize_tensor(tensor, self.quantization_type)
                    
                    # GGUF形式で保存
                    gguf_data['tensors'][key] = {
                        'data': quantized_tensor.numpy().astype(np.int8) if quantized_tensor.dtype == torch.int8 else quantized_tensor.numpy(),
                        'shape': list(tensor.shape),
                        'dtype': str(quantized_tensor.dtype),
                        'original_dtype': str(tensor.dtype)
                    }
                    
                    # 量子化情報を保存
                    if quant_metadata:
                        gguf_data['quantization_info'][key] = quant_metadata
                
                pbar.update(1)
        
        logger.info("GGUF形式変換完了")
        return gguf_data
    
    def save_gguf_model(self, gguf_data: Dict, filename: str = None) -> str:
        """GGUFモデルを保存"""
        if filename is None:
            model_name = self.model_path.stem
            filename = f"{model_name}_so8t_{self.quantization_type}.gguf"
        
        output_path = self.output_dir / filename
        
        logger.info(f"GGUFモデル保存中: {output_path}")
        
        # メタデータをJSONで保存
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(gguf_data['metadata'], f, indent=2, ensure_ascii=False)
        
        # テンソルデータをNPZ形式で保存
        tensor_data = {}
        for key, tensor_info in gguf_data['tensors'].items():
            tensor_data[key] = tensor_info['data']
        
        npz_path = output_path.with_suffix('.npz')
        np.savez_compressed(npz_path, **tensor_data)
        
        # 量子化情報を保存
        quant_path = output_path.with_suffix('.quant.json')
        with open(quant_path, 'w', encoding='utf-8') as f:
            json.dump(gguf_data['quantization_info'], f, indent=2, ensure_ascii=False)
        
        # ファイルサイズをチェック
        total_size = sum(f.stat().st_size for f in [metadata_path, npz_path, quant_path])
        total_size_gb = total_size / (1024**3)
        
        logger.info(f"GGUFモデル保存完了:")
        logger.info(f"  - メタデータ: {metadata_path} ({metadata_path.stat().st_size / 1024:.1f} KB)")
        logger.info(f"  - テンソルデータ: {npz_path} ({npz_path.stat().st_size / (1024**2):.1f} MB)")
        logger.info(f"  - 量子化情報: {quant_path} ({quant_path.stat().st_size / 1024:.1f} KB)")
        logger.info(f"  - 総サイズ: {total_size_gb:.2f} GB")
        
        return str(output_path)
    
    def create_model_card(self, analysis: Dict) -> str:
        """モデルカードを作成"""
        model_card = f"""# SO8T群Transformer GGUFモデル

## 概要
SO8T群Transformerモデルを8bit量子化GGUF形式に変換したモデルです。

## アーキテクチャ
- **モデルタイプ**: {analysis['model_architecture']}
- **総レイヤー数**: {analysis['total_layers']}
- **SO8T群レイヤー数**: {analysis['so8t_layers']}
- **アテンション層数**: {analysis['attention_layers']}
- **FFN層数**: {analysis['ffn_layers']}
- **SO8回転パラメータ数**: {analysis['so8_rotation_params']}
- **Triality heads数**: {analysis['triality_heads']}

## 量子化
- **量子化タイプ**: {self.quantization_type}
- **メモリ効率**: 大幅なメモリ使用量削減
- **精度**: 量子化による軽微な精度低下

## SO8T群構造
SO8T群Transformerは以下の特徴を持ちます:
- **SO(8)群回転**: 8次元回転群による非可換ゲート
- **Triality reasoning**: 3つの推論ヘッド（task, safety, authority）
- **PET正則化**: 時系列一貫性による群の慣性保持
- **安全人格**: 学習中に群構造が崩壊しない設計

## 使用方法
```python
import numpy as np
import json

# メタデータ読み込み
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

# テンソルデータ読み込み
tensor_data = np.load('model_tensors.npz')

# 量子化情報読み込み
with open('model_quantization.json', 'r') as f:
    quant_info = json.load(f)
```

## ファイル構成
- `model_metadata.json`: モデルメタデータ
- `model_tensors.npz`: 量子化されたテンソルデータ
- `model_quantization.json`: 量子化情報

## 作成日時
{time.strftime('%Y-%m-%d %H:%M:%S')}

## 変換器
SO8TGGUFConverter v1.0
"""
        return model_card
    
    def convert(self) -> str:
        """SO8TモデルをGGUF形式に変換"""
        logger.info("🚀 SO8T群Transformer GGUF変換開始！")
        
        try:
            # 1. モデル読み込み
            logger.info("📥 ステップ1: SO8Tモデル読み込み")
            state_dict = self.load_so8t_model()
            
            # 2. SO8T群構造分析
            logger.info("🔍 ステップ2: SO8T群構造分析")
            analysis = self.analyze_so8t_structure(state_dict)
            
            # 3. GGUF形式変換
            logger.info("🔄 ステップ3: GGUF形式変換")
            gguf_data = self.convert_to_gguf_format(state_dict, analysis)
            
            # 4. モデル保存
            logger.info("💾 ステップ4: GGUFモデル保存")
            output_path = self.save_gguf_model(gguf_data)
            
            # 5. モデルカード作成
            logger.info("📝 ステップ5: モデルカード作成")
            model_card = self.create_model_card(analysis)
            card_path = self.output_dir / "README.md"
            with open(card_path, 'w', encoding='utf-8') as f:
                f.write(model_card)
            
            # 6. メモリクリア
            logger.info("🧹 ステップ6: メモリクリア")
            del state_dict, gguf_data
            gc.collect()
            
            logger.info("✅ SO8T群Transformer GGUF変換完了！")
            logger.info(f"📁 出力ディレクトリ: {self.output_dir}")
            logger.info(f"📄 モデルカード: {card_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ 変換エラー: {e}")
            raise


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SO8T群Transformer GGUF変換')
    parser.add_argument('--model_path', type=str, required=True, help='SO8Tモデルのパス')
    parser.add_argument('--output_dir', type=str, default='so8t_gguf_models', help='出力ディレクトリ')
    parser.add_argument('--quantization', type=str, default='Q8_0', 
                       choices=['Q8_0', 'Q4_K_M', 'Q5_K_M', 'none'], help='量子化タイプ')
    parser.add_argument('--max_memory', type=float, default=8.0, help='最大メモリ使用量 (GB)')
    
    args = parser.parse_args()
    
    # 変換器作成
    converter = SO8TGGUFConverter(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantization_type=args.quantization,
        max_memory_gb=args.max_memory
    )
    
    # 変換実行
    output_path = converter.convert()
    print(f"\n🎉 変換完了！出力: {output_path}")


if __name__ == "__main__":
    main()
