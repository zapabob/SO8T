#!/usr/bin/env python3
"""
SO8Tモデルをllama.cppのconvert_hf_to_gguf.pyを使用してGGUF形式に変換

llama.cppの公式変換スクリプトをベースにSO8Tモデル専用の変換を実装
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import argparse
from tqdm import tqdm
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# llama.cppのGGUFライブラリをパスに追加
llama_cpp_path = project_root / "external" / "llama.cpp-master"
sys.path.insert(0, str(llama_cpp_path / "gguf-py"))

try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("[ERROR] GGUFライブラリが見つかりません。llama.cppのgguf-pyディレクトリを確認してください。")
    sys.exit(1)

# Transformers関連のインポート
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)


class SO8TToGGUFConverter:
    """SO8TモデルをGGUF形式に変換するクラス（llama.cppベース）"""
    
    def __init__(
        self,
        input_model_path: str,
        output_gguf_path: str,
        quantization_type: str = "Q8_0",
        model_name: str = "so8t-vl-2b-instruct"
    ):
        """
        Args:
            input_model_path: 入力SO8Tモデルパス
            output_gguf_path: 出力GGUFファイルパス
            quantization_type: 量子化タイプ
            model_name: モデル名
        """
        self.input_model_path = Path(input_model_path)
        self.output_gguf_path = Path(output_gguf_path)
        self.quantization_type = quantization_type
        self.model_name = model_name
        
        # 量子化タイプのマッピング
        self.quantization_mapping = {
            "F16": GGMLQuantizationType.F16,
            "Q8_0": GGMLQuantizationType.Q8_0,
            "Q4_0": GGMLQuantizationType.Q4_0,
            "Q4_1": GGMLQuantizationType.Q4_1,
            "Q5_0": GGMLQuantizationType.Q5_0,
            "Q5_1": GGMLQuantizationType.Q5_1,
            "Q6_K": GGMLQuantizationType.Q6_K,
            "Q8_K": GGMLQuantizationType.Q8_K,
        }
        
        if quantization_type not in self.quantization_mapping:
            raise ValueError(f"サポートされていない量子化タイプ: {quantization_type}")
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用デバイス: {self.device}")
    
    def load_so8t_model(self) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], AutoTokenizer]:
        """SO8Tモデルを読み込み"""
        logger.info(f"SO8Tモデルを読み込み中: {self.input_model_path}")
        
        try:
            # 設定ファイルの読み込み
            config_path = self.input_model_path / "config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 重みファイルの読み込み
            weights = {}
            safetensors_files = list(self.input_model_path.glob("*.safetensors"))
            if safetensors_files:
                from safetensors.torch import load_file
                for safetensors_file in safetensors_files:
                    file_weights = load_file(safetensors_file)
                    weights.update(file_weights)
                    logger.info(f"読み込み完了: {safetensors_file.name}")
            
            # PyTorch形式の重みファイル
            pytorch_file = self.input_model_path / "pytorch_model.bin"
            if pytorch_file.exists():
                pytorch_weights = torch.load(pytorch_file, map_location=self.device)
                weights.update(pytorch_weights)
                logger.info("PyTorch重みファイル読み込み完了")
            
            # トークナイザーの読み込み
            tokenizer = AutoTokenizer.from_pretrained(self.input_model_path)
            
            logger.info(f"総重み数: {len(weights):,}")
            logger.info("SO8Tモデルの読み込み完了")
            return config, weights, tokenizer
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def create_gguf_writer(self) -> GGUFWriter:
        """GGUFライターの作成"""
        logger.info("GGUFライターを作成中...")
        
        # GGUFライターの作成
        gguf_writer = GGUFWriter(self.output_gguf_path, self.model_name)
        
        # 基本メタデータの設定
        gguf_writer.add_type(gguf.GGUFType.MODEL)
        gguf_writer.add_name(self.model_name)
        gguf_writer.add_description("SO8T Safe Agent Model with SO(8) Group Structure")
        gguf_writer.add_author("SO8T Team")
        gguf_writer.add_license("MIT")
        gguf_writer.add_url("https://github.com/so8t/so8t")
        gguf_writer.add_file_type(self.quantization_mapping[self.quantization_type])
        
        logger.info("GGUFライター作成完了")
        return gguf_writer
    
    def add_so8t_metadata(self, gguf_writer: GGUFWriter, config: Dict[str, Any]) -> None:
        """SO8T特有のメタデータを追加"""
        logger.info("SO8Tメタデータを追加中...")
        
        # 基本構造パラメータ
        gguf_writer.add_context_length(config.get("max_position_embeddings", 32768))
        gguf_writer.add_embedding_length(config.get("hidden_size", 1536))
        gguf_writer.add_block_count(config.get("num_hidden_layers", 28))
        gguf_writer.add_feed_forward_length(config.get("intermediate_size", 8960))
        gguf_writer.add_head_count(config.get("num_attention_heads", 12))
        gguf_writer.add_head_count_kv(config.get("num_key_value_heads", 2))
        gguf_writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-6))
        gguf_writer.add_vocab_size(config.get("vocab_size", 151936))
        
        # RoPE設定
        rope_theta = config.get("rope_theta", 1000000.0)
        gguf_writer.add_rope_freq_base(rope_theta)
        
        # RoPE次元数（SO8Tでは8次元）
        rope_dim = config.get("rotation_dim", 8)
        gguf_writer.add_rope_dimension_count(rope_dim)
        
        # SO8T特有の設定
        gguf_writer.add_string("so8t.rotation_dim", str(config.get("rotation_dim", 8)))
        gguf_writer.add_bool("so8t.safety_features", config.get("safety_features", True))
        gguf_writer.add_float32("so8t.pet_lambda", config.get("pet_lambda", 0.1))
        gguf_writer.add_float32("so8t.safety_threshold", config.get("safety_threshold", 0.8))
        gguf_writer.add_string("so8t.group_structure", config.get("group_structure", "SO(8)"))
        
        # ビジョン設定
        if "vision_config" in config:
            vision_config = config["vision_config"]
            gguf_writer.add_string("so8t.vision_config", json.dumps(vision_config))
            gguf_writer.add_int32("so8t.vision_start_token_id", config.get("vision_start_token_id", 151652))
            gguf_writer.add_int32("so8t.vision_end_token_id", config.get("vision_end_token_id", 151653))
            gguf_writer.add_int32("so8t.vision_token_id", config.get("vision_token_id", 151654))
            gguf_writer.add_int32("so8t.image_token_id", config.get("image_token_id", 151655))
            gguf_writer.add_int32("so8t.video_token_id", config.get("video_token_id", 151656))
        
        # トークン設定
        gguf_writer.add_int32("so8t.bos_token_id", config.get("bos_token_id", 151643))
        gguf_writer.add_int32("so8t.eos_token_id", config.get("eos_token_id", 151645))
        gguf_writer.add_int32("so8t.pad_token_id", config.get("pad_token_id", 151643))
        
        # ロープスケーリング
        if "rope_scaling" in config:
            rope_scaling = config["rope_scaling"]
            gguf_writer.add_string("so8t.rope_scaling", json.dumps(rope_scaling))
        
        logger.info("SO8Tメタデータ追加完了")
    
    def add_vocabulary(self, gguf_writer: GGUFWriter, tokenizer: AutoTokenizer) -> None:
        """語彙を追加"""
        logger.info("語彙を追加中...")
        
        try:
            # トークナイザーモデルの設定
            gguf_writer.add_tokenizer_model("gpt2")
            
            # 語彙の取得
            vocab = tokenizer.get_vocab()
            tokens = list(vocab.keys())
            token_ids = list(vocab.values())
            
            # トークンリストの追加
            gguf_writer.add_token_list(tokens)
            
            # 特殊トークンの設定
            if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
                gguf_writer.add_bos_token_id(tokenizer.bos_token_id)
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
                gguf_writer.add_eos_token_id(tokenizer.eos_token_id)
            if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
                gguf_writer.add_pad_token_id(tokenizer.pad_token_id)
            if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
                gguf_writer.add_unk_token_id(tokenizer.unk_token_id)
            
            # 語彙サイズの設定
            gguf_writer.add_vocab_size(len(tokens))
            
            logger.info(f"語彙追加完了: {len(tokens)} トークン")
            
        except Exception as e:
            logger.warning(f"語彙追加エラー: {e}")
    
    def convert_tensors(self, gguf_writer: GGUFWriter, weights: Dict[str, torch.Tensor]) -> None:
        """テンソルをGGUF形式に変換"""
        logger.info("テンソルをGGUF形式に変換中...")
        
        for name, tensor in tqdm(weights.items(), desc="テンソル変換"):
            try:
                # テンソルをCPUに移動
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                
                # 数値型の変換
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                elif tensor.dtype == torch.float16:
                    tensor = tensor.float()
                
                # テンソル名の正規化
                gguf_name = name.replace('.', '_')
                
                # 量子化タイプに応じた変換
                if self.quantization_type == "F16":
                    tensor = tensor.half()
                
                # GGUFに追加
                gguf_writer.add_tensor(
                    name=gguf_name,
                    tensor=tensor.numpy(),
                    data_type=self.quantization_mapping[self.quantization_type]
                )
                
            except Exception as e:
                logger.warning(f"テンソル変換エラー ({name}): {e}")
                continue
        
        logger.info("テンソル変換完了")
    
    def convert(self) -> None:
        """変換の実行"""
        logger.info("=" * 80)
        logger.info("SO8T → GGUF変換開始（llama.cppベース）")
        logger.info("=" * 80)
        
        try:
            # 1. SO8Tモデルの読み込み
            config, weights, tokenizer = self.load_so8t_model()
            
            # 2. GGUFライターの作成
            gguf_writer = self.create_gguf_writer()
            
            # 3. SO8Tメタデータの追加
            self.add_so8t_metadata(gguf_writer, config)
            
            # 4. 語彙の追加
            self.add_vocabulary(gguf_writer, tokenizer)
            
            # 5. テンソルの変換
            self.convert_tensors(gguf_writer, weights)
            
            # 6. GGUFファイルの書き込み
            logger.info("GGUFファイルを書き込み中...")
            gguf_writer.write_header_to_file()
            gguf_writer.write_kv_data_to_file()
            gguf_writer.write_tensors_to_file()
            gguf_writer.close()
            
            logger.info("=" * 80)
            logger.info("GGUF変換完了")
            logger.info("=" * 80)
            logger.info(f"出力ファイル: {self.output_gguf_path}")
            logger.info(f"量子化タイプ: {self.quantization_type}")
            logger.info(f"総パラメータ数: {len(weights):,}")
            
            # ファイルサイズの表示
            file_size = self.output_gguf_path.stat().st_size
            logger.info(f"ファイルサイズ: {file_size / (1024**3):.2f} GB")
            
        except Exception as e:
            logger.error(f"変換中にエラーが発生しました: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='SO8TモデルをGGUF形式に変換（llama.cppベース）')
    parser.add_argument('--input-model', required=True, help='入力SO8Tモデルパス')
    parser.add_argument('--output-gguf', required=True, help='出力GGUFファイルパス')
    parser.add_argument('--quantization', default='Q8_0', 
                       choices=['F16', 'Q8_0', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q6_K', 'Q8_K'],
                       help='量子化タイプ')
    parser.add_argument('--model-name', default='so8t-vl-2b-instruct', help='モデル名')
    parser.add_argument('--verbose', action='store_true', help='詳細ログ出力')
    
    args = parser.parse_args()
    
    # ログ設定
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 変換器の作成と実行
    converter = SO8TToGGUFConverter(
        input_model_path=args.input_model,
        output_gguf_path=args.output_gguf,
        quantization_type=args.quantization,
        model_name=args.model_name
    )
    
    try:
        converter.convert()
        print("\n[SUCCESS] GGUF変換が正常に完了しました")
        return 0
    except Exception as e:
        print(f"\n[ERROR] 変換中にエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
