#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
収集・加工済みデータをHugging Face形式に変換

四値分類データ（four_class_*.jsonl）をHugging Face Dataset形式に変換
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import yaml

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/convert_four_class_to_hf.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FourClassToHFDatasetConverter:
    """四値分類データをHugging Face形式に変換"""
    
    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Args:
            base_model_name: ベースモデル名（トークナイザー取得用）
        """
        self.base_model_name = base_model_name
        logger.info(f"Initializing converter with base model: {base_model_name}")
        
        # トークナイザー読み込み
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"[OK] Tokenizer loaded: {base_model_name}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load tokenizer: {e}")
            raise
    
    def load_jsonl_data(self, data_path: Path) -> List[Dict]:
        """JSONLデータを読み込み"""
        logger.info(f"Loading JSONL data from {data_path}...")
        samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_no}: JSON decode error: {e}")
                    continue
        
        logger.info(f"[OK] Loaded {len(samples):,} samples")
        return samples
    
    def convert_to_hf_format(self, samples: List[Dict], format_type: str = "instruction") -> List[Dict]:
        """
        データをHugging Face形式に変換
        
        Args:
            samples: 元のサンプルリスト
            format_type: フォーマットタイプ ("instruction", "chat", "completion")
        
        Returns:
            hf_samples: Hugging Face形式のサンプルリスト
        """
        logger.info(f"Converting {len(samples):,} samples to {format_type} format...")
        hf_samples = []
        
        for sample in samples:
            try:
                # 元のフィールド取得
                instruction = sample.get("instruction", "")
                input_text = sample.get("input", "")
                output = sample.get("output", "")
                four_class_label = sample.get("four_class_label", "ALLOW")
                
                # フォーマット別に変換
                if format_type == "instruction":
                    # Instruction形式: instruction + input -> output
                    if instruction and input_text:
                        prompt = f"{instruction}\n\n{input_text}"
                    elif instruction:
                        prompt = instruction
                    else:
                        prompt = input_text
                    
                    hf_sample = {
                        "instruction": prompt,
                        "output": output,
                        "text": f"{prompt}\n\n{output}",
                        "four_class_label": four_class_label,
                        "domain_label": sample.get("domain_label", "general"),
                        "safety_label": sample.get("safety_label", "ALLOW"),
                        "thinking_format": sample.get("thinking_format", "quadruple"),
                        "source_sample": sample.get("source_sample", "")
                    }
                
                elif format_type == "chat":
                    # Chat形式: messages形式
                    messages = []
                    if instruction:
                        messages.append({"role": "system", "content": instruction})
                    if input_text:
                        messages.append({"role": "user", "content": input_text})
                    if output:
                        messages.append({"role": "assistant", "content": output})
                    
                    hf_sample = {
                        "messages": messages,
                        "text": self._messages_to_text(messages),
                        "four_class_label": four_class_label,
                        "domain_label": sample.get("domain_label", "general"),
                        "safety_label": sample.get("safety_label", "ALLOW"),
                        "thinking_format": sample.get("thinking_format", "quadruple"),
                        "source_sample": sample.get("source_sample", "")
                    }
                
                elif format_type == "completion":
                    # Completion形式: 単純なテキスト
                    if instruction and input_text:
                        text = f"{instruction}\n\n{input_text}\n\n{output}"
                    elif instruction:
                        text = f"{instruction}\n\n{output}"
                    else:
                        text = f"{input_text}\n\n{output}"
                    
                    hf_sample = {
                        "text": text,
                        "four_class_label": four_class_label,
                        "domain_label": sample.get("domain_label", "general"),
                        "safety_label": sample.get("safety_label", "ALLOW"),
                        "thinking_format": sample.get("thinking_format", "quadruple"),
                        "source_sample": sample.get("source_sample", "")
                    }
                
                else:
                    logger.warning(f"Unknown format type: {format_type}, using instruction format")
                    continue
                
                hf_samples.append(hf_sample)
                
            except Exception as e:
                logger.warning(f"Failed to convert sample: {e}")
                continue
        
        logger.info(f"[OK] Converted {len(hf_samples):,} samples")
        return hf_samples
    
    def _messages_to_text(self, messages: List[Dict]) -> str:
        """メッセージリストをテキストに変換"""
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text_parts.append(f"System: {content}")
            elif role == "user":
                text_parts.append(f"User: {content}")
            elif role == "assistant":
                text_parts.append(f"Assistant: {content}")
        return "\n\n".join(text_parts)
    
    def tokenize_dataset(self, dataset: Dataset, max_length: int = 2048) -> Dataset:
        """データセットをトークナイズ"""
        logger.info(f"Tokenizing dataset (max_length={max_length})...")
        
        def tokenize_function(examples):
            # テキストフィールドをトークナイズ
            texts = examples["text"]
            
            # トークナイズ
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            
            # labelsを追加（言語モデリング用）
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        logger.info("[OK] Tokenization completed")
        return tokenized_dataset
    
    def split_dataset(self, samples: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict]]:
        """
        データセットを分割
        
        Args:
            samples: サンプルリスト
            train_ratio: 訓練データ比率
            val_ratio: 検証データ比率
        
        Returns:
            split_data: 分割されたデータセット
        """
        import random
        random.seed(42)
        random.shuffle(samples)
        
        total = len(samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        split_data = {
            "train": samples[:train_size],
            "val": samples[train_size:train_size + val_size],
            "test": samples[train_size + val_size:]
        }
        
        logger.info(f"[OK] Dataset split: train={len(split_data['train']):,}, val={len(split_data['val']):,}, test={len(split_data['test']):,}")
        
        return split_data
    
    def convert(
        self,
        input_path: Path,
        output_dir: Path,
        format_type: str = "instruction",
        max_length: int = 2048,
        split: bool = True,
        tokenize: bool = True
    ) -> Path:
        """
        データ変換実行
        
        Args:
            input_path: 入力JSONLファイルパス
            output_dir: 出力ディレクトリ
            format_type: フォーマットタイプ
            max_length: 最大シーケンス長
            split: データセット分割するか
            tokenize: トークナイズするか
        
        Returns:
            output_path: 出力パス
        """
        logger.info("="*80)
        logger.info("Four Class to Hugging Face Dataset Converter")
        logger.info("="*80)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Format: {format_type}")
        logger.info(f"Max length: {max_length}")
        
        # 出力ディレクトリ作成
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ読み込み
        samples = self.load_jsonl_data(input_path)
        
        # Hugging Face形式に変換
        hf_samples = self.convert_to_hf_format(samples, format_type=format_type)
        
        # データセット分割
        if split:
            split_data = self.split_dataset(hf_samples)
            
            # DatasetDict作成
            dataset_dict = DatasetDict({
                "train": Dataset.from_list(split_data["train"]),
                "val": Dataset.from_list(split_data["val"]),
                "test": Dataset.from_list(split_data["test"])
            })
        else:
            # 単一データセット
            dataset_dict = DatasetDict({
                "train": Dataset.from_list(hf_samples)
            })
        
        # トークナイズ
        if tokenize:
            dataset_dict = DatasetDict({
                split_name: self.tokenize_dataset(dataset, max_length=max_length)
                for split_name, dataset in dataset_dict.items()
            })
        
        # 保存
        output_path = output_dir / f"hf_dataset_{format_type}"
        dataset_dict.save_to_disk(str(output_path))
        
        logger.info(f"[OK] Dataset saved to {output_path}")
        
        # メタデータ保存
        metadata = {
            "base_model": self.base_model_name,
            "format_type": format_type,
            "max_length": max_length,
            "num_samples": {
                split_name: len(dataset)
                for split_name, dataset in dataset_dict.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Metadata saved to {metadata_path}")
        
        return output_path


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Convert Four Class Data to Hugging Face Dataset")
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help='Base model name for tokenizer'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=["instruction", "chat", "completion"],
        default="instruction",
        help='Output format type'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Do not split dataset'
    )
    parser.add_argument(
        '--no-tokenize',
        action='store_true',
        help='Do not tokenize dataset'
    )
    
    args = parser.parse_args()
    
    # コンバーター初期化
    converter = FourClassToHFDatasetConverter(base_model_name=args.base_model)
    
    # 変換実行
    output_path = converter.convert(
        input_path=args.input,
        output_dir=args.output,
        format_type=args.format,
        max_length=args.max_length,
        split=not args.no_split,
        tokenize=not args.no_tokenize
    )
    
    logger.info("="*80)
    logger.info("[COMPLETE] Conversion completed!")
    logger.info(f"Output: {output_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

