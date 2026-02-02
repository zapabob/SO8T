#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数データセット統合ローダー

複数のJSONLファイルを統合して読み込む

Usage:
    from scripts.training.multi_dataset_loader import MultiDatasetLoader
    
    loader = MultiDatasetLoader(data_paths, tokenizer, max_length=2048)
    dataset = loader.get_dataset()
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset, ConcatDataset

logger = logging.getLogger(__name__)


class SO8TDataset(Dataset):
    """SO8T学習用データセット"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 2048):
        """
        Args:
            data_path: JSONLファイルパス
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading dataset from {data_path}...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    
                    # テキスト構築
                    instruction = sample.get("instruction", "")
                    input_text = sample.get("input", "")
                    output = sample.get("output", "")
                    
                    if instruction and input_text:
                        text = f"{instruction}\n\n{input_text}\n\n{output}"
                    elif instruction:
                        text = f"{instruction}\n\n{output}"
                    else:
                        text = f"{input_text}\n\n{output}" if input_text else output
                    
                    if text.strip():
                        self.samples.append(text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_no}: JSON decode error: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.samples):,} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }


class MultiDatasetLoader:
    """複数データセット統合ローダー"""
    
    def __init__(self, data_paths: List[Path], tokenizer, max_length: int = 2048):
        """
        Args:
            data_paths: データセットパスのリスト
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasets = []
        
        for data_path in data_paths:
            if data_path.exists():
                dataset = SO8TDataset(data_path, tokenizer, max_length)
                if len(dataset) > 0:
                    self.datasets.append(dataset)
                    logger.info(f"[DATASET] Loaded {len(dataset):,} samples from {data_path}")
            else:
                logger.warning(f"[DATASET] File not found: {data_path}")
        
        if len(self.datasets) == 0:
            raise ValueError("No valid datasets found")
        
        # データセット統合
        self.combined_dataset = ConcatDataset(self.datasets)
        logger.info(f"[DATASET] Total samples: {len(self.combined_dataset):,}")
    
    def get_dataset(self) -> Dataset:
        """統合データセット取得"""
        return self.combined_dataset








