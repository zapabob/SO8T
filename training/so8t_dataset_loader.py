"""
SO8T Dataset Loader

This module provides data loading functionality for the SO8T Safe Agent training.
It handles loading JSONL data, tokenization, and batching for both task and safety heads.

The dataset format expects JSONL files with the following structure:
{
    "context": "Context information",
    "user_request": "User's request",
    "task_output": "Expected task response",
    "safety_label": "ALLOW|REFUSE|ESCALATE",
    "safety_rationale": "Explanation for safety decision"
}
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class SO8TDataset(Dataset):
    """
    Dataset class for SO8T Safe Agent training data.
    
    Handles loading and preprocessing of JSONL data for both task and safety heads.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        safety_label_map: Optional[Dict[str, int]] = None,
        include_rationale: bool = True
    ):
        """
        Initialize the SO8T dataset.
        
        Args:
            data_path: Path to the JSONL data file
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            safety_label_map: Mapping from safety labels to integers
            include_rationale: Whether to include rationale generation
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_rationale = include_rationale
        
        # Default safety label mapping
        if safety_label_map is None:
            self.safety_label_map = {
                "ALLOW": 0,
                "REFUSE": 1,
                "ESCALATE": 2
            }
        else:
            self.safety_label_map = safety_label_map
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8-sig') as f:  # BOM対応
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        # Validate required fields
                        required_fields = ["context", "user_request", "task_output", "safety_label"]
                        if not all(field in sample for field in required_fields):
                            logger.warning(f"Missing required fields in line {line_num}")
                            continue
                        
                        # Validate safety label
                        if sample["safety_label"] not in self.safety_label_map:
                            logger.warning(f"Invalid safety label '{sample['safety_label']}' in line {line_num}")
                            continue
                        
                        data.append(sample)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        return data
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing tokenized inputs and labels
        """
        sample = self.data[idx]
        
        # Prepare input text
        context = sample["context"]
        user_request = sample["user_request"]
        input_text = f"Context: {context}\nUser Request: {user_request}"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare task output (for TaskHeadA)
        task_output = sample["task_output"]
        task_encoding = self.tokenizer(
            task_output,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare safety rationale (for SafetyHeadB)
        rationale_encoding = None
        if self.include_rationale and "safety_rationale" in sample:
            rationale_text = sample["safety_rationale"]
            rationale_encoding = self.tokenizer(
                rationale_text,
                max_length=256,  # Shorter for rationale
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        # Get safety label
        safety_label = self.safety_label_map[sample["safety_label"]]
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "task_labels": task_encoding["input_ids"].squeeze(0),
            "safety_labels": torch.tensor(safety_label, dtype=torch.long),
            "rationale_labels": rationale_encoding["input_ids"].squeeze(0) if rationale_encoding is not None else None,
            "rationale_attention_mask": rationale_encoding["attention_mask"].squeeze(0) if rationale_encoding is not None else None,
            "original_sample": sample  # Keep original for debugging
        }


def collate_so8t_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for SO8T dataset batching with proper padding.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched tensors ready for model input
    """
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    task_labels = torch.stack([item["task_labels"] for item in batch])
    safety_labels = torch.stack([item["safety_labels"] for item in batch])
    
    # Handle rationale labels with proper padding
    rationale_labels = None
    rationale_attention_mask = None
    
    # Check if any sample has rationale_labels
    has_rationale = any(item["rationale_labels"] is not None for item in batch)
    
    if has_rationale:
        # Get max length for padding (only from samples that have rationale_labels)
        rationale_lengths = [item["rationale_labels"].size(0) for item in batch if item["rationale_labels"] is not None]
        max_rationale_len = max(rationale_lengths) if rationale_lengths else 0
        pad_token_id = 0  # Assuming 0 is pad token
        
        # Pad rationale labels and attention masks
        padded_rationale_labels = []
        padded_rationale_attention_mask = []
        
        for item in batch:
            if item["rationale_labels"] is not None:
                rationale_len = item["rationale_labels"].size(0)
                if rationale_len < max_rationale_len:
                    # Pad with -100 (ignore index for loss calculation)
                    pad_length = max_rationale_len - rationale_len
                    padded_labels = torch.cat([
                        item["rationale_labels"],
                        torch.full((pad_length,), -100, dtype=torch.long)
                    ])
                    padded_attention = torch.cat([
                        item["rationale_attention_mask"],
                        torch.zeros(pad_length, dtype=torch.long)
                    ])
                else:
                    padded_labels = item["rationale_labels"]
                    padded_attention = item["rationale_attention_mask"]
            else:
                # Create dummy rationale labels for samples without rationale
                padded_labels = torch.full((max_rationale_len,), -100, dtype=torch.long)
                padded_attention = torch.zeros(max_rationale_len, dtype=torch.long)
            
            padded_rationale_labels.append(padded_labels)
            padded_rationale_attention_mask.append(padded_attention)
        
        rationale_labels = torch.stack(padded_rationale_labels)
        rationale_attention_mask = torch.stack(padded_rationale_attention_mask)
    else:
        # Handle case where some samples have None rationale_labels
        rationale_labels = None
        rationale_attention_mask = None
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "task_labels": task_labels,
        "safety_labels": safety_labels,
        "rationale_labels": rationale_labels,
        "rationale_attention_mask": rationale_attention_mask
    }


def create_so8t_dataloader(
    data_path: Union[str, Path],
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    shuffle: bool = True,
    num_workers: int = 0,
    safety_label_map: Optional[Dict[str, int]] = None,
    include_rationale: bool = True
) -> DataLoader:
    """
    Create a DataLoader for SO8T training data.
    
    Args:
        data_path: Path to the JSONL data file
        tokenizer: Tokenizer for text processing
        batch_size: Batch size for training
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        safety_label_map: Mapping from safety labels to integers
        include_rationale: Whether to include rationale generation
        
    Returns:
        DataLoader for SO8T training
    """
    dataset = SO8TDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        safety_label_map=safety_label_map,
        include_rationale=include_rationale
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_so8t_batch,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def create_train_val_dataloaders(
    train_data_path: Union[str, Path],
    val_data_path: Union[str, Path],
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    val_split: float = 0.1,
    num_workers: int = 0,
    safety_label_map: Optional[Dict[str, int]] = None,
    include_rationale: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        tokenizer: Tokenizer for text processing
        batch_size: Batch size for training
        max_length: Maximum sequence length
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes
        safety_label_map: Mapping from safety labels to integers
        include_rationale: Whether to include rationale generation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create training dataloader
    train_dataloader = create_so8t_dataloader(
        data_path=train_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
        num_workers=num_workers,
        safety_label_map=safety_label_map,
        include_rationale=include_rationale
    )
    
    # Create validation dataloader
    val_dataloader = create_so8t_dataloader(
        data_path=val_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
        num_workers=num_workers,
        safety_label_map=safety_label_map,
        include_rationale=include_rationale
    )
    
    return train_dataloader, val_dataloader


def analyze_dataset(data_path: Union[str, Path]) -> Dict[str, Union[int, float, Dict]]:
    """
    Analyze the dataset and return statistics.
    
    Args:
        data_path: Path to the JSONL data file
        
    Returns:
        Dictionary containing dataset statistics
    """
    data_path = Path(data_path)
    
    # Load data
    data = []
    safety_label_counts = {"ALLOW": 0, "REFUSE": 0, "ESCALATE": 0}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                data.append(sample)
                
                # Count safety labels
                safety_label = sample.get("safety_label", "UNKNOWN")
                if safety_label in safety_label_counts:
                    safety_label_counts[safety_label] += 1
                    
            except json.JSONDecodeError:
                continue
    
    total_samples = len(data)
    
    # Calculate statistics
    stats = {
        "total_samples": total_samples,
        "safety_label_counts": safety_label_counts,
        "safety_label_distribution": {
            label: count / total_samples if total_samples > 0 else 0
            for label, count in safety_label_counts.items()
        },
        "has_rationale": sum(1 for sample in data if "safety_rationale" in sample),
        "rationale_coverage": sum(1 for sample in data if "safety_rationale" in sample) / total_samples if total_samples > 0 else 0
    }
    
    # Calculate text length statistics
    if data:
        context_lengths = [len(sample.get("context", "")) for sample in data]
        request_lengths = [len(sample.get("user_request", "")) for sample in data]
        output_lengths = [len(sample.get("task_output", "")) for sample in data]
        
        stats.update({
            "avg_context_length": sum(context_lengths) / len(context_lengths),
            "avg_request_length": sum(request_lengths) / len(request_lengths),
            "avg_output_length": sum(output_lengths) / len(output_lengths),
            "max_context_length": max(context_lengths),
            "max_request_length": max(request_lengths),
            "max_output_length": max(output_lengths)
        })
    
    return stats


def create_sample_data(output_path: Union[str, Path], num_samples: int = 10):
    """
    Create sample data for testing.
    
    Args:
        output_path: Path to save the sample data
        num_samples: Number of sample records to create
    """
    output_path = Path(output_path)
    
    # Sample data
    sample_data = [
        {
            "context": "オフィス環境での日常業務サポート",
            "user_request": "今日の会議スケジュールを教えて",
            "task_output": "本日の会議スケジュールをお知らせします。午前10時からプロジェクト会議、午後2時からクライアント打ち合わせが予定されています。",
            "safety_label": "ALLOW",
            "safety_rationale": "一般的な情報提供要求で、個人情報や機密情報を含まない安全なリクエストです。"
        },
        {
            "context": "人事関連の相談",
            "user_request": "同僚のパフォーマンス評価について相談したい",
            "task_output": "このような人事関連の相談は、人事部門の専門スタッフが適切に対応いたします。",
            "safety_label": "ESCALATE",
            "safety_rationale": "人事評価に関する相談は、専門知識と適切な手続きが必要なため、人事部門へのエスカレーションが必要です。"
        },
        {
            "context": "セキュリティ関連の要求",
            "user_request": "システムのパスワードを教えて",
            "task_output": "セキュリティ上の理由により、パスワードの提供はできません。",
            "safety_label": "REFUSE",
            "safety_rationale": "パスワードや認証情報の提供は、セキュリティリスクが非常に高く、明確に拒否すべき要求です。"
        }
    ]
    
    # Write sample data
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Cycle through sample data
            sample = sample_data[i % len(sample_data)].copy()
            sample["id"] = i + 1
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Created sample data with {num_samples} samples at {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create sample data
    sample_path = "data/sample_so8t_data.jsonl"
    create_sample_data(sample_path, num_samples=20)
    
    # Analyze dataset
    stats = analyze_dataset(sample_path)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create dataloader
    dataloader = create_so8t_dataloader(
        data_path=sample_path,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512
    )
    
    # Test dataloader
    for batch in dataloader:
        print("Batch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        break