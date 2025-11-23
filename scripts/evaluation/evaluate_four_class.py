#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四値分類評価スクリプト

F1macro、誤検知率、正解率、混同行列を計算

Usage:
    python scripts/evaluate_four_class.py --model models/four_class_model --test data/splits/test.jsonl
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ラベルマッピング
LABEL_TO_ID = {"ALLOW": 0, "ESCALATION": 1, "DENY": 2, "REFUSE": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class FourClassDataset:
    """四値分類データセット"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    text = sample.get("text", "")
                    label = sample.get("label", "ALLOW")
                    if text and label in LABEL_TO_ID:
                        self.samples.append({
                            "text": text,
                            "label": LABEL_TO_ID[label]
                        })
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        label = sample["label"]
        
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
            "labels": torch.tensor(label, dtype=torch.long)
        }


def evaluate_model(model, test_dataset, device, batch_size: int = 8):
    """モデル評価"""
    logger.info("Evaluating model...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)


def calculate_metrics(predictions, labels):
    """メトリクス計算"""
    # 基本メトリクス
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_per_class = f1_score(labels, predictions, average=None)
    
    # 誤検知率（危険な要求を誤ってALLOWする率）
    false_positive = 0
    false_positive_total = 0
    for true_label, pred_label in zip(labels, predictions):
        if true_label in [2, 3]:  # DENY or REFUSE
            false_positive_total += 1
            if pred_label == 0:  # ALLOW
                false_positive += 1
    
    false_positive_rate = false_positive / false_positive_total if false_positive_total > 0 else 0.0
    
    # 混同行列
    cm = confusion_matrix(labels, predictions)
    
    # 分類レポート
    report = classification_report(
        labels,
        predictions,
        target_names=["ALLOW", "ESCALATION", "DENY", "REFUSE"],
        output_dict=True
    )
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_allow": f1_per_class[0],
        "f1_escalation": f1_per_class[1],
        "f1_deny": f1_per_class[2],
        "f1_refuse": f1_per_class[3],
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": cm,
        "classification_report": report
    }


def plot_confusion_matrix(cm, output_path: Path):
    """混同行列をプロット"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["ALLOW", "ESCALATION", "DENY", "REFUSE"],
        yticklabels=["ALLOW", "ESCALATION", "DENY", "REFUSE"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Confusion matrix saved to {output_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Four Class Classification Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model directory path"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Test dataset JSONL file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results/four_class_evaluation.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # モデル読み込み
    logger.info(f"Loading model from {args.model}...")
    from scripts.train_four_class_classifier import FourClassModel
    model = torch.load(Path(args.model) / "final_model" / "pytorch_model.bin", map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # テストデータセット
    test_dataset = FourClassDataset(Path(args.test), tokenizer)
    
    # 評価実行
    predictions, labels = evaluate_model(model, test_dataset, device, batch_size=args.batch_size)
    
    # メトリクス計算
    metrics = calculate_metrics(predictions, labels)
    
    # 結果表示
    logger.info("="*80)
    logger.info("Evaluation Results")
    logger.info("="*80)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"F1 ALLOW: {metrics['f1_allow']:.4f}")
    logger.info(f"F1 ESCALATION: {metrics['f1_escalation']:.4f}")
    logger.info(f"F1 DENY: {metrics['f1_deny']:.4f}")
    logger.info(f"F1 REFUSE: {metrics['f1_refuse']:.4f}")
    logger.info(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    logger.info("="*80)
    
    # 結果保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JSON形式で保存（混同行列はリストに変換）
    results = {
        **{k: float(v) for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report"]},
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classification_report": metrics["classification_report"]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    
    # 混同行列プロット
    plot_path = output_path.parent / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], plot_path)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

