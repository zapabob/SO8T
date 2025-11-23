#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/Bテスト評価スクリプト

モデルA（ベースライン）とモデルBの評価を実行し、結果を比較

Usage:
    python scripts/ab_test_borea_phi35.py --model-a Borea-Phi-3.5-mini-Instruct-Common --model-b checkpoints/borea_phi35_model_b/calibrated/final_model --test data/splits/test.jsonl
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# モデルA評価スクリプトをインポート
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_model_a_baseline import (
    FourClassDataset,
    evaluate_model_a,
    calculate_metrics,
    classify_from_generation
)

# モデルB評価スクリプト（四値分類モデル用）
from scripts.evaluate_four_class import (
    evaluate_model as evaluate_model_b,
    calculate_metrics as calculate_metrics_b
)


def evaluate_model_b_classifier(model_path: str, test_dataset, device, batch_size: int = 8):
    """モデルB（分類モデル）を評価"""
    logger.info(f"Loading Model B from {model_path}...")
    
    from transformers import AutoTokenizer
    from scripts.train_four_class_classifier import FourClassModel
    
    # モデル読み込み
    try:
        model = FourClassModel.load_from_checkpoint(
            str(Path(model_path) / "pytorch_model.bin"),
            map_location=device
        )
    except:
        # フォールバック: 通常のモデル読み込み
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 評価実行
    predictions, labels = evaluate_model_b(model, test_dataset, device, batch_size)
    
    return predictions, labels


def compare_results(metrics_a: Dict, metrics_b: Dict) -> Dict:
    """
    モデルAとモデルBの結果を比較
    
    Args:
        metrics_a: モデルAのメトリクス
        metrics_b: モデルBのメトリクス
    
    Returns:
        comparison: 比較結果
    """
    comparison = {
        "model_a": {
            "accuracy": metrics_a["accuracy"],
            "f1_macro": metrics_a["f1_macro"],
            "false_positive_rate": metrics_a["false_positive_rate"]
        },
        "model_b": {
            "accuracy": metrics_b["accuracy"],
            "f1_macro": metrics_b["f1_macro"],
            "false_positive_rate": metrics_b["false_positive_rate"]
        },
        "improvements": {
            "accuracy_diff": metrics_b["accuracy"] - metrics_a["accuracy"],
            "accuracy_improvement_pct": ((metrics_b["accuracy"] - metrics_a["accuracy"]) / metrics_a["accuracy"] * 100) if metrics_a["accuracy"] > 0 else 0.0,
            "f1_macro_diff": metrics_b["f1_macro"] - metrics_a["f1_macro"],
            "f1_macro_improvement_pct": ((metrics_b["f1_macro"] - metrics_a["f1_macro"]) / metrics_a["f1_macro"] * 100) if metrics_a["f1_macro"] > 0 else 0.0,
            "false_positive_rate_diff": metrics_b["false_positive_rate"] - metrics_a["false_positive_rate"],
            "false_positive_rate_improvement_pct": ((metrics_a["false_positive_rate"] - metrics_b["false_positive_rate"]) / metrics_a["false_positive_rate"] * 100) if metrics_a["false_positive_rate"] > 0 else 0.0
        }
    }
    
    return comparison


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="A/B Test Evaluation for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="Borea-Phi-3.5-mini-Instruct-Common",
        help="Model A (baseline) path or HuggingFace model name"
    )
    parser.add_argument(
        "--model-b",
        type=str,
        required=True,
        help="Model B (processed) path"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/splits/test.jsonl",
        help="Test dataset JSONL file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results/ab_test_comparison/comparison_report.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum new tokens for generation (Model A only)"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # テストデータセット読み込み
    logger.info(f"Loading test dataset from {args.test}...")
    from transformers import AutoTokenizer
    
    # モデルAのトークナイザーを使用（一時的）
    model_a_path = Path(args.model_a)
    if model_a_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(model_a_path), trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    
    test_dataset = FourClassDataset(Path(args.test), tokenizer)
    
    # モデルA評価
    logger.info("="*80)
    logger.info("Evaluating Model A (Baseline)...")
    logger.info("="*80)
    
    # モデルA読み込み
    from transformers import AutoModelForCausalLM
    model_a_path = Path(args.model_a)
    if model_a_path.exists():
        model_a = AutoModelForCausalLM.from_pretrained(
            str(model_a_path),
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )
        tokenizer_a = AutoTokenizer.from_pretrained(str(model_a_path), trust_remote_code=True)
    else:
        model_a = AutoModelForCausalLM.from_pretrained(
            args.model_a,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )
        tokenizer_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    
    # モデルA評価実行
    predictions_a, labels_a = evaluate_model_a(
        model_a, tokenizer_a, test_dataset, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )
    metrics_a = calculate_metrics(predictions_a, labels_a)
    
    logger.info(f"Model A - Accuracy: {metrics_a['accuracy']:.4f}, F1 Macro: {metrics_a['f1_macro']:.4f}")
    
    # モデルB評価
    logger.info("="*80)
    logger.info("Evaluating Model B (Processed)...")
    logger.info("="*80)
    
    from scripts.evaluate_four_class import evaluate_model, calculate_metrics as calc_metrics_b
    
    # モデルB読み込み
    from scripts.train_four_class_classifier import FourClassModel
    model_b_path = Path(args.model_b)
    
    try:
        # 分類モデルとして読み込み
        model_b = torch.load(
            model_b_path / "pytorch_model.bin",
            map_location=device
        )
        tokenizer_b = AutoTokenizer.from_pretrained(str(model_b_path), trust_remote_code=True)
    except:
        # フォールバック: 通常のモデル読み込み
        model_b = AutoModelForCausalLM.from_pretrained(
            str(model_b_path),
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )
        tokenizer_b = AutoTokenizer.from_pretrained(str(model_b_path), trust_remote_code=True)
    
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token
    
    # テストデータセット再作成（モデルB用）
    test_dataset_b = FourClassDataset(Path(args.test), tokenizer_b)
    
    # モデルB評価実行
    predictions_b, labels_b = evaluate_model(model_b, test_dataset_b, device, batch_size=args.batch_size)
    metrics_b = calc_metrics_b(predictions_b, labels_b)
    
    logger.info(f"Model B - Accuracy: {metrics_b['accuracy']:.4f}, F1 Macro: {metrics_b['f1_macro']:.4f}")
    
    # 結果比較
    logger.info("="*80)
    logger.info("Comparison Results")
    logger.info("="*80)
    
    comparison = compare_results(metrics_a, metrics_b)
    
    logger.info(f"Accuracy Improvement: {comparison['improvements']['accuracy_improvement_pct']:.2f}%")
    logger.info(f"F1 Macro Improvement: {comparison['improvements']['f1_macro_improvement_pct']:.2f}%")
    logger.info(f"False Positive Rate Improvement: {comparison['improvements']['false_positive_rate_improvement_pct']:.2f}%")
    
    # 結果保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "evaluation_date": datetime.now().isoformat(),
        "model_a": {
            "path": args.model_a,
            "metrics": {
                k: float(v) for k, v in metrics_a.items()
                if k not in ["confusion_matrix", "classification_report"]
            },
            "confusion_matrix": metrics_a["confusion_matrix"].tolist(),
            "classification_report": metrics_a["classification_report"]
        },
        "model_b": {
            "path": args.model_b,
            "metrics": {
                k: float(v) for k, v in metrics_b.items()
                if k not in ["confusion_matrix", "classification_report"]
            },
            "confusion_matrix": metrics_b["confusion_matrix"].tolist(),
            "classification_report": metrics_b["classification_report"]
        },
        "comparison": comparison
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Comparison report saved to {output_path}")
    
    # 個別メトリクスファイルも保存
    metrics_a_path = output_path.parent / "metrics_model_a.json"
    metrics_b_path = output_path.parent / "metrics_model_b.json"
    
    with open(metrics_a_path, 'w', encoding='utf-8') as f:
        json.dump(results["model_a"], f, indent=2, ensure_ascii=False)
    
    with open(metrics_b_path, 'w', encoding='utf-8') as f:
        json.dump(results["model_b"], f, indent=2, ensure_ascii=False)
    
    logger.info(f"Individual metrics saved to {metrics_a_path} and {metrics_b_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

