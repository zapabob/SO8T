"""
SO8T Thinking Model 評価スクリプト

Thinking品質、Safety判定精度、Verifierスコアと実際の品質の相関を評価する。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "so8t-mmllm" / "src"))

from models.so8t_thinking_model import SO8TThinkingModel
from models.safety_aware_so8t import SafetyAwareSO8TConfig
from utils.thinking_utils import (
    load_thinking_dataset,
    extract_thinking_safely,
    parse_safety_label,
)
from transformers import AutoTokenizer


SAFETY_LABEL_MAP = {"ALLOW": 0, "ESCALATE": 1, "REFUSE": 2}
ID_TO_SAFETY_LABEL = {v: k for k, v in SAFETY_LABEL_MAP.items()}


def evaluate_thinking_quality(
    model: SO8TThinkingModel,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Thinking品質を評価
    
    Args:
        model: SO8TThinkingModel
        tokenizer: トークナイザー
        samples: 評価サンプル
        device: デバイス
    
    Returns:
        評価結果の辞書
    """
    results = {
        "thinking_extraction_rate": 0.0,
        "final_extraction_rate": 0.0,
        "format_validity": 0.0,
        "thinking_lengths": [],
        "final_lengths": [],
    }
    
    valid_count = 0
    thinking_extracted = 0
    final_extracted = 0
    
    for sample in samples:
        output = sample.get("output", "")
        
        # ThinkingとFinalを抽出
        thinking, final, _ = extract_thinking_safely(output)
        
        if thinking is not None:
            thinking_extracted += 1
            results["thinking_lengths"].append(len(thinking))
        
        if final is not None:
            final_extracted += 1
            results["final_lengths"].append(len(final))
        
        if thinking is not None and final is not None:
            valid_count += 1
    
    total = len(samples)
    results["thinking_extraction_rate"] = thinking_extracted / total if total > 0 else 0.0
    results["final_extraction_rate"] = final_extracted / total if total > 0 else 0.0
    results["format_validity"] = valid_count / total if total > 0 else 0.0
    
    if results["thinking_lengths"]:
        results["avg_thinking_length"] = np.mean(results["thinking_lengths"])
        results["median_thinking_length"] = np.median(results["thinking_lengths"])
    else:
        results["avg_thinking_length"] = 0.0
        results["median_thinking_length"] = 0.0
    
    if results["final_lengths"]:
        results["avg_final_length"] = np.mean(results["final_lengths"])
        results["median_final_length"] = np.median(results["final_lengths"])
    else:
        results["avg_final_length"] = 0.0
        results["median_final_length"] = 0.0
    
    return results


def evaluate_safety_accuracy(
    model: SO8TThinkingModel,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Safety判定精度を評価
    
    Args:
        model: SO8TThinkingModel
        tokenizer: トークナイザー
        samples: 評価サンプル
        device: デバイス
    
    Returns:
        評価結果の辞書
    """
    true_labels = []
    pred_labels = []
    confidences = []
    
    for sample in samples:
        output = sample.get("output", "")
        thinking, final, _ = extract_thinking_safely(output)
        
        if thinking is None or final is None:
            continue
        
        # 真のラベル
        true_label_str = sample.get("safety_label", "ALLOW")
        true_label = SAFETY_LABEL_MAP.get(true_label_str, 0)
        true_labels.append(true_label)
        
        # モデルの予測
        try:
            eval_result = model.evaluate_safety_and_verifier(
                tokenizer=tokenizer,
                thinking_text=thinking,
                final_text=final,
                device=device,
            )
            
            pred_label_str = eval_result["safety_label"]
            pred_label = SAFETY_LABEL_MAP.get(pred_label_str, 0)
            pred_labels.append(pred_label)
            confidences.append(eval_result["safety_confidence"])
        except Exception as e:
            print(f"[WARNING] Failed to evaluate sample: {e}")
            continue
    
    if len(true_labels) == 0:
        return {"error": "No valid samples"}
    
    # メトリクスを計算
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")
    
    # 混同行列
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 分類レポート
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=["ALLOW", "ESCALATE", "REFUSE"],
        output_dict=True,
    )
    
    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "num_samples": len(true_labels),
    }


def evaluate_verifier_correlation(
    model: SO8TThinkingModel,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Verifierスコアと実際の品質の相関を評価
    
    Args:
        model: SO8TThinkingModel
        tokenizer: トークナイザー
        samples: 評価サンプル
        device: デバイス
    
    Returns:
        評価結果の辞書
    """
    verifier_scores = []
    true_qualities = []
    
    for sample in samples:
        output = sample.get("output", "")
        thinking, final, _ = extract_thinking_safely(output)
        
        if thinking is None or final is None:
            continue
        
        # 真の品質（Verifierラベルから）
        verifier_label = sample.get("verifier_label", {})
        true_logical = verifier_label.get("logical", 1.0)
        true_faithful = verifier_label.get("faithful", 1.0)
        true_quality = (true_logical + true_faithful) / 2.0
        true_qualities.append(true_quality)
        
        # モデルのVerifierスコア
        try:
            eval_result = model.evaluate_safety_and_verifier(
                tokenizer=tokenizer,
                thinking_text=thinking,
                final_text=final,
                device=device,
            )
            
            plausibility = eval_result.get("verifier_plausibility", 0.0)
            self_confidence = eval_result.get("verifier_self_confidence", 0.0)
            pred_quality = (plausibility + self_confidence) / 2.0 if plausibility and self_confidence else 0.0
            
            verifier_scores.append(pred_quality)
        except Exception as e:
            print(f"[WARNING] Failed to evaluate sample: {e}")
            continue
    
    if len(verifier_scores) == 0:
        return {"error": "No valid samples"}
    
    # 相関係数を計算
    correlation = np.corrcoef(true_qualities, verifier_scores)[0, 1]
    
    return {
        "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "num_samples": len(verifier_scores),
        "avg_true_quality": float(np.mean(true_qualities)),
        "avg_pred_quality": float(np.mean(verifier_scores)),
    }


def plot_confusion_matrix(cm: np.ndarray, output_path: Path):
    """混同行列をプロット"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ALLOW", "ESCALATE", "REFUSE"],
        yticklabels=["ALLOW", "ESCALATE", "REFUSE"],
    )
    plt.title("Safety Classification Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SO8T Thinking Model"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        required=True,
        help="Test dataset (JSONL format)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--use-redacted",
        action="store_true",
        help="Use <think> format",
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # データセットをロード
    print(f"[INFO] Loading test dataset: {args.test_dataset}")
    samples = load_thinking_dataset(args.test_dataset)
    print(f"[INFO] Loaded {len(samples)} samples")
    
    # トークナイザーをロード
    print(f"[INFO] Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデルをロード
    print(f"[INFO] Loading model: {args.model_path}")
    so8t_config = SafetyAwareSO8TConfig(
        num_safety_labels=3,
        num_verifier_dims=2,
        use_verifier_head=True,
        use_strict_so8_rotation=True,
    )
    
    model = SO8TThinkingModel(
        base_model_name_or_path=str(args.model_path),
        so8t_config=so8t_config,
        use_redacted_tokens=args.use_redacted,
    )
    model.set_tokenizer(tokenizer)
    model.eval()
    model.to(device)
    
    # 評価を実行
    print("[INFO] Evaluating thinking quality...")
    thinking_results = evaluate_thinking_quality(model, tokenizer, samples, device)
    
    print("[INFO] Evaluating safety accuracy...")
    safety_results = evaluate_safety_accuracy(model, tokenizer, samples, device)
    
    print("[INFO] Evaluating verifier correlation...")
    verifier_results = evaluate_verifier_correlation(model, tokenizer, samples, device)
    
    # 結果を保存
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "thinking_quality": thinking_results,
        "safety_accuracy": safety_results,
        "verifier_correlation": verifier_results,
    }
    
    results_file = args.output_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Results saved to: {results_file}")
    
    # 混同行列をプロット
    if "confusion_matrix" in safety_results:
        cm = np.array(safety_results["confusion_matrix"])
        plot_confusion_matrix(cm, args.output_dir / "confusion_matrix.png")
        print(f"[INFO] Confusion matrix saved to: {args.output_dir / 'confusion_matrix.png'}")
    
    # サマリーを表示
    print("\n[EVALUATION SUMMARY]")
    print(f"Thinking Quality:")
    print(f"  - Format Validity: {thinking_results['format_validity']:.2%}")
    print(f"  - Thinking Extraction Rate: {thinking_results['thinking_extraction_rate']:.2%}")
    print(f"  - Final Extraction Rate: {thinking_results['final_extraction_rate']:.2%}")
    
    if "accuracy" in safety_results:
        print(f"\nSafety Accuracy:")
        print(f"  - Accuracy: {safety_results['accuracy']:.2%}")
        print(f"  - F1 Macro: {safety_results['f1_macro']:.4f}")
        print(f"  - F1 Weighted: {safety_results['f1_weighted']:.4f}")
    
    if "correlation" in verifier_results:
        print(f"\nVerifier Correlation:")
        print(f"  - Correlation: {verifier_results['correlation']:.4f}")
        print(f"  - Avg True Quality: {verifier_results['avg_true_quality']:.4f}")
        print(f"  - Avg Pred Quality: {verifier_results['avg_pred_quality']:.4f}")


if __name__ == "__main__":
    main()

