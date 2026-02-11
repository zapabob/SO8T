#!/usr/bin/env python3
"""
温度較正スクリプト
ECE（Expected Calibration Error）を最小化する最適温度を探索
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def calculate_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) を計算
    
    Args:
        confidences: 信頼度配列 [0, 1]
        accuracies: 正解フラグ配列 {0, 1}
        n_bins: ビン数
    
    Returns:
        ece: ECEスコア
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # ビン内のサンプルを取得
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_brier_score(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """
    Brier Score を計算
    
    Args:
        confidences: 信頼度配列
        accuracies: 正解フラグ配列
    
    Returns:
        brier_score: Brierスコア
    """
    return np.mean((confidences - accuracies) ** 2)


def evaluate_with_temperature(
    model,
    tokenizer,
    dataset,
    temperature: float,
    device: str,
    max_samples: int = 100,
) -> Dict:
    """
    指定温度で評価
    
    Args:
        model: モデル
        tokenizer: トークナイザー
        dataset: 評価データセット
        temperature: 温度パラメータ
        device: デバイス
        max_samples: 最大評価サンプル数
    
    Returns:
        metrics: 評価メトリクス
    """
    model.eval()
    
    confidences = []
    accuracies = []
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc=f"Eval (T={temperature:.2f})", total=min(len(dataset), max_samples))):
            if i >= max_samples:
                break
            
            # 入力準備
            input_text = sample['instruction']
            target_text = sample['output']
            
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)
            
            # 予測
            outputs = model(**inputs)
            logits = outputs.logits / temperature  # 温度適用
            
            # 信頼度計算（softmax後の最大値）
            probs = F.softmax(logits[:, -1, :], dim=-1)
            confidence = probs.max().item()
            
            # 正解性判定（簡易版：次トークン予測が正しいか）
            predicted_id = probs.argmax().item()
            target_id = target_ids[0, 0].item() if target_ids.size(1) > 0 else -1
            accuracy = float(predicted_id == target_id)
            
            confidences.append(confidence)
            accuracies.append(accuracy)
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # メトリクス計算
    ece = calculate_ece(confidences, accuracies)
    brier = calculate_brier_score(confidences, accuracies)
    accuracy_mean = accuracies.mean()
    confidence_mean = confidences.mean()
    
    return {
        'temperature': temperature,
        'ece': ece,
        'brier_score': brier,
        'accuracy': accuracy_mean,
        'avg_confidence': confidence_mean,
    }


def grid_search_temperature(
    model,
    tokenizer,
    dataset,
    device: str,
    temperature_range: tuple = (0.5, 2.0),
    n_steps: int = 16,
    max_samples: int = 100,
) -> Dict:
    """
    グリッドサーチで最適温度を探索
    
    Args:
        model: モデル
        tokenizer: トークナイザー
        dataset: 評価データセット
        device: デバイス
        temperature_range: 温度探索範囲
        n_steps: 探索ステップ数
        max_samples: 最大評価サンプル数
    
    Returns:
        best_result: 最適結果
    """
    temperatures = np.linspace(temperature_range[0], temperature_range[1], n_steps)
    
    results = []
    
    for temp in temperatures:
        logger.info(f"[CALIBRATION] Evaluating temperature: {temp:.2f}")
        
        metrics = evaluate_with_temperature(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            temperature=temp,
            device=device,
            max_samples=max_samples,
        )
        
        results.append(metrics)
        
        logger.info(
            f"[RESULT] T={temp:.2f}: "
            f"ECE={metrics['ece']:.4f}, "
            f"Brier={metrics['brier_score']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}"
        )
    
    # 最適温度を選択（ECE最小）
    best_result = min(results, key=lambda x: x['ece'])
    
    return {
        'best_temperature': best_result['temperature'],
        'best_ece': best_result['ece'],
        'all_results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Temperature calibration")
    parser.add_argument("--model_path", type=str, default="checkpoints/phi4_so8t_japanese_final", help="Model path")
    parser.add_argument("--data_path", type=str, default="data/phi4_japanese_synthetic.jsonl", help="Validation data path")
    parser.add_argument("--output_path", type=str, default="_docs/temperature_calibration_report.json", help="Output report path")
    parser.add_argument("--temperature_min", type=float, default=0.5, help="Min temperature")
    parser.add_argument("--temperature_max", type=float, default=2.0, help="Max temperature")
    parser.add_argument("--n_steps", type=int, default=16, help="Number of temperature steps")
    parser.add_argument("--max_samples", type=int, default=100, help="Max evaluation samples")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Temperature Calibration")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Temperature range: [{args.temperature_min}, {args.temperature_max}]")
    logger.info(f"Steps: {args.n_steps}")
    logger.info("=" * 70)
    
    # デバイス
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[DEVICE] Using {device}")
    
    # モデル読み込み
    logger.info("[STEP 1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # データセット読み込み
    logger.info("[STEP 2] Loading dataset...")
    dataset = load_dataset('json', data_files=args.data_path, split='train')
    
    # 温度較正
    logger.info("[STEP 3] Running temperature calibration...")
    result = grid_search_temperature(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        temperature_range=(args.temperature_min, args.temperature_max),
        n_steps=args.n_steps,
        max_samples=args.max_samples,
    )
    
    # 結果保存
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info("[SUCCESS] Calibration completed!")
    logger.info(f"Best temperature: {result['best_temperature']:.2f}")
    logger.info(f"Best ECE: {result['best_ece']:.4f}")
    logger.info(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()

