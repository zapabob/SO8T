#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AED Calibration Tool

温度スケーリング＋AEDしきい値最適化（小検証セット）

Usage:
    python scripts/training/calibrate_aed.py \
        --model D:/webdataset/gguf_models/so8t_baked_Q5_K_M.gguf \
        --val_data data/val.jsonl \
        --output_dir D:/webdataset/calibration
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, classification_report, brier_score_loss
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def compute_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE)を計算
    
    Args:
        y_true: 真のラベル [N]
        y_pred_proba: 予測確率 [N, 3] (ALLOW, ESCALATE, DENY)
        n_bins: ビン数
    
    Returns:
        ECE値
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    confidences = np.max(y_pred_proba, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    温度スケーリングを適用
    
    Args:
        logits: ロジット [N, 3]
        temperature: 温度パラメータ
    
    Returns:
        スケーリング後のロジット
    """
    return logits / temperature


def predict_aed(
    logits: np.ndarray,
    thresholds: Dict[str, float],
    temperature: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    AED予測（温度スケーリング＋しきい値適用）
    
    Args:
        logits: ロジット [N, 3] (ALLOW, ESCALATE, DENY)
        thresholds: しきい値辞書 {"deny": 0.5, "escalate": 0.7}
        temperature: 温度パラメータ
    
    Returns:
        (予測ラベル, 予測確率)
    """
    # 温度スケーリング
    logits_scaled = logits / temperature
    
    # ソフトマックス
    probs = F.softmax(torch.tensor(logits_scaled), dim=-1).numpy()
    
    # しきい値適用
    # DENY < threshold_deny
    # ESCALATE: threshold_deny <= prob < threshold_escalate
    # ALLOW: prob >= threshold_escalate
    predictions = np.zeros(len(probs), dtype=int)
    
    deny_threshold = thresholds.get("deny", 0.5)
    escalate_threshold = thresholds.get("escalate", 0.7)
    
    # DENYクラス（インデックス2）の確率がしきい値未満
    deny_probs = probs[:, 2]
    escalate_probs = probs[:, 1]
    allow_probs = probs[:, 0]
    
    # 最大確率に基づいて分類
    max_probs = np.max(probs, axis=1)
    
    # DENY: DENY確率がしきい値以上
    predictions[deny_probs >= deny_threshold] = 2
    
    # ESCALATE: DENY未満かつESCALATE確率がしきい値以上
    mask_escalate = (deny_probs < deny_threshold) & (escalate_probs >= escalate_threshold)
    predictions[mask_escalate] = 1
    
    # ALLOW: 残り
    mask_allow = (deny_probs < deny_threshold) & (escalate_probs < escalate_threshold)
    predictions[mask_allow] = 0
    
    return predictions, probs


def objective_function(
    params: np.ndarray,
    logits: np.ndarray,
    y_true: np.ndarray,
    initial_thresholds: Dict[str, float]
) -> float:
    """
    最適化目的関数（ECE + Brier Score + 誤許可率）
    
    Args:
        params: [temperature, deny_threshold, escalate_threshold]
        logits: ロジット [N, 3]
        y_true: 真のラベル [N]
        initial_thresholds: 初期しきい値
    
    Returns:
        損失値（最小化）
    """
    temperature = params[0]
    thresholds = {
        "deny": params[1],
        "escalate": params[2]
    }
    
    # 予測
    y_pred, y_pred_proba = predict_aed(logits, thresholds, temperature)
    
    # ECE
    ece = compute_ece(y_true, y_pred_proba)
    
    # Brier Score
    brier = brier_score_loss(y_true, y_pred_proba[np.arange(len(y_true)), y_true])
    
    # 誤許可率（危険をALLOW）
    # 真のラベルがDENY(2)またはESCALATE(1)なのにALLOW(0)と予測
    false_allow_rate = ((y_true >= 1) & (y_pred == 0)).mean()
    
    # 総合損失（重み付き）
    loss = (
        0.4 * ece +
        0.3 * brier +
        0.3 * false_allow_rate
    )
    
    return loss


def load_validation_data(val_data_path: Path) -> Tuple[List[str], List[int]]:
    """
    検証データを読み込み
    
    Args:
        val_data_path: 検証データパス（JSONL）
    
    Returns:
        (テキストリスト, ラベルリスト)
    """
    texts = []
    labels = []
    
    # ラベルマッピング
    label_map = {"ALLOW": 0, "ESCALATE": 1, "DENY": 2}
    
    logger.info(f"Loading validation data from {val_data_path}...")
    with open(val_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                if "text" in sample:
                    texts.append(sample["text"])
                    # ラベルを取得（デフォルトはALLOW）
                    label_str = sample.get("label", "ALLOW")
                    labels.append(label_map.get(label_str.upper(), 0))
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(texts)} validation samples")
    return texts, labels


def get_logits_from_model(
    model_path: Path,
    texts: List[str]
) -> np.ndarray:
    """
    モデルからロジットを取得（簡易版）
    
    注意: 実際の実装では、llama.cppやOllama APIを使用して
    ロジットを取得する必要があります。
    
    Args:
        model_path: モデルパス
        texts: テキストリスト
    
    Returns:
        ロジット [N, 3]
    """
    # 簡易実装: ダミーロジットを返す
    # 実際の実装では、モデル推論を実行してロジットを取得
    logger.warning("Using dummy logits. Implement actual model inference.")
    n_samples = len(texts)
    logits = np.random.randn(n_samples, 3) * 0.5
    return logits


def main():
    parser = argparse.ArgumentParser(description="AED Calibration")
    parser.add_argument("--model", type=str, required=True,
                       help="GGUF model path")
    parser.add_argument("--val_data", type=str, required=True,
                       help="Validation dataset path (JSONL)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--initial_temperature", type=float, default=1.0,
                       help="Initial temperature")
    parser.add_argument("--initial_deny_threshold", type=float, default=0.5,
                       help="Initial DENY threshold")
    parser.add_argument("--initial_escalate_threshold", type=float, default=0.7,
                       help="Initial ESCALATE threshold")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("AED Calibration")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Validation data: {args.val_data}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 検証データ読み込み
    texts, labels = load_validation_data(Path(args.val_data))
    y_true = np.array(labels)
    
    # モデルからロジットを取得
    logger.info("Getting logits from model...")
    logits = get_logits_from_model(Path(args.model), texts)
    
    # 初期パラメータ
    initial_params = np.array([
        args.initial_temperature,
        args.initial_deny_threshold,
        args.initial_escalate_threshold
    ])
    
    initial_thresholds = {
        "deny": args.initial_deny_threshold,
        "escalate": args.initial_escalate_threshold
    }
    
    # 最適化実行
    logger.info("Optimizing temperature and thresholds...")
    result = minimize(
        objective_function,
        initial_params,
        args=(logits, y_true, initial_thresholds),
        method='L-BFGS-B',
        bounds=[
            (0.1, 5.0),      # temperature
            (0.0, 1.0),      # deny_threshold
            (0.0, 1.0)       # escalate_threshold
        ]
    )
    
    if not result.success:
        logger.warning("Optimization did not converge. Using initial parameters.")
        optimal_params = initial_params
    else:
        optimal_params = result.x
    
    optimal_temperature = optimal_params[0]
    optimal_thresholds = {
        "deny": optimal_params[1],
        "escalate": optimal_params[2]
    }
    
    logger.info(f"Optimal temperature: {optimal_temperature:.4f}")
    logger.info(f"Optimal DENY threshold: {optimal_thresholds['deny']:.4f}")
    logger.info(f"Optimal ESCALATE threshold: {optimal_thresholds['escalate']:.4f}")
    
    # 最適化後の予測
    y_pred, y_pred_proba = predict_aed(logits, optimal_thresholds, optimal_temperature)
    
    # 評価メトリクス
    ece = compute_ece(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba[np.arange(len(y_true)), y_true])
    f1_macro = f1_score(y_true, y_pred, average='macro')
    false_allow_rate = ((y_true >= 1) & (y_pred == 0)).mean()
    
    logger.info("="*80)
    logger.info("Calibration Results")
    logger.info("="*80)
    logger.info(f"ECE: {ece:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    logger.info(f"Macro F1: {f1_macro:.4f}")
    logger.info(f"False Allow Rate: {false_allow_rate:.4f}")
    
    # 分類レポート
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_true, y_pred, target_names=["ALLOW", "ESCALATE", "DENY"]))
    
    # 結果を保存
    calibration_results = {
        "temperature": float(optimal_temperature),
        "thresholds": {
            "deny": float(optimal_thresholds["deny"]),
            "escalate": float(optimal_thresholds["escalate"])
        },
        "metrics": {
            "ece": float(ece),
            "brier_score": float(brier),
            "macro_f1": float(f1_macro),
            "false_allow_rate": float(false_allow_rate)
        }
    }
    
    output_file = output_dir / "calibration_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(calibration_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nCalibration results saved to: {output_file}")
    logger.info("="*80)
    logger.info("Calibration completed!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

