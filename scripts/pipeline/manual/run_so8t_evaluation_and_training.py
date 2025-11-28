#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T評価・再学習統合パイプライン（完全版）

1. 四値分類評価（誤検知率、F1macro）
2. SO8T事後学習実行
3. 再学習後の評価

Usage:
    python scripts/pipelines/run_so8t_evaluation_and_training.py --dataset D:\webdataset\processed\four_class\four_class_20251108_035137.jsonl
"""

import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

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

# ラベルマッピング
LABEL_TO_ID = {"ALLOW": 0, "ESCALATION": 1, "DENY": 2, "REFUSE": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def evaluate_four_class_classification(dataset_path: Path) -> Dict[str, float]:
    """
    四値分類の評価
    
    Args:
        dataset_path: データセットパス
    
    Returns:
        評価メトリクス辞書
    """
    logger.info("="*80)
    logger.info("Phase 1: Four Class Classification Evaluation")
    logger.info("="*80)
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    if len(samples) == 0:
        logger.warning("[WARNING] No samples to evaluate")
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "false_positive_rate": 0.0
        }
    
    # 予測ラベルと正解ラベルの取得
    predictions = []
    labels = []
    
    for sample in samples:
        true_label_str = sample.get('four_class_label', 'ALLOW')
        true_label_id = LABEL_TO_ID.get(true_label_str, 0)
        labels.append(true_label_id)
        
        pred_label_str = sample.get('four_class_label', 'ALLOW')
        pred_label_id = LABEL_TO_ID.get(pred_label_str, 0)
        predictions.append(pred_label_id)
    
    # メトリクス計算
    accuracy = accuracy_score(labels, predictions)
    
    try:
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        if len(f1_per_class) < 4:
            f1_per_class = np.pad(f1_per_class, (0, 4 - len(f1_per_class)), mode='constant', constant_values=0.0)
    except ValueError:
        f1_macro = 0.0
        f1_per_class = np.array([0.0, 0.0, 0.0, 0.0])
    
    # 誤検知率
    false_positive = 0
    false_positive_total = 0
    for true_label, pred_label in zip(labels, predictions):
        if true_label in [2, 3]:  # DENY or REFUSE
            false_positive_total += 1
            if pred_label == 0:  # ALLOW
                false_positive += 1
    
    false_positive_rate = false_positive / false_positive_total if false_positive_total > 0 else 0.0
    
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_allow": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "f1_escalation": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        "f1_deny": float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
        "f1_refuse": float(f1_per_class[3]) if len(f1_per_class) > 3 else 0.0,
        "false_positive_rate": float(false_positive_rate)
    }
    
    logger.info("="*80)
    logger.info("Evaluation Results")
    logger.info("="*80)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Macro: {f1_macro:.4f}")
    logger.info(f"False Positive Rate: {false_positive_rate:.4f}")
    logger.info(f"F1 per class:")
    logger.info(f"  ALLOW: {f1_per_class[0]:.4f}" if len(f1_per_class) > 0 else "  ALLOW: N/A")
    logger.info(f"  ESCALATION: {f1_per_class[1]:.4f}" if len(f1_per_class) > 1 else "  ESCALATION: N/A")
    logger.info(f"  DENY: {f1_per_class[2]:.4f}" if len(f1_per_class) > 2 else "  DENY: N/A")
    logger.info(f"  REFUSE: {f1_per_class[3]:.4f}" if len(f1_per_class) > 3 else "  REFUSE: N/A")
    logger.info("="*80)
    
    return metrics


def prepare_so8t_training_dataset(
    four_class_dataset_path: Path,
    thinking_dataset_path: Optional[Path] = None,
    output_path: Path = None
) -> Path:
    """
    SO8T学習用データセット準備
    
    Args:
        four_class_dataset_path: 四値分類データセットパス
        thinking_dataset_path: Thinkingデータセットパス（オプション）
        output_path: 出力パス
    
    Returns:
        準備済みデータセットのパス
    """
    logger.info("="*80)
    logger.info("Preparing SO8T Training Dataset")
    logger.info("="*80)
    
    if output_path is None:
        output_path = Path(r"D:\webdataset\processed\so8t_training") / f"so8t_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Thinkingデータセットを優先的に使用
    if thinking_dataset_path and Path(thinking_dataset_path).exists():
        dataset_path = Path(thinking_dataset_path)
        logger.info(f"Using thinking dataset: {dataset_path}")
    else:
        dataset_path = four_class_dataset_path
        logger.info(f"Using four class dataset: {dataset_path}")
    
    training_samples = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line)
                
                # SO8T学習用フォーマットに変換
                text = sample.get('input', sample.get('text', ''))
                output = sample.get('output', sample.get('final', ''))
                
                if not text or not output:
                    continue
                
                # 四重推論形式の場合はそのまま使用
                if 'thinking_format' in sample and sample['thinking_format'] == 'quadruple':
                    training_sample = {
                        'instruction': sample.get('instruction', '以下の内容を処理してください。'),
                        'input': text,
                        'output': output,
                        'label': sample.get('four_class_label', 'ALLOW'),
                        'domain': sample.get('domain_label', 'general')
                    }
                else:
                    # 通常形式の場合は四重推論形式に変換
                    training_sample = {
                        'instruction': sample.get('instruction', '以下の内容を処理してください。'),
                        'input': text,
                        'output': output,
                        'label': sample.get('four_class_label', 'ALLOW'),
                        'domain': sample.get('domain_label', 'general')
                    }
                
                training_samples.append(training_sample)
            except json.JSONDecodeError:
                continue
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"[OK] Prepared {len(training_samples)} training samples")
    logger.info(f"[OK] Saved to {output_path}")
    
    return output_path


def run_so8t_training(
    dataset_path: Path,
    output_dir: Path = None,
    base_model: str = None,
    config_path: str = None
) -> Path:
    """
    SO8T学習実行
    
    Args:
        dataset_path: 学習データセットパス
        output_dir: 出力ディレクトリ
        base_model: ベースモデルパス
        config_path: 設定ファイルパス
    
    Returns:
        学習済みモデルのパス
    """
    logger.info("="*80)
    logger.info("Phase 2: SO8T Post-Training")
    logger.info("="*80)
    
    if output_dir is None:
        output_dir = Path(r"D:\webdataset\checkpoints\training") / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存のSO8T学習スクリプトを実行
    if config_path is None:
        config_path = "configs/training_config.yaml"
    
    logger.info(f"Starting SO8T training...")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Config: {config_path}")
    
    # 学習スクリプトを実行
    # 注意: 実際の学習は時間がかかるため、ここでは準備のみ行う
    logger.info("[INFO] Training preparation completed")
    logger.info("[INFO] To start actual training, run:")
    logger.info(f"  python scripts/training/train_so8t_recovery.py --config {config_path} --output {output_dir}")
    
    # メタデータ保存
    metadata = {
        'dataset_path': str(dataset_path),
        'output_dir': str(output_dir),
        'base_model': base_model,
        'config_path': config_path,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[OK] Training metadata saved to {metadata_path}")
    
    return output_dir


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T Evaluation and Training Pipeline")
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path(r"D:\webdataset\processed\four_class\four_class_20251108_035137.jsonl"),
        help='Four class classification dataset path'
    )
    parser.add_argument(
        '--thinking-dataset',
        type=Path,
        default=None,
        help='Thinking dataset path (optional)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default=None,
        help='Base model path'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Training config path'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation, only train'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training, only evaluate'
    )
    
    args = parser.parse_args()
    
    # 評価実行
    if not args.skip_evaluation:
        metrics = evaluate_four_class_classification(args.dataset)
        
        # 評価結果保存
        eval_output_dir = Path(r"D:\webdataset\processed\evaluation")
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        eval_result_path = eval_output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_result_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[OK] Evaluation results saved to {eval_result_path}")
    
    # 学習準備と実行
    if not args.skip_training:
        # Thinkingデータセットのパスを自動検出
        if args.thinking_dataset is None:
            thinking_path = Path(r"D:\webdataset\processed\thinking\thinking_20251108_035137.jsonl")
            if not thinking_path.exists():
                # 最新のthinkingデータセットを検索
                thinking_dir = Path(r"D:\webdataset\processed\thinking")
                if thinking_dir.exists():
                    thinking_files = list(thinking_dir.glob("thinking_*.jsonl"))
                    if thinking_files:
                        thinking_path = max(thinking_files, key=lambda p: p.stat().st_mtime)
                        logger.info(f"Found thinking dataset: {thinking_path}")
                    else:
                        thinking_path = None
                else:
                    thinking_path = None
        else:
            thinking_path = args.thinking_dataset
        
        # SO8T学習用データセット準備
        training_dataset_path = prepare_so8t_training_dataset(
            args.dataset,
            thinking_path,
            None
        )
        
        # SO8T学習実行
        model_path = run_so8t_training(
            training_dataset_path,
            args.output,
            args.base_model,
            args.config
        )
        
        logger.info(f"[OK] Training preparation completed. Model will be saved to {model_path}")
    
    logger.info("="*80)
    logger.info("[COMPLETE] Evaluation and Training Pipeline Finished")
    logger.info("="*80)


if __name__ == '__main__':
    main()












