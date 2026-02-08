#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codexデータセットの統計処理とクレンジングスクリプト

四値分類ラベルの検証と修正、統計処理（平均、標準偏差、外れ値検出）、
品質スコアに基づくフィルタリング、データセットバランス調整を実行
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import Counter
import numpy as np
from scipy import stats

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "cleanse_codex_pairwise_dataset.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class CodexDatasetCleanser:
    """Codexデータセットの統計処理とクレンジング"""
    
    def __init__(self):
        """初期化"""
        self.four_class_labels = ['ALLOW', 'ESCALATION', 'DENY', 'REFUSE']
        logger.info("[INIT] CodexDatasetCleanser initialized")
    
    def _validate_four_class_label(self, label: str) -> str:
        """四値分類ラベルの検証と修正"""
        if label in self.four_class_labels:
            return label
        
        # ラベルの正規化
        label_upper = label.upper()
        if 'ALLOW' in label_upper or '許可' in label:
            return 'ALLOW'
        elif 'ESCALATION' in label_upper or 'エスカレーション' in label or 'ESCALATE' in label_upper:
            return 'ESCALATION'
        elif 'DENY' in label_upper or '拒否' in label:
            return 'DENY'
        elif 'REFUSE' in label_upper or '拒絶' in label:
            return 'REFUSE'
        
        # デフォルトはALLOW
        logger.warning(f"[WARNING] Unknown label '{label}', defaulting to 'ALLOW'")
        return 'ALLOW'
    
    def _detect_outliers(self, values: List[float], z_threshold: float = 3.0) -> List[int]:
        """外れ値を検出（Z-score法）"""
        if len(values) < 3:
            return []
        
        z_scores = np.abs(stats.zscore(values))
        outliers = np.where(z_scores > z_threshold)[0].tolist()
        
        return outliers
    
    def _calculate_statistics(self, samples: List[Dict]) -> Dict[str, Any]:
        """統計情報を計算"""
        quality_scores = [s.get('quality_score', 0.0) for s in samples]
        rejected_quality_scores = [s.get('rejected_quality_score', 0.0) for s in samples]
        
        stats_dict = {
            'total_samples': len(samples),
            'quality_score': {
                'mean': np.mean(quality_scores) if quality_scores else 0.0,
                'std': np.std(quality_scores) if quality_scores else 0.0,
                'min': np.min(quality_scores) if quality_scores else 0.0,
                'max': np.max(quality_scores) if quality_scores else 0.0,
                'median': np.median(quality_scores) if quality_scores else 0.0
            },
            'rejected_quality_score': {
                'mean': np.mean(rejected_quality_scores) if rejected_quality_scores else 0.0,
                'std': np.std(rejected_quality_scores) if rejected_quality_scores else 0.0,
                'min': np.min(rejected_quality_scores) if rejected_quality_scores else 0.0,
                'max': np.max(rejected_quality_scores) if rejected_quality_scores else 0.0,
                'median': np.median(rejected_quality_scores) if rejected_quality_scores else 0.0
            },
            'four_class_distribution': Counter(s.get('four_class_label', 'ALLOW') for s in samples)
        }
        
        return stats_dict
    
    def _calculate_quality_score(self, sample: Dict) -> float:
        """
        サンプルの品質スコアを計算

        Args:
            sample: サンプルデータ

        Returns:
            品質スコア (0.0-1.0)
        """
        text = sample.get('text', '')
        label = sample.get('label', 'ALLOW')

        # 基本スコア
        score = 0.5

        # テキスト長による調整（長すぎず短すぎない）
        text_length = len(text)
        if 50 <= text_length <= 2000:
            score += 0.2
        elif text_length < 20:
            score -= 0.3
        elif text_length > 5000:
            score -= 0.2

        # ラベルによる調整
        if label == 'ALLOW':
            score += 0.1
        elif label == 'ESCALATION':
            score += 0.05
        elif label == 'DENY':
            score += 0.05
        elif label == 'REFUSE':
            score += 0.1

        # テキストの内容チェック
        if any(keyword in text.lower() for keyword in ['python', 'code', 'function', 'class', 'import', 'def']):
            score += 0.1  # コーディング関連

        if any(keyword in text.lower() for keyword in ['math', 'theorem', 'proof', 'calculate']):
            score += 0.1  # 数学関連

        # スコアを0-1の範囲に制限
        return max(0.0, min(1.0, score))

    def cleanse_dataset(
        self,
        dataset_path: Path,
        min_quality_score: float = 0.0,
        balance_classes: bool = True,
        remove_outliers: bool = True,
        z_threshold: float = 3.0
    ) -> Path:
        """
        データセットをクレンジング
        
        Args:
            dataset_path: 入力データセットパス
            min_quality_score: 最小品質スコア
            balance_classes: クラスバランス調整を行うか
            remove_outliers: 外れ値を除去するか
            z_threshold: 外れ値検出のZ-score閾値
        
        Returns:
            クレンジング済みデータセットパス
        """
        logger.info(f"[CLEANSE] Starting dataset cleansing: {dataset_path}")
        
        # データセットを読み込み
        samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        # 品質スコアがなければ計算して追加
                        if 'quality_score' not in sample:
                            sample['quality_score'] = self._calculate_quality_score(sample)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"[WARNING] Failed to parse line {line_num}: {e}")
        
        logger.info(f"[LOAD] Loaded {len(samples)} samples")
        
        # 統計情報を計算
        stats_before = self._calculate_statistics(samples)
        logger.info(f"[STATS] Before cleansing:")
        logger.info(f"  Total samples: {stats_before['total_samples']}")
        logger.info(f"  Quality score: mean={stats_before['quality_score']['mean']:.3f}, std={stats_before['quality_score']['std']:.3f}")
        logger.info(f"  Four-class distribution: {dict(stats_before['four_class_distribution'])}")
        
        # 1. 四値分類ラベルの検証と修正
        logger.info("[STEP 1] Validating and correcting four-class labels...")
        corrected_count = 0
        for sample in samples:
            original_label = sample.get('four_class_label', 'ALLOW')
            corrected_label = self._validate_four_class_label(original_label)
            if original_label != corrected_label:
                sample['four_class_label'] = corrected_label
                corrected_count += 1
        
        logger.info(f"[OK] Corrected {corrected_count} labels")
        
        # 2. 統計処理（外れ値検出）
        if remove_outliers:
            logger.info("[STEP 2] Detecting outliers...")
            quality_scores = [s.get('quality_score', 0.0) for s in samples]
            outlier_indices = self._detect_outliers(quality_scores, z_threshold)
            
            if outlier_indices:
                logger.info(f"[OUTLIERS] Found {len(outlier_indices)} outliers (Z-score > {z_threshold})")
                # 外れ値を除去（後ろから削除してインデックスを維持）
                for idx in sorted(outlier_indices, reverse=True):
                    samples.pop(idx)
                logger.info(f"[OK] Removed {len(outlier_indices)} outliers")
        
        # 3. 品質フィルタリング
        logger.info(f"[STEP 3] Filtering by quality score (min={min_quality_score})...")
        filtered_samples = []
        for sample in samples:
            quality_score = sample.get('quality_score', 0.0)
            if quality_score >= min_quality_score:
                filtered_samples.append(sample)
        
        removed_count = len(samples) - len(filtered_samples)
        samples = filtered_samples
        logger.info(f"[OK] Removed {removed_count} low-quality samples")
        
        # 4. クラスバランス調整
        if balance_classes:
            logger.info("[STEP 4] Balancing classes...")
            four_class_counts = Counter(s.get('four_class_label', 'ALLOW') for s in samples)
            min_count = min(four_class_counts.values()) if four_class_counts else 0
            
            if min_count > 0:
                balanced_samples = []
                class_samples = {label: [] for label in self.four_class_labels}
                
                # クラスごとにサンプルを分類
                for sample in samples:
                    label = sample.get('four_class_label', 'ALLOW')
                    class_samples[label].append(sample)
                
                # 各クラスからmin_count個をランダムに選択
                import random
                random.seed(42)
                for label in self.four_class_labels:
                    if label in class_samples:
                        selected = random.sample(class_samples[label], min(len(class_samples[label]), min_count))
                        balanced_samples.extend(selected)
                
                samples = balanced_samples
                logger.info(f"[OK] Balanced to {min_count} samples per class (total: {len(samples)})")
        
        # 統計情報を再計算
        stats_after = self._calculate_statistics(samples)
        logger.info(f"[STATS] After cleansing:")
        logger.info(f"  Total samples: {stats_after['total_samples']}")
        logger.info(f"  Quality score: mean={stats_after['quality_score']['mean']:.3f}, std={stats_after['quality_score']['std']:.3f}")
        logger.info(f"  Four-class distribution: {dict(stats_after['four_class_distribution'])}")
        
        # クレンジング済みデータセットを保存
        output_path = dataset_path.parent / f"{dataset_path.stem}_cleansed{dataset_path.suffix}"
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SUCCESS] Cleansed dataset saved to {output_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Cleanse Codex pairwise dataset with statistical processing"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Input dataset path (JSONL format)"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.0,
        help="Minimum quality score threshold (default: 0.0)"
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Balance classes (equal number of samples per class)"
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Remove outliers using Z-score method"
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for outlier detection (default: 3.0)"
    )
    
    args = parser.parse_args()
    
    # データセットクレンザーを初期化
    cleanser = CodexDatasetCleanser()
    
    # データセットをクレンジング
    output_path = cleanser.cleanse_dataset(
        dataset_path=args.dataset,
        min_quality_score=args.min_quality_score,
        balance_classes=args.balance_classes,
        remove_outliers=args.remove_outliers,
        z_threshold=args.z_threshold
    )
    
    logger.info(f"[COMPLETE] Dataset cleansing completed: {output_path}")


if __name__ == "__main__":
    main()

