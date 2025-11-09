#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
漸次ラベル付けパイプライン

SO8Tの四重推論を使って段階的にラベル付けを行い、品質を向上させる

Usage:
    python scripts/pipelines/incremental_labeling_pipeline.py --input D:/webdataset/processed --output D:/webdataset/labeled
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter
from tqdm import tqdm
import time

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 既存のクラスをインポート
from scripts.pipelines.web_scraping_data_pipeline import DataLabeler, QuadrupleClassifier

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/incremental_labeling_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IncrementalLabeler:
    """漸次ラベル付けクラス"""
    
    def __init__(
        self,
        use_so8t: bool = True,
        so8t_model_path: Optional[str] = None,
        batch_size: int = 100,
        quality_threshold: float = 0.7
    ):
        """
        初期化
        
        Args:
            use_so8t: SO8T分類を使用するか
            so8t_model_path: SO8Tモデルのパス
            batch_size: バッチサイズ
            quality_threshold: 品質閾値（0.0-1.0）
        """
        self.use_so8t = use_so8t
        self.batch_size = batch_size
        self.quality_threshold = quality_threshold
        
        # 基本ラベル付け（キーワードベース）
        self.basic_labeler = DataLabeler()
        
        # SO8T分類器（使用する場合）
        if use_so8t:
            self.so8t_classifier = QuadrupleClassifier(so8t_model_path)
        else:
            self.so8t_classifier = None
        
        logger.info("="*80)
        logger.info("Incremental Labeler Initialized")
        logger.info("="*80)
        logger.info(f"SO8T enabled: {use_so8t}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Quality threshold: {quality_threshold}")
    
    def phase1_basic_labeling(self, sample: Dict) -> Dict:
        """
        Phase 1: 基本的なラベル付け（キーワードベース）
        
        Args:
            sample: 処理するサンプル
        
        Returns:
            基本的なラベル付けが完了したサンプル
        """
        labeled_sample = self.basic_labeler.label_sample(sample)
        labeled_sample['labeling_phase'] = 'phase1_basic'
        labeled_sample['labeling_quality'] = 0.5  # 基本ラベル付けの品質スコア
        return labeled_sample
    
    def phase2_so8t_labeling(self, sample: Dict) -> Dict:
        """
        Phase 2: SO8Tによる詳細ラベル付け
        
        Args:
            sample: 処理するサンプル
        
        Returns:
            SO8Tによる詳細ラベル付けが完了したサンプル
        """
        if self.so8t_classifier is None:
            logger.warning("[PHASE2] SO8T classifier not available, skipping")
            return sample
        
        try:
            # SO8Tによる四値分類を実行
            classified_sample = self.so8t_classifier.classify_quadruple(sample)
            
            # 品質スコアを計算
            quality_score = self._calculate_quality_score(classified_sample)
            classified_sample['labeling_phase'] = 'phase2_so8t'
            classified_sample['labeling_quality'] = quality_score
            
            return classified_sample
        
        except Exception as e:
            logger.error(f"[PHASE2] SO8T labeling failed: {e}")
            sample['labeling_phase'] = 'phase2_so8t_failed'
            sample['labeling_quality'] = 0.0
            sample['labeling_error'] = str(e)
            return sample
    
    def phase3_quality_improvement(self, sample: Dict) -> Dict:
        """
        Phase 3: ラベル品質の評価と改善
        
        Args:
            sample: 処理するサンプル
        
        Returns:
            品質改善が完了したサンプル
        """
        quality_score = sample.get('labeling_quality', 0.0)
        
        # 品質が閾値以下の場合は再処理
        if quality_score < self.quality_threshold and self.so8t_classifier:
            logger.info(f"[PHASE3] Quality below threshold ({quality_score:.2f} < {self.quality_threshold}), reprocessing...")
            
            try:
                # 再処理（異なるパラメータで）
                improved_sample = self.so8t_classifier.classify_quadruple(sample)
                improved_quality = self._calculate_quality_score(improved_sample)
                
                if improved_quality > quality_score:
                    improved_sample['labeling_phase'] = 'phase3_improved'
                    improved_sample['labeling_quality'] = improved_quality
                    improved_sample['improvement_applied'] = True
                    return improved_sample
                else:
                    sample['labeling_phase'] = 'phase3_no_improvement'
                    sample['improvement_applied'] = False
                    return sample
            
            except Exception as e:
                logger.error(f"[PHASE3] Quality improvement failed: {e}")
                sample['labeling_phase'] = 'phase3_improvement_failed'
                sample['improvement_error'] = str(e)
                return sample
        
        sample['labeling_phase'] = 'phase3_skipped'
        sample['improvement_applied'] = False
        return sample
    
    def _calculate_quality_score(self, sample: Dict) -> float:
        """
        ラベル付けの品質スコアを計算（0.0-1.0）
        
        Args:
            sample: サンプル
        
        Returns:
            品質スコア
        """
        quadruple = sample.get('quadruple_classification', {})
        
        if not quadruple:
            return 0.0
        
        # 信頼度スコアを取得
        confidences = [
            quadruple.get('task_confidence', 0.0),
            quadruple.get('safety_confidence', 0.0),
            quadruple.get('policy_confidence', 0.0),
            quadruple.get('final_confidence', 0.0)
        ]
        
        # 平均信頼度を計算
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # 推論テキストの存在チェック
        has_reasoning = any([
            quadruple.get('task_reasoning'),
            quadruple.get('safety_reasoning'),
            quadruple.get('policy_reasoning'),
            quadruple.get('final_reasoning')
        ])
        
        # 品質スコア = 信頼度 * 0.7 + 推論存在 * 0.3
        quality_score = avg_confidence * 0.7 + (1.0 if has_reasoning else 0.0) * 0.3
        
        return min(quality_score, 1.0)
    
    def process_sample_incremental(self, sample: Dict) -> Dict:
        """
        サンプルを漸次処理（Phase 1 → Phase 2 → Phase 3）
        
        Args:
            sample: 処理するサンプル
        
        Returns:
            処理されたサンプル
        """
        # Phase 1: 基本的なラベル付け
        phase1_sample = self.phase1_basic_labeling(sample)
        
        # Phase 2: SO8Tによる詳細ラベル付け
        if self.use_so8t:
            phase2_sample = self.phase2_so8t_labeling(phase1_sample)
        else:
            phase2_sample = phase1_sample
        
        # Phase 3: 品質改善
        final_sample = self.phase3_quality_improvement(phase2_sample)
        
        # 処理時刻を記録
        final_sample['incremental_labeling_completed_at'] = datetime.now().isoformat()
        
        return final_sample
    
    def process_batch(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        バッチを処理
        
        Args:
            samples: 処理するサンプルのリスト
        
        Returns:
            (処理済みサンプル, 失敗したサンプル)のタプル
        """
        processed_samples = []
        failed_samples = []
        
        for sample in tqdm(samples, desc="Processing batch"):
            try:
                processed_sample = self.process_sample_incremental(sample)
                processed_samples.append(processed_sample)
            except Exception as e:
                logger.error(f"Failed to process sample: {e}")
                failed_samples.append({
                    'sample': sample,
                    'error': str(e)
                })
        
        return processed_samples, failed_samples


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Incremental Labeling Pipeline")
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input directory containing JSONL files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for labeled data'
    )
    parser.add_argument(
        '--so8t-model',
        type=str,
        default=None,
        help='Path to SO8T model'
    )
    parser.add_argument(
        '--use-so8t',
        action='store_true',
        default=True,
        help='Use SO8T classification'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.7,
        help='Quality threshold for improvement (0.0-1.0)'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 漸次ラベル付けを初期化
    labeler = IncrementalLabeler(
        use_so8t=args.use_so8t,
        so8t_model_path=args.so8t_model,
        batch_size=args.batch_size,
        quality_threshold=args.quality_threshold
    )
    
    # JSONLファイルを読み込み
    jsonl_files = list(args.input.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"No JSONL files found in {args.input}")
        return 1
    
    all_samples = []
    for jsonl_file in jsonl_files:
        logger.info(f"Loading {jsonl_file}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(all_samples)} samples from {len(jsonl_files)} files")
    
    # バッチに分割して処理
    batches = [all_samples[i:i + args.batch_size] for i in range(0, len(all_samples), args.batch_size)]
    logger.info(f"Processing {len(batches)} batches...")
    
    all_processed = []
    all_failed = []
    
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)}...")
        processed, failed = labeler.process_batch(batch)
        all_processed.extend(processed)
        all_failed.extend(failed)
    
    # 結果を保存
    output_file = args.output / f"incremental_labeled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    logger.info(f"Saving {len(all_processed)} processed samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_processed:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 統計情報を計算
    quality_scores = [s.get('labeling_quality', 0.0) for s in all_processed]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    logger.info("="*80)
    logger.info("Incremental Labeling Pipeline Completed")
    logger.info("="*80)
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Processed: {len(all_processed)}")
    logger.info(f"Failed: {len(all_failed)}")
    logger.info(f"Average quality score: {avg_quality:.3f}")
    logger.info(f"Output file: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

