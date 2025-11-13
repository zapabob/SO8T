#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット品質改善パイプライン

データセット品質評価レポートで指摘された改善事項を実装し、データセットの品質を向上させる

Usage:
    python scripts/pipelines/improve_dataset_quality.py --input D:/webdataset/processed/four_class/four_class_reclassified_20251109_095554.jsonl --config configs/so8t_auto_data_processing_config.yaml
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import yaml

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/improve_dataset_quality.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from scripts.pipelines.web_scraping_data_pipeline import DataCleaner, QuadrupleClassifier
from scripts.pipelines.so8t_auto_data_processing_pipeline import SO8TAutoDataProcessingPipeline


class DatasetQualityImprovementPipeline:
    """データセット品質改善パイプライン"""
    
    def __init__(self, config_path: Path):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定ファイルを読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 基本設定
        self.output_dir = Path(self.config.get('output_dir', 'D:/webdataset/processed/four_class'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データクリーニング設定
        cleaning_config = self.config.get('data_cleaning', {})
        self.exclude_empty_text = cleaning_config.get('exclude_empty_text', True)
        self.fill_missing_from_content = cleaning_config.get('fill_missing_from_content', True)
        
        # カテゴリ詳細化設定
        category_config = self.config.get('category_refinement', {})
        self.refine_general_category = category_config.get('enabled', True)
        self.category_confidence_threshold = category_config.get('confidence_threshold', 0.3)
        
        # ESCALATION分類設定
        escalation_config = self.config.get('escalation_classification', {})
        self.enable_escalation = escalation_config.get('enabled', True)
        self.long_text_threshold = escalation_config.get('long_text_threshold', 1000)
        
        # SO8T設定
        so8t_config = self.config.get('so8t', {})
        self.use_so8t = so8t_config.get('enabled', True)
        self.so8t_model_path = so8t_config.get('model_path', None)
        
        # コンポーネント初期化
        self.cleaner = DataCleaner(
            exclude_empty_text=self.exclude_empty_text,
            fill_missing_from_content=self.fill_missing_from_content
        )
        self.quadruple_classifier = QuadrupleClassifier(self.so8t_model_path) if self.use_so8t else None
        
        logger.info("="*80)
        logger.info("Dataset Quality Improvement Pipeline Initialized")
        logger.info("="*80)
    
    def load_dataset(self, input_path: Path) -> List[Dict]:
        """
        データセットを読み込み
        
        Args:
            input_path: 入力データセットのパス
        
        Returns:
            サンプルリスト
        """
        logger.info(f"[LOAD] Loading dataset from {input_path}...")
        samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        logger.info(f"[INFO] Loaded {len(samples)} samples")
        return samples
    
    def calculate_quality_metrics(self, samples: List[Dict]) -> Dict:
        """
        データセットの品質メトリクスを計算
        
        Args:
            samples: サンプルリスト
        
        Returns:
            品質メトリクスの辞書
        """
        metrics = {
            'total_samples': len(samples),
            'classification_distribution': {
                'ALLOW': 0,
                'ESCALATION': 0,
                'DENY': 0,
                'REFUSE': 0,
                'unknown': 0
            },
            'missing_fields': {
                'text': 0,
                'nsfw_label': 0,
                'category': 0,
                'domain': 0
            },
            'empty_text': 0,
            'category_distribution': {},
            'language_distribution': {},
            'domain_distribution': {},
            'text_length_stats': {
                'min': float('inf'),
                'max': 0,
                'average': 0,
                'median': 0
            }
        }
        
        text_lengths = []
        
        for sample in samples:
            # 分類分布
            quad_class = sample.get('quadruple_classification', {})
            four_class_label = quad_class.get('four_class_label', 'unknown')
            metrics['classification_distribution'][four_class_label] = metrics['classification_distribution'].get(four_class_label, 0) + 1
            
            # 欠損フィールド
            for field in metrics['missing_fields']:
                if field not in sample or not sample.get(field):
                    metrics['missing_fields'][field] += 1
            
            # テキスト長
            text = sample.get('text', '')
            text_length = len(text) if text else 0
            text_lengths.append(text_length)
            
            if text_length == 0:
                metrics['empty_text'] += 1
            
            # カテゴリ分布
            category = sample.get('category', 'unknown')
            metrics['category_distribution'][category] = metrics['category_distribution'].get(category, 0) + 1
            
            # 言語分布
            language = sample.get('language', 'unknown')
            metrics['language_distribution'][language] = metrics['language_distribution'].get(language, 0) + 1
            
            # ドメイン分布
            domain = sample.get('domain', 'unknown')
            metrics['domain_distribution'][domain] = metrics['domain_distribution'].get(domain, 0) + 1
        
        # テキスト長統計
        if text_lengths:
            metrics['text_length_stats']['min'] = min(text_lengths)
            metrics['text_length_stats']['max'] = max(text_lengths)
            metrics['text_length_stats']['average'] = sum(text_lengths) / len(text_lengths)
            sorted_lengths = sorted(text_lengths)
            metrics['text_length_stats']['median'] = sorted_lengths[len(sorted_lengths) // 2]
        
        return metrics
    
    def improve_dataset(self, input_path: Path) -> Path:
        """
        データセットの品質を改善
        
        Args:
            input_path: 入力データセットのパス
        
        Returns:
            改善されたデータセットのパス
        """
        logger.info("="*80)
        logger.info("Dataset Quality Improvement")
        logger.info("="*80)
        
        # 改善前の品質メトリクスを計算
        original_samples = self.load_dataset(input_path)
        before_metrics = self.calculate_quality_metrics(original_samples)
        
        logger.info("="*80)
        logger.info("Before Improvement Metrics")
        logger.info("="*80)
        logger.info(f"Total samples: {before_metrics['total_samples']}")
        logger.info(f"Classification distribution:")
        for label, count in before_metrics['classification_distribution'].items():
            logger.info(f"  {label}: {count} ({count/before_metrics['total_samples']*100:.1f}%)")
        logger.info(f"Missing fields:")
        for field, count in before_metrics['missing_fields'].items():
            if count > 0:
                logger.info(f"  {field}: {count}")
        logger.info(f"Empty text: {before_metrics['empty_text']}")
        
        # データクリーニング
        logger.info("="*80)
        logger.info("Step 1: Data Cleaning")
        logger.info("="*80)
        cleaned_samples = []
        excluded_count = 0
        
        for sample in tqdm(original_samples, desc="Cleaning"):
            cleaned_sample = self.cleaner.clean_sample(sample)
            if cleaned_sample is not None:
                cleaned_samples.append(cleaned_sample)
            else:
                excluded_count += 1
        
        logger.info(f"[INFO] Excluded {excluded_count} invalid samples")
        logger.info(f"[INFO] Remaining samples: {len(cleaned_samples)}")
        
        # カテゴリ詳細化
        if self.refine_general_category:
            logger.info("="*80)
            logger.info("Step 2: Category Refinement")
            logger.info("="*80)
            
            refined_count = 0
            for sample in tqdm(cleaned_samples, desc="Refining categories"):
                category = sample.get('category', 'general')
                text = sample.get('text', '')
                
                if category in ['general', 'unknown', 'other']:
                    refined_category, confidence = self.quadruple_classifier._refine_category(category, text) if self.quadruple_classifier else (category, 1.0)
                    if refined_category != category and confidence >= self.category_confidence_threshold:
                        sample['category'] = refined_category
                        sample['category_refined'] = True
                        sample['category_confidence'] = confidence
                        refined_count += 1
            
            logger.info(f"[INFO] Refined {refined_count} categories")
        
        # ESCALATION分類の追加
        if self.enable_escalation:
            logger.info("="*80)
            logger.info("Step 3: Adding ESCALATION Classification")
            logger.info("="*80)
            
            escalation_count = 0
            for sample in tqdm(cleaned_samples, desc="Adding ESCALATION"):
                quad_class = sample.get('quadruple_classification', {})
                four_class_label = quad_class.get('four_class_label', 'ALLOW')
                
                # ESCALATION分類を追加
                if four_class_label == 'ALLOW':
                    domain = sample.get('domain', '')
                    text = sample.get('text', '')
                    category = sample.get('category', '')
                    
                    # 機密ドメインまたは長文の場合はESCALATION
                    sensitive_domains = ['defense', 'medical', 'financial']
                    domain_lower = domain.lower()
                    is_sensitive_domain = any(sd in domain_lower for sd in sensitive_domains)
                    
                    if is_sensitive_domain or len(text) > self.long_text_threshold or category in ['medical', 'financial', 'defense']:
                        quad_class['four_class_label'] = 'ESCALATION'
                        quad_class['four_class_label_id'] = 1
                        quad_class['escalation_reason'] = 'sensitive_domain' if is_sensitive_domain else ('long_text' if len(text) > self.long_text_threshold else 'sensitive_category')
                        sample['quadruple_classification'] = quad_class
                        escalation_count += 1
            
            logger.info(f"[INFO] Added ESCALATION classification to {escalation_count} samples")
        
        # 改善後の品質メトリクスを計算
        after_metrics = self.calculate_quality_metrics(cleaned_samples)
        
        logger.info("="*80)
        logger.info("After Improvement Metrics")
        logger.info("="*80)
        logger.info(f"Total samples: {after_metrics['total_samples']}")
        logger.info(f"Classification distribution:")
        for label, count in after_metrics['classification_distribution'].items():
            logger.info(f"  {label}: {count} ({count/after_metrics['total_samples']*100:.1f}%)")
        logger.info(f"Missing fields:")
        for field, count in after_metrics['missing_fields'].items():
            if count > 0:
                logger.info(f"  {field}: {count}")
        logger.info(f"Empty text: {after_metrics['empty_text']}")
        
        # 改善前後の比較
        logger.info("="*80)
        logger.info("Improvement Comparison")
        logger.info("="*80)
        logger.info(f"Samples removed: {before_metrics['total_samples'] - after_metrics['total_samples']}")
        logger.info(f"Missing text fields: {before_metrics['missing_fields']['text']} -> {after_metrics['missing_fields']['text']}")
        logger.info(f"Empty text: {before_metrics['empty_text']} -> {after_metrics['empty_text']}")
        logger.info(f"ESCALATION classification: {before_metrics['classification_distribution']['ESCALATION']} -> {after_metrics['classification_distribution']['ESCALATION']}")
        
        # 結果を保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"four_class_improved_{timestamp}.jsonl"
        logger.info(f"[SAVE] Saving {len(cleaned_samples)} improved samples to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in cleaned_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 品質比較レポートを保存
        report_file = self.output_dir / f"quality_improvement_report_{timestamp}.json"
        report = {
            'before': before_metrics,
            'after': after_metrics,
            'improvements': {
                'samples_removed': before_metrics['total_samples'] - after_metrics['total_samples'],
                'missing_text_reduced': before_metrics['missing_fields']['text'] - after_metrics['missing_fields']['text'],
                'empty_text_reduced': before_metrics['empty_text'] - after_metrics['empty_text'],
                'escalation_added': after_metrics['classification_distribution']['ESCALATION'] - before_metrics['classification_distribution']['ESCALATION']
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[SAVE] Quality improvement report saved to {report_file}")
        logger.info(f"[OK] Dataset quality improvement completed. Output: {output_file}")
        
        return output_file


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Dataset Quality Improvement Pipeline")
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input dataset file path'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/so8t_auto_data_processing_config.yaml'),
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    if not args.input.exists():
        logger.error(f"Input dataset file not found: {args.input}")
        return 1
    
    pipeline = DatasetQualityImprovementPipeline(args.config)
    pipeline.improve_dataset(args.input)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


















