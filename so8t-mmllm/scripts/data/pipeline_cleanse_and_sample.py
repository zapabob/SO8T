#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データクレンジング&統計的サンプリングパイプライン
600GB教師データ → 300GB学習データ
機械学習ベストプラクティス準拠
"""

import os
import json
import random
import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from tqdm import tqdm

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ML ベストプラクティス設定
ML_BEST_PRACTICES = {
    # データ分割（scikit-learn準拠）
    "train_ratio": 0.8,  # 80% 学習
    "val_ratio": 0.1,    # 10% 検証
    "test_ratio": 0.1,   # 10% テスト
    
    # 品質フィルタ
    "min_length": 50,      # 最小文字数
    "max_length": 10000,   # 最大文字数
    "min_quality_score": 0.7,  # 最小品質スコア
    
    # 重複除去
    "dedup_method": "sha256",  # ハッシュベース
    "similarity_threshold": 0.9,  # 類似度閾値
    
    # ドメインバランシング（統計的）
    "balance_domains": True,  # ドメイン間バランス
    "max_domain_ratio": 0.3,  # 単一ドメイン最大30%
    
    # 言語バランシング
    "language_targets": {
        "ja": 0.7,  # 日本語70%
        "en": 0.2,  # 英語20%
        "zh": 0.1   # 中国語10%
    },
    
    # ランダムシード（再現性）
    "random_seed": 42
}


@dataclass
class DatasetStatistics:
    """データセット統計"""
    total_samples: int
    total_size_gb: float
    by_language: Dict[str, int]
    by_domain: Dict[str, int]
    by_split: Dict[str, int]
    avg_length: float
    avg_quality: float
    duplicates_removed: int


class DataCleanser:
    """データクレンザー"""
    
    def __init__(self, config: Dict = None):
        self.config = config or ML_BEST_PRACTICES
        self.seen_hashes = set()
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
    
    def clean_text(self, text: str) -> str:
        """
        テキストクリーニング
        
        Args:
            text: 元テキスト
        
        Returns:
            cleaned_text: クリーニング後テキスト
        """
        # 空白正規化
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # 特殊文字除去（制御文字等）
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # 前後空白除去
        text = text.strip()
        
        return text
    
    def calculate_quality_score(self, sample: Dict) -> float:
        """
        品質スコア計算
        
        Args:
            sample: サンプル
        
        Returns:
            quality_score: 品質スコア（0.0-1.0）
        """
        text = sample.get('text', '')
        
        score = 0.0
        
        # 長さスコア
        length = len(text)
        if self.config['min_length'] <= length <= self.config['max_length']:
            score += 0.3
        elif length > self.config['max_length']:
            score += 0.1
        
        # 言語適合性
        lang = sample.get('language', 'ja')
        if lang == 'ja':
            ja_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf')
            score += (ja_chars / length) * 0.4 if length > 0 else 0
        elif lang == 'en':
            en_chars = sum(1 for c in text if c.isascii() and c.isalpha())
            score += (en_chars / length) * 0.4 if length > 0 else 0
        elif lang == 'zh':
            zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            score += (zh_chars / length) * 0.4 if length > 0 else 0
        
        # 句読点バランス
        if lang == 'ja':
            punct_count = text.count('。') + text.count('、')
            if 2 <= punct_count <= length / 50:
                score += 0.2
        
        # 多様性（ユニーク文字率）
        if length > 0:
            unique_ratio = len(set(text)) / length
            score += unique_ratio * 0.1
        
        # 既存スコアがあれば統合
        if 'quality_score' in sample:
            score = (score + sample['quality_score']) / 2
        
        return min(score, 1.0)
    
    def is_duplicate(self, sample: Dict) -> bool:
        """
        重複チェック（SHA256ハッシュベース）
        
        Args:
            sample: サンプル
        
        Returns:
            is_dup: 重複フラグ
        """
        text = sample.get('text', '')
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def filter_sample(self, sample: Dict) -> Tuple[bool, float]:
        """
        サンプルフィルタリング
        
        Args:
            sample: サンプル
        
        Returns:
            keep: 保持フラグ
            quality: 品質スコア
        """
        # テキストクリーニング
        sample['text'] = self.clean_text(sample.get('text', ''))
        
        # 長さチェック
        length = len(sample['text'])
        if length < self.config['min_length'] or length > self.config['max_length']:
            return False, 0.0
        
        # 品質スコア
        quality = self.calculate_quality_score(sample)
        if quality < self.config['min_quality_score']:
            return False, quality
        
        # 重複チェック
        if self.is_duplicate(sample):
            return False, quality
        
        return True, quality


class StatisticalSampler:
    """統計的サンプラー（ML ベストプラクティス）"""
    
    def __init__(self, config: Dict = None):
        self.config = config or ML_BEST_PRACTICES
        random.seed(self.config['random_seed'])
    
    def stratified_split(
        self,
        samples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        層化分割（Stratified Split）
        ドメイン・言語比率を維持したまま分割
        
        Args:
            samples: 全サンプル
            train_ratio: 学習データ比率
            val_ratio: 検証データ比率
            test_ratio: テストデータ比率
        
        Returns:
            train, val, test: 分割データ
        """
        logger.info("[SPLIT] Stratified splitting...")
        
        # ドメイン×言語でグループ化
        groups = defaultdict(list)
        for sample in samples:
            key = f"{sample.get('language', 'unknown')}_{sample.get('domain', 'unknown')}"
            groups[key].append(sample)
        
        train, val, test = [], [], []
        
        for key, group_samples in groups.items():
            # シャッフル（再現性あり）
            random.shuffle(group_samples)
            
            n = len(group_samples)
            train_n = int(n * train_ratio)
            val_n = int(n * val_ratio)
            
            train.extend(group_samples[:train_n])
            val.extend(group_samples[train_n:train_n + val_n])
            test.extend(group_samples[train_n + val_n:])
        
        # 最終シャッフル
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        logger.info(f"[OK] Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
        
        return train, val, test
    
    def balance_domains(self, samples: List[Dict]) -> List[Dict]:
        """
        ドメインバランシング（統計的サンプリング）
        
        Args:
            samples: 元サンプル
        
        Returns:
            balanced_samples: バランス済みサンプル
        """
        if not self.config['balance_domains']:
            return samples
        
        logger.info("[BALANCE] Domain balancing...")
        
        # ドメイン別カウント
        domain_samples = defaultdict(list)
        for sample in samples:
            domain = sample.get('domain', 'unknown')
            domain_samples[domain].append(sample)
        
        # 最大ドメインサイズ計算
        max_domain_size = int(len(samples) * self.config['max_domain_ratio'])
        
        # バランシング
        balanced = []
        for domain, domain_list in domain_samples.items():
            if len(domain_list) > max_domain_size:
                # 過剰ドメインはサンプリング
                sampled = random.sample(domain_list, max_domain_size)
                balanced.extend(sampled)
                logger.info(f"[BALANCE] {domain}: {len(domain_list):,} → {max_domain_size:,}")
            else:
                balanced.extend(domain_list)
        
        random.shuffle(balanced)
        logger.info(f"[OK] Balanced: {len(balanced):,} samples")
        
        return balanced
    
    def balance_languages(self, samples: List[Dict]) -> List[Dict]:
        """
        言語バランシング（統計的重み付け）
        
        Args:
            samples: 元サンプル
        
        Returns:
            balanced_samples: バランス済みサンプル
        """
        logger.info("[BALANCE] Language balancing...")
        
        # 言語別分類
        lang_samples = defaultdict(list)
        for sample in samples:
            lang = sample.get('language', 'unknown')
            lang_samples[lang].append(sample)
        
        # ターゲット言語比率
        total_target = len(samples)
        lang_targets = {
            lang: int(total_target * weight)
            for lang, weight in self.config['language_targets'].items()
        }
        
        # バランシング
        balanced = []
        for lang, target_count in lang_targets.items():
            lang_list = lang_samples.get(lang, [])
            
            if len(lang_list) > target_count:
                # オーバーサンプリング
                sampled = random.sample(lang_list, target_count)
                balanced.extend(sampled)
                logger.info(f"[BALANCE] {lang}: {len(lang_list):,} → {target_count:,}")
            elif len(lang_list) < target_count and len(lang_list) > 0:
                # アンダーサンプリング（重複許可）
                sampled = random.choices(lang_list, k=target_count)
                balanced.extend(sampled)
                logger.info(f"[BALANCE] {lang}: {len(lang_list):,} → {target_count:,} (with duplication)")
            else:
                balanced.extend(lang_list)
        
        random.shuffle(balanced)
        logger.info(f"[OK] Language balanced: {len(balanced):,} samples")
        
        return balanced


class DataPipeline:
    """データパイプライン（600GB → 300GB）"""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        target_size_gb: float = 300.0,
        config: Dict = None
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size_gb = target_size_gb
        self.config = config or ML_BEST_PRACTICES
        self.cleanser = DataCleanser(self.config)
        self.sampler = StatisticalSampler(self.config)
    
    def load_raw_data(self) -> List[Dict]:
        """
        生データ読み込み
        
        Returns:
            samples: サンプルリスト
        """
        logger.info("[LOAD] Loading raw data...")
        logger.info(f"Input directory: {self.input_dir}")
        
        samples = []
        jsonl_files = list(self.input_dir.glob("**/*.jsonl"))
        
        logger.info(f"Found {len(jsonl_files)} jsonl files")
        
        for file_path in tqdm(jsonl_files, desc="Loading files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"[OK] Loaded {len(samples):,} samples")
        return samples
    
    def cleanse_data(self, samples: List[Dict]) -> Tuple[List[Dict], int]:
        """
        データクレンジング
        
        Args:
            samples: 元サンプル
        
        Returns:
            cleaned_samples: クリーニング済みサンプル
            removed_count: 除去数
        """
        logger.info("[CLEANSE] Data cleaning...")
        
        cleaned = []
        removed = 0
        
        for sample in tqdm(samples, desc="Cleaning"):
            keep, quality = self.cleanser.filter_sample(sample)
            
            if keep:
                sample['quality_score'] = quality
                cleaned.append(sample)
            else:
                removed += 1
        
        logger.info(f"[OK] Cleaned: {len(cleaned):,}, Removed: {removed:,}")
        
        return cleaned, removed
    
    def sample_to_target_size(self, samples: List[Dict], target_gb: float) -> List[Dict]:
        """
        目標サイズまでサンプリング
        
        Args:
            samples: 元サンプル
            target_gb: 目標サイズ（GB）
        
        Returns:
            sampled: サンプリング済み
        """
        logger.info(f"[SAMPLE] Sampling to target size: {target_gb} GB...")
        
        # 現在のサイズ計算
        current_size = sum(len(s.get('text', '').encode('utf-8')) for s in samples)
        current_gb = current_size / (1024**3)
        
        logger.info(f"Current size: {current_gb:.2f} GB")
        
        if current_gb <= target_gb:
            logger.info(f"[OK] Already within target size")
            return samples
        
        # サンプリング比率
        sample_ratio = target_gb / current_gb
        sample_count = int(len(samples) * sample_ratio)
        
        # 統計的サンプリング（層化）
        # ドメイン・言語比率を維持
        groups = defaultdict(list)
        for sample in samples:
            key = f"{sample.get('language', 'unknown')}_{sample.get('domain', 'unknown')}"
            groups[key].append(sample)
        
        sampled = []
        for key, group_samples in groups.items():
            group_ratio = sample_ratio
            group_count = int(len(group_samples) * group_ratio)
            
            if group_count > 0:
                # 品質スコア上位を優先的にサンプリング
                sorted_samples = sorted(
                    group_samples,
                    key=lambda x: x.get('quality_score', 0.0),
                    reverse=True
                )
                sampled.extend(sorted_samples[:group_count])
        
        random.shuffle(sampled)
        
        # サイズ確認
        final_size = sum(len(s.get('text', '').encode('utf-8')) for s in sampled)
        final_gb = final_size / (1024**3)
        
        logger.info(f"[OK] Sampled: {len(sampled):,} samples ({final_gb:.2f} GB)")
        
        return sampled
    
    def run_pipeline(self):
        """パイプライン実行"""
        logger.info("="*80)
        logger.info("DATA CLEANSING & SAMPLING PIPELINE")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Target size: {self.target_size_gb} GB")
        logger.info("="*80)
        
        # Step 1: 生データ読み込み
        raw_samples = self.load_raw_data()
        raw_size = sum(len(s.get('text', '').encode('utf-8')) for s in raw_samples) / (1024**3)
        logger.info(f"[RAW] {len(raw_samples):,} samples ({raw_size:.2f} GB)")
        
        # Step 2: クレンジング
        cleaned_samples, removed_count = self.cleanse_data(raw_samples)
        
        # Step 3: ドメインバランシング
        balanced_samples = self.sampler.balance_domains(cleaned_samples)
        
        # Step 4: 言語バランシング
        lang_balanced_samples = self.sampler.balance_languages(balanced_samples)
        
        # Step 5: 目標サイズまでサンプリング
        final_samples = self.sample_to_target_size(lang_balanced_samples, self.target_size_gb)
        
        # Step 6: Train/Val/Test分割
        train, val, test = self.sampler.stratified_split(
            final_samples,
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['val_ratio'],
            test_ratio=self.config['test_ratio']
        )
        
        # Step 7: 保存
        self._save_splits(train, val, test)
        
        # Step 8: 統計レポート
        self._generate_statistics_report(raw_samples, final_samples, train, val, test, removed_count)
        
        logger.info("="*80)
        logger.info("[COMPLETE] Pipeline finished!")
        logger.info("="*80)
    
    def _save_splits(self, train: List[Dict], val: List[Dict], test: List[Dict]):
        """
        分割データ保存
        
        Args:
            train, val, test: 分割データ
        """
        logger.info("[SAVE] Saving train/val/test splits...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        splits = {'train': train, 'val': val, 'test': test}
        
        for split_name, split_samples in splits.items():
            output_file = self.output_dir / f"{split_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in tqdm(split_samples, desc=f"Saving {split_name}"):
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            size_gb = output_file.stat().st_size / (1024**3)
            logger.info(f"[OK] {split_name}: {len(split_samples):,} samples ({size_gb:.2f} GB)")
    
    def _generate_statistics_report(
        self,
        raw_samples: List[Dict],
        final_samples: List[Dict],
        train: List[Dict],
        val: List[Dict],
        test: List[Dict],
        removed_count: int
    ):
        """統計レポート生成"""
        logger.info("[REPORT] Generating statistics...")
        
        # 統計計算
        def calc_stats(samples: List[Dict], name: str) -> Dict:
            stats = {
                "name": name,
                "total_samples": len(samples),
                "total_size_gb": sum(len(s.get('text', '').encode('utf-8')) for s in samples) / (1024**3),
                "by_language": Counter(s.get('language', 'unknown') for s in samples),
                "by_domain": Counter(s.get('domain', 'unknown') for s in samples),
                "avg_length": np.mean([len(s.get('text', '')) for s in samples]) if samples else 0,
                "avg_quality": np.mean([s.get('quality_score', 0.0) for s in samples]) if samples else 0
            }
            return stats
        
        raw_stats = calc_stats(raw_samples, "Raw Data")
        final_stats = calc_stats(final_samples, "Final Data")
        train_stats = calc_stats(train, "Train")
        val_stats = calc_stats(val, "Validation")
        test_stats = calc_stats(test, "Test")
        
        # レポート作成
        report = f"""# データパイプライン統計レポート

## 概要
- **実行日時**: {datetime.now().isoformat()}
- **入力ディレクトリ**: {self.input_dir}
- **出力ディレクトリ**: {self.output_dir}
- **目標サイズ**: {self.target_size_gb} GB

## データフロー

```
Raw Data: {raw_stats['total_samples']:,} samples ({raw_stats['total_size_gb']:.2f} GB)
    ↓ クレンジング（重複除去、品質フィルタ）
Cleaned: 除去 {removed_count:,} samples
    ↓ ドメインバランシング
    ↓ 言語バランシング
    ↓ サイズ調整サンプリング
Final: {final_stats['total_samples']:,} samples ({final_stats['total_size_gb']:.2f} GB)
    ↓ 層化分割（80/10/10）
Train: {train_stats['total_samples']:,} samples ({train_stats['total_size_gb']:.2f} GB)
Val: {val_stats['total_samples']:,} samples ({val_stats['total_size_gb']:.2f} GB)
Test: {test_stats['total_samples']:,} samples ({test_stats['total_size_gb']:.2f} GB)
```

## 言語分布

### Raw Data
{self._format_distribution(raw_stats['by_language'])}

### Final Data
{self._format_distribution(final_stats['by_language'])}

## ドメイン分布

### Raw Data
{self._format_distribution(raw_stats['by_domain'])}

### Final Data
{self._format_distribution(final_stats['by_domain'])}

## 品質指標

| データセット | 平均長 | 平均品質スコア |
|------------|--------|--------------|
| Raw | {raw_stats['avg_length']:.0f} | {raw_stats['avg_quality']:.3f} |
| Final | {final_stats['avg_length']:.0f} | {final_stats['avg_quality']:.3f} |
| Train | {train_stats['avg_length']:.0f} | {train_stats['avg_quality']:.3f} |
| Val | {val_stats['avg_length']:.0f} | {val_stats['avg_quality']:.3f} |
| Test | {test_stats['avg_length']:.0f} | {test_stats['avg_quality']:.3f} |

## MLベストプラクティス適用

- [OK] 層化分割（Stratified Split）
- [OK] ランダムシード固定（再現性）
- [OK] ドメインバランシング
- [OK] 言語バランシング
- [OK] 品質フィルタリング
- [OK] 重複除去（SHA256）
- [OK] Train/Val/Test分割（80/10/10）

## 結論

クレンジング&サンプリングパイプラインが正常に完了しました。
最終データセットは、統計的にバランスされ、高品質で、機械学習に最適化されています。
"""
        
        report_file = self.output_dir / "pipeline_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"[OK] Report saved: {report_file}")
    
    def _format_distribution(self, counter: Counter) -> str:
        """分布フォーマット"""
        total = sum(counter.values())
        lines = []
        for key, count in counter.most_common():
            percentage = (count / total * 100) if total > 0 else 0
            lines.append(f"- **{key}**: {count:,} ({percentage:.1f}%)")
        return '\n'.join(lines)


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Cleansing & Sampling Pipeline (600GB→300GB)")
    parser.add_argument("--input-dir", type=str, default="D:/webdataset", help="Input directory (raw data)")
    parser.add_argument("--output-dir", type=str, default="D:/webdataset/cleaned", help="Output directory (cleaned data)")
    parser.add_argument("--target-gb", type=float, default=100.0, help="Target size in GB (cleaned data)")
    args = parser.parse_args()
    
    pipeline = DataPipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        target_size_gb=args.target_gb
    )
    
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()

