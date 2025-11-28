#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFaceデータセット統合スクリプト

ダウンロード済みの全HFデータセットを統合して、
SO8T学習用の統一データセットを作成する

Usage:
    python scripts/data/integrate_hf_datasets.py
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from tqdm import tqdm
import random
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetIntegrator:
    """HFデータセット統合クラス"""

    def __init__(self):
        self.stats = defaultdict(int)
        logger.info("Dataset Integrator initialized")

    def scan_datasets(self, base_dir: Path) -> List[Path]:
        """データセットディレクトリをスキャンして有効なファイルを収集"""
        valid_files = []

        if not base_dir.exists():
            logger.error(f"Base directory does not exist: {base_dir}")
            return valid_files

        # データセットディレクトリを走査
        dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        for dataset_dir in tqdm(dataset_dirs, desc="Scanning datasets"):
            # JSON/JSONLファイルを検索
            json_files = list(dataset_dir.glob("**/*.json"))
            jsonl_files = list(dataset_dir.glob("**/*.jsonl"))

            all_files = json_files + jsonl_files

            for file_path in all_files:
                # サイズチェック（空ファイルは除外）
                if file_path.stat().st_size > 1024:  # 1KB以上
                    valid_files.append(file_path)
                    self.stats['total_files'] += 1
                else:
                    self.stats['skipped_empty'] += 1

        logger.info(f"Found {len(valid_files)} valid dataset files")
        return valid_files

    def normalize_sample(self, sample: Dict, source_file: str) -> Optional[Dict]:
        """サンプルを統一フォーマットに正規化"""
        try:
            # 様々なフォーマットに対応
            normalized = {
                "source": source_file,
                "dataset": Path(source_file).parent.name
            }

            # テキスト抽出
            text = self._extract_text(sample)
            if not text or len(text.strip()) < 10:
                self.stats['skipped_short_text'] += 1
                return None

            normalized["text"] = text.strip()

            # プロンプト/レスポンスの抽出（可能な場合）
            prompt = self._extract_prompt(sample)
            chosen = self._extract_chosen(sample)
            rejected = self._extract_rejected(sample)

            if prompt:
                normalized["prompt"] = prompt
            if chosen:
                normalized["chosen"] = chosen
            if rejected:
                normalized["rejected"] = rejected

            # 言語判定
            normalized["language"] = self._detect_language(text)

            return normalized

        except Exception as e:
            logger.warning(f"Failed to normalize sample from {source_file}: {e}")
            self.stats['skipped_error'] += 1
            return None

    def _extract_text(self, sample: Dict) -> str:
        """テキストを抽出"""
        # 様々なフィールド名に対応
        text_fields = ['text', 'content', 'message', 'response', 'output', 'answer']

        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                return sample[field]

        # conversations形式の場合
        if 'conversations' in sample:
            texts = []
            for conv in sample['conversations']:
                if isinstance(conv, dict) and 'value' in conv:
                    texts.append(conv['value'])
            if texts:
                return ' '.join(texts)

        # instruction/response形式
        if 'instruction' in sample and 'response' in sample:
            return f"{sample['instruction']} {sample['response']}"

        # 最初の文字列値を使用
        for key, value in sample.items():
            if isinstance(value, str) and len(value.strip()) > 10:
                return value

        return ""

    def _extract_prompt(self, sample: Dict) -> Optional[str]:
        """プロンプトを抽出"""
        prompt_fields = ['instruction', 'input', 'prompt', 'question', 'query']
        for field in prompt_fields:
            if field in sample and isinstance(sample[field], str):
                return sample[field].strip()
        return None

    def _extract_chosen(self, sample: Dict) -> Optional[str]:
        """chosenレスポンスを抽出"""
        if 'chosen' in sample:
            return sample['chosen']
        if 'response' in sample:
            return sample['response']
        if 'output' in sample:
            return sample['output']
        return None

    def _extract_rejected(self, sample: Dict) -> Optional[str]:
        """rejectedレスポンスを抽出"""
        if 'rejected' in sample:
            return sample['rejected']
        return None

    def _detect_language(self, text: str) -> str:
        """言語を判定"""
        # 日本語文字の割合で判定
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))
        total_chars = len(re.sub(r'\s+', '', text))

        if total_chars == 0:
            return "unknown"

        japanese_ratio = japanese_chars / total_chars

        if japanese_ratio > 0.1:
            return "ja"
        else:
            return "en"

    def load_and_normalize_file(self, file_path: Path) -> List[Dict]:
        """ファイルを読み込んで正規化"""
        normalized_samples = []

        try:
            # JSONLファイル
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            sample = json.loads(line)
                            normalized = self.normalize_sample(sample, str(file_path))
                            if normalized:
                                normalized_samples.append(normalized)
                                self.stats['processed_samples'] += 1

                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON at line {line_num} in {file_path}: {e}")
                            self.stats['skipped_invalid_json'] += 1
                            continue

            # JSONファイル
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)

                        # リスト形式の場合
                        if isinstance(data, list):
                            for sample in data:
                                normalized = self.normalize_sample(sample, str(file_path))
                                if normalized:
                                    normalized_samples.append(normalized)
                                    self.stats['processed_samples'] += 1

                        # 辞書形式の場合
                        elif isinstance(data, dict):
                            # データがネストされている場合
                            if 'data' in data and isinstance(data['data'], list):
                                for sample in data['data']:
                                    normalized = self.normalize_sample(sample, str(file_path))
                                    if normalized:
                                        normalized_samples.append(normalized)
                                        self.stats['processed_samples'] += 1
                            else:
                                # 単一サンプルとして扱う
                                normalized = self.normalize_sample(data, str(file_path))
                                if normalized:
                                    normalized_samples.append(normalized)
                                    self.stats['processed_samples'] += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON file {file_path}: {e}")
                        self.stats['skipped_invalid_json'] += 1

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            self.stats['skipped_error'] += 1

        return normalized_samples

    def integrate_datasets(self, base_dir: Path, output_file: Path, max_samples: Optional[int] = None) -> Dict[str, int]:
        """データセット統合を実行"""
        logger.info("="*80)
        logger.info("HF Dataset Integration")
        logger.info("="*80)

        # データセットファイル収集
        logger.info("Scanning dataset files...")
        dataset_files = self.scan_datasets(base_dir)

        if not dataset_files:
            logger.error("No valid dataset files found")
            return {}

        # 全サンプル統合
        all_samples = []
        logger.info("Loading and normalizing samples...")

        for file_path in tqdm(dataset_files, desc="Processing files"):
            samples = self.load_and_normalize_file(file_path)
            all_samples.extend(samples)

            # メモリ管理のため定期的にシャッフル
            if len(all_samples) > 100000:
                random.shuffle(all_samples)

        logger.info(f"Total normalized samples: {len(all_samples)}")

        # サンプル数制限
        if max_samples and len(all_samples) > max_samples:
            logger.info(f"Limiting to {max_samples} samples...")
            random.shuffle(all_samples)
            all_samples = all_samples[:max_samples]

        # 統計出力
        self._print_stats(all_samples)

        # 保存
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving integrated dataset to {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(all_samples, desc="Saving samples"):
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        # 最終統計
        final_stats = {
            'total_files_processed': len(dataset_files),
            'total_samples': len(all_samples),
            'output_file': str(output_file),
            'file_size_mb': round(output_file.stat().st_size / 1024 / 1024, 2)
        }

        logger.info("Integration completed!")
        logger.info(f"Output: {output_file} ({final_stats['file_size_mb']} MB)")
        logger.info(f"Samples: {final_stats['total_samples']}")

        return final_stats

    def _print_stats(self, samples: List[Dict]):
        """統計情報を出力"""
        logger.info("Dataset Statistics:")

        # 言語分布
        languages = Counter(s.get('language', 'unknown') for s in samples)
        logger.info(f"  Languages: {dict(languages)}")

        # データセット分布（トップ10）
        datasets = Counter(s.get('dataset', 'unknown') for s in samples)
        logger.info("  Top datasets:")
        for dataset, count in datasets.most_common(10):
            logger.info(f"    {dataset}: {count}")

        # フィールド存在率
        total = len(samples)
        has_prompt = sum(1 for s in samples if 'prompt' in s)
        has_chosen = sum(1 for s in samples if 'chosen' in s)
        has_rejected = sum(1 for s in samples if 'rejected' in s)

        logger.info("  Field coverage:")
        logger.info(f"    prompt: {has_prompt}/{total} ({has_prompt/total*100:.1f}%)")
        logger.info(f"    chosen: {has_chosen}/{total} ({has_chosen/total*100:.1f}%)")
        logger.info(f"    rejected: {has_rejected}/{total} ({has_rejected/total*100:.1f}%)")

        # テキスト長統計
        text_lengths = [len(s.get('text', '')) for s in samples]
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            max_length = max(text_lengths)
            min_length = min(text_lengths)
            logger.info("  Text length stats:")
            logger.info(f"    Average: {avg_length:.0f} chars")
            logger.info(f"    Min: {min_length} chars")
            logger.info(f"    Max: {max_length} chars")


def main():
    parser = argparse.ArgumentParser(description="Integrate HF datasets")
    parser.add_argument(
        '--input-dir',
        type=str,
        default='D:/webdataset/datasets',
        help='Input dataset directory'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='D:/webdataset/integrated_dataset.jsonl',
        help='Output integrated dataset file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to include'
    )

    args = parser.parse_args()

    integrator = DatasetIntegrator()
    stats = integrator.integrate_datasets(
        Path(args.input_dir),
        Path(args.output_file),
        args.max_samples
    )

    # 統計をJSONで保存
    stats_file = Path(args.output_file).with_suffix('.stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Statistics saved to {stats_file}")


if __name__ == '__main__':
    main()
