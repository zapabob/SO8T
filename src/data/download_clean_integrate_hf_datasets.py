#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T HF Datasets Download, Clean and Integrate
選択されたHFデータセットのダウンロード・クレンジング・統合システム

機能:
- 選択されたデータセットのダウンロード
- データクレンジングと正規化
- マルチリンガル統合データセット作成
- 既存データセットとの重複除去
- 品質評価と統計レポート
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import re
from tqdm import tqdm
import numpy as np

# Hugging Face imports
try:
    from datasets import load_dataset, Dataset
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError as e:
    HF_AVAILABLE = False
    print(f"Warning: HF libraries not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hf_download_integrate.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DownloadedDataset:
    """ダウンロード済みデータセット"""
    dataset_id: str
    local_path: str
    size_mb: float
    num_samples: int
    languages: List[str]
    download_time: float
    quality_score: float
    processing_status: str  # 'downloaded', 'cleaned', 'integrated'

@dataclass
class IntegratedEntry:
    """統合データセットエントリ"""
    id: str
    text: str
    language: str
    source_dataset: str
    category: str
    quality_score: float
    created_at: str

class HFDatasetDownloader:
    """HFデータセットダウンローダー"""

    def __init__(self, download_dir: str = "data/hf_downloads",
                 output_dir: str = "data/hf_integrated"):
        self.download_dir = Path(download_dir)
        self.output_dir = Path(output_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 既存ハッシュセット
        self.existing_hashes = self._load_existing_hashes()

        # 品質フィルタリング設定
        self.min_text_length = 10
        self.max_text_length = 10000
        self.nsfw_filter_strength = 'medium'  # 'low', 'medium', 'high'

        logger.info(f"Initialized HFDatasetDownloader with download_dir: {download_dir}")

    def _load_existing_hashes(self) -> set:
        """既存データセットのハッシュをロード"""
        hashes = set()
        existing_dirs = [Path("data/hf_multilingual"), Path("data/nobel_fields_cot/cleansed")]

        for data_dir in existing_dirs:
            if data_dir.exists():
                for jsonl_file in data_dir.rglob("*.jsonl"):
                    try:
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        data = json.loads(line.strip())
                                        text = data.get('text', data.get('content', ''))
                                        if text:
                                            text_hash = hashlib.md5(text.encode()).hexdigest()
                                            hashes.add(text_hash)
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        logger.warning(f"Failed to process {jsonl_file}: {e}")

        logger.info(f"Loaded {len(hashes)} existing text hashes")
        return hashes

    def download_selected_datasets(self, selection_file: str) -> List[DownloadedDataset]:
        """選択されたデータセットをダウンロード"""
        logger.info(f"Downloading datasets from {selection_file}")

        # 選択ファイル読み込み
        try:
            with open(selection_file, 'r', encoding='utf-8') as f:
                selection_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load selection file: {e}")
            return []

        selected_datasets = selection_data.get('datasets', [])
        downloaded_datasets = []

        for dataset_info in tqdm(selected_datasets, desc="Downloading datasets"):
            dataset_id = dataset_info['id']

            try:
                downloaded = self._download_single_dataset(dataset_id, dataset_info)
                if downloaded:
                    downloaded_datasets.append(downloaded)
                    logger.info(f"Downloaded {dataset_id}: {downloaded.num_samples} samples")
                else:
                    logger.warning(f"Failed to download {dataset_id}")

            except Exception as e:
                logger.error(f"Error downloading {dataset_id}: {e}")
                continue

        logger.info(f"Downloaded {len(downloaded_datasets)} datasets successfully")
        return downloaded_datasets

    def _download_single_dataset(self, dataset_id: str, dataset_info: Dict) -> Optional[DownloadedDataset]:
        """単一データセットのダウンロード"""
        import time
        start_time = time.time()

        try:
            # データセットダウンロード
            dataset = load_dataset(dataset_id, split='train', trust_remote_code=True)

            # サンプル数制限（メモリ節約）
            max_samples = 50000
            if len(dataset) > max_samples:
                indices = np.random.choice(len(dataset), max_samples, replace=False)
                dataset = dataset.select(indices)

            # ローカル保存
            local_path = self.download_dir / f"{dataset_id.replace('/', '_')}.jsonl"
            with open(local_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    json.dump(dict(item), f, ensure_ascii=False)
                    f.write('\n')

            # サイズ計算
            size_mb = local_path.stat().st_size / (1024 * 1024)
            download_time = time.time() - start_time

            downloaded = DownloadedDataset(
                dataset_id=dataset_id,
                local_path=str(local_path),
                size_mb=size_mb,
                num_samples=len(dataset),
                languages=dataset_info.get('languages', []),
                download_time=download_time,
                quality_score=dataset_info.get('quality_score', 0.5),
                processing_status='downloaded'
            )

            return downloaded

        except Exception as e:
            logger.error(f"Failed to download {dataset_id}: {e}")
            return None

    def clean_and_integrate_datasets(self, downloaded_datasets: List[DownloadedDataset]) -> Dict[str, Any]:
        """データセットのクレンジングと統合"""
        logger.info("Cleaning and integrating datasets...")

        integrated_entries = []
        processing_stats = {
            'total_processed': 0,
            'duplicates_removed': 0,
            'quality_filtered': 0,
            'nsfw_filtered': 0,
            'language_detected': {},
            'category_distribution': {}
        }

        for downloaded in tqdm(downloaded_datasets, desc="Processing datasets"):
            try:
                entries = self._process_single_downloaded_dataset(downloaded, processing_stats)
                integrated_entries.extend(entries)
                downloaded.processing_status = 'processed'

            except Exception as e:
                logger.error(f"Failed to process {downloaded.dataset_id}: {e}")
                continue

        # 最終統合
        integrated_dataset = self._create_integrated_dataset(integrated_entries, processing_stats)

        logger.info(f"Integration completed: {len(integrated_entries)} entries")
        return integrated_dataset

    def _process_single_downloaded_dataset(self, downloaded: DownloadedDataset,
                                         processing_stats: Dict) -> List[IntegratedEntry]:
        """ダウンロード済みデータセットの処理"""
        entries = []

        try:
            with open(downloaded.local_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            item = json.loads(line.strip())
                            processing_stats['total_processed'] += 1

                            # テキスト抽出とクレンジング
                            cleaned_text = self._clean_text_item(item)
                            if not cleaned_text:
                                continue

                            # 長さチェック
                            if not (self.min_text_length <= len(cleaned_text) <= self.max_text_length):
                                processing_stats['quality_filtered'] += 1
                                continue

                            # 重複チェック
                            text_hash = hashlib.md5(cleaned_text.encode()).hexdigest()
                            if text_hash in self.existing_hashes:
                                processing_stats['duplicates_removed'] += 1
                                continue

                            # NSFWフィルタリング
                            if self._contains_nsfw(cleaned_text):
                                processing_stats['nsfw_filtered'] += 1
                                continue

                            # 言語検出
                            language = self._detect_language(cleaned_text, downloaded.languages)

                            # カテゴリ判定
                            category = self._determine_category(cleaned_text, downloaded.dataset_id)

                            # エントリ作成
                            entry = IntegratedEntry(
                                id=f"{downloaded.dataset_id}_{line_num}_{hashlib.md5(cleaned_text.encode()).hexdigest()[:8]}",
                                text=cleaned_text,
                                language=language,
                                source_dataset=downloaded.dataset_id,
                                category=category,
                                quality_score=downloaded.quality_score,
                                created_at=datetime.now().isoformat()
                            )

                            entries.append(entry)

                            # 統計更新
                            processing_stats['language_detected'][language] = processing_stats['language_detected'].get(language, 0) + 1
                            processing_stats['category_distribution'][category] = processing_stats['category_distribution'].get(category, 0) + 1

                            # ハッシュ追加（今後の重複チェック用）
                            self.existing_hashes.add(text_hash)

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Failed to process file {downloaded.local_path}: {e}")

        return entries

    def _clean_text_item(self, item: Dict) -> str:
        """テキストアイテムのクレンジング"""
        # 一般的なテキストフィールド
        text_fields = ['text', 'content', 'instruction', 'response', 'input', 'output',
                      'question', 'answer', 'context', 'dialogue', 'utterance']

        text = ""
        for field in text_fields:
            if field in item and item[field]:
                candidate = str(item[field]).strip()
                if len(candidate) > len(text):
                    text = candidate

        if not text:
            return ""

        # HTMLタグ除去
        text = re.sub(r'<[^>]+>', '', text)

        # URL除去
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # 連続空白の正規化
        text = re.sub(r'\s+', ' ', text)

        # 特殊文字の正規化
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        return text.strip()

    def _contains_nsfw(self, text: str) -> bool:
        """NSFWコンテンツ判定"""
        if self.nsfw_filter_strength == 'low':
            return False  # フィルタリングなし

        text_lower = text.lower()

        # 基本的なNSFWキーワードチェック
        nsfw_count = sum(1 for keyword in ['sex', 'porn', 'nude', 'fuck', 'shit', 'damn']
                        if keyword in text_lower)

        if self.nsfw_filter_strength == 'high':
            # 厳格フィルタリング
            return nsfw_count > 0
        else:
            # 中程度フィルタリング（文脈依存）
            return nsfw_count > 2

    def _detect_language(self, text: str, dataset_languages: List[str]) -> str:
        """言語検出"""
        # データセット情報からの言語
        if dataset_languages:
            if 'ja' in dataset_languages or 'japanese' in str(dataset_languages).lower():
                return 'ja'
            elif 'en' in dataset_languages or 'english' in str(dataset_languages).lower():
                return 'en'

        # テキストベースの検出
        if self._contains_japanese(text):
            return 'ja'
        else:
            return 'en'  # デフォルト

    def _contains_japanese(self, text: str) -> bool:
        """日本語を含むかチェック"""
        japanese_pattern = r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]'
        return bool(re.search(japanese_pattern, text))

    def _determine_category(self, text: str, dataset_id: str) -> str:
        """カテゴリ判定"""
        text_lower = text.lower()

        # データセットIDベースの判定
        if 'question' in dataset_id.lower() or 'qa' in dataset_id.lower():
            return 'question_answering'
        elif 'dialog' in dataset_id.lower() or 'conversation' in dataset_id.lower():
            return 'dialogue'
        elif 'science' in dataset_id.lower() or 'sciq' in dataset_id.lower():
            return 'science'
        elif 'math' in dataset_id.lower():
            return 'mathematics'

        # テキスト内容ベースの判定
        if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question_answering'
        elif any(word in text_lower for word in ['theorem', 'proof', 'equation', 'formula']):
            return 'mathematics'
        elif any(word in text_lower for word in ['experiment', 'hypothesis', 'theory']):
            return 'science'
        else:
            return 'general'

    def _create_integrated_dataset(self, entries: List[IntegratedEntry],
                                 processing_stats: Dict) -> Dict[str, Any]:
        """統合データセット作成"""
        logger.info("Creating integrated dataset...")

        # カテゴリ別ファイル作成
        categories = {}
        for entry in entries:
            cat = entry.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry)

        # 各カテゴリを別ファイルに保存
        integrated_files = {}
        for category, cat_entries in categories.items():
            category_file = self.output_dir / f"hf_integrated_{category}.jsonl"
            with open(category_file, 'w', encoding='utf-8') as f:
                for entry in cat_entries:
                    json.dump(asdict(entry), f, ensure_ascii=False, indent=None)
                    f.write('\n')

            integrated_files[category] = {
                'file_path': str(category_file),
                'count': len(cat_entries)
            }

        # 統合統計
        integration_result = {
            'timestamp': datetime.now().isoformat(),
            'total_entries': len(entries),
            'categories': integrated_files,
            'processing_stats': processing_stats,
            'quality_metrics': self._calculate_integration_quality(entries),
            'language_distribution': dict(sorted(processing_stats['language_detected'].items(),
                                               key=lambda x: x[1], reverse=True)),
            'category_distribution': dict(sorted(processing_stats['category_distribution'].items(),
                                               key=lambda x: x[1], reverse=True))
        }

        # 統計ファイル保存
        stats_file = self.output_dir / "integration_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(integration_result, f, ensure_ascii=False, indent=2)

        logger.info(f"Integrated dataset created with {len(entries)} entries")
        return integration_result

    def _calculate_integration_quality(self, entries: List[IntegratedEntry]) -> Dict[str, Any]:
        """統合品質計算"""
        if not entries:
            return {}

        quality_scores = [entry.quality_score for entry in entries]
        text_lengths = [len(entry.text) for entry in entries]

        return {
            'avg_quality_score': np.mean(quality_scores),
            'quality_score_std': np.std(quality_scores),
            'avg_text_length': np.mean(text_lengths),
            'text_length_std': np.std(text_lengths),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths),
            'unique_sources': len(set(entry.source_dataset for entry in entries)),
            'language_diversity': len(set(entry.language for entry in entries))
        }

def main():
    """メイン実行関数"""
    print("SO8T HF Datasets Download, Clean and Integrate")
    print("=" * 55)

    if not HF_AVAILABLE:
        print("ERROR: Hugging Face libraries not available")
        print("Install with: pip install datasets huggingface_hub")
        return

    downloader = HFDatasetDownloader()

    try:
        # 選択ファイルからデータセットダウンロード
        selection_file = "data/top_hf_selected/top_hf_datasets_selected.json"

        if not Path(selection_file).exists():
            print(f"ERROR: Selection file not found: {selection_file}")
            return

        print("\n[1/2] Downloading selected datasets...")
        downloaded_datasets = downloader.download_selected_datasets(selection_file)

        if not downloaded_datasets:
            print("No datasets downloaded")
            return

        print("\n[2/2] Cleaning and integrating datasets...")
        integration_result = downloader.clean_and_integrate_datasets(downloaded_datasets)

        # 結果表示
        print("\n" + "="*60)
        print("INTEGRATION RESULTS SUMMARY")
        print("="*60)
        print(f"Total Entries: {integration_result['total_entries']}")
        print(f"Categories: {len(integration_result['categories'])}")
        print(f"Language Diversity: {integration_result['quality_metrics']['language_diversity']}")
        print(f"Average Quality Score: {integration_result['quality_metrics']['avg_quality_score']:.3f}")

        print(f"\nProcessing Statistics:")
        stats = integration_result['processing_stats']
        print(f"- Total Processed: {stats['total_processed']}")
        print(f"- Duplicates Removed: {stats['duplicates_removed']}")
        print(f"- Quality Filtered: {stats['quality_filtered']}")
        print(f"- NSFW Filtered: {stats['nsfw_filtered']}")

        print(f"\nLanguage Distribution:")
        for lang, count in list(integration_result['language_distribution'].items())[:5]:
            print(f"- {lang}: {count}")

        print(f"\nCategory Distribution:")
        for cat, count in list(integration_result['category_distribution'].items())[:5]:
            print(f"- {cat}: {count}")

        print(f"\n[DIR] Results saved to: {downloader.output_dir}")

        # 音声通知
        try:
            import winsound
            winsound.Beep(1500, 600)  # 成功音
            print("[AUDIO] Dataset integration completed successfully")
        except ImportError:
            print("[AUDIO] Dataset integration completed (winsound not available)")

    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()
