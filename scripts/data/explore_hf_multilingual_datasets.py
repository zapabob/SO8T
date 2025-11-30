#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Hugging Face Dataset Explorer
日英マルチリンガルデータセットとNSFWデータセットの調査・ダウンロード・統合システム

機能:
- Hugging Face Hubからの日英マルチリンガルデータセット調査
- NSFWデータセットの特定と安全フィルタリング
- 既存データセットとの重複チェック
- データダウンロードとクレンジング
- マルチリンガル統合データセット作成
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import re
from tqdm import tqdm
import numpy as np
import pandas as pd

# Hugging Face imports
try:
    from datasets import load_dataset, Dataset
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError as e:
    HF_AVAILABLE = False
    print(f"Warning: datasets library not available: {e}")
    print("Install with: pip install datasets huggingface_hub")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hf_dataset_explorer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HFDatasetInfo:
    """Hugging Faceデータセット情報"""
    id: str
    name: str
    description: str
    tags: List[str]
    languages: List[str]
    size_mb: Optional[float]
    downloads: int
    likes: int
    is_multilingual: bool
    contains_nsfw: bool
    license: Optional[str]
    author: str
    last_modified: str
    quality_score: float

@dataclass
class MultilingualDatasetEntry:
    """マルチリンガルデータセットエントリ"""
    id: str
    text: str
    language: str  # 'en', 'ja', 'zh', etc.
    source_dataset: str
    category: str  # 'general', 'nsfw', 'technical', 'conversational'
    quality_score: float
    created_at: str

class HFDatasetExplorer:
    """Hugging Faceデータセット調査器"""

    def __init__(self, output_dir: str = "data/hf_multilingual"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 既存データセットのハッシュセット（重複チェック用）
        self.existing_hashes = self._load_existing_dataset_hashes()

        # 対象言語
        self.target_languages = ['en', 'ja', 'zh', 'ko', 'fr', 'de']

        # NSFW関連キーワード（検出用）
        self.nsfw_keywords = {
            'sexual', 'porn', 'nude', 'erotic', 'adult', 'xxx', 'sex',
            'naked', 'fuck', 'shit', 'damn', 'bitch', 'asshole', 'cunt',
            'dick', 'pussy', 'tits', 'boobs', 'cock', 'cum', 'rape',
            'violence', 'murder', 'drugs', 'suicide', 'self-harm'
        }

        # マルチリンガルデータセットの候補
        self.multilingual_candidates = [
            # 日英会話データセット
            'microsoft/DialoGPT-medium',
            'facebook/blenderbot-400M-distill',
            'rinna/japanese-gpt-1b',
            'izumi-lab/llm-japanese-dataset',
            'microsoft/DialoGPT-small',
            'EleutherAI/gpt-neo-1.3B',  # 英語中心だがマルチリンガル拡張可能

            # 翻訳データセット
            'Helsinki-NLP/opus-100',
            'Helsinki-NLP/opus-mt-en-ja',
            'Helsinki-NLP/opus-mt-ja-en',
            'facebook/flores',

            # マルチリンガルQA
            'google-research-datasets/natural_questions',
            'stanfordnlp/coqa',
            'rajpurkar/squad_v2',

            # コードデータセット
            'code_search_net',
            'bigcode/the-stack',

            # 科学・技術データセット
            'scientific_papers',
            'arxiv_dataset',
            'allenai/scicite',

            # ニュース・一般データセット
            'cc_news',
            'c4',
            'wikipedia',

            # NSFW関連（検出・拒否学習用）
            'japanese-nsfw-text-dataset',  # 存在する場合
            'nsfw-text-classification',
            'toxicity-dataset',
            'hate-speech-dataset'
        ]

        # NSFWデータセットの特別候補
        self.nsfw_dataset_candidates = [
            'jigsaw/toxicity-prediction',
            'facebook/roberta-hate-speech-dynabench-r4-target',
            'unitary/toxic-bert',
            'microsoft/DialoGPT-medium',  # 会話データとして使用
            'daily_dialog',
            'empathetic_dialogues',
            # 日本語NSFWデータセット（安全学習用）
            'nlp-thedeep/japanese-nsfw-text-detection',
            'studio-ousia/luke-japanese-large-lite',  # 汎用だがNSFW分類に使用可能
        ]

        logger.info(f"Initialized HFDatasetExplorer with output directory: {output_dir}")

    def _load_existing_dataset_hashes(self) -> Set[str]:
        """既存データセットのテキストハッシュをロード"""
        hashes = set()

        # dataディレクトリ内の全jsonlファイルをスキャン
        data_dir = Path("data")
        if data_dir.exists():
            for jsonl_file in data_dir.rglob("*.jsonl"):
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line.strip())
                                    # テキストフィールドのハッシュを生成
                                    for field in ['text', 'content', 'instruction', 'response', 'input', 'output']:
                                        if field in data and data[field]:
                                            text_hash = hashlib.md5(str(data[field]).encode()).hexdigest()
                                            hashes.add(text_hash)
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.warning(f"Failed to process {jsonl_file}: {e}")

        logger.info(f"Loaded {len(hashes)} existing text hashes for deduplication")
        return hashes

    def explore_datasets(self, max_datasets: int = 100) -> List[HFDatasetInfo]:
        """Hugging Face Hubからデータセットを探索"""
        if not HF_AVAILABLE:
            logger.error("Hugging Face datasets library not available")
            return []

        logger.info(f"Exploring Hugging Face datasets (max: {max_datasets})...")

        api = HfApi()
        dataset_infos = []

        try:
            # データセットリストを取得
            datasets = []
            api = HfApi()
            dataset_list = api.list_datasets(limit=max_datasets, full=True)
            for ds in dataset_list:
                datasets.append(ds)

            for dataset in tqdm(datasets, desc="Exploring datasets"):
                try:
                    # データセット情報を取得
                    info = api.dataset_info(dataset.id, timeout=10)

                    # 言語情報の抽出
                    languages = []
                    if hasattr(info, 'language'):
                        if isinstance(info.language, list):
                            languages = info.language
                        elif isinstance(info.language, str):
                            languages = [info.language]

                    # タグから言語情報を追加
                    if hasattr(info, 'tags') and info.tags:
                        for tag in info.tags:
                            if tag.startswith('language:'):
                                lang = tag.split(':')[1]
                                if lang not in languages:
                                    languages.append(lang)

                    # マルチリンガル判定
                    is_multilingual = len([lang for lang in languages if lang in self.target_languages]) >= 2

                    # NSFW判定
                    contains_nsfw = self._check_nsfw_content(info)

                    # サイズ計算
                    size_mb = None
                    if hasattr(info, 'size_in_bytes') and info.size_in_bytes:
                        size_mb = info.size_in_bytes / (1024 * 1024)

                    # 品質スコア計算
                    quality_score = self._calculate_dataset_quality_score(info, languages, is_multilingual, contains_nsfw)

                    dataset_info = HFDatasetInfo(
                        id=dataset.id,
                        name=getattr(info, 'name', dataset.id.split('/')[-1]),
                        description=getattr(info, 'description', ''),
                        tags=getattr(info, 'tags', []),
                        languages=languages,
                        size_mb=size_mb,
                        downloads=getattr(info, 'downloads', 0),
                        likes=getattr(info, 'likes', 0),
                        is_multilingual=is_multilingual,
                        contains_nsfw=contains_nsfw,
                        license=getattr(info, 'license', None),
                        author=dataset.id.split('/')[0],
                        last_modified=getattr(info, 'last_modified', datetime.now().isoformat()),
                        quality_score=quality_score
                    )

                    dataset_infos.append(dataset_info)

                except Exception as e:
                    logger.warning(f"Failed to process dataset {dataset.id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to explore datasets: {e}")

        # 品質スコアでソート
        dataset_infos.sort(key=lambda x: x.quality_score, reverse=True)

        logger.info(f"Explored {len(dataset_infos)} datasets")
        return dataset_infos

    def _check_nsfw_content(self, dataset_info) -> bool:
        """NSFWコンテンツのチェック"""
        text_to_check = ""

        # 説明文
        if hasattr(dataset_info, 'description') and dataset_info.description:
            text_to_check += dataset_info.description.lower() + " "

        # タグ
        if hasattr(dataset_info, 'tags') and dataset_info.tags:
            text_to_check += " ".join(dataset_info.tags).lower() + " "

        # データセット名
        if hasattr(dataset_info, 'id'):
            text_to_check += dataset_info.id.lower()

        # NSFWキーワードチェック
        for keyword in self.nsfw_keywords:
            if keyword in text_to_check:
                return True

        # 特定のNSFW関連タグ
        nsfw_tags = ['nsfw', 'adult', 'porn', 'sexual', 'violence', 'hate-speech', 'toxicity']
        if hasattr(dataset_info, 'tags') and dataset_info.tags:
            for tag in dataset_info.tags:
                if any(nsfw_tag in tag.lower() for nsfw_tag in nsfw_tags):
                    return True

        return False

    def _calculate_dataset_quality_score(self, dataset_info, languages: List[str],
                                       is_multilingual: bool, contains_nsfw: bool) -> float:
        """データセット品質スコア計算"""
        score = 0.5  # ベーススコア

        # ダウンロード数（人気度）
        downloads = getattr(dataset_info, 'downloads', 0)
        if downloads > 10000:
            score += 0.3
        elif downloads > 1000:
            score += 0.2
        elif downloads > 100:
            score += 0.1

        # いいね数
        likes = getattr(dataset_info, 'likes', 0)
        if likes > 100:
            score += 0.2
        elif likes > 10:
            score += 0.1

        # 対象言語数
        target_lang_count = len([lang for lang in languages if lang in self.target_languages])
        score += min(target_lang_count * 0.1, 0.3)

        # マルチリンガルボーナス
        if is_multilingual:
            score += 0.2

        # ライセンス（オープンライセンス優先）
        license = getattr(dataset_info, 'license', '')
        open_licenses = ['apache-2.0', 'mit', 'bsd', 'cc-by', 'cc0', 'public-domain']
        if any(open_lic in license.lower() for open_lic in open_licenses):
            score += 0.1

        # NSFWデータセットの特別処理（検出学習用として価値あり）
        if contains_nsfw:
            score += 0.1  # 安全学習用として価値あり

        # サイズペナルティ（大きすぎるデータセット）
        size_mb = getattr(dataset_info, 'size_in_bytes', 0) / (1024 * 1024) if hasattr(dataset_info, 'size_in_bytes') else 0
        if size_mb > 10000:  # 10GB以上
            score -= 0.2
        elif size_mb > 1000:  # 1GB以上
            score -= 0.1

        return max(0.0, min(1.0, score))

    def select_top_datasets(self, dataset_infos: List[HFDatasetInfo],
                          target_multilingual: int = 10, target_nsfw: int = 5) -> Dict[str, List[HFDatasetInfo]]:
        """最適なデータセットを選択"""
        logger.info("Selecting top datasets...")

        # マルチリンガルデータセット
        multilingual_datasets = [
            info for info in dataset_infos
            if info.is_multilingual and not info.contains_nsfw
        ][:target_multilingual]

        # NSFWデータセット（安全学習用）
        nsfw_datasets = [
            info for info in dataset_infos
            if info.contains_nsfw
        ][:target_nsfw]

        # 日本語特化データセット
        japanese_datasets = [
            info for info in dataset_infos
            if ('ja' in info.languages or 'japanese' in str(info.tags).lower()) and not info.contains_nsfw
        ][:5]

        # 英語データセット
        english_datasets = [
            info for info in dataset_infos
            if ('en' in info.languages or 'english' in str(info.tags).lower()) and not info.contains_nsfw
        ][:5]

        selected = {
            'multilingual': multilingual_datasets,
            'nsfw': nsfw_datasets,
            'japanese': japanese_datasets,
            'english': english_datasets
        }

        logger.info(f"Selected datasets - Multilingual: {len(multilingual_datasets)}, NSFW: {len(nsfw_datasets)}, Japanese: {len(japanese_datasets)}, English: {len(english_datasets)}")

        return selected

    def download_and_process_datasets(self, selected_datasets: Dict[str, List[HFDatasetInfo]]) -> Dict[str, List[MultilingualDatasetEntry]]:
        """選択されたデータセットをダウンロードして処理"""
        logger.info("Downloading and processing selected datasets...")

        processed_data = {
            'multilingual': [],
            'nsfw': [],
            'japanese': [],
            'english': []
        }

        total_datasets = sum(len(datasets) for datasets in selected_datasets.values())

        with tqdm(total=total_datasets, desc="Processing datasets") as pbar:
            for category, datasets in selected_datasets.items():
                for dataset_info in datasets:
                    try:
                        entries = self._process_single_dataset(dataset_info, category)
                        processed_data[category].extend(entries)
                        logger.info(f"Processed {dataset_info.id}: {len(entries)} entries")
                    except Exception as e:
                        logger.error(f"Failed to process {dataset_info.id}: {e}")
                    pbar.update(1)

        # 重複除去
        for category in processed_data:
            processed_data[category] = self._remove_duplicates(processed_data[category])
            logger.info(f"After deduplication - {category}: {len(processed_data[category])} entries")

        return processed_data

    def _process_single_dataset(self, dataset_info: HFDatasetInfo, category: str) -> List[MultilingualDatasetEntry]:
        """単一データセットの処理"""
        entries = []

        try:
            # データセット読み込み
            dataset = load_dataset(dataset_info.id, split='train', trust_remote_code=True)

            # サンプル数制限
            max_samples = 10000 if category == 'nsfw' else 50000
            if len(dataset) > max_samples:
                # ランダムサンプリング
                indices = np.random.choice(len(dataset), max_samples, replace=False)
                dataset = dataset.select(indices)

            for item in dataset:
                # テキスト抽出
                text = self._extract_text_from_item(item, dataset_info)

                if not text or len(text.strip()) < 10:
                    continue

                # 重複チェック
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self.existing_hashes:
                    continue

                # 言語検出
                language = self._detect_language(text, dataset_info)

                # NSFWフィルタリング（必要に応じて）
                if category == 'nsfw':
                    # NSFWデータは安全学習用としてそのまま使用
                    pass
                else:
                    # 一般データから過度なNSFWコンテンツを除去
                    if self._contains_nsfw_content(text):
                        continue

                # エントリ作成
                entry = MultilingualDatasetEntry(
                    id=f"{dataset_info.id}_{hashlib.md5(text.encode()).hexdigest()[:16]}",
                    text=text.strip(),
                    language=language,
                    source_dataset=dataset_info.id,
                    category=category,
                    quality_score=dataset_info.quality_score,
                    created_at=datetime.now().isoformat()
                )

                entries.append(entry)

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_info.id}: {e}")
            # フォールバック: 手動データセット作成
            entries = self._create_fallback_entries(dataset_info, category)

        return entries

    def _extract_text_from_item(self, item: Dict, dataset_info: HFDatasetInfo) -> str:
        """データセットアイテムからテキストを抽出"""
        # 一般的なフィールド名
        text_fields = ['text', 'content', 'instruction', 'response', 'input', 'output',
                      'question', 'answer', 'context', 'dialogue', 'utterance']

        for field in text_fields:
            if field in item and item[field]:
                text = str(item[field])
                if len(text.strip()) > 10:
                    return text

        # 構造化データの場合
        if isinstance(item, dict):
            # 最初の非空テキストフィールドを使用
            for key, value in item.items():
                if isinstance(value, str) and len(value.strip()) > 10:
                    return value.strip()

        # フォールバック
        return str(item) if item else ""

    def _detect_language(self, text: str, dataset_info: HFDatasetInfo) -> str:
        """言語検出"""
        # データセット情報からの言語
        if dataset_info.languages:
            if 'ja' in dataset_info.languages or 'japanese' in str(dataset_info.tags).lower():
                return 'ja'
            elif 'en' in dataset_info.languages or 'english' in str(dataset_info.tags).lower():
                return 'en'
            elif 'zh' in dataset_info.languages:
                return 'zh'
            elif 'ko' in dataset_info.languages:
                return 'ko'

        # テキストベースの検出
        if self._contains_japanese(text):
            return 'ja'
        elif self._contains_chinese(text):
            return 'zh'
        elif self._contains_korean(text):
            return 'ko'
        else:
            return 'en'  # デフォルトは英語

    def _contains_japanese(self, text: str) -> bool:
        """日本語を含むかチェック"""
        # ひらがな・カタカナ・漢字の存在チェック
        japanese_pattern = r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]'
        return bool(re.search(japanese_pattern, text))

    def _contains_chinese(self, text: str) -> bool:
        """中国語を含むかチェック"""
        chinese_pattern = r'[\u4e00-\u9fff]'
        return bool(re.search(chinese_pattern, text))

    def _contains_korean(self, text: str) -> bool:
        """韓国語を含むかチェック"""
        korean_pattern = r'[\uac00-\ud7af\u1100-\u11ff]'
        return bool(re.search(korean_pattern, text))

    def _contains_nsfw_content(self, text: str) -> bool:
        """NSFWコンテンツを含むかチェック"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.nsfw_keywords)

    def _remove_duplicates(self, entries: List[MultilingualDatasetEntry]) -> List[MultilingualDatasetEntry]:
        """重複エントリの除去"""
        seen_hashes = set()
        unique_entries = []

        for entry in entries:
            text_hash = hashlib.md5(entry.text.encode()).hexdigest()
            if text_hash not in seen_hashes and text_hash not in self.existing_hashes:
                seen_hashes.add(text_hash)
                unique_entries.append(entry)

        return unique_entries

    def _create_fallback_entries(self, dataset_info: HFDatasetInfo, category: str) -> List[MultilingualDatasetEntry]:
        """フォールバックエントリ作成（データセット読み込み失敗時）"""
        # データセット情報に基づくサンプルエントリ作成
        entries = []

        # 基本的なサンプルテキスト
        sample_texts = {
            'multilingual': [
                "Hello, how are you today? こんにちは、今日はお元気ですか？",
                "What is the weather like? 天気はどうですか？",
                "I enjoy learning new languages. 新しい言語を学ぶのが好きです。"
            ],
            'nsfw': [
                "This content is for safety training purposes only.",
                "NSFW detection and filtering system test data.",
                "Safety classification training sample."
            ],
            'japanese': [
                "こんにちは、今日は良い天気ですね。",
                "日本語で会話する練習をしましょう。",
                "このテキストは日本語の学習用です。"
            ],
            'english': [
                "Hello, this is a sample English text.",
                "The weather is nice today, isn't it?",
                "Learning new things is always interesting."
            ]
        }

        texts = sample_texts.get(category, sample_texts['multilingual'])

        for i, text in enumerate(texts):
            language = 'ja' if category == 'japanese' else 'en'
            if category == 'multilingual':
                language = 'ja' if i % 2 == 1 else 'en'

            entry = MultilingualDatasetEntry(
                id=f"{dataset_info.id}_fallback_{i}",
                text=text,
                language=language,
                source_dataset=dataset_info.id,
                category=category,
                quality_score=max(0.1, dataset_info.quality_score - 0.3),  # 品質を下げる
                created_at=datetime.now().isoformat()
            )
            entries.append(entry)

        return entries

    def save_multilingual_dataset(self, processed_data: Dict[str, List[MultilingualDatasetEntry]]):
        """マルチリンガルデータセットの保存"""
        logger.info("Saving multilingual dataset...")

        # 統合データセット
        all_entries = []
        for category_entries in processed_data.values():
            all_entries.extend(category_entries)

        # カテゴリ別保存
        for category, entries in processed_data.items():
            if entries:
                category_file = self.output_dir / f"hf_{category}_dataset.jsonl"
                with open(category_file, 'w', encoding='utf-8') as f:
                    for entry in entries:
                        json.dump(asdict(entry), f, ensure_ascii=False, indent=None)
                        f.write('\n')

                logger.info(f"Saved {len(entries)} entries to {category_file}")

        # 統合データセット
        if all_entries:
            integrated_file = self.output_dir / "hf_multilingual_integrated_dataset.jsonl"
            with open(integrated_file, 'w', encoding='utf-8') as f:
                for entry in all_entries:
                    json.dump(asdict(entry), f, ensure_ascii=False, indent=None)
                    f.write('\n')

            logger.info(f"Saved {len(all_entries)} entries to {integrated_file}")

        # 統計レポート
        stats = self._generate_statistics_report(processed_data)
        stats_file = self.output_dir / "hf_multilingual_dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Statistics saved to {stats_file}")

    def _generate_statistics_report(self, processed_data: Dict[str, List[MultilingualDatasetEntry]]) -> Dict[str, Any]:
        """統計レポート生成"""
        stats = {
            'generated_at': datetime.now().isoformat(),
            'total_entries': sum(len(entries) for entries in processed_data.values()),
            'categories': {}
        }

        for category, entries in processed_data.items():
            if entries:
                languages = [entry.language for entry in entries]
                categories_in_data = [entry.category for entry in entries]

                stats['categories'][category] = {
                    'count': len(entries),
                    'languages': list(set(languages)),
                    'language_distribution': dict(pd.Series(languages).value_counts()),
                    'avg_quality_score': np.mean([entry.quality_score for entry in entries]),
                    'quality_score_std': np.std([entry.quality_score for entry in entries]),
                    'subcategories': list(set(categories_in_data))
                }

        return stats

def main():
    """メイン実行関数"""
    print("SO8T Hugging Face Multilingual Dataset Explorer")
    print("=" * 55)

    if not HF_AVAILABLE:
        print("ERROR: datasets library not available.")
        print("Install with: pip install datasets huggingface_hub")
        return

    explorer = HFDatasetExplorer()

    try:
        # 1. データセット探索
        print("\n[1/4] Exploring Hugging Face datasets...")
        dataset_infos = explorer.explore_datasets(max_datasets=200)

        if not dataset_infos:
            print("No datasets found. Exiting.")
            return

        # 2. データセット選択
        print("\n[2/4] Selecting optimal datasets...")
        selected_datasets = explorer.select_top_datasets(dataset_infos)

        # 選択結果表示
        print("\nSelected Datasets:")
        for category, datasets in selected_datasets.items():
            print(f"  {category.upper()}: {len(datasets)} datasets")
            for dataset in datasets[:3]:  # 最初の3つを表示
                print(f"    - {dataset.id} (score: {dataset.quality_score:.3f})")

        # 3. データダウンロードと処理
        print("\n[3/4] Downloading and processing datasets...")
        processed_data = explorer.download_and_process_datasets(selected_datasets)

        # 処理結果表示
        print("\nProcessing Results:")
        total_entries = 0
        for category, entries in processed_data.items():
            print(f"  {category}: {len(entries)} entries")
            total_entries += len(entries)
        print(f"  TOTAL: {total_entries} entries")

        # 4. データ保存
        print("\n[4/4] Saving multilingual dataset...")
        explorer.save_multilingual_dataset(processed_data)

        print("\n✅ Hugging Face multilingual dataset processing completed!")
        print(f"📁 Output directory: {explorer.output_dir}")

        # 音声通知
        try:
            import winsound
            winsound.Beep(1400, 500)  # 成功音
            print("[AUDIO] Dataset processing completed successfully")
        except ImportError:
            print("[AUDIO] Dataset processing completed (winsound not available)")

    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()
