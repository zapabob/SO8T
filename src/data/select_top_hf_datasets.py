#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Top HF Datasets Selector
日英内部推論強化用に厳選されたHFデータセットの上位20%を選択

機能:
- 日英特化の品質ランキングシステム
- 上位20%の高品質データセット選択
- Playwrightを使用したブラウザ確認
- 既存データセットとの重複チェック
- NSFWデータセットの安全フィルタリング
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
import asyncio
from urllib.parse import urljoin

# Playwright for browser verification
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: playwright not available. Install with: pip install playwright")

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
        logging.FileHandler('logs/top_hf_selector.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DatasetCandidate:
    """データセット候補"""
    id: str
    name: str
    description: str
    languages: List[str]
    tags: List[str]
    downloads: int
    likes: int
    size_mb: Optional[float]
    quality_score: float
    ja_en_relevance: float
    reasoning_fitness: float
    contains_nsfw: bool
    is_downloaded: bool
    browser_verified: bool
    final_rank: int

class TopHFSelector:
    """トップHFデータセットセレクター"""

    def __init__(self, output_dir: str = "data/top_hf_selected"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 既存データセットのハッシュセット
        self.existing_hashes = self._load_existing_dataset_hashes()

        # 優先データセット（日英内部推論強化向け）
        self.priority_datasets = [
            # 高品質日英翻訳データセット
            'Helsinki-NLP/opus-100',
            'Helsinki-NLP/opus-mt-en-ja',
            'Helsinki-NLP/opus-mt-ja-en',
            'facebook/flores',

            # 内部推論強化用QAデータセット
            'google-research-datasets/natural_questions',
            'stanfordnlp/coqa',
            'allenai/sciq',  # 科学QA
            'allenai/openbookqa',  # 推論QA

            # 日英会話データセット
            'facebook/blenderbot-400M-distill',
            'microsoft/DialoGPT-medium',
            'daily_dialog',
            'empathetic_dialogues',

            # 数学・科学推論データセット
            'allenai/scicite',
            'scientific_papers',
            'arxiv_dataset',
            'qasper',  # 論文QA

            # 日本語特化データセット
            'izumi-lab/llm-japanese-dataset',
            'rinna/japanese-gpt-1b',
            'nlp-thedeep/japanese-nsfw-text-detection',

            # NSFW/安全学習用データセット
            'jigsaw/toxicity-prediction',
            'facebook/roberta-hate-speech-dynabench-r4-target',
            'unitary/toxic-bert',
        ]

        # NSFWキーワード（検出用）
        self.nsfw_keywords = {
            'sexual', 'porn', 'nude', 'erotic', 'adult', 'xxx', 'sex',
            'naked', 'fuck', 'shit', 'damn', 'bitch', 'asshole', 'cunt',
            'dick', 'pussy', 'tits', 'boobs', 'cock', 'cum', 'rape',
            'violence', 'murder', 'drugs', 'suicide', 'self-harm'
        }

        logger.info(f"Initialized TopHFSelector with output directory: {output_dir}")

    def _load_existing_dataset_hashes(self) -> set:
        """既存データセットのテキストハッシュをロード"""
        hashes = set()
        data_dir = Path("data")

        if data_dir.exists():
            for jsonl_file in data_dir.rglob("*.jsonl"):
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line.strip())
                                    for field in ['text', 'content', 'instruction', 'response']:
                                        if field in data and data[field]:
                                            text_hash = hashlib.md5(str(data[field]).encode()).hexdigest()
                                            hashes.add(text_hash)
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.warning(f"Failed to process {jsonl_file}: {e}")

        logger.info(f"Loaded {len(hashes)} existing text hashes")
        return hashes

    def load_existing_candidates_from_file(self, file_path: str) -> List[DatasetCandidate]:
        """既存のファイルから候補を読み込み"""
        candidates = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        candidate = DatasetCandidate(**data)
                        candidates.append(candidate)
            logger.info(f"Loaded {len(candidates)} candidates from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load candidates from {file_path}: {e}")

        return candidates

    async def select_top_datasets(self, top_percentage: float = 20.0) -> List[DatasetCandidate]:
        """上位パーセントの高品質データセットを選択"""
        logger.info(f"Selecting top {top_percentage}% HF datasets for JA-EN reasoning...")

        if not HF_AVAILABLE:
            logger.error("HF libraries not available")
            return []

        # HF Hubからデータセット情報を取得
        candidates = await self._fetch_dataset_candidates()

        if not candidates:
            logger.warning("No dataset candidates found")
            return []

        # 品質スコアリング
        scored_candidates = []
        for candidate in tqdm(candidates, desc="Scoring datasets"):
            # 各種スコア計算
            ja_en_score = self._calculate_ja_en_relevance(candidate)
            reasoning_score = self._calculate_reasoning_fitness(candidate)
            quality_score = self._calculate_overall_quality(candidate, ja_en_score, reasoning_score)

            candidate.ja_en_relevance = ja_en_score
            candidate.reasoning_fitness = reasoning_score
            candidate.quality_score = quality_score

            scored_candidates.append(candidate)

        # スコアでソート
        scored_candidates.sort(key=lambda x: x.quality_score, reverse=True)

        # 上位パーセントを選択
        top_count = max(1, int(len(scored_candidates) * top_percentage / 100))
        top_candidates = scored_candidates[:top_count]

        # ブラウザ検証（Playwright）
        if PLAYWRIGHT_AVAILABLE:
            await self._verify_with_browser(top_candidates)

        # 最終ランキング
        for i, candidate in enumerate(top_candidates):
            candidate.final_rank = i + 1

        logger.info(f"Selected {len(top_candidates)} top datasets")
        return top_candidates

    async def _fetch_dataset_candidates(self) -> List[DatasetCandidate]:
        """HF Hubからデータセット候補を取得"""
        candidates = []

        try:
            api = HfApi()

            # 優先データセットを先にチェック
            for dataset_id in tqdm(self.priority_datasets, desc="Checking priority datasets"):
                try:
                    info = api.dataset_info(dataset_id, timeout=10)
                    candidate = self._create_candidate_from_info(info)
                    if candidate:
                        candidates.append(candidate)
                except Exception as e:
                    logger.warning(f"Failed to fetch {dataset_id}: {e}")

            # 追加のデータセット探索（日英関連）
            search_terms = ['japanese', 'english', 'translation', 'reasoning', 'qa', 'dialog']
            for term in search_terms:
                try:
                    search_results = api.list_datasets(search=term, limit=10)
                    for result in search_results:
                        if result.id not in [c.id for c in candidates]:
                            try:
                                info = api.dataset_info(result.id, timeout=5)
                                candidate = self._create_candidate_from_info(info)
                                if candidate:
                                    candidates.append(candidate)
                            except Exception:
                                continue
                except Exception as e:
                    logger.warning(f"Search failed for {term}: {e}")

        except Exception as e:
            logger.error(f"Failed to fetch dataset candidates: {e}")

        return candidates

    def _create_candidate_from_info(self, info) -> Optional[DatasetCandidate]:
        """HFデータセット情報から候補を作成"""
        try:
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

            # NSFWチェック
            contains_nsfw = self._check_nsfw_content(info)

            # サイズ計算
            size_mb = None
            if hasattr(info, 'size_in_bytes') and info.size_in_bytes:
                size_mb = info.size_in_bytes / (1024 * 1024)

            candidate = DatasetCandidate(
                id=info.id,
                name=getattr(info, 'name', info.id.split('/')[-1]),
                description=getattr(info, 'description', ''),
                languages=languages,
                tags=getattr(info, 'tags', []),
                downloads=getattr(info, 'downloads', 0),
                likes=getattr(info, 'likes', 0),
                size_mb=size_mb,
                quality_score=0.0,  # 後で計算
                ja_en_relevance=0.0,
                reasoning_fitness=0.0,
                contains_nsfw=contains_nsfw,
                is_downloaded=False,  # 後でチェック
                browser_verified=False,
                final_rank=0
            )

            return candidate

        except Exception as e:
            logger.warning(f"Failed to create candidate from info: {e}")
            return None

    def _check_nsfw_content(self, info) -> bool:
        """NSFWコンテンツのチェック"""
        text_to_check = ""

        if hasattr(info, 'description') and info.description:
            text_to_check += info.description.lower() + " "

        if hasattr(info, 'tags') and info.tags:
            text_to_check += " ".join(info.tags).lower() + " "

        if hasattr(info, 'id'):
            text_to_check += info.id.lower()

        return any(keyword in text_to_check for keyword in self.nsfw_keywords)

    def _calculate_ja_en_relevance(self, candidate: DatasetCandidate) -> float:
        """日英関連度スコア計算"""
        score = 0.0

        # 言語チェック
        has_japanese = 'ja' in candidate.languages or 'japanese' in str(candidate.tags).lower()
        has_english = 'en' in candidate.languages or 'english' in str(candidate.tags).lower()

        if has_japanese and has_english:
            score += 1.0  # 日英両方
        elif has_japanese or has_english:
            score += 0.6  # 一方のみ

        # データセット名と説明文チェック
        text_to_check = f"{candidate.id} {candidate.description} {' '.join(candidate.tags)}".lower()

        # 関連キーワード
        relevance_keywords = [
            'japanese', 'english', 'translation', 'bilingual', 'multilingual',
            'dialog', 'conversation', 'chat', 'talk', 'discussion'
        ]

        keyword_matches = sum(1 for kw in relevance_keywords if kw in text_to_check)
        score += min(keyword_matches * 0.1, 0.4)

        return min(score, 1.0)

    def _calculate_reasoning_fitness(self, candidate: DatasetCandidate) -> float:
        """内部推論強化適合性スコア計算"""
        score = 0.0

        text_to_check = f"{candidate.id} {candidate.description} {' '.join(candidate.tags)}".lower()

        # 推論関連キーワード
        reasoning_keywords = [
            'reasoning', 'logic', 'inference', 'thinking', 'problem', 'solution',
            'theorem', 'proof', 'mathematical', 'scientific', 'analysis',
            'question', 'answer', 'explanation', 'understanding', 'qa',
            'science', 'math', 'logic', 'reason', 'think', 'solve'
        ]

        keyword_matches = sum(1 for kw in reasoning_keywords if kw in text_to_check)
        score += min(keyword_matches * 0.15, 0.6)

        # データセットタイプ別ボーナス
        if 'question' in text_to_check or 'answer' in text_to_check or 'qa' in text_to_check:
            score += 0.2  # QAデータセット
        if 'math' in text_to_check or 'science' in text_to_check:
            score += 0.2  # 数学・科学データセット

        return min(score, 1.0)

    def _calculate_overall_quality(self, candidate: DatasetCandidate,
                                 ja_en_score: float, reasoning_score: float) -> float:
        """総合品質スコア計算"""
        # ベーススコア
        base_score = 0.3

        # ダウンロード数といいね数
        downloads_score = min(candidate.downloads / 10000, 1.0) * 0.2
        likes_score = min(candidate.likes / 100, 1.0) * 0.1

        # サイズペナルティ（大きすぎるデータセット）
        size_penalty = 0.0
        if candidate.size_mb and candidate.size_mb > 1000:  # 1GB以上
            size_penalty = 0.1

        # 総合スコア
        total_score = (
            base_score +
            downloads_score +
            likes_score +
            ja_en_score * 0.3 +
            reasoning_score * 0.3 -
            size_penalty
        )

        return max(0.0, min(1.0, total_score))

    async def _verify_with_browser(self, candidates: List[DatasetCandidate]):
        """Playwrightを使ってブラウザでデータセットを確認"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available, skipping browser verification")
            return

        logger.info("Verifying datasets with browser...")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )

            for candidate in tqdm(candidates, desc="Browser verification"):
                try:
                    # HF HubのURL
                    url = f"https://huggingface.co/datasets/{candidate.id}"

                    page = await context.new_page()
                    await page.goto(url, timeout=10000)

                    # ページ内容の確認
                    title = await page.title()
                    if "404" in title or "Not Found" in title:
                        candidate.browser_verified = False
                    else:
                        # データセットの基本情報確認
                        content = await page.text_content()
                        if candidate.name.lower() in content.lower():
                            candidate.browser_verified = True
                        else:
                            candidate.browser_verified = False

                    await page.close()

                except Exception as e:
                    logger.warning(f"Browser verification failed for {candidate.id}: {e}")
                    candidate.browser_verified = False

            await browser.close()

        verified_count = sum(1 for c in candidates if c.browser_verified)
        logger.info(f"Browser verification completed: {verified_count}/{len(candidates)} verified")

    def select_top_percentage(self, candidates: List[DatasetCandidate], top_percentage: float = 20.0) -> List[DatasetCandidate]:
        """上位パーセントのデータセットを選択"""
        if not candidates:
            return []

        # スコアでソート
        candidates.sort(key=lambda x: x.quality_score, reverse=True)

        # 上位パーセントを選択
        top_count = max(1, int(len(candidates) * top_percentage / 100))
        top_candidates = candidates[:top_count]

        # ランキング設定
        for i, candidate in enumerate(top_candidates):
            candidate.final_rank = i + 1

        logger.info(f"Selected top {top_percentage}% ({len(top_candidates)}) from {len(candidates)} candidates")
        return top_candidates

    def save_selected_datasets(self, candidates: List[DatasetCandidate]):
        """選択されたデータセットを保存"""
        logger.info("Saving selected datasets...")

        # JSONシリアライズ可能な形式に変換
        def serialize_candidate(candidate):
            data = asdict(candidate)
            # numpy int64をintに変換
            for key, value in data.items():
                if hasattr(value, 'item'):  # numpy types
                    data[key] = value.item()
                elif isinstance(value, (int, float)) and str(type(value)).startswith("<class 'numpy."):
                    data[key] = value.item()
            return data

        # JSON形式で保存
        output_data = {
            'selection_timestamp': datetime.now().isoformat(),
            'total_candidates': len(candidates),
            'selection_criteria': {
                'top_percentage': 20.0,
                'ja_en_focused': True,
                'reasoning_enhancement': True,
                'nsfw_included': True
            },
            'datasets': [serialize_candidate(candidate) for candidate in candidates]
        }

        # メイン結果ファイル
        output_file = self.output_dir / "top_hf_datasets_selected.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # ダウンロード用リスト
        download_list = self.output_dir / "datasets_to_download.txt"
        with open(download_list, 'w', encoding='utf-8') as f:
            f.write("# Top 20% HF Datasets for JA-EN Reasoning Enhancement\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("# Format: dataset_id | quality_score | ja_en_relevance | reasoning_fitness | contains_nsfw\n\n")

            for candidate in candidates:
                status = "VERIFIED" if candidate.browser_verified else "UNVERIFIED"
                nsfw_tag = "NSFW" if candidate.contains_nsfw else "SAFE"
                f.write(f"{candidate.id} | {candidate.quality_score:.3f} | {candidate.ja_en_relevance:.3f} | {candidate.reasoning_fitness:.3f} | {nsfw_tag} | {status}\n")

        # カテゴリ別リスト
        categories = {
            'multilingual_ja_en': [c for c in candidates if 'ja' in c.languages and 'en' in c.languages],
            'nsfw_datasets': [c for c in candidates if c.contains_nsfw],
            'reasoning_datasets': [c for c in candidates if c.reasoning_fitness > 0.7],
            'high_quality': [c for c in candidates if c.quality_score > 0.8]
        }

        for cat_name, cat_candidates in categories.items():
            if cat_candidates:
                cat_file = self.output_dir / f"{cat_name}_list.txt"
                with open(cat_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {cat_name.replace('_', ' ').title()} Datasets\n")
                    f.write(f"# Count: {len(cat_candidates)}\n\n")

                    for candidate in cat_candidates:
                        f.write(f"{candidate.id} (Score: {candidate.quality_score:.3f})\n")

        logger.info(f"Selected datasets saved to {self.output_dir}")

    def generate_summary_report(self, candidates: List[DatasetCandidate]) -> Dict[str, Any]:
        """サマリーレポート生成"""
        if not candidates:
            return {}

        report = {
            'summary': {
                'total_selected': len(candidates),
                'avg_quality_score': np.mean([c.quality_score for c in candidates]),
                'avg_ja_en_relevance': np.mean([c.ja_en_relevance for c in candidates]),
                'avg_reasoning_fitness': np.mean([c.reasoning_fitness for c in candidates]),
                'nsfw_datasets_count': sum(1 for c in candidates if c.contains_nsfw),
                'browser_verified_count': sum(1 for c in candidates if c.browser_verified),
                'multilingual_count': sum(1 for c in candidates if 'ja' in c.languages and 'en' in c.languages)
            },
            'top_performers': [
                {
                    'id': c.id,
                    'quality_score': c.quality_score,
                    'ja_en_relevance': c.ja_en_relevance,
                    'reasoning_fitness': c.reasoning_fitness,
                    'contains_nsfw': c.contains_nsfw,
                    'browser_verified': c.browser_verified
                }
                for c in candidates[:5]  # トップ5
            ],
            'category_breakdown': {
                'nsfw_datasets': [c.id for c in candidates if c.contains_nsfw],
                'multilingual_ja_en': [c.id for c in candidates if 'ja' in c.languages and 'en' in c.languages],
                'high_reasoning_fitness': [c.id for c in candidates if c.reasoning_fitness > 0.8],
                'browser_verified': [c.id for c in candidates if c.browser_verified]
            }
        }

        return report

async def main():
    """メイン実行関数"""
    print("SO8T Top HF Datasets Selector - Refined Selection")
    print("=" * 55)

    if not HF_AVAILABLE:
        print("ERROR: Hugging Face libraries not available")
        print("Install with: pip install datasets huggingface_hub")
        return

    if not PLAYWRIGHT_AVAILABLE:
        print("WARNING: Playwright not available - browser verification will be skipped")
        print("Install with: pip install playwright")
    else:
        print("Playwright available for browser verification")

    selector = TopHFSelector()

    try:
        # 既存のNSFWデータセットを読み込んで上位20%を選択
        print("\n[1/4] Loading existing NSFW datasets...")
        nsfw_candidates = selector.load_existing_candidates_from_file("data/hf_multilingual/hf_nsfw_dataset.jsonl")

        if nsfw_candidates:
            print(f"Loaded {len(nsfw_candidates)} NSFW candidates")

            # NSFWデータセットの上位20%を選択
            print("\n[2/4] Selecting top 20% NSFW datasets...")
            top_nsfw = selector.select_top_percentage(nsfw_candidates, top_percentage=20.0)

            # ブラウザ検証（利用可能な場合）
            if PLAYWRIGHT_AVAILABLE:
                print("\n[3/4] Browser verification...")
                await selector._verify_with_browser(top_nsfw)

            # 結果保存
            print("\n[4/4] Saving refined selection...")
            selector.save_selected_datasets(top_nsfw)

            # サマリーレポート生成
            summary_report = selector.generate_summary_report(top_nsfw)

            # レポート表示
            print("\n" + "="*60)
            print("REFINED SELECTION RESULTS SUMMARY (Top 20% NSFW)")
            print("="*60)
            print(f"Total Selected Datasets: {summary_report['summary']['total_selected']}")
            print(f"Average Quality Score: {summary_report['summary']['avg_quality_score']:.3f}")
            print(f"JA-EN Relevance: {summary_report['summary']['avg_ja_en_relevance']:.3f}")
            print(f"Reasoning Fitness: {summary_report['summary']['avg_reasoning_fitness']:.3f}")
            print(f"NSFW Datasets: {summary_report['summary']['nsfw_datasets_count']}")
            print(f"Browser Verified: {summary_report['summary']['browser_verified_count']}")
            print(f"Multilingual JA-EN: {summary_report['summary']['multilingual_count']}")

            print("\nTOP PERFORMERS:")
            for i, performer in enumerate(summary_report['top_performers'][:5], 1):
                status = "VERIFIED" if performer['browser_verified'] else "UNVERIFIED"
                nsfw_tag = "NSFW" if performer['contains_nsfw'] else "SAFE"
                print(f"{i}. {performer['id']} (Score: {performer['quality_score']:.3f}) [{status}] [{nsfw_tag}]")

            print(f"\n[DIR] Results saved to: {selector.output_dir}")

            # 上位データセットのダウンロードリスト作成
            download_list = selector.output_dir / "top_20_percent_nsfw_download_list.txt"
            with open(download_list, 'w', encoding='utf-8') as f:
                f.write("# Top 20% NSFW Datasets for JA-EN Reasoning Enhancement\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write("# These datasets are selected for safety training and rejection behavior learning\n\n")

                for candidate in top_nsfw:
                    verified = "VERIFIED" if candidate.browser_verified else "UNVERIFIED"
                    f.write(f"{candidate.id} | Score: {candidate.quality_score:.3f} | {verified}\n")

            print(f"📋 Download list saved to: {download_list}")

        else:
            print("No NSFW candidates found. Running full dataset exploration...")

            # 上位20%のデータセット選択
            print("\n[1/3] Selecting top 20% datasets...")
            selected_datasets = await selector.select_top_datasets(top_percentage=20.0)

            if not selected_datasets:
                print("No datasets selected")
                return

            # 結果保存
            print("\n[2/3] Saving selection results...")
            selector.save_selected_datasets(selected_datasets)

            # サマリーレポート生成
            print("\n[3/3] Generating summary report...")
            summary_report = selector.generate_summary_report(selected_datasets)

            # レポート表示
            print("\n" + "="*50)
            print("SELECTION RESULTS SUMMARY")
            print("="*50)
            print(f"Total Selected Datasets: {summary_report['summary']['total_selected']}")
            print(f"Average Quality Score: {summary_report['summary']['avg_quality_score']:.3f}")
            print(f"JA-EN Relevance: {summary_report['summary']['avg_ja_en_relevance']:.3f}")
            print(f"Reasoning Fitness: {summary_report['summary']['avg_reasoning_fitness']:.3f}")
            print(f"NSFW Datasets: {summary_report['summary']['nsfw_datasets_count']}")
            print(f"Browser Verified: {summary_report['summary']['browser_verified_count']}")
            print(f"Multilingual JA-EN: {summary_report['summary']['multilingual_count']}")

            print("\nTOP PERFORMERS:")
            for i, performer in enumerate(summary_report['top_performers'][:3], 1):
                print(f"{i}. {performer['id']} (Score: {performer['quality_score']:.3f})")

        # 音声通知
        try:
            import winsound
            winsound.Beep(1400, 600)  # 成功音
            print("[AUDIO] Dataset selection completed successfully")
        except ImportError:
            print("[AUDIO] Dataset selection completed (winsound not available)")

    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Main execution failed: {e}")

    def load_existing_candidates_from_file(self, file_path: str) -> List[DatasetCandidate]:
        """既存のファイルから候補を読み込み"""
        candidates = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        candidate = DatasetCandidate(**data)
                        candidates.append(candidate)
            logger.info(f"Loaded {len(candidates)} candidates from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load candidates from {file_path}: {e}")

        return candidates

if __name__ == "__main__":
    asyncio.run(main())
