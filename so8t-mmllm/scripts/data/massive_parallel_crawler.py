#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超大規模並列Webクローラー
- 国会図書館、Wikipedia（日中英）、官庁、eGov、日経225企業等
- 真の並列処理（multiprocessing）
- バックグラウンド実行対応
- 統計的重み付け（日本語優先）
- 目標: 10,000,000+ samples
"""

import os
import sys
import json
import time
import logging
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
from dataclasses import dataclass, asdict
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import re
from tqdm import tqdm
import numpy as np

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/massive_crawl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 超大規模クロール設定（200GB版）
MASSIVE_CRAWL_CONFIG = {
    "target_samples": 17_000_000,  # 1700万サンプル（~200GB）
    "max_workers": 16,  # 並列ワーカー数
    "max_depth": 4,  # クロール深度
    "delay_per_domain": 1.0,  # ドメインごとの遅延
    "timeout": 15,  # タイムアウト
    "max_pages_per_site": 1000,  # サイトあたり最大ページ数
    "checkpoint_interval": 180,  # 3分
    "max_checkpoints": 5,  # チェックポイント数（最大5個ローテーション）
    
    # 言語別重み（日本語優先）
    "language_weights": {
        "ja": 0.7,  # 日本語70%
        "en": 0.2,  # 英語20%
        "zh": 0.1   # 中国語10%
    }
}


# 包括的クロールソース
COMPREHENSIVE_SOURCES = {
    # 国会図書館・公文書
    "ndl": {
        "urls": [
            "https://www.ndl.go.jp/",  # 国立国会図書館
            "https://dl.ndl.go.jp/",  # デジタルコレクション
        ],
        "domain": "library",
        "language": "ja",
        "priority": "high"
    },
    
    # Wikipedia（多言語）
    "wikipedia_ja": {
        "urls": ["https://ja.wikipedia.org/wiki/"],
        "domain": "encyclopedia",
        "language": "ja",
        "priority": "high"
    },
    "wikipedia_en": {
        "urls": ["https://en.wikipedia.org/wiki/"],
        "domain": "encyclopedia",
        "language": "en",
        "priority": "medium"
    },
    "wikipedia_zh": {
        "urls": ["https://zh.wikipedia.org/wiki/"],
        "domain": "encyclopedia",
        "language": "zh",
        "priority": "low"
    },
    
    # 官庁公式
    "cabinet": {
        "urls": [
            "https://www.kantei.go.jp/",  # 首相官邸
            "https://www.cas.go.jp/",  # 内閣官房
        ],
        "domain": "government",
        "language": "ja",
        "priority": "high"
    },
    "mod": {
        "urls": ["https://www.mod.go.jp/"],  # 防衛省
        "domain": "defense",
        "language": "ja",
        "priority": "high"
    },
    "mof": {
        "urls": ["https://www.mof.go.jp/"],  # 財務省
        "domain": "finance",
        "language": "ja",
        "priority": "high"
    },
    "mext": {
        "urls": ["https://www.mext.go.jp/"],  # 文部科学省
        "domain": "education",
        "language": "ja",
        "priority": "high"
    },
    "mhlw": {
        "urls": ["https://www.mhlw.go.jp/"],  # 厚生労働省
        "domain": "medical",
        "language": "ja",
        "priority": "high"
    },
    "meti": {
        "urls": ["https://www.meti.go.jp/"],  # 経済産業省
        "domain": "business",
        "language": "ja",
        "priority": "high"
    },
    "mlit": {
        "urls": ["https://www.mlit.go.jp/"],  # 国土交通省
        "domain": "transport",
        "language": "ja",
        "priority": "high"
    },
    "fsa": {
        "urls": ["https://www.fsa.go.jp/"],  # 金融庁
        "domain": "finance",
        "language": "ja",
        "priority": "high"
    },
    
    # eGov（電子政府）
    "egov": {
        "urls": [
            "https://www.e-gov.go.jp/",  # e-Gov
            "https://elaws.e-gov.go.jp/",  # 法令データベース
        ],
        "domain": "legal",
        "language": "ja",
        "priority": "high"
    },
    
    # 宇宙・航空
    "jaxa": {
        "urls": ["https://www.jaxa.jp/"],  # JAXA
        "domain": "aerospace",
        "language": "ja",
        "priority": "medium"
    },
    
    # 日経225企業（代表例）
    "toyota": {
        "urls": ["https://global.toyota/jp/"],
        "domain": "automotive",
        "language": "ja",
        "priority": "medium"
    },
    "sony": {
        "urls": ["https://www.sony.com/ja/"],
        "domain": "electronics",
        "language": "ja",
        "priority": "medium"
    },
    "mitsubishi": {
        "urls": ["https://www.mitsubishicorp.com/jp/ja/"],
        "domain": "trading",
        "language": "ja",
        "priority": "medium"
    },
    "softbank": {
        "urls": ["https://www.softbank.jp/"],
        "domain": "telecom",
        "language": "ja",
        "priority": "medium"
    },
    
    # 文化・民俗
    "culture": {
        "urls": [
            "https://www.bunka.go.jp/",  # 文化庁
            "https://www.minpaku.ac.jp/",  # 国立民族学博物館
        ],
        "domain": "culture",
        "language": "ja",
        "priority": "medium"
    },
    
    # 教育
    "education": {
        "urls": [
            "https://www.u-tokyo.ac.jp/ja/",  # 東京大学
            "https://www.kyoto-u.ac.jp/ja",  # 京都大学
        ],
        "domain": "education",
        "language": "ja",
        "priority": "medium"
    },
}


@dataclass
class CrawlTask:
    """クロールタスク"""
    source_id: str
    urls: List[str]
    domain: str
    language: str
    priority: str
    max_pages: int


class ParallelWebCrawler:
    """超大規模並列Webクローラー（チェックポイント機能付き）"""
    
    def __init__(self, config: Dict = None):
        self.config = config or MASSIVE_CRAWL_CONFIG
        self.output_dir = Path("data/web_crawled")
        self.checkpoint_dir = Path("data/crawl_checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # プロセス間共有データ
        self.manager = mp.Manager()
        self.visited_urls = self.manager.dict()  # 訪問済みURL
        self.collected_count = self.manager.Value('i', 0)  # 収集カウンタ
        self.progress_lock = self.manager.Lock()
        
        # チェックポイント管理
        self.checkpoint_deque = deque(maxlen=self.config['max_checkpoints'])  # 最大5個
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter = 0
        
        # 日経225企業読み込み
        self._load_nikkei225_companies()
    
    def _load_nikkei225_companies(self):
        """日経225企業をCOMPREHENSIVE_SOURCESに追加"""
        nikkei_file = Path("scripts/data/nikkei225_sources.json")
        
        if not nikkei_file.exists():
            logger.warning(f"Nikkei225 file not found: {nikkei_file}")
            return
        
        try:
            with open(nikkei_file, 'r', encoding='utf-8') as f:
                nikkei_data = json.load(f)
            
            for company in nikkei_data.get('nikkei225_companies', []):
                source_id = f"nikkei_{company['name'].replace(' ', '_')}"
                
                COMPREHENSIVE_SOURCES[source_id] = {
                    "urls": [company['url']],
                    "domain": company['domain'],
                    "language": "ja",
                    "priority": "medium"
                }
            
            logger.info(f"[OK] Loaded {len(nikkei_data.get('nikkei225_companies', []))} Nikkei225 companies")
        
        except Exception as e:
            logger.error(f"Failed to load Nikkei225: {e}")
    
    def _crawl_single_url(self, url: str, domain: str, language: str) -> Dict:
        """
        単一URL クロール（ワーカープロセス用）
        
        Args:
            url: クロール対象URL
            domain: ドメイン分類
            language: 言語
        
        Returns:
            sample: 収集サンプル
        """
        # 訪問済みチェック
        with self.progress_lock:
            if url in self.visited_urls:
                return None
            self.visited_urls[url] = True
        
        try:
            # レート制限
            time.sleep(self.config['delay_per_domain'])
            
            # HTTPリクエスト
            response = requests.get(
                url,
                timeout=self.config['timeout'],
                headers={'User-Agent': 'SO8T-MassiveCrawler/1.0 (Research)'},
                allow_redirects=True
            )
            response.raise_for_status()
            
            # HTML解析
            soup = BeautifulSoup(response.content, 'lxml')
            
            # テキスト抽出
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()
            
            if len(text) < 200:
                return None
            
            # 言語検証
            if not self._verify_language(text, language):
                return None
            
            # サンプル作成
            sample = {
                "text": text,
                "url": url,
                "domain": domain,
                "language": language,
                "source": "web_crawl",
                "crawled_at": datetime.now().isoformat(),
                "text_length": len(text)
            }
            
            # カウンタ更新
            with self.progress_lock:
                self.collected_count.value += 1
            
            return sample
        
        except Exception as e:
            logger.debug(f"Crawl failed {url}: {e}")
            return None
    
    def _verify_language(self, text: str, expected_lang: str) -> bool:
        """
        言語検証
        
        Args:
            text: テキスト
            expected_lang: 期待言語
        
        Returns:
            valid: 言語が一致するか
        """
        if expected_lang == "ja":
            # 日本語文字の割合
            ja_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf')
            return (ja_chars / len(text)) > 0.3
        elif expected_lang == "en":
            # アルファベットの割合
            en_chars = sum(1 for c in text if c.isascii() and c.isalpha())
            return (en_chars / len(text)) > 0.5
        elif expected_lang == "zh":
            # 中国語文字の割合
            zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            return (zh_chars / len(text)) > 0.3
        else:
            return True
    
    def _crawl_task(self, task: CrawlTask) -> List[Dict]:
        """
        タスク実行（プロセスプール用）
        
        Args:
            task: クロールタスク
        
        Returns:
            samples: 収集サンプル
        """
        logger.info(f"[START] {task.source_id} ({task.language})")
        
        samples = []
        queue = deque(task.urls)
        depth_map = {url: 0 for url in task.urls}
        
        while queue and len(samples) < task.max_pages:
            url = queue.popleft()
            current_depth = depth_map.get(url, 0)
            
            if current_depth >= self.config['max_depth']:
                continue
            
            # クロール
            sample = self._crawl_single_url(url, task.domain, task.language)
            
            if sample:
                samples.append(sample)
                
                # 新規リンク抽出（深度制限内）
                if current_depth < self.config['max_depth'] - 1:
                    try:
                        response = requests.get(url, timeout=self.config['timeout'])
                        soup = BeautifulSoup(response.content, 'lxml')
                        
                        for a_tag in soup.find_all('a', href=True):
                            new_url = urljoin(url, a_tag['href'])
                            
                            # 同一ドメインチェック
                            if urlparse(new_url).netloc == urlparse(url).netloc:
                                if new_url not in depth_map:
                                    queue.append(new_url)
                                    depth_map[new_url] = current_depth + 1
                    except:
                        pass
            
            # 進捗表示
            if len(samples) % 10 == 0:
                logger.info(f"[{task.source_id}] {len(samples)}/{task.max_pages} pages")
        
        logger.info(f"[DONE] {task.source_id}: {len(samples)} samples")
        return samples
    
    def create_tasks(self) -> List[CrawlTask]:
        """
        クロールタスク生成
        
        Returns:
            tasks: タスクリスト
        """
        tasks = []
        
        # 言語別サンプル配分
        lang_targets = {
            "ja": int(self.config['target_samples'] * self.config['language_weights']['ja']),
            "en": int(self.config['target_samples'] * self.config['language_weights']['en']),
            "zh": int(self.config['target_samples'] * self.config['language_weights']['zh'])
        }
        
        # 各ソースのタスク作成
        for source_id, source_config in COMPREHENSIVE_SOURCES.items():
            language = source_config['language']
            priority = source_config['priority']
            
            # 優先度別ページ数
            if priority == "high":
                max_pages = 1000
            elif priority == "medium":
                max_pages = 500
            else:
                max_pages = 100
            
            task = CrawlTask(
                source_id=source_id,
                urls=source_config['urls'],
                domain=source_config['domain'],
                language=language,
                priority=priority,
                max_pages=max_pages
            )
            tasks.append(task)
        
        logger.info(f"[TASKS] Created {len(tasks)} crawl tasks")
        return tasks
    
    def _save_checkpoint(self, samples: List[Dict]):
        """
        チェックポイント保存（3分間隔、最大5個ローテーション）
        
        Args:
            samples: 現在収集済みサンプル
        """
        logger.info(f"[CHECKPOINT] Saving checkpoint {self.checkpoint_counter}...")
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.checkpoint_counter:04d}.pkl"
        
        checkpoint_data = {
            "samples": samples,
            "visited_urls": dict(self.visited_urls),
            "collected_count": self.collected_count.value,
            "checkpoint_time": datetime.now().isoformat(),
            "checkpoint_id": self.checkpoint_counter
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # チェックポイントリストに追加（FIFO、最大5個）
        self.checkpoint_deque.append(str(checkpoint_file))
        
        # 古いチェックポイント削除（6個目以降）
        if len(self.checkpoint_deque) > self.config['max_checkpoints']:
            old_checkpoint = Path(self.checkpoint_deque[0])
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                logger.info(f"[CLEANUP] Deleted old checkpoint: {old_checkpoint.name}")
        
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter += 1
        
        logger.info(f"[OK] Checkpoint saved: {checkpoint_file.name}")
        logger.info(f"[INFO] Active checkpoints: {len(self.checkpoint_deque)}/{self.config['max_checkpoints']}")
    
    def _check_and_save_checkpoint(self, samples: List[Dict]):
        """チェックポイント時間チェック&保存"""
        if time.time() - self.last_checkpoint_time >= self.config['checkpoint_interval']:
            self._save_checkpoint(samples)
    
    def run_parallel_crawl(self):
        """並列クロール実行（チェックポイント対応）"""
        logger.info("="*80)
        logger.info("MASSIVE PARALLEL WEB CRAWLER (with 3-min checkpoints)")
        logger.info(f"Target: {self.config['target_samples']:,} samples")
        logger.info(f"Max workers: {self.config['max_workers']}")
        logger.info(f"Sources: {len(COMPREHENSIVE_SOURCES)}")
        logger.info(f"Checkpoint: Every {self.config['checkpoint_interval']}s ({self.config['checkpoint_interval']/60:.1f}min)")
        logger.info(f"Max checkpoints: {self.config['max_checkpoints']}")
        logger.info(f"Language weights: JA={self.config['language_weights']['ja']:.0%}, "
                   f"EN={self.config['language_weights']['en']:.0%}, "
                   f"ZH={self.config['language_weights']['zh']:.0%}")
        logger.info("="*80)
        
        # タスク作成
        tasks = self.create_tasks()
        
        # 並列実行
        all_samples = []
        
        with ProcessPoolExecutor(max_workers=self.config['max_workers']) as executor:
            # タスク投入
            future_to_task = {
                executor.submit(self._crawl_task, task): task
                for task in tasks
            }
            
            # 進捗表示
            with tqdm(total=len(tasks), desc="Crawling sources") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        samples = future.result()
                        all_samples.extend(samples)
                        pbar.update(1)
                        pbar.set_postfix({'total_samples': len(all_samples)})
                        
                        # チェックポイントチェック
                        self._check_and_save_checkpoint(all_samples)
                        
                    except Exception as e:
                        logger.error(f"Task {task.source_id} failed: {e}")
        
        # 最終チェックポイント
        self._save_checkpoint(all_samples)
        
        # 保存
        self._save_results(all_samples)
        
        # RAGパイプライン自動実行
        self._run_rag_pipeline(all_samples)
        
        logger.info("="*80)
        logger.info(f"[COMPLETE] Collected {len(all_samples):,} samples")
        logger.info(f"[COMPLETE] Checkpoints: {len(self.checkpoint_deque)}")
        logger.info("="*80)
    
    def _save_results(self, samples: List[Dict]):
        """
        結果保存
        
        Args:
            samples: 収集サンプル
        """
        logger.info("[SAVE] Saving collected data...")
        
        # 言語別・ドメイン別に分割
        categorized = {}
        for sample in samples:
            lang = sample['language']
            domain = sample['domain']
            key = f"{lang}_{domain}"
            
            if key not in categorized:
                categorized[key] = []
            categorized[key].append(sample)
        
        # ファイル保存
        for key, samples_list in categorized.items():
            output_file = self.output_dir / f"web_crawled_{key}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples_list:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[OK] Saved {len(samples_list):,} samples to {output_file}")
        
        # 統計レポート
        stats = {
            "total_samples": len(samples),
            "by_language": {},
            "by_domain": {},
            "crawled_at": datetime.now().isoformat()
        }
        
        for sample in samples:
            lang = sample['language']
            domain = sample['domain']
            
            stats['by_language'][lang] = stats['by_language'].get(lang, 0) + 1
            stats['by_domain'][domain] = stats['by_domain'].get(domain, 0) + 1
        
        stats_file = self.output_dir / f"crawl_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Stats saved to {stats_file}")
    
    def _run_rag_pipeline(self, samples: List[Dict]):
        """
        RAG用自動パイプライン
        クロールデータ → チャンク分割 → ベクトル化 → インデックス作成
        
        Args:
            samples: 収集サンプル
        """
        logger.info("="*80)
        logger.info("RAG PIPELINE AUTOMATION")
        logger.info("="*80)
        
        # RAG用ディレクトリ作成
        rag_dir = self.output_dir / "rag_ready"
        rag_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("[STEP 1] Chunking documents...")
        chunked_samples = self._chunk_documents(samples)
        
        logger.info("[STEP 2] Saving RAG-ready data...")
        self._save_rag_data(chunked_samples, rag_dir)
        
        logger.info("[STEP 3] Creating metadata index...")
        self._create_metadata_index(chunked_samples, rag_dir)
        
        logger.info("="*80)
        logger.info(f"[RAG] Ready for vector DB ingestion")
        logger.info(f"[RAG] Output: {rag_dir}")
        logger.info(f"[RAG] Chunks: {len(chunked_samples):,}")
        logger.info("="*80)
    
    def _chunk_documents(self, samples: List[Dict], chunk_size: int = 512) -> List[Dict]:
        """
        ドキュメントチャンク分割（RAG用）
        
        Args:
            samples: 元サンプル
            chunk_size: チャンクサイズ（文字数）
        
        Returns:
            chunks: チャンク済みサンプル
        """
        chunks = []
        
        for sample in tqdm(samples, desc="Chunking"):
            text = sample.get('text', '')
            
            # chunk_sizeごとに分割（オーバーラップあり）
            overlap = chunk_size // 4  # 25% オーバーラップ
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                
                if len(chunk_text) < 100:  # 短すぎるチャンクはスキップ
                    continue
                
                chunk = {
                    "chunk_text": chunk_text,
                    "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest(),
                    "source_url": sample.get('url', ''),
                    "domain": sample.get('domain', ''),
                    "language": sample.get('language', ''),
                    "chunk_index": i // (chunk_size - overlap),
                    "source_length": len(text),
                    "crawled_at": sample.get('crawled_at', '')
                }
                
                chunks.append(chunk)
        
        logger.info(f"[OK] Created {len(chunks):,} chunks from {len(samples):,} documents")
        return chunks
    
    def _save_rag_data(self, chunks: List[Dict], rag_dir: Path):
        """
        RAG用データ保存（言語・ドメイン別）
        
        Args:
            chunks: チャンク
            rag_dir: 出力ディレクトリ
        """
        # 言語×ドメインで分類
        categorized = {}
        for chunk in chunks:
            key = f"{chunk['language']}_{chunk['domain']}"
            if key not in categorized:
                categorized[key] = []
            categorized[key].append(chunk)
        
        # 保存
        for key, chunk_list in categorized.items():
            output_file = rag_dir / f"rag_chunks_{key}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for chunk in chunk_list:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            logger.info(f"[RAG] Saved {len(chunk_list):,} chunks to {output_file.name}")
    
    def _create_metadata_index(self, chunks: List[Dict], rag_dir: Path):
        """
        メタデータインデックス作成（RAG検索用）
        
        Args:
            chunks: チャンク
            rag_dir: 出力ディレクトリ
        """
        metadata = {
            "total_chunks": len(chunks),
            "by_language": {},
            "by_domain": {},
            "chunk_size": 512,
            "overlap": 128,
            "created_at": datetime.now().isoformat(),
            "ready_for_vectorization": True,
            "recommended_embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        }
        
        for chunk in chunks:
            lang = chunk['language']
            domain = chunk['domain']
            
            metadata['by_language'][lang] = metadata['by_language'].get(lang, 0) + 1
            metadata['by_domain'][domain] = metadata['by_domain'].get(domain, 0) + 1
        
        # メタデータ保存
        metadata_file = rag_dir / "rag_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[RAG] Metadata index created: {metadata_file.name}")
        
        # RAG統合ガイド作成
        guide_content = f"""# RAG Integration Guide

## データセット情報
- **総チャンク数**: {metadata['total_chunks']:,}
- **作成日時**: {metadata['created_at']}
- **チャンクサイズ**: {metadata['chunk_size']} 文字
- **オーバーラップ**: {metadata['overlap']} 文字

## 言語分布
{self._format_dict(metadata['by_language'])}

## ドメイン分布
{self._format_dict(metadata['by_domain'])}

## RAG統合手順

### Step 1: ベクトルDB準備

```python
# Chroma DB使用例
from chromadb import Client
from chromadb.config import Settings

client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="{rag_dir}/vector_db"
))

collection = client.create_collection("so8t_knowledge")
```

### Step 2: 埋め込み生成

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('{metadata['recommended_embedding_model']}')

# チャンク読み込み&ベクトル化
import json

for file in Path('{rag_dir}').glob('rag_chunks_*.jsonl'):
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            embedding = model.encode(chunk['chunk_text'])
            
            collection.add(
                ids=[chunk['chunk_id']],
                embeddings=[embedding],
                documents=[chunk['chunk_text']],
                metadatas=[{{
                    'url': chunk['source_url'],
                    'domain': chunk['domain'],
                    'language': chunk['language']
                }}]
            )
```

### Step 3: RAG検索

```python
# クエリ検索
query = "防衛装備品の調達について"
query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# 結果を SO8T モデルに渡す
context = "\\n\\n".join(results['documents'][0])
prompt = f"Context:\\n{{context}}\\n\\nQuery: {{query}}"
```

## 推奨設定

- **Vector DB**: ChromaDB または FAISS
- **Embedding Model**: paraphrase-multilingual-mpnet-base-v2
- **Chunk size**: 512文字（調整可能）
- **Top-k**: 5-10チャンク
- **Re-ranking**: Optional（精度向上）

## 完了
データはRAG統合準備完了です。
"""
        
        guide_file = rag_dir / "RAG_INTEGRATION_GUIDE.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"[RAG] Integration guide created: {guide_file.name}")
    
    def _format_dict(self, d: Dict) -> str:
        """辞書フォーマット"""
        return '\n'.join([f"- **{k}**: {v:,}" for k, v in d.items()])


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Massive Parallel Web Crawler (200GB scale)")
    parser.add_argument("--target", type=int, default=17_000_000, help="Target samples (17M for ~200GB)")
    parser.add_argument("--output-dir", type=str, default="D:/webdataset", help="Output directory (D: drive recommended)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--ja-weight", type=float, default=0.7, help="Japanese weight")
    parser.add_argument("--en-weight", type=float, default=0.2, help="English weight")
    parser.add_argument("--zh-weight", type=float, default=0.1, help="Chinese weight")
    args = parser.parse_args()
    
    # 設定更新
    MASSIVE_CRAWL_CONFIG['target_samples'] = args.target
    MASSIVE_CRAWL_CONFIG['max_workers'] = args.workers
    MASSIVE_CRAWL_CONFIG['language_weights'] = {
        'ja': args.ja_weight,
        'en': args.en_weight,
        'zh': args.zh_weight
    }
    
    # 出力ディレクトリ
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("SO8T MASSIVE PARALLEL WEB CRAWLER")
    logger.info(f"Target: {args.target:,} samples (~{args.target * 12 / 1024 / 1024 / 1024:.0f} GB estimated)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Workers: {args.workers}")
    logger.info("="*80)
    
    # ディスク容量確認
    if os.name == 'nt':  # Windows
        import shutil
        drive = str(output_dir)[0] + ":"
        try:
            total, used, free = shutil.disk_usage(drive)
            free_gb = free / (1024**3)
            logger.info(f"[DISK] Drive {drive} free space: {free_gb:.1f} GB")
            
            if free_gb < 1000:
                logger.warning(f"[WARNING] Low disk space: {free_gb:.1f} GB (recommended: 1000GB+)")
        except:
            pass
    
    # クローラー初期化（出力ディレクトリ指定）
    crawler = ParallelWebCrawler(MASSIVE_CRAWL_CONFIG)
    crawler.output_dir = output_dir
    crawler.checkpoint_dir = output_dir / "checkpoints"
    crawler.output_dir.mkdir(parents=True, exist_ok=True)
    crawler.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        crawler.run_parallel_crawl()
    except KeyboardInterrupt:
        logger.warning("[INTERRUPTED] Saving progress...")
    except Exception as e:
        logger.error(f"[ERROR] Crawl failed: {e}")
        raise


if __name__ == "__main__":
    # ログディレクトリ作成
    Path("logs").mkdir(exist_ok=True)
    main()

