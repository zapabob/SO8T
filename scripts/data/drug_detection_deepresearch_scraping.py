#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
違法薬物検知目的DeepResearch Webスクレイピングスクリプト

10ブラウザ×10タブ（合計100タブ）の並列構成で、違法薬物検知目的に特化した
DeepResearch Webスクレイピングを実行します。

重要: この実装は検知目的のみで、生成目的ではない。安全判定と拒否挙動の学習を目的とする。

データソース:
- PMDA (医薬品医療機器総合機構)
- FDA (Food and Drug Administration)
- e-Gov (日本の法令データベース)
- WHO (World Health Organization)
- UNODC (United Nations Office on Drugs and Crime)
- EMCDDA (European Monitoring Centre for Drugs and Drug Addiction)
- Wikipedia (違法薬物・医薬品関連記事)
- 技術ドキュメントサイト（GitHub、Stack Overflow、Qiita、Zenn、Dev.to等）
- コーディング教育サイト（freeCodeCamp、Codecademy等）

Usage:
    python scripts/data/drug_detection_deepresearch_scraping.py --output D:/webdataset/drug_detection_deepresearch
"""

import sys
import json
import logging
import asyncio
import argparse
import random
import time
import psutil
import os
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse, quote, urljoin
from collections import deque, Counter
from dataclasses import dataclass

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[ERROR] Playwright not installed. Install with: pip install playwright")
    sys.exit(1)

# BeautifulSoupインポート
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("[ERROR] BeautifulSoup not installed. Install with: pip install beautifulsoup4")
    sys.exit(1)

# ParallelTabProcessorインポート
try:
    from scripts.data.parallel_tab_processor import ParallelTabProcessor
    PARALLEL_TAB_PROCESSOR_AVAILABLE = True
except ImportError:
    PARALLEL_TAB_PROCESSOR_AVAILABLE = False
    print("[WARNING] ParallelTabProcessor not available")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/drug_detection_deepresearch_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 違法薬物検知用キーワードカテゴリ
DRUG_DETECTION_KEYWORDS = {
    'illegal_drugs': {
        'keywords_ja': ['違法薬物', '麻薬', '覚醒剤', '大麻', 'コカイン', 'ヘロイン', 'MDMA', 'LSD', '幻覚剤', '向精神薬'],
        'keywords_en': ['illegal drugs', 'narcotics', 'amphetamine', 'cannabis', 'cocaine', 'heroin', 'MDMA', 'LSD', 'hallucinogen', 'psychotropic'],
        'severity': 'high',
        'legal_status': 'illegal'
    },
    'prescription_drugs_abuse': {
        'keywords_ja': ['処方薬乱用', 'オピオイド乱用', '鎮痛剤乱用', '睡眠薬乱用', '抗不安薬乱用'],
        'keywords_en': ['prescription drug abuse', 'opioid abuse', 'painkiller abuse', 'sleeping pill abuse', 'anxiety medication abuse'],
        'severity': 'high',
        'legal_status': 'illegal_when_abused'
    },
    'controlled_substances': {
        'keywords_ja': ['指定薬物', '規制薬物', '向精神薬', '麻薬指定', '覚醒剤指定'],
        'keywords_en': ['controlled substance', 'scheduled drug', 'psychotropic substance', 'narcotic', 'amphetamine'],
        'severity': 'high',
        'legal_status': 'controlled'
    },
    'drug_trafficking': {
        'keywords_ja': ['薬物密輸', '薬物取引', '薬物販売', '薬物密売'],
        'keywords_en': ['drug trafficking', 'drug trade', 'drug dealing', 'drug smuggling'],
        'severity': 'critical',
        'legal_status': 'illegal'
    },
    'drug_manufacturing': {
        'keywords_ja': ['薬物製造', '覚醒剤製造', '麻薬製造', '違法製造'],
        'keywords_en': ['drug manufacturing', 'amphetamine manufacturing', 'narcotic manufacturing', 'illegal manufacturing'],
        'severity': 'critical',
        'legal_status': 'illegal'
    }
}

# ドメイン別知識キーワード
DOMAIN_KNOWLEDGE_KEYWORDS = {
    'technology': {
        'keywords_ja': ['人工知能', '機械学習', '深層学習', '自然言語処理', 'コンピュータビジョン', 'ブロックチェーン', '暗号通貨', '量子コンピュータ', 'IoT', '5G'],
        'keywords_en': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'blockchain', 'cryptocurrency', 'quantum computing', 'IoT', '5G']
    },
    'programming_languages': {
        'keywords_ja': ['Python', 'JavaScript', 'TypeScript', 'Go', 'Rust', 'C++', 'C#', 'Swift', 'Kotlin', 'Java', 'SQL'],
        'keywords_en': ['Python', 'JavaScript', 'TypeScript', 'Go', 'Rust', 'C++', 'C#', 'Swift', 'Kotlin', 'Java', 'SQL']
    },
    'algorithms': {
        'keywords_ja': ['アルゴリズム', 'データ構造', 'ソート', '検索', 'グラフ理論', '動的計画法', '貪欲法', '分割統治法'],
        'keywords_en': ['algorithm', 'data structure', 'sorting', 'searching', 'graph theory', 'dynamic programming', 'greedy algorithm', 'divide and conquer']
    },
    'hardware': {
        'keywords_ja': ['ハードウェア', 'CPU', 'メモリ', 'RAM', 'GPU', 'TPU', 'FPGA', 'ASIC', 'マイクロプロセッサ'],
        'keywords_en': ['hardware', 'CPU', 'memory', 'RAM', 'GPU', 'TPU', 'FPGA', 'ASIC', 'microprocessor']
    },
    'coding_best_practices': {
        'keywords_ja': ['ベストプラクティス', 'コーディング規約', 'コードレビュー', 'リファクタリング', 'デザインパターン', 'SOLID原則', 'テスト駆動開発'],
        'keywords_en': ['best practices', 'coding standards', 'code review', 'refactoring', 'design patterns', 'SOLID principles', 'test-driven development']
    }
}

# コーディング能力向上用キーワード
CODING_ABILITY_KEYWORDS = {
    'code_examples': {
        'keywords_ja': ['コード例', 'サンプルコード', '実装例', '使い方', 'チュートリアル'],
        'keywords_en': ['code example', 'sample code', 'implementation example', 'how to', 'tutorial']
    },
    'tutorials': {
        'keywords_ja': ['チュートリアル', '入門', '初心者', '上級者', 'エキスパート', 'ガイド'],
        'keywords_en': ['tutorial', 'beginner', 'advanced', 'expert', 'guide', 'getting started']
    },
    'documentation': {
        'keywords_ja': ['ドキュメント', 'APIリファレンス', '仕様書', 'マニュアル', 'リファレンス'],
        'keywords_en': ['documentation', 'API reference', 'specification', 'manual', 'reference']
    }
}

# NSFW検知用キーワード（検知目的のみ、生成目的ではない）
NSFW_DETECTION_KEYWORDS = {
    'nsfw_detection': {
        'keywords_ja': ['成人向け', 'アダルト', '18禁', 'R18', 'エロ', '性的'],
        'keywords_en': ['adult', '18+', 'NSFW', 'explicit', 'sexual', 'mature']
    }
}

# データソース定義
DATA_SOURCES = {
    'pmda': {
        'base_url': 'https://www.pmda.go.jp',
        'name': 'PMDA (医薬品医療機器総合機構)',
        'language': 'ja',
        'enabled': True
    },
    'fda': {
        'base_url': 'https://www.fda.gov',
        'name': 'FDA (Food and Drug Administration)',
        'language': 'en',
        'enabled': True
    },
    'egov': {
        'base_url': 'https://elaws.e-gov.go.jp',
        'name': 'e-Gov (日本の法令データベース)',
        'language': 'ja',
        'enabled': True
    },
    'who': {
        'base_url': 'https://www.who.int',
        'name': 'WHO (World Health Organization)',
        'language': 'en',
        'enabled': True
    },
    'unodc': {
        'base_url': 'https://www.unodc.org',
        'name': 'UNODC (United Nations Office on Drugs and Crime)',
        'language': 'en',
        'enabled': True
    },
    'emcdda': {
        'base_url': 'https://www.emcdda.europa.eu',
        'name': 'EMCDDA (European Monitoring Centre for Drugs and Drug Addiction)',
        'language': 'en',
        'enabled': True
    },
    'wikipedia': {
        'base_url': 'https://ja.wikipedia.org',
        'name': 'Wikipedia',
        'language': 'ja',
        'enabled': True
    },
    'github': {
        'base_url': 'https://github.com',
        'name': 'GitHub',
        'language': 'en',
        'enabled': True
    },
    'stackoverflow': {
        'base_url': 'https://stackoverflow.com',
        'name': 'Stack Overflow',
        'language': 'en',
        'enabled': True
    },
    'qiita': {
        'base_url': 'https://qiita.com',
        'name': 'Qiita',
        'language': 'ja',
        'enabled': True
    },
    'zenn': {
        'base_url': 'https://zenn.dev',
        'name': 'Zenn',
        'language': 'ja',
        'enabled': True
    },
    'devto': {
        'base_url': 'https://dev.to',
        'name': 'Dev.to',
        'language': 'en',
        'enabled': True
    },
    'freecodecamp': {
        'base_url': 'https://www.freecodecamp.org',
        'name': 'freeCodeCamp',
        'language': 'en',
        'enabled': True
    },
    'codecademy': {
        'base_url': 'https://www.codecademy.com',
        'name': 'Codecademy',
        'language': 'en',
        'enabled': True
    }
}


@dataclass
class KeywordTask:
    """キーワードタスク"""
    keyword: str
    category: str
    language: str
    data_source: str
    related_keywords: List[str] = None
    url: str = None
    detection_purpose: str = 'safety_training'


class ResourceManager:
    """動的リソース管理クラス"""
    
    def __init__(self, max_memory_gb: float = 8.0, max_cpu_percent: float = 80.0):
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent
    
    def get_memory_usage_gb(self) -> float:
        """現在のメモリ使用量を取得（GB）"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb / 1024
    
    def get_cpu_percent(self) -> float:
        """現在のCPU使用率を取得（%）"""
        return psutil.cpu_percent(interval=0.1)
    
    def can_allocate_browser(self) -> bool:
        """ブラウザを割り当て可能かチェック"""
        memory_gb = self.get_memory_usage_gb()
        cpu_percent = self.get_cpu_percent()
        return memory_gb < self.max_memory_gb and cpu_percent < self.max_cpu_percent
    
    def get_resource_status(self) -> Dict:
        """リソース状態を取得"""
        return {
            'memory_gb': self.get_memory_usage_gb(),
            'cpu_percent': self.get_cpu_percent(),
            'max_memory_gb': self.max_memory_gb,
            'max_cpu_percent': self.max_cpu_percent
        }


class DrugDetectionDeepResearchScraper:
    """違法薬物検知目的DeepResearch Webスクレイピングクラス"""
    
    def __init__(
        self,
        output_dir: Path,
        num_browsers: int = 10,
        num_tabs: int = 10,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        delay_per_action: float = 1.5,
        timeout: int = 30000,
        max_memory_gb: float = 8.0,
        max_cpu_percent: float = 80.0
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            num_browsers: ブラウザ数（デフォルト: 10）
            num_tabs: 各ブラウザのタブ数（デフォルト: 10）
            use_cursor_browser: Cursorのブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_action: アクション間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_memory_gb: 最大メモリ使用量（GB）
            max_cpu_percent: 最大CPU使用率（%）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_browsers = num_browsers
        self.num_tabs = num_tabs
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_action = delay_per_action
        self.timeout = timeout
        
        self.resource_manager = ResourceManager(max_memory_gb, max_cpu_percent)
        
        self.all_samples: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # キーワードキュー
        self.keyword_queue: deque = deque()
        self.completed_keywords: Set[str] = set()
        
        # ブラウザ管理
        self.browsers: List[Browser] = []
        self.contexts: List[BrowserContext] = []
        
        # カテゴリ別サンプル
        self.samples_by_category: Dict[str, List[Dict]] = {
            'illegal_drugs': [],
            'prescription_drugs_abuse': [],
            'controlled_substances': [],
            'drug_trafficking': [],
            'drug_manufacturing': [],
            'domain_knowledge': [],
            'coding_ability': [],
            'nsfw_detection': []
        }
        
        logger.info("="*80)
        logger.info("Drug Detection DeepResearch Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of browsers: {self.num_browsers}")
        logger.info(f"Number of tabs per browser: {self.num_tabs}")
        logger.info(f"Total parallel tasks: {self.num_browsers * self.num_tabs}")
        logger.info(f"Detection purpose: safety_training (NOT for generation)")
    
    def generate_deepresearch_url(self, keyword: str, language: str, data_source: str) -> str:
        """
        DeepResearchクエリURLを生成
        
        Args:
            keyword: キーワード
            language: 言語
            data_source: データソース
        
        Returns:
            DeepResearch URL
        """
        # DeepResearchのベースURL（実際のDeepResearch APIを使用する場合は適宜変更）
        # ここでは検索エンジンを使用したDeepResearchクエリを生成
        keyword_encoded = quote(keyword)
        
        # データソースに応じたURL生成
        if data_source == 'pmda':
            return f"https://www.google.com/search?hl=ja&q={keyword_encoded}+site:pmda.go.jp"
        elif data_source == 'fda':
            return f"https://www.google.com/search?q={keyword_encoded}+site:fda.gov"
        elif data_source == 'egov':
            return f"https://www.google.com/search?hl=ja&q={keyword_encoded}+site:elaws.e-gov.go.jp"
        elif data_source == 'who':
            return f"https://www.google.com/search?q={keyword_encoded}+site:who.int"
        elif data_source == 'unodc':
            return f"https://www.google.com/search?q={keyword_encoded}+site:unodc.org"
        elif data_source == 'emcdda':
            return f"https://www.google.com/search?q={keyword_encoded}+site:emcdda.europa.eu"
        elif data_source == 'wikipedia':
            if language == 'ja':
                return f"https://ja.wikipedia.org/wiki/{keyword_encoded}"
            else:
                return f"https://en.wikipedia.org/wiki/{keyword_encoded}"
        elif data_source == 'github':
            return f"https://github.com/search?q={keyword_encoded}"
        elif data_source == 'stackoverflow':
            return f"https://stackoverflow.com/search?q={keyword_encoded}"
        elif data_source == 'qiita':
            return f"https://qiita.com/search?q={keyword_encoded}"
        elif data_source == 'zenn':
            return f"https://zenn.dev/search?q={keyword_encoded}"
        elif data_source == 'devto':
            return f"https://dev.to/search?q={keyword_encoded}"
        elif data_source == 'freecodecamp':
            return f"https://www.freecodecamp.org/news/search/?query={keyword_encoded}"
        elif data_source == 'codecademy':
            return f"https://www.codecademy.com/search?query={keyword_encoded}"
        else:
            # デフォルトはGoogle検索
            if language == 'ja':
                return f"https://www.google.com/search?hl=ja&q={keyword_encoded}"
            else:
                return f"https://www.google.com/search?q={keyword_encoded}"
    
    def initialize_keyword_queue(self):
        """キーワードキューを初期化"""
        logger.info("[QUEUE] Initializing keyword queue...")
        
        # 違法薬物検知用キーワード
        for category, info in DRUG_DETECTION_KEYWORDS.items():
            for language in ['ja', 'en']:
                keywords = info['keywords_ja'] if language == 'ja' else info['keywords_en']
                for keyword in keywords:
                    # データソースごとにタスクを作成
                    for source_id, source_info in DATA_SOURCES.items():
                        if source_info['enabled'] and source_info['language'] == language:
                            task = KeywordTask(
                                keyword=keyword,
                                category=category,
                                language=language,
                                data_source=source_id,
                                url=self.generate_deepresearch_url(keyword, language, source_id)
                            )
                            self.keyword_queue.append(task)
        
        # ドメイン別知識キーワード
        for category, info in DOMAIN_KNOWLEDGE_KEYWORDS.items():
            for language in ['ja', 'en']:
                keywords = info['keywords_ja'] if language == 'ja' else info['keywords_en']
                for keyword in keywords:
                    # 技術ドキュメントサイトに限定
                    tech_sources = ['github', 'stackoverflow', 'qiita', 'zenn', 'devto']
                    for source_id in tech_sources:
                        if source_id in DATA_SOURCES and DATA_SOURCES[source_id]['enabled']:
                            task = KeywordTask(
                                keyword=keyword,
                                category='domain_knowledge',
                                language=language,
                                data_source=source_id,
                                url=self.generate_deepresearch_url(keyword, language, source_id)
                            )
                            self.keyword_queue.append(task)
        
        # コーディング能力向上用キーワード
        for category, info in CODING_ABILITY_KEYWORDS.items():
            for language in ['ja', 'en']:
                keywords = info['keywords_ja'] if language == 'ja' else info['keywords_en']
                for keyword in keywords:
                    # コーディング教育サイトに限定
                    coding_sources = ['freecodecamp', 'codecademy', 'github', 'stackoverflow']
                    for source_id in coding_sources:
                        if source_id in DATA_SOURCES and DATA_SOURCES[source_id]['enabled']:
                            task = KeywordTask(
                                keyword=keyword,
                                category='coding_ability',
                                language=language,
                                data_source=source_id,
                                url=self.generate_deepresearch_url(keyword, language, source_id)
                            )
                            self.keyword_queue.append(task)
        
        # NSFW検知用キーワード（検知目的のみ、生成目的ではない）
        for category, info in NSFW_DETECTION_KEYWORDS.items():
            for language in ['ja', 'en']:
                keywords = info['keywords_ja'] if language == 'ja' else info['keywords_en']
                for keyword in keywords:
                    # 検知目的のため、一般的な検索エンジンを使用
                    task = KeywordTask(
                        keyword=keyword,
                        category='nsfw_detection',
                        language=language,
                        data_source='google',
                        url=self.generate_deepresearch_url(keyword, language, 'google')
                    )
                    self.keyword_queue.append(task)
        
        logger.info(f"[QUEUE] Initialized {len(self.keyword_queue)} keyword tasks")
    
    async def check_cursor_browser_running(self, port: int) -> bool:
        """Cursorブラウザが起動しているかチェック"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/json/version", timeout=aiohttp.ClientTimeout(total=2.0)) as response:
                    if response.status == 200:
                        return True
        except Exception:
            pass
        return False
    
    async def launch_cursor_browser_background(self, port: int) -> bool:
        """Cursorブラウザをバックグラウンドで起動"""
        try:
            cursor_paths = [
                r"C:\Users\{}\AppData\Local\Programs\cursor\Cursor.exe".format(os.environ.get('USERNAME', 'downl')),
                r"C:\Program Files\Cursor\Cursor.exe",
                r"C:\Program Files (x86)\Cursor\Cursor.exe",
            ]
            
            cursor_exe = None
            for path in cursor_paths:
                if Path(path).exists():
                    cursor_exe = path
                    break
            
            if not cursor_exe:
                logger.warning(f"[BROWSER] Cursor executable not found")
                return False
            
            cmd = [
                cursor_exe,
                f"--remote-debugging-port={port}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding"
            ]
            
            logger.info(f"[BROWSER] Launching Cursor browser on port {port}...")
            if platform.system() == "Windows":
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            for i in range(20):
                await asyncio.sleep(0.5)
                if await self.check_cursor_browser_running(port):
                    logger.info(f"[OK] Cursor browser launched on port {port}")
                    return True
            
            logger.warning(f"[BROWSER] Cursor browser may not have started")
            return False
            
        except Exception as e:
            logger.error(f"[BROWSER] Failed to launch Cursor browser: {e}")
            return False
    
    async def connect_to_cursor_browser(self, playwright, browser_index: int) -> Optional[Browser]:
        """Cursorのブラウザに接続"""
        if not self.use_cursor_browser:
            logger.info(f"[BROWSER {browser_index}] Launching new browser...")
            browser = await playwright.chromium.launch(headless=False)
            logger.info(f"[OK] Browser {browser_index} launched")
            return browser
        
        port = self.remote_debugging_port + browser_index
        
        try:
            logger.info(f"[BROWSER {browser_index}] Connecting to Cursor browser on port {port}...")
            cdp_endpoint = f"http://127.0.0.1:{port}"
            browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
            
            contexts = browser.contexts
            if contexts:
                logger.info(f"[OK] Browser {browser_index} connected ({len(contexts)} contexts)")
            else:
                await browser.new_context()
                logger.info(f"[OK] Browser {browser_index}: New context created")
            
            return browser
            
        except Exception as e:
            logger.warning(f"[BROWSER {browser_index}] Failed to connect: {e}")
            
            browser_running = await self.check_cursor_browser_running(port)
            if not browser_running:
                launch_success = await self.launch_cursor_browser_background(port)
                if launch_success:
                    await asyncio.sleep(2.0)
                    try:
                        cdp_endpoint = f"http://127.0.0.1:{port}"
                        browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
                        contexts = browser.contexts
                        if contexts:
                            logger.info(f"[OK] Browser {browser_index} connected after launch")
                        else:
                            await browser.new_context()
                            logger.info(f"[OK] Browser {browser_index}: New context created")
                        return browser
                    except Exception as reconnect_error:
                        logger.warning(f"[BROWSER {browser_index}] Failed to connect after launch: {reconnect_error}")
            
            logger.info(f"[BROWSER {browser_index}] Falling back to launching new browser...")
            browser = await playwright.chromium.launch(headless=False)
            logger.info(f"[OK] Browser {browser_index} launched")
            return browser
    
    async def scrape_url(self, page: Page, url: str, task: KeywordTask) -> Optional[Dict]:
        """
        URLをスクレイピング
        
        Args:
            page: ページインスタンス
            url: スクレイピングするURL
            task: キーワードタスク
        
        Returns:
            スクレイピング結果
        """
        try:
            logger.info(f"[SCRAPE] Scraping URL: {url}")
            
            await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            html = await page.content()
            
            if BS4_AVAILABLE:
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                title = soup.title.string if soup.title else ""
            else:
                text = html
                title = ""
            
            sample = {
                'text': text[:10000],  # 最大10000文字
                'url': url,
                'title': title,
                'keyword': task.keyword,
                'category': task.category,
                'language': task.language,
                'data_source': task.data_source,
                'source_name': DATA_SOURCES.get(task.data_source, {}).get('name', 'Unknown'),
                'collected_at': datetime.now().isoformat(),
                'detection_purpose': 'safety_training'
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"[SCRAPE] Failed to scrape URL {url}: {e}")
            return None
    
    async def browser_worker(self, browser_index: int, playwright):
        """ブラウザワーカー（10タブ並列処理）"""
        logger.info(f"[WORKER {browser_index}] Starting browser worker...")
        
        browser = None
        for retry in range(3):
            try:
                browser = await self.connect_to_cursor_browser(playwright, browser_index)
                if browser:
                    break
            except Exception as e:
                logger.warning(f"[WORKER {browser_index}] Browser connection failed (attempt {retry + 1}/3): {e}")
                if retry < 2:
                    await asyncio.sleep(5.0 * (retry + 1))
        
        if not browser:
            logger.error(f"[WORKER {browser_index}] Failed to connect browser")
            return
        
        self.browsers.append(browser)
        
        try:
            # コンテキスト作成
            contexts = browser.contexts
            if contexts:
                context = contexts[0]
            else:
                context = await browser.new_context()
            
            self.contexts.append(context)
            
            # ParallelTabProcessorを使用して10タブを初期化
            if PARALLEL_TAB_PROCESSOR_AVAILABLE:
                tab_processor = ParallelTabProcessor(
                    context,
                    num_tabs=self.num_tabs,
                    delay_per_action=self.delay_per_action,
                    timeout=self.timeout
                )
                
                await tab_processor.initialize_tabs()
                
                # キーワードタスクを取得してURLリストを作成
                urls = []
                tasks = []
                
                while len(urls) < self.num_tabs * 10:  # 各タブに10個のURLを割り当て
                    if not self.keyword_queue:
                        break
                    
                    task = self.keyword_queue.popleft()
                    
                    if task.keyword in self.completed_keywords:
                        continue
                    
                    self.completed_keywords.add(task.keyword)
                    urls.append(task.url)
                    tasks.append(task)
                
                # カスタム処理関数を定義
                async def process_url_func(page: Page, tab_index: int, url: str) -> Optional[Dict]:
                    # URLに対応するタスクを取得
                    task_index = urls.index(url) if url in urls else None
                    if task_index is not None and task_index < len(tasks):
                        task = tasks[task_index]
                        return await self.scrape_url(page, url, task)
                    return None
                
                # 並列処理実行
                results = await tab_processor.process_tabs_parallel(urls, process_func=process_url_func)
                
                # 結果を統合
                for result in results:
                    if result:
                        self.all_samples.append(result)
                        
                        # カテゴリ別に分類
                        category = result.get('category', 'unknown')
                        if category in self.samples_by_category:
                            self.samples_by_category[category].append(result)
                        elif category.startswith('illegal') or category.startswith('prescription') or category.startswith('controlled') or category.startswith('drug'):
                            # 違法薬物関連カテゴリ
                            self.samples_by_category['illegal_drugs'].append(result)
                
                # タブを閉じる
                await tab_processor.close_tabs()
                
            else:
                logger.warning(f"[WORKER {browser_index}] ParallelTabProcessor not available, using single tab")
                page = await context.new_page()
                
                # 単一タブで処理
                for _ in range(10):
                    if not self.keyword_queue:
                        break
                    
                    task = self.keyword_queue.popleft()
                    if task.keyword in self.completed_keywords:
                        continue
                    
                    self.completed_keywords.add(task.keyword)
                    result = await self.scrape_url(page, task.url, task)
                    
                    if result:
                        self.all_samples.append(result)
                        category = result.get('category', 'unknown')
                        if category in self.samples_by_category:
                            self.samples_by_category[category].append(result)
                
                await page.close()
        
        except Exception as e:
            logger.error(f"[WORKER {browser_index}] Worker error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"[WORKER {browser_index}] Browser worker finished")
    
    async def run_parallel_scraping(self):
        """並列スクレイピング実行"""
        logger.info("="*80)
        logger.info("Starting Drug Detection DeepResearch Scraping")
        logger.info("="*80)
        
        # キーワードキュー初期化
        self.initialize_keyword_queue()
        
        # Playwright起動
        async with async_playwright() as playwright:
            # 並列ブラウザワーカー起動
            tasks = []
            for i in range(self.num_browsers):
                task = asyncio.create_task(self.browser_worker(i, playwright))
                tasks.append(task)
                await asyncio.sleep(1.0)  # 起動間隔
            
            # すべてのワーカーが完了するまで待機
            await asyncio.gather(*tasks)
        
        logger.info(f"[TOTAL] Collected {len(self.all_samples)} samples")
        
        # カテゴリ別統計
        for category, samples in self.samples_by_category.items():
            logger.info(f"[CATEGORY {category}] Collected {len(samples)} samples")
    
    def save_samples(self, samples: List[Dict], filename: str = None) -> Path:
        """サンプルを保存"""
        if filename is None:
            filename = f"drug_detection_deepresearch_{self.session_id}.jsonl"
        
        output_file = self.output_dir / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Saved {len(samples)} samples to {output_file}")
        return output_file
    
    def save_samples_by_category(self):
        """カテゴリ別にサンプルを保存"""
        for category, samples in self.samples_by_category.items():
            if samples:
                filename = f"drug_detection_deepresearch_{category}_{self.session_id}.jsonl"
                self.save_samples(samples, filename)
        
        # メタデータを保存
        metadata = {
            'total_samples': len(self.all_samples),
            'samples_by_category': {cat: len(samples) for cat, samples in self.samples_by_category.items()},
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'detection_purpose': 'safety_training',
            'num_browsers': self.num_browsers,
            'num_tabs': self.num_tabs,
            'total_parallel_tasks': self.num_browsers * self.num_tabs
        }
        
        metadata_file = self.output_dir / f"metadata_{self.session_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[OK] Metadata saved to {metadata_file}")


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Drug Detection DeepResearch Web Scraping')
    parser.add_argument('--output', type=Path, default=Path('D:/webdataset/drug_detection_deepresearch'), help='Output directory')
    parser.add_argument('--num-browsers', type=int, default=10, help='Number of browsers (default: 10)')
    parser.add_argument('--num-tabs', type=int, default=10, help='Number of tabs per browser (default: 10)')
    parser.add_argument('--use-cursor-browser', action='store_true', default=True, help='Use Cursor browser')
    parser.add_argument('--remote-debugging-port', type=int, default=9222, help='Remote debugging port (base port)')
    parser.add_argument('--delay', type=float, default=1.5, help='Delay between actions (seconds)')
    parser.add_argument('--timeout', type=int, default=30000, help='Page load timeout (milliseconds)')
    parser.add_argument('--max-memory-gb', type=float, default=8.0, help='Maximum memory usage (GB)')
    parser.add_argument('--max-cpu-percent', type=float, default=80.0, help='Maximum CPU usage (%)')
    
    args = parser.parse_args()
    
    scraper = DrugDetectionDeepResearchScraper(
        output_dir=args.output,
        num_browsers=args.num_browsers,
        num_tabs=args.num_tabs,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_action=args.delay,
        timeout=args.timeout,
        max_memory_gb=args.max_memory_gb,
        max_cpu_percent=args.max_cpu_percent
    )
    
    # 並列スクレイピング実行
    await scraper.run_parallel_scraping()
    
    # 保存
    output_file = scraper.save_samples(scraper.all_samples)
    scraper.save_samples_by_category()
    
    logger.info(f"[SUCCESS] Drug Detection DeepResearch scraping completed. Output: {output_file}")
    return output_file


if __name__ == '__main__':
    asyncio.run(main())
