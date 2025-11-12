#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ドメイン別知識サイトPlaywrightスクレイピング（SO8T統合版）

既定のドメイン別知識サイト（防衛・航空宇宙・運輸・一般・NSFW検知用・違法薬物検知用）を
Playwrightで訪問し、人間を模倣した動きでWebスクレイピングして学習用データを収集。
SO8T/thinkingモデルで四重推論と四値分類によるラベル付け、データクレンジングも実行。

Usage:
    python scripts/data/collect_domain_knowledge_with_playwright.py \
        --output D:/webdataset/domain_knowledge_collected \
        --so8t_model_path models/so8t_thinking \
        --domains defense,aerospace,transport,general,nsfw_detection,drug_detection
"""

import sys
import json
import logging
import argparse
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from collections import Counter
from urllib.parse import urlparse, urljoin
import re

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.error("[ERROR] Playwright not installed. Install with: pip install playwright && playwright install chromium")

# BeautifulSoupインポート
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.error("[ERROR] BeautifulSoup not installed. Install with: pip install beautifulsoup4")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_domain_knowledge_playwright.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# HumanLikeScraperをインポート
try:
    from scripts.data.human_like_web_scraping import HumanLikeScraper
    HUMAN_LIKE_SCRAPER_AVAILABLE = True
except ImportError:
    HUMAN_LIKE_SCRAPER_AVAILABLE = False
    logger.warning("[WARNING] HumanLikeScraper not available, will implement basic scraping")


# ドメイン別サイトURL定義（統合版）
DOMAIN_SITES = {
    "defense": {
        "name": "防衛",
        "base_url": "https://www.mod.go.jp",
        "start_urls": [
            "https://www.mod.go.jp/j/press/wp/index.html",  # 防衛白書
            "https://www.mod.go.jp/j/policy/agenda/guideline/index.html",  # 防衛3文書
            "https://www.mod.go.jp/j/approach/defense/index.html",  # 防衛政策
            "https://www.nids.mod.go.jp/",  # 防衛研究所
        ],
        "selectors": {
            "title": "h1, .title, .page-title",
            "content": "main, .content, .main-content, #main-content",
            "links": "a[href^='/'], a[href^='https://www.mod.go.jp']"
        },
        "min_text_length": 500,
        "max_text_length": 15000,
        "language": "ja"
    },
    "aerospace": {
        "name": "航空宇宙",
        "base_url": "https://www.jaxa.jp",
        "start_urls": [
            "https://www.jaxa.jp/",
            "https://www.jaxa.jp/about/index_j.html",
            "https://www.jaxa.jp/projects/index_j.html",
            "https://www.jaxa.jp/news/index_j.html",
            "https://ntrs.nasa.gov/",  # NASA Technical Reports Server
        ],
        "selectors": {
            "title": "h1, .title, .page-title",
            "content": "main, .content, .main-content, article",
            "links": "a[href^='/'], a[href^='https://www.jaxa.jp']"
        },
        "min_text_length": 400,
        "max_text_length": 12000,
        "language": "ja"
    },
    "transport": {
        "name": "運輸",
        "base_url": "https://www.jreast.co.jp",
        "start_urls": [
            "https://www.jreast.co.jp/",  # JR東日本
            "https://jr-central.co.jp/",  # JR東海
            "https://www.westjr.co.jp/",  # JR西日本
            "https://www.ana.co.jp/ja/jp/",  # ANA
            "https://www.jal.co.jp/ja/",  # JAL
        ],
        "selectors": {
            "title": "h1, .title, .page-title",
            "content": "main, .content, .main-content, article",
            "links": "a[href^='/'], a[href^='https://']"
        },
        "min_text_length": 300,
        "max_text_length": 10000,
        "language": "ja"
    },
    "general": {
        "name": "一般",
        "base_url": "https://ja.wikipedia.org",
        "start_urls": [
            "https://ja.wikipedia.org/wiki/メインページ",
            "https://ja.wikipedia.org/wiki/Category:技術",
            "https://ja.wikipedia.org/wiki/Category:科学",
            "https://qiita.com/",
            "https://zenn.dev/",
        ],
        "selectors": {
            "title": "h1.firstHeading, h1, .title",
            "content": "#mw-content-text, main, .content, article",
            "links": "a[href^='/wiki/'], a[href^='/items/'], a[href^='/articles/']"
        },
        "min_text_length": 500,
        "max_text_length": 10000,
        "language": "ja"
    },
    "nsfw_detection": {
        "name": "NSFW検知用",
        "base_url": "https://www.fanza.co.jp",
        "start_urls": [
            "https://www.fanza.co.jp/",  # Fanza（検知目的のみ）
            "https://www.dmm.co.jp/",  # DMM（検知目的のみ）
            "https://live.fc2.com/",  # FC2（検知目的のみ）
            "https://missav.ai/",  # Missav（検知目的のみ）
        ],
        "selectors": {
            "title": "h1, .title, .page-title",
            "content": "main, .content, .main-content, article, .description",
            "links": "a[href^='/'], a[href^='https://']"
        },
        "min_text_length": 200,
        "max_text_length": 5000,
        "language": "ja",
        "purpose": "detection_only"  # 検知目的のみ
    },
    "drug_detection": {
        "name": "違法薬物検知用",
        "base_url": "https://www.pmda.go.jp",
        "start_urls": [
            "https://www.pmda.go.jp/",  # PMDA（検知目的のみ）
            "https://www.health.vic.gov.au/",  # GoV（検知目的のみ）
            "https://www.unodc.org/",  # UNODC（検知目的のみ）
            "https://www.emcdda.europa.eu/",  # EMCDDA（検知目的のみ）
            "https://ja.wikipedia.org/wiki/Category:薬物",  # Wikipedia薬物関連（検知目的のみ）
        ],
        "selectors": {
            "title": "h1, .title, .page-title",
            "content": "main, .content, .main-content, article",
            "links": "a[href^='/'], a[href^='https://']"
        },
        "min_text_length": 300,
        "max_text_length": 8000,
        "language": "ja",
        "purpose": "detection_only"  # 検知目的のみ
    }
}


class DomainKnowledgeCollector:
    """ドメイン別知識サイト収集クラス（SO8T統合版）"""
    
    def __init__(
        self,
        output_dir: Path,
        domains: List[str],
        so8t_model_path: Optional[str] = None,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        delay_per_request: float = 2.0,
        timeout: int = 30000,
        max_pages_per_domain: int = 100,
        max_depth: int = 3,
        quality_threshold: float = 0.7
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            domains: 収集するドメインリスト
            so8t_model_path: SO8T/thinkingモデルパス
            use_cursor_browser: Cursorブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_request: リクエスト間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_pages_per_domain: ドメインごとの最大ページ数
            max_depth: 最大クロール深度
            quality_threshold: 品質スコア閾値
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available. Install with: pip install playwright && playwright install chromium")
        if not BS4_AVAILABLE:
            raise RuntimeError("BeautifulSoup not available. Install with: pip install beautifulsoup4")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.domains = domains
        self.so8t_model_path = so8t_model_path
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_request = delay_per_request
        self.timeout = timeout
        self.max_pages_per_domain = max_pages_per_domain
        self.max_depth = max_depth
        self.quality_threshold = quality_threshold
        
        self.visited_urls: Set[str] = set()
        self.collected_samples: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # SO8T/thinkingモデル初期化
        self.so8t_pipeline = None
        self.quadruple_classifier = None
        if self.so8t_model_path:
            self._initialize_so8t_model()
        
        # HumanLikeScraper初期化
        if HUMAN_LIKE_SCRAPER_AVAILABLE:
            self.human_scraper = HumanLikeScraper(
                output_dir=self.output_dir / "raw",
                use_cursor_browser=self.use_cursor_browser,
                remote_debugging_port=self.remote_debugging_port,
                delay_per_request=self.delay_per_request,
                timeout=self.timeout,
                max_pages=1000,
                follow_links=True,
                max_depth=self.max_depth
            )
        else:
            self.human_scraper = None
            logger.warning("[WARNING] HumanLikeScraper not available, using basic scraping")
    
    def _initialize_so8t_model(self):
        """SO8T/thinkingモデルを初期化"""
        try:
            from scripts.agents.integrated_reasoning_pipeline import IntegratedReasoningPipeline
            
            logger.info(f"[SO8T] Initializing SO8T/thinking model from {self.so8t_model_path}...")
            self.so8t_pipeline = IntegratedReasoningPipeline(
                model_path=self.so8t_model_path
            )
            logger.info("[SO8T] SO8T/thinking model initialized successfully")
        except Exception as e:
            logger.warning(f"[SO8T] Failed to initialize SO8T/thinking model: {e}")
            logger.warning("[SO8T] Will continue without SO8T labeling")
            self.so8t_pipeline = None
        
        # QuadrupleClassifierも試す
        if not self.so8t_pipeline:
            try:
                from scripts.pipelines.web_scraping_data_pipeline import QuadrupleClassifier
                
                logger.info(f"[SO8T] Trying QuadrupleClassifier from {self.so8t_model_path}...")
                self.quadruple_classifier = QuadrupleClassifier(
                    model_path=self.so8t_model_path
                )
                logger.info("[SO8T] QuadrupleClassifier initialized successfully")
            except Exception as e:
                logger.warning(f"[SO8T] Failed to initialize QuadrupleClassifier: {e}")
                self.quadruple_classifier = None
    
    def extract_text_from_html(self, html: str, selectors: Dict[str, str]) -> Dict[str, str]:
        """HTMLからテキストを抽出"""
        soup = BeautifulSoup(html, 'lxml')
        
        result = {}
        
        # タイトル抽出
        if "title" in selectors:
            title_elem = soup.select_one(selectors["title"])
            if title_elem:
                result["title"] = title_elem.get_text(strip=True)
        
        # コンテンツ抽出
        if "content" in selectors:
            content_elem = soup.select_one(selectors["content"])
            if content_elem:
                # スクリプト・スタイル・ナビゲーション要素を削除
                for elem in content_elem.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    elem.decompose()
                
                text = content_elem.get_text(separator='\n', strip=True)
                # 余分な空白を削除
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                result["content"] = text.strip()
        
        # リンク抽出
        if "links" in selectors:
            links = []
            for link_elem in soup.select(selectors["links"]):
                href = link_elem.get('href', '')
                if href:
                    links.append(href)
            result["links"] = links
        
        return result
    
    async def scrape_page(
        self,
        page: Page,
        url: str,
        domain_config: Dict,
        depth: int = 0
    ) -> Optional[Dict]:
        """
        単一ページをスクレイピング
        
        Args:
            page: Playwright Pageオブジェクト
            url: スクレイピングするURL
            domain_config: ドメイン設定
            depth: 現在の深度
        
        Returns:
            サンプル辞書（成功時）またはNone
        """
        if url in self.visited_urls:
            return None
        
        if depth > self.max_depth:
            return None
        
        try:
            logger.info(f"[SCRAPE] [{depth}] {url}")
            
            # ページに移動
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")
            
            # 人間を模倣した動作（HumanLikeScraperを使用）
            if self.human_scraper:
                await self.human_scraper.human_like_wait(1.0, 2.5, longer=True)
                await self.human_scraper.human_like_mouse_move(page, use_bezier=True)
                await self.human_scraper.human_like_wait(0.5, 1.0)
                await self.human_scraper.human_like_scroll(page, gradual=True)
                await self.human_scraper.human_like_wait(0.5, 1.0)
                await self.human_scraper.human_like_hover(page, multiple_elements=True)
                await self.human_scraper.human_like_wait(0.5, 1.0)
            else:
                # 基本的な待機
                await asyncio.sleep(self.delay_per_request)
            
            # HTML取得
            html = await page.content()
            
            # テキスト抽出
            extracted = self.extract_text_from_html(html, domain_config["selectors"])
            
            if not extracted.get("content"):
                logger.debug(f"[SKIP] No content found: {url}")
                return None
            
            content = extracted["content"]
            min_length = domain_config.get("min_text_length", 200)
            max_length = domain_config.get("max_text_length", 10000)
            
            if len(content) < min_length:
                logger.debug(f"[SKIP] Text too short: {url} ({len(content)} chars)")
                return None
            
            if len(content) > max_length:
                content = content[:max_length]
            
            # サンプル作成
            sample = {
                "instruction": f"以下の{extracted.get('title', 'コンテンツ')}について説明してください。",
                "output": content,
                "title": extracted.get("title", ""),
                "url": url,
                "domain": domain_config.get("name", "unknown"),
                "source": domain_config.get("base_url", "unknown"),
                "depth": depth,
                "timestamp": datetime.now().isoformat()
            }
            
            self.visited_urls.add(url)
            return sample
            
        except PlaywrightTimeoutError:
            logger.warning(f"[TIMEOUT] {url}")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Failed to scrape {url}: {e}")
            return None
    
    def label_with_so8t(self, sample: Dict) -> Dict:
        """
        SO8T/thinkingモデルで四重推論と四値分類によるラベル付け
        
        Args:
            sample: サンプル辞書
        
        Returns:
            ラベル付け済みサンプル辞書
        """
        labeled_sample = sample.copy()
        
        if not self.so8t_pipeline and not self.quadruple_classifier:
            logger.debug("[SO8T] SO8T model not available, skipping labeling")
            labeled_sample["four_class_label"] = "ALLOW"
            labeled_sample["quality_score"] = 0.5
            return labeled_sample
        
        try:
            # テキストを取得
            text = sample.get("output", sample.get("text", ""))
            if not text:
                labeled_sample["four_class_label"] = "DENY"
                labeled_sample["quality_score"] = 0.0
                return labeled_sample
            
            # 四重推論と四値分類を実行
            if self.so8t_pipeline:
                # IntegratedReasoningPipelineを使用
                result = self.so8t_pipeline.process_with_integrated_reasoning(
                    query=text[:1000],  # 最初の1000文字を使用
                    use_knowledge=False,
                    use_classification=True,
                    use_rag=False
                )
                
                # 四重推論結果を追加
                if result.get("quadruple_thinking"):
                    labeled_sample["quadruple_thinking"] = result["quadruple_thinking"]
                
                # 四値分類ラベルを追加
                labeled_sample["four_class_label"] = result.get("four_class_label", "ALLOW")
                
                # 品質スコア（信頼度から推定）
                quality_score = 0.8  # デフォルト
                if result.get("four_class_classification"):
                    classification = result["four_class_classification"]
                    # 信頼度スコアから品質スコアを推定
                    confidence = classification.get("confidence", 0.8)
                    quality_score = confidence
                labeled_sample["quality_score"] = quality_score
                
            elif self.quadruple_classifier:
                # QuadrupleClassifierを使用
                classification_result = self.quadruple_classifier.classify_quadruple(sample)
                
                # 四重推論結果を追加
                if classification_result.get("quadruple_thinking"):
                    labeled_sample["quadruple_thinking"] = classification_result["quadruple_thinking"]
                
                # 四値分類ラベルを追加
                labeled_sample["four_class_label"] = classification_result.get("four_class_label", "ALLOW")
                
                # 品質スコア
                quality_score = classification_result.get("quality_score", 0.7)
                labeled_sample["quality_score"] = quality_score
            
            logger.debug(f"[SO8T] Labeled: {labeled_sample['four_class_label']}, Quality: {labeled_sample.get('quality_score', 0.0):.2f}")
            
        except Exception as e:
            logger.warning(f"[SO8T] Labeling failed: {e}")
            labeled_sample["four_class_label"] = "ALLOW"
            labeled_sample["quality_score"] = 0.5
        
        return labeled_sample
    
    def clean_data(self, samples: List[Dict]) -> List[Dict]:
        """
        SO8T/thinkingモデルを使用したデータクレンジング
        
        Args:
            samples: サンプルリスト
        
        Returns:
            クレンジング済みサンプルリスト
        """
        logger.info(f"[CLEAN] Starting data cleaning for {len(samples)} samples...")
        
        cleaned_samples = []
        seen_hashes: Set[str] = set()
        
        for sample in samples:
            # 1. 重複検出（URLとテキストハッシュ）
            text = sample.get("output", sample.get("text", ""))
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            url = sample.get("url", "")
            
            if text_hash in seen_hashes:
                logger.debug(f"[CLEAN] Duplicate detected (hash): {url}")
                continue
            seen_hashes.add(text_hash)
            
            # 2. ノイズ除去（短すぎる/長すぎるテキスト）
            if len(text) < 100:
                logger.debug(f"[CLEAN] Text too short: {url} ({len(text)} chars)")
                continue
            
            if len(text) > 50000:
                logger.debug(f"[CLEAN] Text too long: {url} ({len(text)} chars)")
                continue
            
            # 3. 品質スコアリング（SO8T/thinkingモデルを使用）
            if self.so8t_pipeline or self.quadruple_classifier:
                # 既にラベル付けされている場合はその品質スコアを使用
                quality_score = sample.get("quality_score", 0.5)
            else:
                # 基本的な品質スコアリング
                quality_score = 0.7  # デフォルト
            
            # 4. 閾値によるフィルタリング
            if quality_score < self.quality_threshold:
                logger.debug(f"[CLEAN] Quality score too low: {url} ({quality_score:.2f})")
                continue
            
            # 5. 四値分類ラベルによるフィルタリング（REFUSEは除外）
            four_class_label = sample.get("four_class_label", "ALLOW")
            if four_class_label == "REFUSE":
                logger.debug(f"[CLEAN] REFUSE label detected: {url}")
                continue
            
            cleaned_samples.append(sample)
        
        logger.info(f"[CLEAN] Cleaned: {len(cleaned_samples)}/{len(samples)} samples passed")
        return cleaned_samples
    
    async def crawl_domain(
        self,
        browser: Browser,
        domain_name: str,
        domain_config: Dict
    ) -> List[Dict]:
        """
        単一ドメインをクロール
        
        Args:
            browser: ブラウザオブジェクト
            domain_name: ドメイン名
            domain_config: ドメイン設定
        
        Returns:
            収集されたサンプルリスト
        """
        logger.info("="*80)
        logger.info(f"Crawling domain: {domain_name} ({domain_config.get('name', domain_name)})")
        logger.info("="*80)
        
        samples = []
        urls_to_visit = list(domain_config["start_urls"])
        visited_count = 0
        
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            while urls_to_visit and visited_count < self.max_pages_per_domain:
                url = urls_to_visit.pop(0)
                
                # 絶対URLに変換
                if not url.startswith("http"):
                    url = urljoin(domain_config["base_url"], url)
                
                # ページスクレイピング
                sample = await self.scrape_page(page, url, domain_config, depth=0)
                
                if sample:
                    # SO8T/thinkingモデルでラベル付け
                    labeled_sample = self.label_with_so8t(sample)
                    samples.append(labeled_sample)
                    visited_count += 1
                    
                    # リンクを追加（深度制限内）
                    if "links" in sample and sample.get("depth", 0) < self.max_depth:
                        for link in sample.get("links", [])[:10]:  # 最大10リンク
                            if link.startswith("/"):
                                full_url = urljoin(domain_config["base_url"], link)
                            elif link.startswith("http"):
                                full_url = link
                            else:
                                continue
                            
                            if full_url not in self.visited_urls:
                                urls_to_visit.append(full_url)
                
                # 進捗表示
                if visited_count % 10 == 0:
                    logger.info(f"[PROGRESS] {domain_name}: {visited_count} pages, {len(samples)} samples")
        
        finally:
            await context.close()
        
        logger.info(f"[COMPLETE] {domain_name}: {len(samples)} samples collected")
        return samples
    
    async def collect_all(self) -> List[Dict]:
        """すべてのドメインからデータを収集"""
        logger.info("="*80)
        logger.info("Starting Domain Knowledge Collection with Playwright (SO8T Integrated)")
        logger.info("="*80)
        logger.info(f"Domains: {', '.join(self.domains)}")
        logger.info(f"SO8T model: {self.so8t_model_path or 'Not specified'}")
        logger.info("")
        
        all_samples = []
        
        async with async_playwright() as playwright:
            browser = await self._connect_to_browser(playwright)
            
            try:
                for domain_name in self.domains:
                    if domain_name not in DOMAIN_SITES:
                        logger.warning(f"[SKIP] Unknown domain: {domain_name}")
                        continue
                    
                    domain_config = DOMAIN_SITES[domain_name].copy()
                    samples = await self.crawl_domain(browser, domain_name, domain_config)
                    all_samples.extend(samples)
                    
                    logger.info(f"[TOTAL] {len(all_samples)} samples collected so far")
                
            finally:
                await browser.close()
        
        # データクレンジング
        logger.info("="*80)
        logger.info("Starting data cleaning...")
        logger.info("="*80)
        cleaned_samples = self.clean_data(all_samples)
        
        return cleaned_samples
    
    async def _connect_to_browser(self, playwright) -> Browser:
        """ブラウザに接続"""
        if self.use_cursor_browser:
            try:
                cdp_endpoint = f"http://127.0.0.1:{self.remote_debugging_port}"
                browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
                logger.info(f"[BROWSER] Connected to Cursor browser at {cdp_endpoint}")
                return browser
            except Exception as e:
                logger.warning(f"[BROWSER] Failed to connect to Cursor browser: {e}")
                logger.info("[BROWSER] Launching new browser...")
        
        browser = await playwright.chromium.launch(headless=False)
        logger.info("[BROWSER] Browser launched")
        return browser
    
    def save_samples(self, samples: List[Dict], suffix: str = ""):
        """サンプルを保存"""
        # JSONL形式で保存
        output_file = self.output_dir / f"domain_knowledge_{self.session_id}{suffix}.jsonl"
        
        logger.info(f"Saving {len(samples):,} samples to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Saved to {output_file}")
        
        # 統計情報を保存
        stats_file = self.output_dir / f"stats_{self.session_id}{suffix}.json"
        stats = {
            "total_samples": len(samples),
            "domains": Counter(s["domain"] for s in samples),
            "four_class_labels": Counter(s.get("four_class_label", "ALLOW") for s in samples),
            "avg_quality_score": sum(s.get("quality_score", 0.5) for s in samples) / len(samples) if samples else 0,
            "avg_content_length": sum(len(s.get("output", "")) for s in samples) / len(samples) if samples else 0,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[STATS] Statistics saved to {stats_file}")
        logger.info(f"  Total samples: {stats['total_samples']:,}")
        logger.info(f"  Domains: {dict(stats['domains'])}")
        logger.info(f"  Four-class labels: {dict(stats['four_class_labels'])}")
        logger.info(f"  Avg quality score: {stats['avg_quality_score']:.2f}")
        logger.info(f"  Avg content length: {stats['avg_content_length']:.0f} chars")


async def main():
    parser = argparse.ArgumentParser(description="Domain Knowledge Collection with Playwright (SO8T Integrated)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--domains", type=str, default="defense,aerospace,transport,general",
                       help="Comma-separated list of domains")
    parser.add_argument("--so8t_model_path", type=str, default=None,
                       help="SO8T/thinking model path")
    parser.add_argument("--use_cursor_browser", action="store_true",
                       help="Use Cursor browser (CDP connection)")
    parser.add_argument("--remote_debugging_port", type=int, default=9222,
                       help="Remote debugging port (for Cursor browser)")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between requests (seconds)")
    parser.add_argument("--timeout", type=int, default=30000,
                       help="Page load timeout (milliseconds)")
    parser.add_argument("--max_pages_per_domain", type=int, default=100,
                       help="Maximum pages per domain")
    parser.add_argument("--max_depth", type=int, default=3,
                       help="Maximum crawl depth")
    parser.add_argument("--quality_threshold", type=float, default=0.7,
                       help="Quality score threshold for filtering")
    
    args = parser.parse_args()
    
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("[ERROR] Playwright not installed. Install with: pip install playwright && playwright install chromium")
        return 1
    
    domains = [d.strip() for d in args.domains.split(",")]
    
    collector = DomainKnowledgeCollector(
        output_dir=Path(args.output),
        domains=domains,
        so8t_model_path=args.so8t_model_path,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_request=args.delay,
        timeout=args.timeout,
        max_pages_per_domain=args.max_pages_per_domain,
        max_depth=args.max_depth,
        quality_threshold=args.quality_threshold
    )
    
    # データ収集実行
    samples = await collector.collect_all()
    
    # サンプル保存
    collector.save_samples(samples, suffix="_cleaned")
    
    logger.info("="*80)
    logger.info("Data collection completed!")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))






