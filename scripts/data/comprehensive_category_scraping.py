#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
包括的カテゴリWebスクレイピングスクリプト

日本語・英語の広範なカテゴリ（NSFW含む）とArxivを含む包括的なWebスクレイピングを実行します。
人間を模倣した動作でCursorブラウザを使用します。

Usage:
    python scripts/data/comprehensive_category_scraping.py --output D:\webdataset\processed
"""

import sys
import json
import logging
import asyncio
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
except ImportError:
    print("[ERROR] Playwright not installed. Install with: pip install playwright")
    sys.exit(1)

# BeautifulSoupインポート
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("[ERROR] BeautifulSoup not installed. Install with: pip install beautifulsoup4")
    sys.exit(1)

# ArxivCrawlerインポート
try:
    from so8t_mmllm.scripts.data.arxiv_crawler import ArxivCrawler
    ARXIV_AVAILABLE = True
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "arxiv_crawler",
            PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "arxiv_crawler.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ArxivCrawler = module.ArxivCrawler
        ARXIV_AVAILABLE = True
    except Exception:
        ARXIV_AVAILABLE = False
        logger.warning("Arxiv crawler not available")

# NSFW分類器インポート
try:
    from scripts.data.train_nsfw_classifier import NSFWClassifier
    NSFW_CLASSIFIER_AVAILABLE = True
except ImportError:
    NSFW_CLASSIFIER_AVAILABLE = False

# HumanLikeScraperインポート
try:
    from scripts.data.human_like_web_scraping import HumanLikeScraper
    HUMAN_LIKE_SCRAPER_AVAILABLE = True
except ImportError:
    HUMAN_LIKE_SCRAPER_AVAILABLE = False

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_category_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 広範なカテゴリURLリスト
COMPREHENSIVE_CATEGORY_URLS = {
    # 日本語サイト
    'ja': {
        'encyclopedia': [
            'https://ja.wikipedia.org/wiki/メインページ',
            'https://kotobank.jp/',
            'https://www.weblio.jp/',
        ],
        'academic': [
            'https://www.jstage.jst.go.jp/',
            'https://ci.nii.ac.jp/',
            'https://www.ndl.go.jp/',
        ],
        'news': [
            'https://www3.nhk.or.jp/news/',
            'https://www.asahi.com/',
            'https://www.yomiuri.co.jp/',
        ],
        'technology': [
            'https://qiita.com/',
            'https://zenn.dev/',
            'https://atmarkit.itmedia.co.jp/',
        ],
        'government': [
            'https://www.e-gov.go.jp/',
            'https://www.cao.go.jp/',
            'https://www.mhlw.go.jp/',
        ],
        'education': [
            'https://www.mext.go.jp/',
            'https://www.jasso.go.jp/',
        ],
        'culture': [
            'https://www.bunka.go.jp/',
            'https://www.nippon.com/ja/',
        ],
        'nsfw_detection': [  # 検知目的のみ
            'https://ja.wikipedia.org/wiki/Category:成人向けコンテンツ',
            'https://ja.wikipedia.org/wiki/Category:性に関する記事',
            # 注意: NSFWコンテンツは検知目的のみで、生成目的ではありません
        ]
    },
    # 英語サイト
    'en': {
        'encyclopedia': [
            'https://en.wikipedia.org/wiki/Main_Page',
            'https://www.britannica.com/',
            'https://www.encyclopedia.com/',
        ],
        'academic': [
            'https://www.nature.com/',
            'https://www.science.org/',
            'https://www.cell.com/',
        ],
        'news': [
            'https://www.bbc.com/news',
            'https://www.reuters.com/',
            'https://www.theguardian.com/',
        ],
        'technology': [
            'https://github.com/',
            'https://stackoverflow.com/',
            'https://medium.com/',
        ],
        'education': [
            'https://www.khanacademy.org/',
            'https://ocw.mit.edu/',
            'https://www.coursera.org/',
        ],
        'science': [
            'https://www.nasa.gov/',
            'https://www.nsf.gov/',
            'https://www.scientificamerican.com/',
        ],
        'nsfw_detection': [  # 検知目的のみ
            'https://en.wikipedia.org/wiki/Category:Adult_content',
            'https://en.wikipedia.org/wiki/Category:Sexuality',
            # 注意: NSFWコンテンツは検知目的のみで、生成目的ではありません
        ]
    }
}


class ComprehensiveCategoryScraper:
    """包括的カテゴリWebスクレイピングクラス"""
    
    def __init__(
        self,
        output_dir: Path,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        delay_per_request: float = 2.0,
        timeout: int = 30000,
        max_pages_per_category: int = 50,
        include_nsfw: bool = True,
        include_arxiv: bool = True
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            use_cursor_browser: Cursorのブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_request: リクエスト間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_pages_per_category: カテゴリあたりの最大ページ数
            include_nsfw: NSFWカテゴリを含めるか（検知目的）
            include_arxiv: Arxivを含めるか
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_request = delay_per_request
        self.timeout = timeout
        self.max_pages_per_category = max_pages_per_category
        self.include_nsfw = include_nsfw
        self.include_arxiv = include_arxiv
        
        self.all_samples: List[Dict] = []
        self.nsfw_samples: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # NSFW分類器初期化
        self.nsfw_classifier = None
        if NSFW_CLASSIFIER_AVAILABLE:
            try:
                nsfw_model_path = Path("models/nsfw_classifier.joblib")
                if nsfw_model_path.exists():
                    self.nsfw_classifier = NSFWClassifier(model_path=nsfw_model_path)
                    logger.info("[NSFW] NSFW classifier loaded")
                else:
                    logger.warning("[NSFW] NSFW model not found, will use rule-based detection")
            except Exception as e:
                logger.warning(f"[NSFW] Failed to load NSFW classifier: {e}")
        
        logger.info("="*80)
        logger.info("Comprehensive Category Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Include NSFW (detection purpose): {self.include_nsfw}")
        logger.info(f"Include Arxiv: {self.include_arxiv}")
        logger.info(f"Max pages per category: {self.max_pages_per_category}")
    
    def detect_nsfw(self, text: str, url: str = None) -> tuple:
        """NSFW検知"""
        if self.nsfw_classifier:
            try:
                label, confidence = self.nsfw_classifier.predict(text)
                return label, confidence
            except Exception as e:
                logger.warning(f"[NSFW] Detection failed: {e}")
                return self._rule_based_nsfw_detection(text)
        else:
            return self._rule_based_nsfw_detection(text)
    
    def _rule_based_nsfw_detection(self, text: str) -> tuple:
        """ルールベースNSFW検知（フォールバック）"""
        import re
        
        nsfw_keywords = [
            r'性的', r'ポルノ', r'アダルト', r'エロ', r'わいせつ',
            r'暴力', r'殺人', r'自殺', r'テロ', r'爆弾',
            r'差別', r'ヘイト', r'誹謗', r'中傷',
            r'sexual', r'porn', r'adult', r'violence', r'hate'
        ]
        
        text_lower = text.lower()
        for keyword in nsfw_keywords:
            if re.search(keyword, text_lower):
                return ('nsfw_detected', 0.7)
        
        return ('safe', 1.0)
    
    async def scrape_category_urls(
        self,
        urls: List[str],
        category: str,
        language: str
    ) -> List[Dict]:
        """カテゴリURLをスクレイピング"""
        if not HUMAN_LIKE_SCRAPER_AVAILABLE:
            logger.error("[ERROR] HumanLikeScraper not available")
            return []
        
        scraper = HumanLikeScraper(
            output_dir=self.output_dir,
            use_cursor_browser=self.use_cursor_browser,
            remote_debugging_port=self.remote_debugging_port,
            delay_per_request=self.delay_per_request,
            timeout=self.timeout,
            max_pages=self.max_pages_per_category,
            follow_links=True,
            max_depth=2
        )
        
        samples = await scraper.scrape_with_auto_navigation(
            start_urls=urls[:5],  # カテゴリあたり最大5開始URL
            max_pages=self.max_pages_per_category
        )
        
        # カテゴリ情報を追加
        for sample in samples:
            sample['category'] = category
            sample['language'] = language
            
            # NSFW検知
            nsfw_label, nsfw_confidence = self.detect_nsfw(sample.get('text', ''), sample.get('url', ''))
            sample['nsfw_label'] = nsfw_label
            sample['nsfw_confidence'] = float(nsfw_confidence)
            sample['nsfw_detection_purpose'] = 'safety_training'
            
            if nsfw_label != 'safe':
                self.nsfw_samples.append(sample)
        
        return samples
    
    def collect_arxiv_papers(self, categories: List[str] = None, max_papers: int = 1000) -> List[Dict]:
        """Arxiv論文を収集"""
        if not ARXIV_AVAILABLE:
            logger.warning("[ARXIV] Arxiv crawler not available")
            return []
        
        if categories is None:
            categories = ['cs.AI', 'cs.CL', 'cs.LG', 'math', 'physics']
        
        logger.info(f"[ARXIV] Collecting papers from categories: {categories}")
        
        arxiv_config = {
            'categories': categories,
            'max_papers': max_papers,
            'date_from': '2020-01-01',
            'delay': self.delay_per_request
        }
        
        crawler = ArxivCrawler(arxiv_config)
        samples = crawler.crawl()
        
        # NSFW検知を適用
        for sample in samples:
            text = sample.get('content', sample.get('text', ''))
            url = sample.get('url', '')
            
            nsfw_label, nsfw_confidence = self.detect_nsfw(text, url)
            sample['nsfw_label'] = nsfw_label
            sample['nsfw_confidence'] = float(nsfw_confidence)
            sample['nsfw_detection_purpose'] = 'safety_training'
            
            if nsfw_label != 'safe':
                self.nsfw_samples.append(sample)
        
        logger.info(f"[ARXIV] Collected {len(samples)} papers")
        return samples
    
    async def scrape_all_categories(self) -> List[Dict]:
        """全カテゴリをスクレイピング"""
        logger.info("="*80)
        logger.info("Starting Comprehensive Category Scraping")
        logger.info("="*80)
        
        all_samples = []
        
        # 日本語カテゴリ
        logger.info("[JA] Starting Japanese category scraping...")
        for category, urls in COMPREHENSIVE_CATEGORY_URLS['ja'].items():
            if category == 'nsfw_detection' and not self.include_nsfw:
                continue
            
            logger.info(f"[JA] Scraping category: {category} ({len(urls)} URLs)")
            samples = await self.scrape_category_urls(urls, category, 'ja')
            all_samples.extend(samples)
            logger.info(f"[JA] Collected {len(samples)} samples from {category}")
            
            # カテゴリ間の待機
            await asyncio.sleep(self.delay_per_request * 2)
        
        # 英語カテゴリ
        logger.info("[EN] Starting English category scraping...")
        for category, urls in COMPREHENSIVE_CATEGORY_URLS['en'].items():
            if category == 'nsfw_detection' and not self.include_nsfw:
                continue
            
            logger.info(f"[EN] Scraping category: {category} ({len(urls)} URLs)")
            samples = await self.scrape_category_urls(urls, category, 'en')
            all_samples.extend(samples)
            logger.info(f"[EN] Collected {len(samples)} samples from {category}")
            
            # カテゴリ間の待機
            await asyncio.sleep(self.delay_per_request * 2)
        
        # Arxiv論文収集
        if self.include_arxiv:
            logger.info("[ARXIV] Starting Arxiv paper collection...")
            arxiv_samples = self.collect_arxiv_papers(max_papers=1000)
            all_samples.extend(arxiv_samples)
            logger.info(f"[ARXIV] Collected {len(arxiv_samples)} papers")
        
        self.all_samples = all_samples
        logger.info(f"[TOTAL] Collected {len(self.all_samples)} samples")
        logger.info(f"[NSFW] Detected {len(self.nsfw_samples)} NSFW samples (detection purpose only)")
        
        return all_samples
    
    def save_samples(self, samples: List[Dict], filename: str = None) -> Path:
        """サンプルを保存"""
        if filename is None:
            filename = f"comprehensive_category_scraped_{self.session_id}.jsonl"
        
        output_file = self.output_dir / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Saved {len(samples)} samples to {output_file}")
        return output_file
    
    def save_nsfw_samples(self, samples: List[Dict], filename: str = None) -> Optional[Path]:
        """NSFW検知サンプルを保存（検知目的のみ）"""
        if not samples:
            return None
        
        if filename is None:
            filename = f"nsfw_detected_{self.session_id}.jsonl"
        
        output_file = self.output_dir / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[NSFW] Saved {len(samples)} NSFW-detected samples to {output_file} (detection purpose only)")
        return output_file


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Comprehensive Category Web Scraping")
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('D:/webdataset/processed'),
        help='Output directory'
    )
    parser.add_argument(
        '--use-cursor-browser',
        action='store_true',
        default=True,
        help='Use Cursor browser'
    )
    parser.add_argument(
        '--remote-debugging-port',
        type=int,
        default=9222,
        help='Remote debugging port'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay between requests (seconds)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30000,
        help='Page load timeout (milliseconds)'
    )
    parser.add_argument(
        '--max-pages-per-category',
        type=int,
        default=50,
        help='Maximum pages per category'
    )
    parser.add_argument(
        '--include-nsfw',
        action='store_true',
        default=True,
        help='Include NSFW categories (detection purpose only)'
    )
    parser.add_argument(
        '--include-arxiv',
        action='store_true',
        default=True,
        help='Include Arxiv papers'
    )
    
    args = parser.parse_args()
    
    # スクレイパー作成
    scraper = ComprehensiveCategoryScraper(
        output_dir=args.output,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_request=args.delay,
        timeout=args.timeout,
        max_pages_per_category=args.max_pages_per_category,
        include_nsfw=args.include_nsfw,
        include_arxiv=args.include_arxiv
    )
    
    # スクレイピング実行
    samples = await scraper.scrape_all_categories()
    
    # 保存
    output_file = scraper.save_samples(samples)
    
    # NSFW検知データ保存（検知目的のみ）
    if scraper.nsfw_samples:
        nsfw_file = scraper.save_nsfw_samples(scraper.nsfw_samples)
        logger.info(f"[NSFW] NSFW samples saved (detection purpose only): {nsfw_file}")
    
    logger.info(f"[SUCCESS] Comprehensive category scraping completed. Output: {output_file}")
    return output_file


if __name__ == "__main__":
    asyncio.run(main())





