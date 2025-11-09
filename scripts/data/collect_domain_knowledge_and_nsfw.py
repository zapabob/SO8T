#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ドメイン別知識とNSFW検知用データ収集スクリプト

Cursorブラウザを使って、ドメイン別知識とNSFW検知用データを収集します。
NSFWデータは検知目的のみで、生成目的ではありません。

Usage:
    python scripts/data/collect_domain_knowledge_and_nsfw.py --output D:\webdataset\processed
"""

import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page
except ImportError:
    print("[ERROR] Playwright not installed. Install with: pip install playwright")
    sys.exit(1)

# ドメイン知識クローラーインポート
try:
    from so8t_mmllm.scripts.data.domain_knowledge_crawler import DomainKnowledgeCrawler
    DOMAIN_CRAWLER_AVAILABLE = True
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "domain_knowledge_crawler",
            PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "domain_knowledge_crawler.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DomainKnowledgeCrawler = module.DomainKnowledgeCrawler
        DOMAIN_CRAWLER_AVAILABLE = True
    except Exception:
        DOMAIN_CRAWLER_AVAILABLE = False
        logger.warning("Domain knowledge crawler not available")

# NSFW分類器インポート
try:
    from scripts.data.train_nsfw_classifier import NSFWClassifier
    NSFW_CLASSIFIER_AVAILABLE = True
except ImportError:
    NSFW_CLASSIFIER_AVAILABLE = False

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/domain_knowledge_nsfw_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DomainKnowledgeNSFWCollector:
    """ドメイン別知識とNSFW検知用データ収集クラス"""
    
    def __init__(
        self,
        output_dir: Path,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        delay_per_request: float = 2.0,
        timeout: int = 30000
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            use_cursor_browser: Cursorのブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_request: リクエスト間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_request = delay_per_request
        self.timeout = timeout
        
        self.domain_samples: List[Dict] = []
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
        logger.info("Domain Knowledge & NSFW Data Collector Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Use Cursor browser: {self.use_cursor_browser}")
        logger.info(f"NSFW classifier available: {self.nsfw_classifier is not None}")
    
    async def connect_to_cursor_browser(self, playwright) -> Optional[Browser]:
        """Cursorのブラウザに接続"""
        if not self.use_cursor_browser:
            logger.info("[BROWSER] Launching new browser...")
            browser = await playwright.chromium.launch(headless=False)
            logger.info("[OK] Browser launched")
            return browser
        
        try:
            logger.info(f"[BROWSER] Connecting to Cursor browser on port {self.remote_debugging_port}...")
            cdp_endpoint = f"http://127.0.0.1:{self.remote_debugging_port}"
            browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
            
            contexts = browser.contexts
            if contexts:
                logger.info(f"[OK] Connected to Cursor browser (found {len(contexts)} contexts)")
            else:
                logger.info("[INFO] No existing contexts found, creating new context...")
                await browser.new_context()
                logger.info("[OK] New context created")
            
            return browser
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to Cursor browser: {e}")
            logger.info("[INFO] Falling back to launching new browser...")
            browser = await playwright.chromium.launch(headless=False)
            logger.info("[OK] New browser launched")
            return browser
    
    def detect_nsfw(self, text: str, url: str = None) -> tuple:
        """
        NSFW検知
        
        Args:
            text: テキスト
            url: URL（オプション）
        
        Returns:
            (label, confidence) タプル
        """
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
        
        # NSFWキーワード（検知目的）
        nsfw_keywords = [
            r'性的', r'ポルノ', r'アダルト', r'エロ', r'わいせつ',
            r'暴力', r'殺人', r'自殺', r'テロ', r'爆弾',
            r'差別', r'ヘイト', r'誹謗', r'中傷'
        ]
        
        text_lower = text.lower()
        for keyword in nsfw_keywords:
            if re.search(keyword, text_lower):
                return ('nsfw_detected', 0.7)
        
        return ('safe', 1.0)
    
    async def scrape_domain_knowledge_urls(
        self,
        urls: List[Dict],
        wait_for_user: bool = False
    ) -> List[Dict]:
        """
        ドメイン別知識URLをスクレイピング
        
        Args:
            urls: URL辞書リスト（url, domain, subdomain, languageを含む）
            wait_for_user: ユーザー確認を待つか
        
        Returns:
            収集したサンプルリスト
        """
        samples = []
        
        async with async_playwright() as playwright:
            browser = await self.connect_to_cursor_browser(playwright)
            
            try:
                contexts = browser.contexts
                if contexts:
                    context = contexts[0]
                else:
                    context = await browser.new_context()
                
                page = await context.new_page()
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })
                
                logger.info(f"[SCRAPE] Starting to scrape {len(urls)} domain knowledge URLs...")
                
                for i, url_info in enumerate(urls, 1):
                    url = url_info.get('url', '')
                    domain = url_info.get('domain', 'unknown')
                    subdomain = url_info.get('subdomain', 'unknown')
                    language = url_info.get('language', 'ja')
                    
                    logger.info(f"[PROGRESS] [{i}/{len(urls)}] Processing: {url}")
                    
                    try:
                        await page.goto(url, timeout=self.timeout, wait_until="networkidle")
                        
                        if wait_for_user:
                            logger.info("[INTERACTIVE] Page loaded. Press Enter to continue...")
                            await asyncio.to_thread(input, "> ")
                        
                        # タイトル取得
                        title = await page.title()
                        
                        # HTML取得
                        html = await page.content()
                        
                        # テキスト抽出
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, 'lxml')
                        
                        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
                            tag.decompose()
                        
                        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=lambda x: x and ('content' in x.lower() or 'main' in x.lower()))
                        
                        if main_content:
                            text = main_content.get_text(separator='\n', strip=True)
                        else:
                            text = soup.get_text(separator='\n', strip=True)
                        
                        import re
                        text = re.sub(r'\n\n+', '\n\n', text)
                        text = re.sub(r'[ \t]+', ' ', text)
                        text = text.strip()
                        
                        if len(text) < 200:
                            logger.warning(f"[SKIP] Text too short: {len(text)} chars")
                            continue
                        
                        # NSFW検知
                        nsfw_label, nsfw_confidence = self.detect_nsfw(text, url)
                        
                        sample = {
                            "text": text,
                            "url": url,
                            "domain": domain,
                            "subdomain": subdomain,
                            "language": language,
                            "title": title,
                            "source": "domain_knowledge_collector",
                            "crawled_at": datetime.now().isoformat(),
                            "text_length": len(text),
                            "nsfw_label": nsfw_label,
                            "nsfw_confidence": float(nsfw_confidence),
                            "nsfw_detection_purpose": "safety_training"  # 検知目的
                        }
                        
                        samples.append(sample)
                        self.domain_samples.append(sample)
                        
                        # NSFW検知された場合は別リストにも追加
                        if nsfw_label != 'safe':
                            self.nsfw_samples.append(sample)
                            logger.info(f"[NSFW] Detected: {nsfw_label} (confidence: {nsfw_confidence:.2f})")
                        
                        logger.info(f"[OK] Scraped {len(text)} characters from {url}")
                        
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to scrape {url}: {e}")
                    
                    if i < len(urls):
                        await asyncio.sleep(self.delay_per_request)
                
                await page.close()
                
            finally:
                if not self.use_cursor_browser:
                    await browser.close()
                else:
                    logger.info("[BROWSER] Keeping Cursor browser connection open")
        
        logger.info(f"[OK] Scraped {len(samples)} domain knowledge pages")
        return samples
    
    def collect_domain_knowledge(self, config: Dict) -> List[Dict]:
        """
        ドメイン別知識を収集（既存クローラーを使用）
        
        Args:
            config: 設定辞書
        
        Returns:
            収集したサンプルリスト
        """
        if not DOMAIN_CRAWLER_AVAILABLE:
            logger.error("[ERROR] Domain knowledge crawler not available")
            return []
        
        logger.info("[DOMAIN] Starting domain knowledge collection...")
        
        crawler_config = {
            'japanese_sites': config.get('japanese_sites', ['kotobank', 'weblio', 'jstage', 'cinii']),
            'english_sites': config.get('english_sites', ['britannica', 'khan_academy', 'mit_ocw']),
            'max_samples_per_site': config.get('max_samples_per_site', 1000),
            'delay': self.delay_per_request,
            'timeout': self.timeout // 1000,
            'user_agent': 'SO8T-DataCollector/1.0 (Research Purpose)'
        }
        
        crawler = DomainKnowledgeCrawler(crawler_config)
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
        
        self.domain_samples.extend(samples)
        logger.info(f"[DOMAIN] Collected {len(samples)} domain knowledge samples")
        
        return samples
    
    def save_samples(self, samples: List[Dict], filename: str) -> Path:
        """サンプルを保存"""
        output_file = self.output_dir / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Saved {len(samples)} samples to {output_file}")
        return output_file


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Collect Domain Knowledge & NSFW Data")
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
        '--max-samples-per-site',
        type=int,
        default=1000,
        help='Maximum samples per site'
    )
    parser.add_argument(
        '--japanese-sites',
        nargs='+',
        default=['kotobank', 'weblio', 'jstage', 'cinii'],
        help='Japanese sites to crawl'
    )
    parser.add_argument(
        '--english-sites',
        nargs='+',
        default=['britannica', 'khan_academy', 'mit_ocw'],
        help='English sites to crawl'
    )
    
    args = parser.parse_args()
    
    # コレクター作成
    collector = DomainKnowledgeNSFWCollector(
        output_dir=args.output,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_request=args.delay,
        timeout=args.timeout
    )
    
    # ドメイン別知識収集設定
    domain_config = {
        'japanese_sites': args.japanese_sites,
        'english_sites': args.english_sites,
        'max_samples_per_site': args.max_samples_per_site
    }
    
    # ドメイン別知識収集
    domain_samples = collector.collect_domain_knowledge(domain_config)
    
    # 保存
    domain_file = collector.save_samples(
        collector.domain_samples,
        f"domain_knowledge_{collector.session_id}.jsonl"
    )
    
    # NSFW検知データ保存（検知目的）
    if collector.nsfw_samples:
        nsfw_file = collector.save_samples(
            collector.nsfw_samples,
            f"nsfw_detected_{collector.session_id}.jsonl"
        )
        logger.info(f"[NSFW] Saved {len(collector.nsfw_samples)} NSFW-detected samples (detection purpose only)")
    
    logger.info(f"[SUCCESS] Collection completed. Domain knowledge: {domain_file}")
    
    return domain_file


if __name__ == "__main__":
    asyncio.run(main())





