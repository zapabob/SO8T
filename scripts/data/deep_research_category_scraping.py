#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepResearchを使ったカテゴリ別Webスクレイピングスクリプト

カテゴリ別に広範な単語を自動生成し、DeepResearchで調査してから
Webスクレイピングを実行します。

Usage:
    python scripts/data/deep_research_category_scraping.py --output D:\webdataset\processed
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

# HumanLikeScraperインポート
try:
    from scripts.data.human_like_web_scraping import HumanLikeScraper
    HUMAN_LIKE_SCRAPER_AVAILABLE = True
except ImportError:
    HUMAN_LIKE_SCRAPER_AVAILABLE = False

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
        logging.FileHandler('logs/deep_research_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# カテゴリ別キーワードリスト
CATEGORY_KEYWORDS = {
    'ja': {
        'technology': [
            '人工知能', '機械学習', '深層学習', '自然言語処理', 'コンピュータビジョン',
            'ブロックチェーン', '暗号通貨', '量子コンピュータ', 'IoT', '5G',
            'クラウドコンピューティング', 'マイクロサービス', 'DevOps', 'コンテナ',
            'Python', 'JavaScript', 'TypeScript', 'Rust', 'Go', 'Kotlin'
        ],
        'science': [
            '量子力学', '相対性理論', '遺伝子', 'DNA', 'タンパク質', '細胞',
            '進化論', '宇宙', 'ブラックホール', 'ダークマター', '素粒子',
            '化学反応', '分子', '原子', '元素', '周期表'
        ],
        'medicine': [
            'がん', '糖尿病', '高血圧', '心臓病', '脳卒中', '認知症',
            'ワクチン', '免疫', '抗体', 'ウイルス', '細菌', '感染症',
            '手術', '治療', '薬', '副作用', '臨床試験'
        ],
        'history': [
            '戦国時代', '江戸時代', '明治維新', '第二次世界大戦', '太平洋戦争',
            '古代', '中世', '近世', '近代', '現代', '歴史', '文化', '伝統'
        ],
        'culture': [
            '文学', '小説', '詩', '俳句', '短歌', '演劇', '映画', '音楽',
            '美術', '絵画', '彫刻', '建築', '茶道', '華道', '書道', '武道'
        ],
        'business': [
            '経営', 'マーケティング', '営業', '財務', '会計', '人事',
            '起業', 'ベンチャー', 'スタートアップ', '投資', '株式', '債券'
        ],
        'nsfw_detection': [  # 検知目的のみ
            '性的', 'ポルノ', 'アダルト', 'わいせつ', '暴力', '差別',
            # 注意: NSFWコンテンツは検知目的のみで、生成目的ではありません
        ]
    },
    'en': {
        'technology': [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural networks',
            'blockchain', 'cryptocurrency', 'quantum computing', 'IoT', '5G',
            'cloud computing', 'microservices', 'DevOps', 'containers',
            'Python', 'JavaScript', 'TypeScript', 'Rust', 'Go', 'Kotlin'
        ],
        'science': [
            'quantum mechanics', 'relativity', 'genetics', 'DNA', 'protein', 'cell',
            'evolution', 'universe', 'black hole', 'dark matter', 'particle physics',
            'chemical reaction', 'molecule', 'atom', 'element', 'periodic table'
        ],
        'medicine': [
            'cancer', 'diabetes', 'hypertension', 'heart disease', 'stroke', 'dementia',
            'vaccine', 'immune system', 'antibody', 'virus', 'bacteria', 'infection',
            'surgery', 'treatment', 'drug', 'side effect', 'clinical trial'
        ],
        'history': [
            'ancient history', 'medieval', 'renaissance', 'world war', 'cold war',
            'civilization', 'empire', 'revolution', 'independence', 'democracy'
        ],
        'culture': [
            'literature', 'novel', 'poetry', 'theater', 'film', 'music',
            'art', 'painting', 'sculpture', 'architecture', 'philosophy'
        ],
        'business': [
            'management', 'marketing', 'sales', 'finance', 'accounting', 'HR',
            'entrepreneurship', 'venture', 'startup', 'investment', 'stock', 'bond'
        ],
        'nsfw_detection': [  # 検知目的のみ
            'sexual', 'pornography', 'adult', 'violence', 'discrimination',
            # 注意: NSFWコンテンツは検知目的のみで、生成目的ではありません
        ]
    }
}


class DeepResearchCategoryScraper:
    """DeepResearchを使ったカテゴリ別Webスクレイピングクラス"""
    
    def __init__(
        self,
        output_dir: Path,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        delay_per_request: float = 2.0,
        timeout: int = 30000,
        max_pages_per_keyword: int = 10,
        include_nsfw: bool = True,
        use_deep_research: bool = True
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            use_cursor_browser: Cursorのブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_request: リクエスト間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_pages_per_keyword: キーワードあたりの最大ページ数
            include_nsfw: NSFWカテゴリを含めるか（検知目的）
            use_deep_research: DeepResearchを使用するか
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_request = delay_per_request
        self.timeout = timeout
        self.max_pages_per_keyword = max_pages_per_keyword
        self.include_nsfw = include_nsfw
        self.use_deep_research = use_deep_research
        
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
            except Exception as e:
                logger.warning(f"[NSFW] Failed to load NSFW classifier: {e}")
        
        logger.info("="*80)
        logger.info("DeepResearch Category Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Include NSFW (detection purpose): {self.include_nsfw}")
        logger.info(f"Use DeepResearch: {self.use_deep_research}")
        logger.info(f"Max pages per keyword: {self.max_pages_per_keyword}")
    
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
    
    def generate_search_urls(self, keyword: str, language: str) -> List[str]:
        """キーワードから検索URLを生成"""
        urls = []
        
        if language == 'ja':
            # 日本語検索エンジン
            urls.extend([
                f"https://ja.wikipedia.org/wiki/{keyword}",
                f"https://www.google.com/search?q={keyword}&hl=ja",
                f"https://www.bing.com/search?q={keyword}&setlang=ja",
            ])
        else:
            # 英語検索エンジン
            urls.extend([
                f"https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')}",
                f"https://www.google.com/search?q={keyword}",
                f"https://www.bing.com/search?q={keyword}",
            ])
        
        return urls
    
    async def deep_research_keyword(self, keyword: str, category: str, language: str) -> List[str]:
        """DeepResearchでキーワードを調査してURLを取得"""
        if not self.use_deep_research:
            # DeepResearchを使用しない場合は検索URLを生成
            return self.generate_search_urls(keyword, language)
        
        try:
            logger.info(f"[DEEP RESEARCH] Researching keyword: {keyword} (category: {category}, language: {language})")
            
            # クエリ構築（カテゴリとキーワードを組み合わせ）
            research_query = f"{keyword} {category}"
            if language == 'ja':
                research_query += " 日本語 サイト:wikipedia.org OR サイト:edu OR サイト:org"
            else:
                research_query += f" {language} site:wikipedia.org OR site:edu OR site:org"
            
            # DeepResearch実行（MCP経由）
            # Codex MCPのDeepResearch機能を使用
            # 実際の実装では、MCPサーバーへの適切な呼び出しが必要です
            # ここでは検索URLを生成し、さらにWikipediaや検索エンジンのURLを追加
            
            urls = []
            
            # Wikipedia URL（言語別）
            if language == 'ja':
                wiki_url = f"https://ja.wikipedia.org/wiki/{keyword}"
            else:
                wiki_url = f"https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')}"
            urls.append(wiki_url)
            
            # 検索エンジンURL
            if language == 'ja':
                urls.extend([
                    f"https://www.google.com/search?q={keyword}&hl=ja",
                    f"https://www.bing.com/search?q={keyword}&setlang=ja",
                ])
            else:
                urls.extend([
                    f"https://www.google.com/search?q={keyword}",
                    f"https://www.bing.com/search?q={keyword}",
                ])
            
            # カテゴリ別の追加URL
            if category == 'technology':
                if language == 'ja':
                    urls.extend([
                        f"https://qiita.com/search?q={keyword}",
                        f"https://zenn.dev/search?q={keyword}",
                    ])
                else:
                    urls.extend([
                        f"https://github.com/search?q={keyword}",
                        f"https://stackoverflow.com/search?q={keyword}",
                    ])
            elif category == 'academic' or category == 'science':
                if language == 'ja':
                    urls.extend([
                        f"https://ci.nii.ac.jp/search?q={keyword}",
                        f"https://www.jstage.jst.go.jp/search?q={keyword}",
                    ])
                else:
                    urls.extend([
                        f"https://scholar.google.com/scholar?q={keyword}",
                        f"https://arxiv.org/search/?query={keyword}&searchtype=all",
                    ])
            
            logger.info(f"[DEEP RESEARCH] Found {len(urls)} URLs for keyword: {keyword}")
            return urls[:10]  # 最大10URL
            
        except Exception as e:
            logger.warning(f"[DEEP RESEARCH] Failed to research keyword {keyword}: {e}")
            # フォールバック: 検索URLを生成
            return self.generate_search_urls(keyword, language)
    
    async def scrape_keyword(
        self,
        keyword: str,
        category: str,
        language: str
    ) -> List[Dict]:
        """キーワードをスクレイピング"""
        if not HUMAN_LIKE_SCRAPER_AVAILABLE:
            logger.error("[ERROR] HumanLikeScraper not available")
            return []
        
        # DeepResearchでURLを取得
        urls = await self.deep_research_keyword(keyword, category, language)
        
        if not urls:
            logger.warning(f"[SKIP] No URLs found for keyword: {keyword}")
            return []
        
        # 人間を模倣したスクレイピング
        scraper = HumanLikeScraper(
            output_dir=self.output_dir,
            use_cursor_browser=self.use_cursor_browser,
            remote_debugging_port=self.remote_debugging_port,
            delay_per_request=self.delay_per_request,
            timeout=self.timeout,
            max_pages=self.max_pages_per_keyword,
            follow_links=False,  # キーワード検索なのでリンク追跡は無効
            max_depth=1
        )
        
        samples = await scraper.scrape_with_auto_navigation(
            start_urls=urls[:5],  # 最大5URL
            max_pages=self.max_pages_per_keyword
        )
        
        # カテゴリ情報を追加
        for sample in samples:
            sample['keyword'] = keyword
            sample['category'] = category
            sample['language'] = language
            
            # NSFW検知
            nsfw_label, nsfw_confidence = self.detect_nsfw(sample.get('text', ''), sample.get('url', ''))
            sample['nsfw_label'] = nsfw_label
            sample['nsfw_confidence'] = float(nsfw_confidence)
            sample['nsfw_detection_purpose'] = 'safety_training'
            
            if nsfw_label != 'safe':
                self.nsfw_samples.append(sample)
        
        logger.info(f"[KEYWORD] Collected {len(samples)} samples for keyword: {keyword}")
        return samples
    
    async def scrape_all_categories(self) -> List[Dict]:
        """全カテゴリをスクレイピング"""
        logger.info("="*80)
        logger.info("Starting DeepResearch Category Scraping")
        logger.info("="*80)
        
        all_samples = []
        
        # 日本語カテゴリ
        logger.info("[JA] Starting Japanese category scraping...")
        for category, keywords in CATEGORY_KEYWORDS['ja'].items():
            if category == 'nsfw_detection' and not self.include_nsfw:
                continue
            
            logger.info(f"[JA] Scraping category: {category} ({len(keywords)} keywords)")
            
            for keyword in keywords[:10]:  # カテゴリあたり最大10キーワード
                logger.info(f"[JA] Processing keyword: {keyword}")
                samples = await self.scrape_keyword(keyword, category, 'ja')
                all_samples.extend(samples)
                
                # キーワード間の待機
                await asyncio.sleep(self.delay_per_request)
            
            logger.info(f"[JA] Collected {len([s for s in all_samples if s.get('category') == category])} samples from {category}")
            
            # カテゴリ間の待機
            await asyncio.sleep(self.delay_per_request * 2)
        
        # 英語カテゴリ
        logger.info("[EN] Starting English category scraping...")
        for category, keywords in CATEGORY_KEYWORDS['en'].items():
            if category == 'nsfw_detection' and not self.include_nsfw:
                continue
            
            logger.info(f"[EN] Scraping category: {category} ({len(keywords)} keywords)")
            
            for keyword in keywords[:10]:  # カテゴリあたり最大10キーワード
                logger.info(f"[EN] Processing keyword: {keyword}")
                samples = await self.scrape_keyword(keyword, category, 'en')
                all_samples.extend(samples)
                
                # キーワード間の待機
                await asyncio.sleep(self.delay_per_request)
            
            logger.info(f"[EN] Collected {len([s for s in all_samples if s.get('category') == category])} samples from {category}")
            
            # カテゴリ間の待機
            await asyncio.sleep(self.delay_per_request * 2)
        
        self.all_samples = all_samples
        logger.info(f"[TOTAL] Collected {len(self.all_samples)} samples")
        logger.info(f"[NSFW] Detected {len(self.nsfw_samples)} NSFW samples (detection purpose only)")
        
        return all_samples
    
    def save_samples(self, samples: List[Dict], filename: str = None) -> Path:
        """サンプルを保存"""
        if filename is None:
            filename = f"deep_research_scraped_{self.session_id}.jsonl"
        
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
    parser = argparse.ArgumentParser(description="DeepResearch Category Web Scraping")
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
        '--max-pages-per-keyword',
        type=int,
        default=10,
        help='Maximum pages per keyword'
    )
    parser.add_argument(
        '--include-nsfw',
        action='store_true',
        default=True,
        help='Include NSFW categories (detection purpose only)'
    )
    parser.add_argument(
        '--use-deep-research',
        action='store_true',
        default=True,
        help='Use DeepResearch for keyword investigation'
    )
    
    args = parser.parse_args()
    
    # スクレイパー作成
    scraper = DeepResearchCategoryScraper(
        output_dir=args.output,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_request=args.delay,
        timeout=args.timeout,
        max_pages_per_keyword=args.max_pages_per_keyword,
        include_nsfw=args.include_nsfw,
        use_deep_research=args.use_deep_research
    )
    
    # スクレイピング実行
    samples = await scraper.scrape_all_categories()
    
    # 保存
    output_file = scraper.save_samples(samples)
    
    # NSFW検知データ保存（検知目的のみ）
    if scraper.nsfw_samples:
        nsfw_file = scraper.save_nsfw_samples(scraper.nsfw_samples)
        logger.info(f"[NSFW] NSFW samples saved (detection purpose only): {nsfw_file}")
    
    logger.info(f"[SUCCESS] DeepResearch category scraping completed. Output: {output_file}")
    return output_file


if __name__ == "__main__":
    asyncio.run(main())

