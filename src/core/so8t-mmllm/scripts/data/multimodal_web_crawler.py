#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチモーダルwebスクレイピングモジュール

画像+テキストの同時収集機能を提供します。

Usage:
    from so8t_mmllm.scripts.data.multimodal_web_crawler import MultimodalWebCrawler
    crawler = MultimodalWebCrawler()
    samples = crawler.crawl_multimodal(urls, output_dir)
"""

import sys
import json
import time
import logging
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# エラーハンドリングとリトライ機構のインポート
try:
    from scripts.data.crawler_error_handler import CrawlerErrorHandler, classify_exception
    from scripts.data.retry_handler import RetryHandler
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultimodalWebCrawler:
    """マルチモーダルwebスクレイピングクラス"""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Args:
            config: 設定辞書
            output_dir: 出力ディレクトリ
        """
        self.config = config or {
            "delay": 1.0,
            "timeout": 15,
            "max_depth": 3,
            "min_text_length": 200,
            # 画像設定
            "image_min_width": 100,
            "image_min_height": 100,
            "image_max_size_mb": 10,
            "image_formats": ["jpg", "jpeg", "png", "webp", "gif"],
            "save_images": True,
        }
        
        # 出力ディレクトリ
        if output_dir is None:
            output_dir = Path(r"D:\webdataset\multimodal")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像保存ディレクトリ
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # エラーハンドリングとリトライ機構
        if ERROR_HANDLING_AVAILABLE:
            error_log_dir = self.output_dir / "error_logs"
            self.error_handler = CrawlerErrorHandler(log_dir=error_log_dir)
            self.retry_handler = RetryHandler(
                max_retries=3,
                initial_delay=1.0,
                backoff_factor=2.0,
                max_delay=10.0
            )
        else:
            self.error_handler = None
            self.retry_handler = None
        
        # 訪問済みURL管理
        self.visited_urls: Set[str] = set()
        self.collected_samples: List[Dict] = []
        
        # robots.txt管理
        self.robots_parsers: Dict[str, RobotFileParser] = {}
    
    def _check_robots_txt(self, url: str) -> bool:
        """robots.txtをチェック"""
        try:
            parsed = urlparse(url)
            domain = f"{parsed.scheme}://{parsed.netloc}"
            
            if domain not in self.robots_parsers:
                robots_url = urljoin(domain, "/robots.txt")
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self.robots_parsers[domain] = rp
                except Exception:
                    return True
            
            rp = self.robots_parsers[domain]
            return rp.can_fetch("SO8T-MultimodalCrawler/1.0", url)
        
        except Exception as e:
            logger.debug(f"Robots.txt check failed for {url}: {e}")
            return True
    
    def _extract_image_urls(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """
        HTMLから画像URLを抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
            base_url: ベースURL
        
        Returns:
            image_info_list: 画像情報のリスト
        """
        image_info_list = []
        
        # <img>タグから抽出
        for img_tag in soup.find_all('img'):
            src = img_tag.get('src') or img_tag.get('data-src') or img_tag.get('data-lazy-src')
            if not src:
                continue
            
            # 絶対URLに変換
            image_url = urljoin(base_url, src)
            
            # altテキスト取得
            alt_text = img_tag.get('alt', '')
            
            image_info = {
                "url": image_url,
                "alt_text": alt_text,
                "source": "img_tag"
            }
            
            image_info_list.append(image_info)
        
        # <picture>タグから抽出
        for picture_tag in soup.find_all('picture'):
            for source_tag in picture_tag.find_all('source'):
                srcset = source_tag.get('srcset', '')
                if srcset:
                    # srcsetから最初のURLを取得
                    first_url = srcset.split(',')[0].split()[0]
                    image_url = urljoin(base_url, first_url)
                    
                    image_info = {
                        "url": image_url,
                        "alt_text": "",
                        "source": "picture_tag"
                    }
                    
                    image_info_list.append(image_info)
        
        # CSS背景画像から抽出（簡易実装）
        for element in soup.find_all(style=True):
            style = element.get('style', '')
            # background-image: url(...) を検索
            match = re.search(r'background-image:\s*url\(["\']?([^"\']+)["\']?\)', style)
            if match:
                image_url = urljoin(base_url, match.group(1))
                
                image_info = {
                    "url": image_url,
                    "alt_text": "",
                    "source": "css_background"
                }
                
                image_info_list.append(image_info)
        
        return image_info_list
    
    def _download_image(self, image_url: str) -> Optional[Dict]:
        """
        画像をダウンロードして保存
        
        Args:
            image_url: 画像URL
        
        Returns:
            image_info: 画像情報（失敗時はNone）
        """
        def _fetch_image():
            # Chromeヘッダーを使用
            chrome_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': url if 'url' in locals() else '',
            }
            response = requests.get(
                image_url,
                timeout=self.config['timeout'],
                headers=chrome_headers,
                stream=True
            )
            response.raise_for_status()
            
            # 画像データ読み込み
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            
            # 画像サイズチェック
            width, height = image.size
            if width < self.config['image_min_width'] or height < self.config['image_min_height']:
                return None
            
            # ファイルサイズチェック
            file_size_mb = len(response.content) / (1024 * 1024)
            if file_size_mb > self.config['image_max_size_mb']:
                return None
            
            # 画像形式チェック
            image_format = image.format.lower() if image.format else ""
            if image_format not in self.config['image_formats']:
                return None
            
            # 画像保存
            image_filename = None
            if self.config['save_images']:
                image_hash = hashlib.md5(response.content).hexdigest()
                image_filename = f"{image_hash}.{image_format}"
                image_path = self.images_dir / image_filename
                
                image.save(image_path, format=image_format)
            
            return {
                "url": image_url,
                "width": width,
                "height": height,
                "format": image_format,
                "file_size_mb": file_size_mb,
                "filename": image_filename,
                "path": str(self.images_dir / image_filename) if image_filename else None
            }
        
        try:
            if self.retry_handler:
                retryable_exceptions = (
                    requests.exceptions.RequestException,
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                )
                
                try:
                    image_info = self.retry_handler.retry_sync(
                        _fetch_image,
                        operation_name=f"download_image_{image_url[:50]}",
                        retryable_exceptions=retryable_exceptions
                    )
                    return image_info
                except Exception as e:
                    if self.error_handler:
                        error_type = classify_exception(e)
                        self.error_handler.handle_error(
                            error_type,
                            image_url,
                            e,
                            context={"type": "image_download"},
                            log_traceback=False
                        )
                    return None
            else:
                return _fetch_image()
        
        except Exception as e:
            if self.error_handler:
                error_type = classify_exception(e)
                self.error_handler.handle_error(
                    error_type,
                    image_url,
                    e,
                    context={"type": "image_download"},
                    log_traceback=False
                )
            return None
    
    def _crawl_multimodal_url(self, url: str, depth: int = 0) -> Optional[Dict]:
        """
        マルチモーダルURLをクロール（テキスト+画像）
        
        Args:
            url: クロール対象URL
            depth: 現在の深度
        
        Returns:
            sample: 収集サンプル（失敗時はNone）
        """
        # 訪問済みチェック
        if url in self.visited_urls:
            return None
        self.visited_urls.add(url)
        
        # robots.txtチェック
        if not self._check_robots_txt(url):
            logger.debug(f"[ROBOTS] Disallowed: {url}")
            return None
        
        def _fetch_and_parse():
            """HTTPリクエストとHTML解析"""
            time.sleep(self.config['delay'])
            
            # Chromeヘッダーを使用
            chrome_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
            }
            response = requests.get(
                url,
                timeout=self.config['timeout'],
                headers=chrome_headers,
                allow_redirects=True
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # テキスト抽出
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()
            
            if len(text) < self.config['min_text_length']:
                return None
            
            # 画像URL抽出
            image_urls = self._extract_image_urls(soup, url)
            
            # 画像ダウンロード
            downloaded_images = []
            for image_info in image_urls[:10]:  # 最大10画像まで
                downloaded_image = self._download_image(image_info['url'])
                if downloaded_image:
                    downloaded_image['alt_text'] = image_info.get('alt_text', '')
                    downloaded_image['source'] = image_info.get('source', '')
                    downloaded_images.append(downloaded_image)
            
            # サンプル作成
            sample = {
                "text": text,
                "url": url,
                "source": "multimodal_web_crawl",
                "crawled_at": datetime.now().isoformat(),
                "text_length": len(text),
                "images": downloaded_images,
                "image_count": len(downloaded_images),
                "depth": depth
            }
            
            return sample
        
        try:
            # リトライ機構を使用
            if self.retry_handler:
                retryable_exceptions = (
                    requests.exceptions.RequestException,
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                )
                
                try:
                    sample = self.retry_handler.retry_sync(
                        _fetch_and_parse,
                        operation_name=f"multimodal_crawl_{url[:50]}",
                        retryable_exceptions=retryable_exceptions
                    )
                    return sample
                except Exception as e:
                    if self.error_handler:
                        error_type = classify_exception(e)
                        self.error_handler.handle_error(
                            error_type,
                            url,
                            e,
                            context={"type": "multimodal_crawl", "depth": depth},
                            log_traceback=False
                        )
                    return None
            else:
                return _fetch_and_parse()
        
        except Exception as e:
            if self.error_handler:
                error_type = classify_exception(e)
                self.error_handler.handle_error(
                    error_type,
                    url,
                    e,
                    context={"type": "multimodal_crawl", "depth": depth},
                    log_traceback=False
                )
            return None
    
    def crawl_multimodal(
        self,
        urls: List[str],
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        マルチモーダルデータ収集
        
        Args:
            urls: クロール対象URLリスト
            max_samples: 最大サンプル数
        
        Returns:
            samples: 収集サンプル
        """
        logger.info(f"[START] Multimodal web crawling from {len(urls):,} URLs")
        
        queue = [(url, 0) for url in urls]  # (url, depth)
        max_samples = max_samples or float('inf')
        
        with tqdm(desc="Crawling multimodal data") as pbar:
            while queue and len(self.collected_samples) < max_samples:
                url, depth = queue.pop(0)
                
                if depth >= self.config['max_depth']:
                    continue
                
                # クロール
                sample = self._crawl_multimodal_url(url, depth)
                
                if sample:
                    self.collected_samples.append(sample)
                    pbar.update(1)
                    pbar.set_postfix({
                        'samples': len(self.collected_samples),
                        'images': sum(s.get('image_count', 0) for s in self.collected_samples)
                    })
                    
                    # 深く探索（関連URLを発見）
                    if depth < self.config['max_depth'] - 1:
                        try:
                            response = requests.get(url, timeout=self.config['timeout'])
                            soup = BeautifulSoup(response.content, 'lxml')
                            
                            for a_tag in soup.find_all('a', href=True):
                                new_url = urljoin(url, a_tag['href'])
                                if urlparse(new_url).netloc == urlparse(url).netloc:
                                    if new_url not in self.visited_urls:
                                        queue.append((new_url, depth + 1))
                        except Exception:
                            pass
        
        logger.info(f"[COMPLETE] Collected {len(self.collected_samples):,} multimodal samples")
        logger.info(f"[COMPLETE] Total images: {sum(s.get('image_count', 0) for s in self.collected_samples):,}")
        
        return self.collected_samples
    
    def save(self, filename: Optional[str] = None):
        """
        収集データを保存
        
        Args:
            filename: 出力ファイル名（Noneの場合は自動生成）
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multimodal_crawled_{timestamp}.jsonl"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.collected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(self.collected_samples):,} samples to {output_file}")
        
        # 統計レポート
        stats = {
            "total_samples": len(self.collected_samples),
            "total_images": sum(s.get('image_count', 0) for s in self.collected_samples),
            "samples_with_images": sum(1 for s in self.collected_samples if s.get('image_count', 0) > 0),
            "collected_at": datetime.now().isoformat()
        }
        
        stats_file = self.output_dir / "multimodal_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[STATS] Statistics saved to {stats_file}")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal web crawling")
    parser.add_argument("--urls", type=str, nargs='+', required=True,
                        help="URLs to crawl")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory")
    args = parser.parse_args()
    
    crawler = MultimodalWebCrawler(output_dir=args.output_dir)
    
    try:
        samples = crawler.crawl_multimodal(args.urls, max_samples=args.max_samples)
        crawler.save()
        print(f"\n[SUCCESS] Collected {len(samples):,} multimodal samples")
    except Exception as e:
        logger.error(f"[ERROR] Crawling failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()

