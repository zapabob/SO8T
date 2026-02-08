#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特定ドメイン特化データ収集モジュール

防衛、航空宇宙、半導体、精密機器、インフラ、運輸などの特定ドメインに特化した
データ収集機能を提供します。

Usage:
    from so8t_mmllm.scripts.data.domain_specialized_collector import DomainSpecializedCollector
    collector = DomainSpecializedCollector(domain="defense")
    samples = collector.collect(target_samples=1000)
"""

import os
import sys
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote
from urllib.robotparser import RobotFileParser
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# エラーハンドリングとリトライ機構のインポート
try:
    from scripts.data.crawler_error_handler import CrawlerErrorHandler, ErrorType, classify_exception
    from scripts.data.retry_handler import RetryHandler
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False

logger = logging.getLogger(__name__)


# ドメイン別キーワード定義
DOMAIN_KEYWORDS = {
    "defense": {
        "keywords": [
            "防衛", "軍事", "安全保障", "国防", "自衛隊", "武器", "戦略",
            "防衛装備", "防衛省", "統合幕僚監部", "陸上自衛隊", "海上自衛隊", "航空自衛隊"
        ],
        "sources": [
            "https://www.mod.go.jp/",
            "https://www.jda.go.jp/",
            "https://www.nids.mod.go.jp/",  # 防衛研究所
        ],
        "priority": "high"
    },
    "aerospace": {
        "keywords": [
            "航空", "宇宙", "ロケット", "衛星", "飛行", "航空機", "ジェット",
            "JAXA", "宇宙開発", "人工衛星", "国際宇宙ステーション", "ISS"
        ],
        "sources": [
            "https://www.jaxa.jp/",
            "https://www.mext.go.jp/a_menu/kaihatu/space/",  # 文部科学省 宇宙開発
        ],
        "priority": "high"
    },
    "semiconductor": {
        "keywords": [
            "半導体", "チップ", "IC", "LSI", "メモリ", "プロセッサ",
            "TSMC", "インテル", "AMD", "NVIDIA", "半導体製造", "ファウンドリ"
        ],
        "sources": [
            "https://www.meti.go.jp/policy/mono_info_service/mono/electronics/",  # 経済産業省
        ],
        "priority": "medium"
    },
    "precision_machinery": {
        "keywords": [
            "精密機器", "工作機械", "NC", "CNC", "ロボット", "FA",
            "産業用ロボット", "工作機械", "測定器", "光学機器"
        ],
        "sources": [],
        "priority": "medium"
    },
    "infrastructure": {
        "keywords": [
            "インフラ", "インフラストラクチャ", "社会資本", "公共事業",
            "道路", "橋", "トンネル", "ダム", "港湾", "空港"
        ],
        "sources": [
            "https://www.mlit.go.jp/",  # 国土交通省
        ],
        "priority": "high"
    },
    "transport": {
        "keywords": [
            "運輸", "交通", "鉄道", "輸送", "物流", "道路", "港湾",
            "JR", "新幹線", "在来線", "貨物", "物流センター"
        ],
        "sources": [
            "https://www.mlit.go.jp/",  # 国土交通省
        ],
        "priority": "high"
    }
}


class DomainSpecializedCollector:
    """特定ドメイン特化データ収集クラス"""
    
    def __init__(
        self,
        domain: str,
        config: Optional[Dict] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Args:
            domain: ドメイン名（"defense", "aerospace", "semiconductor", etc.）
            config: 設定辞書
            output_dir: 出力ディレクトリ
        """
        if domain not in DOMAIN_KEYWORDS:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_KEYWORDS.keys())}")
        
        self.domain = domain
        self.domain_config = DOMAIN_KEYWORDS[domain]
        self.config = config or {
            "max_depth": 3,
            "delay": 1.0,
            "timeout": 15,
            "max_pages": 1000,
            "min_text_length": 200,
            "min_keyword_density": 0.02,  # キーワード密度の最小値（2%）
        }
        
        # 出力ディレクトリ
        if output_dir is None:
            output_dir = Path("data/domain_specialized") / domain
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # エラーハンドリングとリトライ機構
        if ERROR_HANDLING_AVAILABLE:
            error_log_dir = self.output_dir / "error_logs"
            self.error_handler = CrawlerErrorHandler(log_dir=error_log_dir)
            self.retry_handler = RetryHandler(
                max_retries=self.config.get('max_retries', 3),
                initial_delay=self.config.get('retry_initial_delay', 1.0),
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
        """
        robots.txtをチェック
        
        Args:
            url: チェック対象URL
        
        Returns:
            allowed: クロールが許可されているか
        """
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
                except:
                    # robots.txtが存在しない場合は許可
                    return True
            
            rp = self.robots_parsers[domain]
            return rp.can_fetch("SO8T-DomainCollector/1.0", url)
        
        except Exception as e:
            logger.debug(f"Robots.txt check failed for {url}: {e}")
            return True  # エラー時は許可
    
    def _calculate_domain_relevance(self, text: str) -> Tuple[float, int]:
        """
        ドメイン関連性スコアリング
        
        Args:
            text: テキスト
        
        Returns:
            (relevance_score, keyword_count): 関連性スコア（0.0-1.0）とキーワード出現数
        """
        keywords = self.domain_config["keywords"]
        text_lower = text.lower()
        
        # キーワード出現数
        keyword_count = sum(1 for kw in keywords if kw.lower() in text_lower)
        
        # キーワード密度
        keyword_density = keyword_count / len(keywords)
        
        # テキスト長に対するキーワード出現率
        text_length = len(text)
        if text_length > 0:
            keyword_ratio = keyword_count / (text_length / 100)  # 100文字あたりの出現数
        else:
            keyword_ratio = 0
        
        # 関連性スコア計算（キーワード密度と出現率の組み合わせ）
        relevance_score = min(
            keyword_density * 0.6 + min(keyword_ratio / 10.0, 1.0) * 0.4,
            1.0
        )
        
        return relevance_score, keyword_count
    
    def _crawl_url(self, url: str, depth: int = 0) -> Optional[Dict]:
        """
        単一URLをクロール
        
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
            
            response = requests.get(
                url,
                timeout=self.config['timeout'],
                headers={'User-Agent': 'SO8T-DomainCollector/1.0 (Research)'},
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
            
            # ドメイン関連性スコアリング
            relevance_score, keyword_count = self._calculate_domain_relevance(text)
            
            # 最小キーワード密度チェック
            if relevance_score < self.config['min_keyword_density']:
                return None
            
            # サンプル作成
            sample = {
                "text": text,
                "url": url,
                "domain": self.domain,
                "source": "domain_specialized_crawl",
                "crawled_at": datetime.now().isoformat(),
                "text_length": len(text),
                "relevance_score": relevance_score,
                "keyword_count": keyword_count,
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
                        operation_name=f"domain_crawl_{url[:50]}",
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
                            context={"domain": self.domain, "depth": depth},
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
                    context={"domain": self.domain, "depth": depth},
                    log_traceback=False
                )
            return None
    
    def _discover_urls(self, base_url: str, max_urls: int = 100) -> List[str]:
        """
        キーワードベースURL発見
        
        Args:
            base_url: ベースURL
            max_urls: 最大URL数
        
        Returns:
            discovered_urls: 発見されたURLリスト
        """
        discovered_urls = []
        queue = [base_url]
        visited = set()
        
        while queue and len(discovered_urls) < max_urls:
            url = queue.pop(0)
            
            if url in visited:
                continue
            visited.add(url)
            
            try:
                response = requests.get(url, timeout=self.config['timeout'])
                soup = BeautifulSoup(response.content, 'lxml')
                
                # リンク抽出
                for a_tag in soup.find_all('a', href=True):
                    new_url = urljoin(url, a_tag['href'])
                    parsed = urlparse(new_url)
                    
                    # 同一ドメインのみ
                    if parsed.netloc == urlparse(base_url).netloc:
                        # キーワードがURLまたはリンクテキストに含まれる場合
                        link_text = a_tag.get_text().lower()
                        url_lower = new_url.lower()
                        
                        keywords_lower = [kw.lower() for kw in self.domain_config["keywords"]]
                        if any(kw in link_text or kw in url_lower for kw in keywords_lower):
                            if new_url not in visited and new_url not in discovered_urls:
                                discovered_urls.append(new_url)
                                queue.append(new_url)
            
            except Exception as e:
                logger.debug(f"URL discovery failed for {url}: {e}")
                continue
        
        return discovered_urls[:max_urls]
    
    def collect(self, target_samples: int = 1000) -> List[Dict]:
        """
        ドメイン特化データ収集
        
        Args:
            target_samples: 目標サンプル数
        
        Returns:
            samples: 収集サンプル
        """
        logger.info(f"[START] Domain-specialized collection: {self.domain}")
        logger.info(f"[TARGET] {target_samples:,} samples")
        
        # ソースURLから開始
        source_urls = self.domain_config.get("sources", [])
        
        # キーワードベースURL発見
        discovered_urls = []
        for source_url in source_urls:
            logger.info(f"[DISCOVER] Discovering URLs from {source_url}...")
            urls = self._discover_urls(source_url, max_urls=500)
            discovered_urls.extend(urls)
            logger.info(f"[OK] Discovered {len(urls):,} URLs")
        
        # 重複除去
        discovered_urls = list(set(discovered_urls))
        logger.info(f"[OK] Total unique URLs: {len(discovered_urls):,}")
        
        # クロール実行
        queue = [(url, 0) for url in discovered_urls]  # (url, depth)
        
        with tqdm(total=target_samples, desc=f"Collecting {self.domain}") as pbar:
            while queue and len(self.collected_samples) < target_samples:
                url, depth = queue.pop(0)
                
                if depth >= self.config['max_depth']:
                    continue
                
                # クロール
                sample = self._crawl_url(url, depth)
                
                if sample:
                    self.collected_samples.append(sample)
                    pbar.update(1)
                    pbar.set_postfix({
                        'relevance': f"{sample['relevance_score']:.2f}",
                        'keywords': sample['keyword_count']
                    })
                    
                    # 関連性が高い場合は深く探索
                    if sample['relevance_score'] > 0.3 and depth < self.config['max_depth'] - 1:
                        try:
                            response = requests.get(url, timeout=self.config['timeout'])
                            soup = BeautifulSoup(response.content, 'lxml')
                            
                            for a_tag in soup.find_all('a', href=True):
                                new_url = urljoin(url, a_tag['href'])
                                if urlparse(new_url).netloc == urlparse(url).netloc:
                                    if new_url not in self.visited_urls:
                                        queue.append((new_url, depth + 1))
                        except:
                            pass
        
        logger.info(f"[COMPLETE] Collected {len(self.collected_samples):,} samples")
        
        # 関連性スコアでソート
        self.collected_samples.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return self.collected_samples
    
    def save(self, filename: Optional[str] = None):
        """
        収集データを保存
        
        Args:
            filename: 出力ファイル名（Noneの場合は自動生成）
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.domain}_specialized_{timestamp}.jsonl"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.collected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(self.collected_samples):,} samples to {output_file}")
        
        # 統計レポート
        stats = {
            "domain": self.domain,
            "total_samples": len(self.collected_samples),
            "avg_relevance_score": sum(s['relevance_score'] for s in self.collected_samples) / len(self.collected_samples) if self.collected_samples else 0.0,
            "avg_keyword_count": sum(s['keyword_count'] for s in self.collected_samples) / len(self.collected_samples) if self.collected_samples else 0.0,
            "collected_at": datetime.now().isoformat()
        }
        
        stats_file = self.output_dir / f"{self.domain}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[STATS] Statistics saved to {stats_file}")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain-specialized data collection")
    parser.add_argument("--domain", type=str, required=True,
                        choices=list(DOMAIN_KEYWORDS.keys()),
                        help="Domain to collect")
    parser.add_argument("--target", type=int, default=1000,
                        help="Target number of samples")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory")
    args = parser.parse_args()
    
    collector = DomainSpecializedCollector(
        domain=args.domain,
        output_dir=args.output_dir
    )
    
    try:
        samples = collector.collect(target_samples=args.target)
        collector.save()
        print(f"\n[SUCCESS] Collected {len(samples):,} samples for domain: {args.domain}")
    except Exception as e:
        logger.error(f"[ERROR] Collection failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()

