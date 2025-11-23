#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robots.txt遵守とレート制限管理モジュール

各データソースのrobots.txt確認とレート制限を管理
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RobotsComplianceManager:
    """robots.txt遵守管理クラス"""
    
    def __init__(self):
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.last_fetch_time: Dict[str, float] = {}
        self.user_agent = "SO8T-DataCollector/1.0 (Research Purpose)"
    
    def get_robots_parser(self, base_url: str) -> Optional[RobotFileParser]:
        """robots.txtパーサー取得（キャッシュ付き）"""
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        if domain in self.robots_cache:
            return self.robots_cache[domain]
        
        try:
            robots_url = f"{domain}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.read()
            
            self.robots_cache[domain] = parser
            logger.info(f"[ROBOTS] Loaded robots.txt from {robots_url}")
            return parser
        
        except Exception as e:
            logger.warning(f"[ROBOTS] Failed to load robots.txt from {domain}: {e}")
            return None
    
    def can_fetch(self, url: str) -> bool:
        """URL取得が許可されているか確認"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        parser = self.get_robots_parser(domain)
        if not parser:
            return True
        
        try:
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return True
    
    def get_crawl_delay(self, url: str) -> float:
        """robots.txtからクロール遅延を取得"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        parser = self.get_robots_parser(domain)
        if not parser:
            return 1.0  # デフォルト1秒
        
        try:
            delay = parser.crawl_delay(self.user_agent)
            return delay if delay else 1.0
        except Exception:
            return 1.0


class RateLimiter:
    """レート制限管理クラス"""
    
    def __init__(self, default_delay: float = 1.0):
        """
        Args:
            default_delay: デフォルト遅延（秒）
        """
        self.default_delay = default_delay
        self.domain_delays: Dict[str, float] = defaultdict(lambda: default_delay)
        self.last_request_time: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.request_windows: Dict[str, timedelta] = defaultdict(lambda: timedelta(seconds=60))
    
    def set_domain_delay(self, domain: str, delay: float):
        """ドメイン別遅延設定"""
        self.domain_delays[domain] = delay
        logger.info(f"[RATE] Set delay for {domain}: {delay}s")
    
    def wait_if_needed(self, url: str):
        """必要に応じて待機"""
        parsed = urlparse(url)
        domain = parsed.netloc
        
        delay = self.domain_delays.get(domain, self.default_delay)
        last_time = self.last_request_time.get(domain, 0)
        
        elapsed = time.time() - last_time
        if elapsed < delay:
            wait_time = delay - elapsed
            time.sleep(wait_time)
        
        self.last_request_time[domain] = time.time()
        self.request_counts[domain] += 1
    
    def get_delay(self, domain: str) -> float:
        """ドメインの遅延時間を取得"""
        return self.domain_delays.get(domain, self.default_delay)
    
    def reset_domain(self, domain: str):
        """ドメインのリクエストカウントをリセット"""
        self.request_counts[domain] = 0
        self.last_request_time[domain] = 0


class ComplianceChecker:
    """遵守チェッカー（robots.txt + レート制限統合）"""
    
    def __init__(self, default_delay: float = 1.0):
        """
        Args:
            default_delay: デフォルト遅延（秒）
        """
        self.robots_manager = RobotsComplianceManager()
        self.rate_limiter = RateLimiter(default_delay)
    
    def check_and_wait(self, url: str) -> bool:
        """
        robots.txt確認とレート制限待機
        
        Args:
            url: チェック対象URL
        
        Returns:
            can_fetch: 取得可能かどうか
        """
        # robots.txt確認
        if not self.robots_manager.can_fetch(url):
            logger.debug(f"[COMPLIANCE] robots.txt disallows: {url}")
            return False
        
        # レート制限待機
        self.rate_limiter.wait_if_needed(url)
        
        # robots.txtから遅延時間を取得して設定
        parsed = urlparse(url)
        domain = parsed.netloc
        crawl_delay = self.robots_manager.get_crawl_delay(url)
        self.rate_limiter.set_domain_delay(domain, crawl_delay)
        
        return True
    
    def set_domain_delay(self, domain: str, delay: float):
        """ドメイン別遅延設定"""
        self.rate_limiter.set_domain_delay(domain, delay)
    
    def get_stats(self) -> Dict:
        """統計情報取得"""
        return {
            'robots_cache_size': len(self.robots_manager.robots_cache),
            'domain_delays': dict(self.rate_limiter.domain_delays),
            'request_counts': dict(self.rate_limiter.request_counts)
        }


# デフォルトのドメイン別遅延設定
DEFAULT_DOMAIN_DELAYS = {
    'kanpou.4web.jp': 2.0,
    'www.e-gov.go.jp': 1.5,
    'elaws.e-gov.go.jp': 1.5,
    'zenn.dev': 1.0,
    'qiita.com': 1.0,
    'ja.wikipedia.org': 0.5,  # WikipediaはAPI推奨
    'en.wikipedia.org': 0.5,
    'zh.wikipedia.org': 0.5,
}


def create_compliance_checker(config: Dict = None) -> ComplianceChecker:
    """遵守チェッカー作成"""
    default_delay = config.get('default_delay', 1.0) if config else 1.0
    checker = ComplianceChecker(default_delay)
    
    # デフォルト遅延設定を適用
    for domain, delay in DEFAULT_DOMAIN_DELAYS.items():
        checker.set_domain_delay(domain, delay)
    
    return checker

















