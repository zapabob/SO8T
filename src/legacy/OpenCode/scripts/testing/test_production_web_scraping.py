#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本番環境webスクレイピングテストスクリプト

各データソースのクローラー単体テスト、robots.txt遵守確認テスト、レート制限動作確認テスト
"""

import os
import sys
import json
import logging
import unittest
from pathlib import Path
from typing import Dict, List

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# テスト対象モジュールのインポート
try:
    from scripts.data.specialized_crawlers import (
        Kanpou4WebCrawler, EGovCrawler, ZennCrawler, QiitaCrawler, WikipediaCrawler
    )
    from scripts.data.robots_compliance import (
        RobotsComplianceManager, RateLimiter, ComplianceChecker, create_compliance_checker
    )
    from scripts.data.massive_parallel_crawler import COMPREHENSIVE_SOURCES
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)


class TestRobotsCompliance(unittest.TestCase):
    """robots.txt遵守テスト"""
    
    def setUp(self):
        self.robots_manager = RobotsComplianceManager()
    
    def test_robots_txt_loading(self):
        """robots.txt読み込みテスト"""
        parser = self.robots_manager.get_robots_parser("https://www.e-gov.go.jp/")
        self.assertIsNotNone(parser)
    
    def test_can_fetch_check(self):
        """URL取得許可確認テスト"""
        # 許可されているURL
        can_fetch = self.robots_manager.can_fetch("https://www.e-gov.go.jp/")
        self.assertIsInstance(can_fetch, bool)
    
    def test_crawl_delay_retrieval(self):
        """クロール遅延取得テスト"""
        delay = self.robots_manager.get_crawl_delay("https://www.e-gov.go.jp/")
        self.assertIsInstance(delay, float)
        self.assertGreaterEqual(delay, 0)


class TestRateLimiter(unittest.TestCase):
    """レート制限テスト"""
    
    def setUp(self):
        self.rate_limiter = RateLimiter(default_delay=1.0)
    
    def test_domain_delay_setting(self):
        """ドメイン別遅延設定テスト"""
        self.rate_limiter.set_domain_delay("example.com", 2.0)
        delay = self.rate_limiter.get_delay("example.com")
        self.assertEqual(delay, 2.0)
    
    def test_wait_if_needed(self):
        """待機機能テスト"""
        import time
        start_time = time.time()
        self.rate_limiter.wait_if_needed("https://example.com/test")
        elapsed = time.time() - start_time
        # デフォルト遅延以上かどうか確認
        self.assertGreaterEqual(elapsed, 0.9)  # 1秒 - 誤差


class TestComplianceChecker(unittest.TestCase):
    """遵守チェッカーテスト"""
    
    def setUp(self):
        self.checker = create_compliance_checker({'default_delay': 1.0})
    
    def test_check_and_wait(self):
        """チェック&待機テスト"""
        result = self.checker.check_and_wait("https://www.e-gov.go.jp/")
        self.assertIsInstance(result, bool)
    
    def test_get_stats(self):
        """統計情報取得テスト"""
        stats = self.checker.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('robots_cache_size', stats)
        self.assertIn('domain_delays', stats)
        self.assertIn('request_counts', stats)


class TestSpecializedCrawlers(unittest.TestCase):
    """専用クローラーテスト"""
    
    def setUp(self):
        self.base_config = {
            'delay': 1.0,
            'timeout': 10,
            'max_pages': 10,  # テスト用に少なく設定
            'user_agent': 'SO8T-DataCollector/1.0 (Research Purpose)'
        }
    
    def test_kanpou_crawler_init(self):
        """4web版官報クローラー初期化テスト"""
        config = self.base_config.copy()
        config.update({
            'base_url': 'https://kanpou.4web.jp/',
            'domain': 'official_gazette',
            'language': 'ja'
        })
        crawler = Kanpou4WebCrawler(config)
        self.assertIsNotNone(crawler)
        self.assertEqual(crawler.base_url, 'https://kanpou.4web.jp/')
    
    def test_egov_crawler_init(self):
        """e-Govクローラー初期化テスト"""
        config = self.base_config.copy()
        config.update({
            'base_url': 'https://www.e-gov.go.jp/',
            'domain': 'government',
            'language': 'ja'
        })
        crawler = EGovCrawler(config)
        self.assertIsNotNone(crawler)
        self.assertEqual(crawler.base_url, 'https://www.e-gov.go.jp/')
    
    def test_zenn_crawler_init(self):
        """zennクローラー初期化テスト"""
        config = self.base_config.copy()
        config.update({
            'base_url': 'https://zenn.dev/',
            'domain': 'tech_blog',
            'language': 'ja',
            'api_enabled': False
        })
        crawler = ZennCrawler(config)
        self.assertIsNotNone(crawler)
        self.assertEqual(crawler.base_url, 'https://zenn.dev/')
    
    def test_qiita_crawler_init(self):
        """Qiitaクローラー初期化テスト"""
        config = self.base_config.copy()
        config.update({
            'base_url': 'https://qiita.com/',
            'domain': 'tech_blog',
            'language': 'ja',
            'api_enabled': False,
            'api_token': None
        })
        crawler = QiitaCrawler(config)
        self.assertIsNotNone(crawler)
        self.assertEqual(crawler.base_url, 'https://qiita.com/')
    
    def test_wikipedia_crawler_init(self):
        """ウィキペディアクローラー初期化テスト"""
        config = self.base_config.copy()
        config.update({
            'base_url': 'https://ja.wikipedia.org/wiki/',
            'domain': 'encyclopedia',
            'language': 'ja'
        })
        crawler = WikipediaCrawler(config)
        self.assertIsNotNone(crawler)
        self.assertEqual(crawler.base_url, 'https://ja.wikipedia.org/wiki/')


class TestComprehensiveSources(unittest.TestCase):
    """包括的ソース設定テスト"""
    
    def test_comprehensive_sources_contains_new_sources(self):
        """新データソースがCOMPREHENSIVE_SOURCESに含まれているか確認"""
        self.assertIn('kanpou_4web', COMPREHENSIVE_SOURCES)
        self.assertIn('egov', COMPREHENSIVE_SOURCES)
        self.assertIn('zenn', COMPREHENSIVE_SOURCES)
        self.assertIn('qiita', COMPREHENSIVE_SOURCES)
        self.assertIn('wikipedia_ja', COMPREHENSIVE_SOURCES)
    
    def test_source_config_structure(self):
        """ソース設定の構造確認"""
        for source_id, source_config in COMPREHENSIVE_SOURCES.items():
            self.assertIn('urls', source_config)
            self.assertIn('domain', source_config)
            self.assertIn('language', source_config)
            self.assertIn('priority', source_config)
    
    def test_new_sources_have_crawler_type(self):
        """新データソースにcrawler_typeが設定されているか確認"""
        new_sources = ['kanpou_4web', 'egov', 'zenn', 'qiita']
        for source_id in new_sources:
            if source_id in COMPREHENSIVE_SOURCES:
                self.assertIn('crawler_type', COMPREHENSIVE_SOURCES[source_id])


class TestNikkei225Sources(unittest.TestCase):
    """日経225企業リストテスト"""
    
    def setUp(self):
        nikkei_file = PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "nikkei225_sources.json"
        with open(nikkei_file, 'r', encoding='utf-8') as f:
            self.nikkei_data = json.load(f)
    
    def test_nikkei225_file_exists(self):
        """日経225ファイルが存在するか確認"""
        nikkei_file = PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "nikkei225_sources.json"
        self.assertTrue(nikkei_file.exists())
    
    def test_nikkei225_companies_count(self):
        """日経225企業数が225社以上か確認"""
        companies = self.nikkei_data.get('nikkei225_companies', [])
        self.assertGreaterEqual(len(companies), 225)
    
    def test_nikkei225_company_structure(self):
        """日経225企業データの構造確認"""
        companies = self.nikkei_data.get('nikkei225_companies', [])
        for company in companies[:10]:  # 最初の10社をチェック
            self.assertIn('name', company)
            self.assertIn('url', company)
            self.assertIn('domain', company)


def run_integration_test():
    """統合テスト（実際のクロールは実行しない）"""
    logger.info("="*80)
    logger.info("INTEGRATION TEST: Production Web Scraping")
    logger.info("="*80)
    
    # 1. robots.txt遵守チェッカー作成
    logger.info("[TEST 1] Creating compliance checker...")
    checker = create_compliance_checker({'default_delay': 1.0})
    stats = checker.get_stats()
    logger.info(f"[OK] Compliance checker created: {stats}")
    
    # 2. 各データソースの設定確認
    logger.info("[TEST 2] Checking data source configurations...")
    required_sources = ['kanpou_4web', 'egov', 'zenn', 'qiita', 'wikipedia_ja']
    for source_id in required_sources:
        if source_id in COMPREHENSIVE_SOURCES:
            logger.info(f"[OK] {source_id}: {COMPREHENSIVE_SOURCES[source_id]}")
        else:
            logger.error(f"[FAIL] {source_id} not found in COMPREHENSIVE_SOURCES")
    
    # 3. 日経225企業リスト確認
    logger.info("[TEST 3] Checking Nikkei225 sources...")
    nikkei_file = PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "nikkei225_sources.json"
    if nikkei_file.exists():
        with open(nikkei_file, 'r', encoding='utf-8') as f:
            nikkei_data = json.load(f)
        companies_count = len(nikkei_data.get('nikkei225_companies', []))
        logger.info(f"[OK] Nikkei225 companies: {companies_count}")
    else:
        logger.error("[FAIL] Nikkei225 file not found")
    
    logger.info("="*80)
    logger.info("[COMPLETE] Integration test finished")
    logger.info("="*80)


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Web Scraping Test")
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run unit tests'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run integration test'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests'
    )
    
    args = parser.parse_args()
    
    if args.all or (not args.unit and not args.integration):
        # デフォルト: すべて実行
        args.unit = True
        args.integration = True
    
    if args.unit:
        logger.info("Running unit tests...")
        unittest.main(argv=[''], exit=False, verbosity=2)
    
    if args.integration:
        logger.info("Running integration test...")
        run_integration_test()


if __name__ == '__main__':
    main()

