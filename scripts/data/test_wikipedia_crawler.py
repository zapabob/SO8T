#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipediaクローラーのテストスクリプト

小さなサンプル数でクローラーをテスト実行
"""

import asyncio
import sys
import logging
from pathlib import Path

# ロギングレベルをDEBUGに設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data.wikipedia_chromium_crawler import (
    WikipediaChromiumCrawler,
    load_domain_keywords,
)


async def test_crawler():
    """クローラーのテスト実行"""
    print("="*80)
    print("Wikipedia Chromium Crawler Test")
    print("="*80)
    
    # キーワード設定読み込み
    keywords_config = load_domain_keywords()
    
    # テスト用の出力ディレクトリ
    test_output_dir = Path("D:/webdataset/test")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # クローラー作成（テスト用に少ないサンプル数）
    crawler = WikipediaChromiumCrawler(
        output_dir=test_output_dir,
        keywords_config=keywords_config,
        target_samples_per_domain=5,  # テスト用に5サンプル
    )
    
    try:
        # クロール実行
        await crawler.crawl()
        
        print("="*80)
        print("[SUCCESS] Test crawling completed")
        print("="*80)
        return 0
    
    except Exception as e:
        print(f"[ERROR] Test crawling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_crawler())
    sys.exit(exit_code)

