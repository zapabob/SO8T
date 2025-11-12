#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T/thinkingモデル統制Webスクレイピング再開機能

CursorブラウザによるバックグラウンドwebスクレイピングをSO8T/thinkingモデルで統制

Usage:
    python scripts/data/so8t_thinking_controlled_scraping.py --output D:/webdataset/processed --num-browsers 10
"""

import sys
import json
import logging
import asyncio
import argparse
import signal
import random
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import deque
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "agents"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "audit"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipelines"))

# 既存の並列スクレイピングをインポート
try:
    from scripts.data.parallel_deep_research_scraping import (
        ParallelDeepResearchScraper,
        ResourceManager,
        KeywordTask
    )
    PARALLEL_SCRAPER_AVAILABLE = True
except ImportError as e:
    PARALLEL_SCRAPER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Parallel scraper not available: {e}")

# スクレイピング推論エージェントインポート
try:
    from scripts.agents.scraping_reasoning_agent import ScrapingReasoningAgent
    REASONING_AGENT_AVAILABLE = True
except ImportError:
    REASONING_AGENT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Scraping reasoning agent not available")

# 監査ログインポート
try:
    from scripts.audit.scraping_audit_logger import ScrapingAuditLogger, ScrapingSession, ScrapingEvent
    AUDIT_LOGGER_AVAILABLE = True
except ImportError:
    AUDIT_LOGGER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Scraping audit logger not available")

# 電源断保護インポート
try:
    from scripts.pipelines.power_failure_protected_scraping_pipeline import PowerFailureRecovery
    POWER_FAILURE_RECOVERY_AVAILABLE = True
except ImportError:
    POWER_FAILURE_RECOVERY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Power failure recovery not available")

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error("Playwright not available. Install with: pip install playwright")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/so8t_thinking_controlled_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SO8TThinkingControlledScraper(ParallelDeepResearchScraper):
    """SO8T/thinkingモデル統制Webスクレイピングクラス"""
    
    def __init__(
        self,
        output_dir: Path,
        num_browsers: int = 10,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        delay_per_action: float = 1.5,
        timeout: int = 30000,
        max_pages_per_keyword: int = 5,
        max_memory_gb: float = 8.0,
        max_cpu_percent: float = 80.0,
        use_so8t_control: bool = True,
        so8t_model_path: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
        resume: bool = True,
        target_samples: Optional[int] = None,
        min_samples_per_keyword: int = 10,
        max_samples_per_keyword: int = 100
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            num_browsers: ブラウザ数
            use_cursor_browser: Cursorのブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_action: アクション間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_pages_per_keyword: キーワードあたりの最大ページ数
            max_memory_gb: 最大メモリ使用量（GB）
            max_cpu_percent: 最大CPU使用率（%）
            use_so8t_control: SO8Tモデルで統制するか
            so8t_model_path: SO8Tモデルのパス
            checkpoint_dir: チェックポイントディレクトリ
            resume: チェックポイントから再開するか
            target_samples: 目標サンプル数（Noneの場合は無制限）
            min_samples_per_keyword: キーワードあたりの最小サンプル数
            max_samples_per_keyword: キーワードあたりの最大サンプル数
        """
        # 親クラスの初期化
        super().__init__(
            output_dir=output_dir,
            num_browsers=num_browsers,
            use_cursor_browser=use_cursor_browser,
            remote_debugging_port=remote_debugging_port,
            delay_per_action=delay_per_action,
            timeout=timeout,
            max_pages_per_keyword=max_pages_per_keyword,
            max_memory_gb=max_memory_gb,
            max_cpu_percent=max_cpu_percent,
            use_so8t_control=use_so8t_control,
            so8t_model_path=so8t_model_path
        )
        
        # 監査ロガー初期化
        self.audit_logger = None
        if AUDIT_LOGGER_AVAILABLE:
            try:
                self.audit_logger = ScrapingAuditLogger()
                logger.info("[AUDIT] Audit logger initialized")
            except Exception as e:
                logger.warning(f"[AUDIT] Failed to initialize audit logger: {e}")
        
        # スクレイピング推論エージェント初期化
        self.reasoning_agent = None
        if REASONING_AGENT_AVAILABLE and use_so8t_control:
            try:
                self.reasoning_agent = ScrapingReasoningAgent(
                    model_path=so8t_model_path,
                    audit_logger=self.audit_logger
                )
                logger.info("[REASONING] Reasoning agent initialized")
            except Exception as e:
                logger.warning(f"[REASONING] Failed to initialize reasoning agent: {e}")
        
        # 電源断保護システム初期化
        self.power_recovery = None
        if POWER_FAILURE_RECOVERY_AVAILABLE:
            checkpoint_dir = checkpoint_dir or (output_dir / "checkpoints")
            try:
                self.power_recovery = PowerFailureRecovery(
                    checkpoint_dir=checkpoint_dir,
                    max_checkpoints=10,
                    checkpoint_interval=300.0,  # 5分
                    audit_logger=self.audit_logger
                )
                logger.info("[POWER] Power failure recovery initialized")
                
                # 自動チェックポイント開始
                self.power_recovery.start_auto_checkpoint()
                
                # セッション復旧
                if resume:
                    recovered_sessions = self.power_recovery.recover_sessions()
                    if recovered_sessions:
                        logger.info(f"[POWER] Recovered {len(recovered_sessions)} sessions")
            except Exception as e:
                logger.warning(f"[POWER] Failed to initialize power recovery: {e}")
        
        # データ量拡大設定
        self.target_samples = target_samples
        self.min_samples_per_keyword = min_samples_per_keyword
        self.max_samples_per_keyword = max_samples_per_keyword
        self.current_sample_count = 0
        self.keyword_sample_counts: Dict[str, int] = {}
        
        logger.info("="*80)
        logger.info("SO8T Thinking Controlled Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Audit logger: {self.audit_logger is not None}")
        logger.info(f"Reasoning agent: {self.reasoning_agent is not None}")
        logger.info(f"Power recovery: {self.power_recovery is not None}")
        if self.target_samples:
            logger.info(f"Target samples: {self.target_samples}")
            logger.info(f"Min samples per keyword: {self.min_samples_per_keyword}")
            logger.info(f"Max samples per keyword: {self.max_samples_per_keyword}")
    
    async def scrape_keyword_with_browser(
        self,
        page: Page,
        browser_index: int,
        task: KeywordTask
    ) -> List[Dict]:
        """
        キーワードをスクレイピング（SO8T統制版）
        
        Args:
            page: Playwrightページ
            browser_index: ブラウザインデックス
            task: キーワードタスク
            
        Returns:
            samples: 収集したサンプルのリスト
        """
        session_id = f"{self.session_id}_browser_{browser_index}"
        keyword = task.keyword
        
        # セッション作成
        if self.power_recovery:
            session = self.power_recovery.create_session(
                session_id=session_id,
                browser_index=browser_index,
                keyword=keyword
            )
        else:
            session = None
        
        # キーワード評価（SO8T統制）
        if self.reasoning_agent:
            keyword_result = self.reasoning_agent.evaluate_keyword(
                keyword=keyword,
                context=f"Browser {browser_index} scraping",
                session_id=session_id
            )
            
            if not keyword_result.get('should_scrape', True):
                logger.warning(f"[SO8T] Keyword '{keyword}' denied: {keyword_result.get('decision', 'UNKNOWN')}")
                logger.warning(f"[SO8T] Reasoning: {keyword_result.get('reasoning', 'No reasoning')[:200]}")
                
                # セッション状態を更新
                if self.power_recovery and session:
                    self.power_recovery.update_session(session_id, status="denied")
                
                return []
        
        # ブラウザ状態更新
        if browser_index not in self.browser_status:
            self.browser_status[browser_index] = {
                'status': 'active',
                'current_keyword': keyword,
                'samples_collected': 0,
                'last_activity': datetime.now().isoformat()
            }
        else:
            self.browser_status[browser_index]['current_keyword'] = keyword
            self.browser_status[browser_index]['last_activity'] = datetime.now().isoformat()
        
        samples = []
        
        try:
            # 検索実行
            search_success = await self.human_like_search(page, keyword, task.language)
            
            if not search_success:
                logger.warning(f"[SCRAPE] Search failed for keyword: {keyword}")
                return []
            
            # 検索結果からURLを抽出
            search_results = await self.extract_search_results(page, keyword)
            
            # キーワードあたりのサンプル数チェック
            keyword_count = self.keyword_sample_counts.get(keyword, 0)
            if keyword_count >= self.max_samples_per_keyword:
                logger.info(f"[SAMPLES] Keyword '{keyword}' reached max samples ({self.max_samples_per_keyword}), skipping")
                return samples
            
            # 目標サンプル数チェック
            if self.target_samples and self.current_sample_count >= self.target_samples:
                logger.info(f"[SAMPLES] Target samples reached ({self.target_samples}), stopping")
                return samples
            
            # URLごとにスクレイピング
            for url_info in search_results[:self.max_pages_per_keyword]:
                # 目標サンプル数チェック（ループ内）
                if self.target_samples and self.current_sample_count >= self.target_samples:
                    logger.info(f"[SAMPLES] Target samples reached ({self.target_samples}), stopping")
                    break
                
                # キーワードあたりのサンプル数チェック（ループ内）
                keyword_count = self.keyword_sample_counts.get(keyword, 0)
                if keyword_count >= self.max_samples_per_keyword:
                    logger.info(f"[SAMPLES] Keyword '{keyword}' reached max samples ({self.max_samples_per_keyword}), skipping")
                    break
                
                url = url_info.get('url', '')
                if not url:
                    continue
                
                # 重複チェック
                if self.power_recovery:
                    is_duplicate = self.power_recovery.check_duplicate(url, keyword, session_id)
                    if is_duplicate:
                        logger.debug(f"[DUPLICATE] Skipping duplicate URL: {url[:50]}...")
                        continue
                
                # URL評価（SO8T統制）
                if self.reasoning_agent:
                    url_result = self.reasoning_agent.should_scrape(
                        url=url,
                        keyword=keyword,
                        context=f"Browser {browser_index} scraping",
                        session_id=session_id
                    )
                    
                    if not url_result.get('should_scrape', True):
                        logger.warning(f"[SO8T] URL '{url[:50]}...' denied: {url_result.get('decision', 'UNKNOWN')}")
                        continue
                
                # ページスクレイピング
                try:
                    sample = await self.scrape_page_with_so8t_control(
                        page=page,
                        url=url,
                        keyword=keyword,
                        browser_index=browser_index,
                        session_id=session_id
                    )
                    
                    if sample:
                        # 重複登録
                        if self.power_recovery:
                            self.power_recovery.register_duplicate(url, keyword, session_id)
                        
                        samples.append(sample)
                        
                        # サンプル数カウント更新
                        self.current_sample_count += 1
                        self.keyword_sample_counts[keyword] = self.keyword_sample_counts.get(keyword, 0) + 1
                        
                        # 進捗レポート（100サンプルごと）
                        if self.target_samples and self.current_sample_count % 100 == 0:
                            progress = (self.current_sample_count / self.target_samples * 100) if self.target_samples else 0
                            logger.info(f"[PROGRESS] Collected {self.current_sample_count} samples (target: {self.target_samples}, progress: {progress:.1f}%)")
                            logger.info(f"[PROGRESS] Keyword '{keyword}': {self.keyword_sample_counts[keyword]} samples")
                        
                        # セッション更新
                        if self.power_recovery and session:
                            self.power_recovery.update_session(
                                session_id,
                                url=url,
                                samples_collected=len(samples)
                            )
                        
                        # ブラウザ状態更新
                        self.browser_status[browser_index]['samples_collected'] = len(samples)
                        self.browser_status[browser_index]['last_activity'] = datetime.now().isoformat()
                
                except Exception as e:
                    logger.error(f"[SCRAPE] Failed to scrape {url}: {e}")
                    continue
                
                # アクション間の待機
                await asyncio.sleep(random.uniform(self.delay_per_action, self.delay_per_action * 2))
        
        except Exception as e:
            logger.error(f"[SCRAPE] Failed to scrape keyword '{keyword}': {e}")
        
        # セッション状態を更新
        if self.power_recovery and session:
            self.power_recovery.update_session(
                session_id,
                samples_collected=len(samples),
                status="completed" if samples else "error"
            )
        
        logger.info(f"[SCRAPE] Collected {len(samples)} samples for keyword: {keyword}")
        return samples
    
    async def scrape_page_with_so8t_control(
        self,
        page: Page,
        url: str,
        keyword: str,
        browser_index: int,
        session_id: str
    ) -> Optional[Dict]:
        """
        ページをスクレイピング（SO8T統制版）
        
        Args:
            page: Playwrightページ
            url: URL
            keyword: キーワード
            browser_index: ブラウザインデックス
            session_id: セッションID
            
        Returns:
            sample: 収集したサンプル（Noneの場合は失敗）
        """
        try:
            # ページ遷移
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")
            
            # 人間らしいページ閲覧
            await self.human_like_page_view(page)
            
            # ページコンテンツ抽出
            content = await self.extract_page_content(page, url)
            
            if not content or not content.get('text'):
                return None
            
            # サンプル作成
            sample = {
                'text': content['text'],
                'url': url,
                'keyword': keyword,
                'title': content.get('title', ''),
                'language': content.get('language', 'unknown'),
                'category': content.get('category', 'general'),
                'domain': content.get('domain', 'unknown'),
                'scraped_at': datetime.now().isoformat(),
                'browser_index': browser_index,
                'session_id': session_id,
                'source': 'so8t_thinking_controlled_scraping'
            }
            
            # NSFW検知
            if self.nsfw_classifier:
                nsfw_label, nsfw_confidence = self.nsfw_classifier.predict(content['text'])
                sample['nsfw_label'] = nsfw_label
                sample['nsfw_confidence'] = float(nsfw_confidence)
                sample['nsfw_detection_purpose'] = 'safety_training'
            
            # 監査ログ記録
            if self.audit_logger:
                event = ScrapingEvent(
                    event_id=f"{session_id}_url_{int(time.time())}",
                    session_id=session_id,
                    timestamp=datetime.now().isoformat(),
                    event_type="data_collected",
                    url=url,
                    keyword=keyword,
                    data_hash=hashlib.sha256(content['text'].encode()).hexdigest()[:16]
                )
                self.audit_logger.log_scraping_event(event)
            
            return sample
            
        except Exception as e:
            logger.error(f"[SCRAPE] Failed to scrape page {url}: {e}")
            return None
    
    async def extract_search_results(self, page: Page, keyword: str) -> List[Dict]:
        """
        検索結果からURLを抽出
        
        Args:
            page: Playwrightページ
            keyword: キーワード
            
        Returns:
            results: URL情報のリスト
        """
        results = []
        
        try:
            # 検索結果リンクを抽出
            result_selectors = [
                'a[href*="/url?q="]',  # Google
                'a[href*="bing.com/search"]',  # Bing
                'a.result__a',  # DuckDuckGo
                'h3 a',  # 一般的な検索結果
            ]
            
            for selector in result_selectors:
                links = await page.query_selector_all(selector)
                if links:
                    for link in links[:10]:  # 最大10件
                        try:
                            href = await link.get_attribute('href')
                            if href:
                                # GoogleのURLパラメータから実際のURLを抽出
                                if '/url?q=' in href:
                                    from urllib.parse import urlparse, parse_qs
                                    parsed = urlparse(href)
                                    qs = parse_qs(parsed.query)
                                    if 'q' in qs:
                                        href = qs['q'][0]
                                
                                title = await link.inner_text()
                                results.append({
                                    'url': href,
                                    'title': title[:200] if title else '',
                                    'keyword': keyword
                                })
                        except Exception:
                            continue
                    
                    if results:
                        break
            
        except Exception as e:
            logger.error(f"[EXTRACT] Failed to extract search results: {e}")
        
        return results
    
    async def extract_page_content(self, page: Page, url: str) -> Dict:
        """
        ページコンテンツを抽出（親クラスのメソッドを使用）
        
        Args:
            page: Playwrightページ
            url: URL
            
        Returns:
            content: コンテンツ辞書
        """
        # 親クラスのメソッドを使用
        return await super().extract_page_content(page, url, None, None, [])
    
    def __del__(self):
        """デストラクタ"""
        if self.power_recovery:
            try:
                self.power_recovery.stop_auto_checkpoint()
                self.power_recovery.emergency_save()
            except Exception:
                pass


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T Thinking Controlled Web Scraping")
    parser.add_argument('--output', type=Path, default=Path('D:/webdataset/processed'), help='Output directory')
    parser.add_argument('--num-browsers', type=int, default=10, help='Number of parallel browsers')
    parser.add_argument('--use-cursor-browser', action='store_true', default=True, help='Use Cursor browser')
    parser.add_argument('--remote-debugging-port', type=int, default=9222, help='Remote debugging port')
    parser.add_argument('--delay', type=float, default=1.5, help='Delay between actions (seconds)')
    parser.add_argument('--timeout', type=int, default=30000, help='Page load timeout (milliseconds)')
    parser.add_argument('--max-pages-per-keyword', type=int, default=5, help='Maximum pages per keyword')
    parser.add_argument('--max-memory-gb', type=float, default=8.0, help='Maximum memory usage (GB)')
    parser.add_argument('--max-cpu-percent', type=float, default=80.0, help='Maximum CPU usage (%)')
    parser.add_argument('--so8t-model-path', type=str, help='SO8T model path')
    parser.add_argument('--checkpoint-dir', type=Path, help='Checkpoint directory')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon mode (background)')
    parser.add_argument('--target-samples', type=int, default=None, help='Target number of samples to collect')
    parser.add_argument('--min-samples-per-keyword', type=int, default=10, help='Minimum samples per keyword')
    parser.add_argument('--max-samples-per-keyword', type=int, default=100, help='Maximum samples per keyword')
    
    args = parser.parse_args()
    
    # スクレイパー初期化
    scraper = SO8TThinkingControlledScraper(
        output_dir=args.output,
        num_browsers=args.num_browsers,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_action=args.delay,
        timeout=args.timeout,
        max_pages_per_keyword=args.max_pages_per_keyword,
        max_memory_gb=args.max_memory_gb,
        max_cpu_percent=args.max_cpu_percent,
        use_so8t_control=True,
        so8t_model_path=args.so8t_model_path,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        target_samples=args.target_samples,
        min_samples_per_keyword=args.min_samples_per_keyword,
        max_samples_per_keyword=args.max_samples_per_keyword
    )
    
    # シグナルハンドラー設定
    def signal_handler(signum, frame):
        logger.warning(f"[SIGNAL] Received signal {signum}, shutting down gracefully...")
        if scraper.power_recovery:
            scraper.power_recovery.emergency_save()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)
    
    try:
        # 並列スクレイピング実行
        await scraper.run_parallel_scraping()
        
        # サンプル保存
        if scraper.all_samples:
            scraper.save_samples(scraper.all_samples)
        
        # NSFWサンプル保存
        if scraper.nsfw_samples:
            scraper.save_nsfw_samples(scraper.nsfw_samples)
        
        logger.info("[OK] Scraping completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("[INTERRUPT] Scraping interrupted by user")
        if scraper.power_recovery:
            scraper.power_recovery.emergency_save()
    except Exception as e:
        logger.error(f"[ERROR] Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        if scraper.power_recovery:
            scraper.power_recovery.emergency_save()
        raise


if __name__ == "__main__":
    asyncio.run(main())

