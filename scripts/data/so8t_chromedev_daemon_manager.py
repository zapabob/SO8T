#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
SO8T統制ChromeDev並列ブラウザCUDA分散処理統合マネージャー

Chrome DevTools起動、ブラウザ起動、タブ処理、CUDA分散処理を統合管理します。

Usage:
    from scripts.data.so8t_chromedev_daemon_manager import SO8TChromeDevDaemonManager
    
    manager = SO8TChromeDevDaemonManager(
        num_browsers=10,
        num_tabs=10,
        base_port=9222
    )
    await manager.start_all()
    results = await manager.scrape_with_so8t_control(urls)
"""

import sys
import json
import logging
import asyncio
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/so8t_chromedev_daemon_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 統合サイトリスト（環境変数から読み込み）
try:
    from scripts.utils.env_loader import get_comprehensive_site_lists
    COMPREHENSIVE_SITE_LISTS = get_comprehensive_site_lists()
except ImportError:
    logger.warning("[ENV] Failed to import env_loader, using default site lists")
    # フォールバック: デフォルト値（後方互換性のため）
    COMPREHENSIVE_SITE_LISTS = {
        'encyclopedia': {
            'wikipedia_ja': [
                "https://ja.wikipedia.org/wiki/メインページ",
                "https://ja.wikipedia.org/wiki/Category:コンピュータ",
                "https://ja.wikipedia.org/wiki/Category:プログラミング言語",
                "https://ja.wikipedia.org/wiki/Category:ソフトウェア",
                "https://ja.wikipedia.org/wiki/Category:軍事",
                "https://ja.wikipedia.org/wiki/Category:航空宇宙",
                "https://ja.wikipedia.org/wiki/Category:インフラ",
                "https://ja.wikipedia.org/wiki/Category:日本企業",
                "https://ja.wikipedia.org/wiki/防衛省",
                "https://ja.wikipedia.org/wiki/航空宇宙",
                "https://ja.wikipedia.org/wiki/医療",
            ],
            'wikipedia_en': [
                "https://en.wikipedia.org/wiki/Main_Page",
                "https://en.wikipedia.org/wiki/Category:Computer_science",
                "https://en.wikipedia.org/wiki/Category:Programming_languages",
                "https://en.wikipedia.org/wiki/Category:Software",
                "https://en.wikipedia.org/wiki/Category:Military",
                "https://en.wikipedia.org/wiki/Category:Aerospace",
                "https://en.wikipedia.org/wiki/Category:Infrastructure",
                "https://en.wikipedia.org/wiki/Defense",
                "https://en.wikipedia.org/wiki/Aerospace",
                "https://en.wikipedia.org/wiki/Medicine",
            ],
            'kotobank': [
                "https://kotobank.jp/",
                "https://kotobank.jp/word/プログラミング",
                "https://kotobank.jp/word/コンピュータ",
                "https://kotobank.jp/word/ソフトウェア",
                "https://kotobank.jp/word/軍事",
                "https://kotobank.jp/word/航空宇宙",
                "https://kotobank.jp/word/インフラ",
            ],
            'britannica': [
                "https://www.britannica.com/",
                "https://www.britannica.com/technology/computer",
                "https://www.britannica.com/technology/software",
                "https://www.britannica.com/technology/programming-language",
                "https://www.britannica.com/topic/military",
                "https://www.britannica.com/topic/aerospace-industry",
                "https://www.britannica.com/topic/infrastructure",
            ]
        },
        'coding': {
            'github': [
                "https://github.com/trending",
                "https://github.com/trending/python",
                "https://github.com/trending/rust",
                "https://github.com/trending/typescript",
                "https://github.com/trending/java",
                "https://github.com/trending/cpp",
                "https://github.com/trending/swift",
                "https://github.com/trending/kotlin",
                "https://github.com/trending/csharp",
                "https://github.com/trending/php",
                "https://github.com/explore",
            ],
            'stack_overflow': [
                "https://stackoverflow.com/questions/tagged/python",
                "https://stackoverflow.com/questions/tagged/rust",
                "https://stackoverflow.com/questions/tagged/typescript",
                "https://stackoverflow.com/questions/tagged/javascript",
                "https://stackoverflow.com/questions/tagged/java",
                "https://stackoverflow.com/questions/tagged/c%2b%2b",
                "https://stackoverflow.com/questions/tagged/c",
                "https://stackoverflow.com/questions/tagged/swift",
                "https://stackoverflow.com/questions/tagged/kotlin",
                "https://stackoverflow.com/questions/tagged/c%23",
                "https://stackoverflow.com/questions/tagged/unity3d",
                "https://stackoverflow.com/questions/tagged/php",
            ],
            'documentation': [
                "https://pytorch.org/",
                "https://pytorch.org/docs/stable/index.html",
                "https://pytorch.org/tutorials/",
                "https://www.tensorflow.org/",
                "https://www.tensorflow.org/api_docs",
                "https://www.tensorflow.org/tutorials",
                "https://docs.python.org/",
                "https://developer.mozilla.org/",
                "https://react.dev/",
                "https://vuejs.org/",
                "https://angular.io/",
                "https://docs.microsoft.com/en-us/dotnet/",
                "https://docs.microsoft.com/en-us/cpp/",
                "https://developer.apple.com/swift/",
                "https://kotlinlang.org/docs/home.html",
            ],
            'learning_sites': [
                "https://www.freecodecamp.org/",
                "https://www.codecademy.com/",
                "https://leetcode.com/",
                "https://www.codewars.com/",
            ],
            'tech_blogs': [
                "https://techcrunch.com/",
                "https://www.infoq.com/",
                "https://www.oreilly.com/",
            ],
            'reddit': [
                "https://www.reddit.com/r/programming/",
                "https://www.reddit.com/r/Python/",
                "https://www.reddit.com/r/rust/",
                "https://www.reddit.com/r/typescript/",
                "https://www.reddit.com/r/java/",
                "https://www.reddit.com/r/cpp/",
                "https://www.reddit.com/r/swift/",
                "https://www.reddit.com/r/Kotlin/",
                "https://www.reddit.com/r/csharp/",
                "https://www.reddit.com/r/Unity3D/",
                "https://www.reddit.com/r/PHP/",
            ],
            'hacker_news': [
                "https://news.ycombinator.com/",
            ],
            'engineer_sites': [
                "https://qiita.com/",
                "https://zenn.dev/",
                "https://dev.to/",
                "https://medium.com/tag/programming",
            ]
        },
        'nsfw_detection': {
            'fanza': [
                "https://www.fanza.co.jp/",
                "https://www.dmm.co.jp/",
                "https://www.dmm.co.jp/digital/videoa/",
                "https://www.dmm.co.jp/digital/videoc/",
                "https://www.dmm.co.jp/rental/",
                "https://www.dmm.co.jp/rental/videoa/",
            ],
            'fc2': [
                "https://live.fc2.com/",
                "https://live.fc2.com/category/",
                "https://live.fc2.com/ranking/",
            ],
            'missav': [
                "https://missav.ai/",
                "https://missav.ai/genre/",
                "https://missav.ai/ranking/",
            ],
            'adult_sites': [
                "https://www.pornhub.com/",
                "https://www.pornhub.com/video",
                "https://www.xvideos.com/",
                "https://www.xvideos.com/video",
                "https://www.xhamster.com/",
                "https://www.xhamster.com/video",
            ]
        },
        'government': {
            'japan': [
                "https://www.mod.go.jp/",
                "https://www.jaxa.jp/",
                "https://www.mhlw.go.jp/",
            ],
            'us': [
                "https://www.defense.gov/",
                "https://www.nasa.gov/",
                "https://www.fda.gov/",
            ]
        },
        'tech_blogs': {
            'qiita': [
                "https://qiita.com/",
            ],
            'zenn': [
                "https://zenn.dev/",
            ],
            'note': [
                "https://note.com/tech",
            ]
        }
    }


class SO8TChromeDevDaemonManager:
    """SO8T統制ChromeDev並列ブラウザCUDA分散処理統合マネージャー"""
    
    def __init__(
        self,
        num_browsers: int = 10,
        num_tabs: int = 10,
        base_port: int = 9222,
        config_path: Optional[Path] = None,
        use_daemon_mode: Optional[bool] = None
    ):
        """
        初期化
        
        Args:
            num_browsers: ブラウザ数
            num_tabs: タブ数（各ブラウザ）
            base_port: ベースリモートデバッグポート
            config_path: 設定ファイルパス
            use_daemon_mode: デーモンモードを使用するか（Noneの場合は設定ファイルから読み込み、デフォルト: False）
        """
        self.num_browsers = num_browsers
        self.num_tabs = num_tabs
        self.base_port = base_port
        
        # 設定ファイルを読み込み
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "so8t_chromedev_daemon_config.yaml"
        
        self.config = self._load_config(config_path)
        
        # デーモンモード設定（設定ファイルから読み込み、引数で上書き可能）
        if use_daemon_mode is None:
            self.use_daemon_mode = self.config.get('browser', {}).get('use_daemon_mode', False)
        else:
            self.use_daemon_mode = use_daemon_mode
        
        # コンポーネント
        self.chrome_devtools_launcher = None
        self.browser_manager = None
        self.playwright = None  # ブラウザ直接実行モード用
        self.direct_browsers: Dict[int, Any] = {}  # ブラウザ直接実行モード用
        self.direct_browser_contexts: Dict[int, Any] = {}  # ブラウザ直接実行モード用
        self.tab_processors: Dict[int, Any] = {}
        self.so8t_scrapers: Dict[int, Any] = {}
        self.cuda_processor = None
        
        # 協調機能
        self.browser_coordinator = None
        self.keyword_coordinator = None
        self.use_coordination = self.config.get('coordination', {}).get('enabled', False)
        
        # ジャンル分類
        self.data_labeler = None
        self.so8t_classifier = None
        
        # 状態管理
        self.status: Dict[str, Any] = {
            'initialized': False,
            'chrome_devtools_started': False,
            'browsers_started': False,
            'tabs_initialized': False,
            'cuda_available': False,
            'started_at': None
        }
        
        logger.info("="*80)
        logger.info("SO8T ChromeDev Daemon Manager Initialized")
        logger.info("="*80)
        logger.info(f"Number of browsers: {self.num_browsers}")
        logger.info(f"Number of tabs per browser: {self.num_tabs}")
        logger.info(f"Base port: {self.base_port}")
        logger.info(f"Use daemon mode: {self.use_daemon_mode}")
        logger.info(f"Browser mode: {'Daemon' if self.use_daemon_mode else 'Direct Browser'}")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        設定ファイルを読み込み
        
        Args:
            config_path: 設定ファイルパス
        
        Returns:
            config: 設定辞書
        """
        default_config = {
            'chrome_devtools': {
                'enabled': True,
                'transport': 'stdio',
                'command': 'npx',
                'args': ['-y', '@modelcontextprotocol/server-chrome-devtools'],
                'timeout': 30000,
                'max_instances': 10
            },
            'browsers': {
                'use_daemon_mode': False,  # デーモンモードを使用するか（falseの場合はブラウザ直接実行）
                'headless': False,  # ブラウザの表示モード（falseの場合は表示）
                'use_cursor_browser': True,
                'auto_restart': True,
                'restart_delay': 60.0,
                'max_memory_gb': 8.0,
                'max_cpu_percent': 80.0
            },
            'cuda': {
                'enabled': True,
                'device_id': 0,
                'batch_size': 32,
                'max_memory_fraction': 0.8,
                'num_workers': 4
            },
            'so8t': {
                'enabled': True,
                'model_path': None
            }
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    # デフォルト設定とマージ
                    for key, value in loaded_config.items():
                        if key in default_config and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                    logger.info(f"[CONFIG] Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"[CONFIG] Failed to load config: {e}, using defaults")
        else:
            logger.info("[CONFIG] Config file not found, using defaults")
        
        return default_config
    
    async def initialize(self) -> bool:
        """
        すべてのコンポーネントを初期化
        
        Returns:
            success: 成功フラグ
        """
        try:
            logger.info("[INIT] Initializing all components...")
            
            # 1. Chrome DevTools起動
            if self.config['chrome_devtools'].get('enabled', True):
                await self._initialize_chrome_devtools()
            
            # 2. ブラウザマネージャー初期化
            await self._initialize_browser_manager()
            
            # 3. CUDA分散処理初期化
            if self.config['cuda'].get('enabled', True):
                await self._initialize_cuda_processor()
            
            # 4. 協調機能初期化
            if self.use_coordination:
                await self._initialize_coordination()
            
            # 5. ジャンル分類初期化
            await self._initialize_genre_classification()
            
            self.status['initialized'] = True
            logger.info("[OK] All components initialized")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _initialize_chrome_devtools(self):
        """Chrome DevToolsを初期化"""
        try:
            from scripts.utils.chrome_devtools_launcher import ChromeDevToolsLauncher
            
            chrome_config = self.config['chrome_devtools']
            self.chrome_devtools_launcher = ChromeDevToolsLauncher(
                transport=chrome_config.get('transport', 'stdio'),
                command=chrome_config.get('command', 'npx'),
                args=chrome_config.get('args', ['-y', '@modelcontextprotocol/server-chrome-devtools']),
                timeout=chrome_config.get('timeout', 30000),
                max_instances=chrome_config.get('max_instances', 10)
            )
            
            # すべてのインスタンスを起動
            success = await self.chrome_devtools_launcher.start_all_instances()
            self.status['chrome_devtools_started'] = success
            
            logger.info(f"[CHROMEDEV] Chrome DevTools initialized: {success}")
            
        except Exception as e:
            logger.error(f"[CHROMEDEV] Failed to initialize: {e}")
            self.status['chrome_devtools_started'] = False
    
    async def _initialize_browser_manager(self):
        """ブラウザマネージャーを初期化"""
        try:
            browser_config = self.config['browsers']
            
            if self.use_daemon_mode:
                # デーモンモード: DaemonBrowserManagerを使用
                from scripts.data.daemon_browser_manager import DaemonBrowserManager
                
                self.browser_manager = DaemonBrowserManager(
                    num_browsers=self.num_browsers,
                    base_port=self.base_port,
                    headless=browser_config.get('headless', False),
                    use_cursor_browser=browser_config.get('use_cursor_browser', True),
                    auto_restart=browser_config.get('auto_restart', True),
                    restart_delay=browser_config.get('restart_delay', 60.0),
                    max_memory_gb=browser_config.get('max_memory_gb', 8.0),
                    max_cpu_percent=browser_config.get('max_cpu_percent', 80.0)
                )
                
                logger.info("[BROWSER] Daemon browser manager initialized")
            else:
                # ブラウザ直接実行モード: Playwrightで直接ブラウザを起動
                from playwright.async_api import async_playwright, Browser, BrowserContext
                
                self.playwright = await async_playwright().start()
                self.direct_browsers: Dict[int, Browser] = {}
                self.direct_browser_contexts: Dict[int, BrowserContext] = {}
                
                browser_config = self.config['browsers']
                headless = browser_config.get('headless', False)
                use_cursor_browser = browser_config.get('use_cursor_browser', True)
                
                # ブラウザを直接起動
                for browser_index in range(self.num_browsers):
                    port = self.base_port + browser_index
                    
                    if use_cursor_browser:
                        # Cursorブラウザに接続を試みる
                        try:
                            cdp_endpoint = f"http://127.0.0.1:{port}"
                            browser = await self.playwright.chromium.connect_over_cdp(cdp_endpoint)
                            logger.info(f"[BROWSER {browser_index}] Connected to Cursor browser on port {port}")
                        except Exception as e:
                            logger.warning(f"[BROWSER {browser_index}] Failed to connect to Cursor browser: {e}, launching new browser")
                            browser = await self.playwright.chromium.launch(
                                headless=headless,
                                args=[
                                    f"--remote-debugging-port={port}",
                                    "--disable-blink-features=AutomationControlled",
                                    "--disable-dev-shm-usage"
                                ]
                            )
                    else:
                        # 通常のブラウザを起動
                        browser = await self.playwright.chromium.launch(
                            headless=headless,
                            args=[
                                f"--remote-debugging-port={port}",
                                "--disable-blink-features=AutomationControlled",
                                "--disable-dev-shm-usage"
                            ]
                        )
                    
                    self.direct_browsers[browser_index] = browser
                    
                    # ブラウザコンテキストを作成
                    context = await browser.new_context(
                        viewport={'width': 1920, 'height': 1080},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    )
                    self.direct_browser_contexts[browser_index] = context
                    
                    logger.info(f"[BROWSER {browser_index}] Direct browser launched on port {port} (headless={headless})")
                
                logger.info(f"[BROWSER] Direct browser mode initialized: {len(self.direct_browsers)} browsers")
            
        except Exception as e:
            logger.error(f"[BROWSER] Failed to initialize: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def _initialize_cuda_processor(self):
        """CUDA分散処理を初期化"""
        try:
            from scripts.utils.cuda_distributed_processor import CUDADistributedProcessor
            
            cuda_config = self.config['cuda']
            self.cuda_processor = CUDADistributedProcessor(
                device_id=cuda_config.get('device_id', 0),
                batch_size=cuda_config.get('batch_size', 32),
                max_memory_fraction=cuda_config.get('max_memory_fraction', 0.8),
                num_workers=cuda_config.get('num_workers', 4)
            )
            
            # SO8Tモデルをロード
            if self.config['so8t'].get('enabled', True):
                model_path = self.config['so8t'].get('model_path')
                success = self.cuda_processor.load_so8t_model(model_path)
                if success:
                    logger.info("[CUDA] SO8T model loaded for CUDA inference")
                else:
                    logger.warning("[CUDA] Failed to load SO8T model")
            
            # GPUメモリ情報を表示
            memory_info = self.cuda_processor.get_gpu_memory_info()
            if memory_info.get('available'):
                logger.info(f"[CUDA] GPU: {memory_info.get('device_name')}")
                logger.info(f"[CUDA] Total memory: {memory_info.get('memory_total_gb'):.2f} GB")
                logger.info(f"[CUDA] Free memory: {memory_info.get('memory_free_gb'):.2f} GB")
            
            self.status['cuda_available'] = memory_info.get('available', False)
            
        except Exception as e:
            logger.error(f"[CUDA] Failed to initialize: {e}")
            self.cuda_processor = None
            self.status['cuda_available'] = False
    
    async def _initialize_coordination(self):
        """協調機能を初期化"""
        try:
            from scripts.utils.keyword_coordinator import KeywordCoordinator
            from scripts.data.browser_coordinator import BrowserCoordinator
            
            coordination_config = self.config.get('coordination', {})
            mcp_config = self.config.get('mcp_server', {})
            
            # キーワードコーディネーターを初期化
            keyword_queue_file = coordination_config.get('keyword_queue_file', 'D:/webdataset/checkpoints/keyword_queue.json')
            self.keyword_coordinator = KeywordCoordinator(keyword_queue_file=keyword_queue_file)
            
            # ブラウザコーディネーターを初期化（各ブラウザごとに）
            # メインマネージャーでは1つのコーディネーターを共有
            if mcp_config.get('enabled', True):
                self.browser_coordinator = BrowserCoordinator(
                    browser_id=0,  # メインマネージャーは0として扱う
                    keyword_coordinator=self.keyword_coordinator,
                    mcp_config=mcp_config,
                    heartbeat_interval=coordination_config.get('heartbeat_interval', 30.0),
                    broadcast_channel=coordination_config.get('broadcast_channel', 'browser_coordination')
                )
                await self.browser_coordinator.start()
                logger.info("[COORDINATION] Browser coordinator initialized")
            else:
                logger.info("[COORDINATION] MCP coordination disabled, using local keyword coordinator only")
            
        except Exception as e:
            logger.warning(f"[COORDINATION] Failed to initialize coordination: {e}")
            self.browser_coordinator = None
            self.keyword_coordinator = None
    
    async def _initialize_genre_classification(self):
        """ジャンル分類を初期化（DataLabeler + SO8T）"""
        try:
            from scripts.pipelines.web_scraping_data_pipeline import DataLabeler
            
            # DataLabelerを初期化（高速なキーワードベース分類）
            self.data_labeler = DataLabeler()
            logger.info("[GENRE] DataLabeler initialized")
            
            # SO8T分類器はオプション（高精度が必要な場合のみ）
            so8t_config = self.config.get('so8t', {})
            if so8t_config.get('enabled', True) and so8t_config.get('use_for_classification', False):
                try:
                    from scripts.pipelines.web_scraping_data_pipeline import QuadrupleClassifier
                    model_path = so8t_config.get('model_path')
                    self.so8t_classifier = QuadrupleClassifier(so8t_model_path=model_path)
                    logger.info("[GENRE] SO8T classifier initialized")
                except Exception as e:
                    logger.warning(f"[GENRE] Failed to initialize SO8T classifier: {e}")
                    self.so8t_classifier = None
            else:
                logger.info("[GENRE] SO8T classifier disabled (using DataLabeler only)")
            
        except Exception as e:
            logger.warning(f"[GENRE] Failed to initialize genre classification: {e}")
            self.data_labeler = None
            self.so8t_classifier = None
    
    async def start_all(self) -> bool:
        """
        すべてのコンポーネントを起動
        
        Returns:
            success: 成功フラグ
        """
        try:
            logger.info("[START] Starting all components...")
            
            # 初期化
            if not self.status['initialized']:
                success = await self.initialize()
                if not success:
                    return False
            
            # ブラウザを起動
            if self.use_daemon_mode:
                # デーモンモード: DaemonBrowserManagerを使用
                success = await self.browser_manager.start_all_browsers()
                self.status['browsers_started'] = success
                
                if not success:
                    logger.error("[START] Failed to start browsers")
                    return False
            else:
                # ブラウザ直接実行モード: 既に初期化時に起動済み
                self.status['browsers_started'] = len(self.direct_browsers) > 0
                
                if not self.status['browsers_started']:
                    logger.error("[START] Failed to start direct browsers")
                    return False
            
            # 各ブラウザでタブを初期化
            await self._initialize_tabs()
            
            self.status['started_at'] = datetime.now().isoformat()
            logger.info("[OK] All components started")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to start: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _initialize_tabs(self):
        """各ブラウザでタブを初期化"""
        try:
            logger.info(f"[TABS] Initializing tabs for {self.num_browsers} browsers...")
            
            from scripts.data.parallel_tab_processor import ParallelTabProcessor
            from scripts.data.so8t_controlled_browser_scraper import SO8TControlledBrowserScraper
            from scripts.agents.scraping_reasoning_agent import ScrapingReasoningAgent
            
            # SO8T統制エージェントを初期化
            so8t_agent = None
            if self.config['so8t'].get('enabled', True):
                try:
                    so8t_agent = ScrapingReasoningAgent()
                    logger.info("[SO8T] ScrapingReasoningAgent initialized")
                except Exception as e:
                    logger.warning(f"[SO8T] Failed to initialize agent: {e}")
            
            for browser_index in range(self.num_browsers):
                # ブラウザコンテキストを取得
                if self.use_daemon_mode:
                    browser_context = self.browser_manager.get_browser_context(browser_index)
                else:
                    # ブラウザ直接実行モード: 直接ブラウザコンテキストを使用
                    browser_context = self.direct_browser_contexts.get(browser_index)
                
                if browser_context is None:
                    logger.warning(f"[BROWSER {browser_index}] Browser context not available")
                    continue
                
                # タブプロセッサーを初期化
                tab_processor = ParallelTabProcessor(
                    browser_context=browser_context,
                    num_tabs=self.num_tabs,
                    delay_per_action=1.5,
                    timeout=30000
                )
                await tab_processor.initialize_tabs()
                self.tab_processors[browser_index] = tab_processor
                
                # SO8T統制スクレイパーを初期化
                so8t_scraper = SO8TControlledBrowserScraper(
                    browser_context=browser_context,
                    agent=so8t_agent,
                    num_tabs=self.num_tabs,
                    delay_per_action=1.5,
                    timeout=30000
                )
                await so8t_scraper.initialize_tabs()
                self.so8t_scrapers[browser_index] = so8t_scraper
                
                logger.info(f"[BROWSER {browser_index}] Tabs initialized")
            
            self.status['tabs_initialized'] = True
            logger.info(f"[OK] Initialized tabs for {len(self.tab_processors)} browsers")
            
        except Exception as e:
            logger.error(f"[TABS] Failed to initialize tabs: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _generate_deep_research_urls(
        self, 
        keyword: str, 
        language: str = 'ja',
        category: Optional[str] = None
    ) -> List[str]:
        """
        DeepResearchでキーワードからURLを生成（既存サイトリスト統合対応）
        
        Args:
            keyword: キーワード
            language: 言語（'ja'または'en'）
            category: カテゴリ（'encyclopedia', 'coding', 'nsfw_detection', 'government', 'tech_blogs'、Noneの場合は全カテゴリ）
        
        Returns:
            urls: 生成されたURLのリスト
        """
        try:
            from urllib.parse import quote
            
            urls = []
            
            # 設定から既存サイトリスト統合設定を取得
            deep_research_config = self.config.get('deep_research', {})
            use_existing_site_lists = deep_research_config.get('use_existing_site_lists', True)
            site_list_categories = deep_research_config.get('site_list_categories', {})
            
            # 既存サイトリストからのURL生成（設定が有効な場合）
            if use_existing_site_lists:
                site_list_urls = []
                
                # カテゴリが指定されている場合はそのカテゴリのみ、指定されていない場合は全カテゴリ
                categories_to_process = [category] if category else list(COMPREHENSIVE_SITE_LISTS.keys())
                
                for cat in categories_to_process:
                    # カテゴリが有効かチェック
                    if cat not in COMPREHENSIVE_SITE_LISTS:
                        continue
                    
                    # 設定でカテゴリが無効化されている場合はスキップ
                    if cat in site_list_categories and not site_list_categories.get(cat, True):
                        continue
                    
                    # カテゴリ内のすべてのサブカテゴリからURLを取得
                    for subcat, subcat_urls in COMPREHENSIVE_SITE_LISTS[cat].items():
                        site_list_urls.extend(subcat_urls)
                
                if site_list_urls:
                    urls.extend(site_list_urls)
                    logger.info(f"[DEEP RESEARCH] Added {len(site_list_urls)} URLs from existing site lists (category: {category or 'all'})")
            
            # 検索エンジンURLを生成（Google、Bing、DuckDuckGo）
            keyword_encoded = quote(keyword)
            
            if language == 'ja':
                urls.extend([
                    f"https://www.google.com/search?hl=ja&q={keyword_encoded}",
                    f"https://www.bing.com/search?q={keyword_encoded}&setlang=ja",
                    f"https://html.duckduckgo.com/html/?q={keyword_encoded}"
                ])
            else:
                urls.extend([
                    f"https://www.google.com/search?q={keyword_encoded}",
                    f"https://www.bing.com/search?q={keyword_encoded}",
                    f"https://html.duckduckgo.com/html/?q={keyword_encoded}"
                ])
            
            # Wikipedia URL（言語別）
            if language == 'ja':
                wiki_url = f"https://ja.wikipedia.org/wiki/{keyword.replace(' ', '_')}"
            else:
                wiki_url = f"https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')}"
            urls.append(wiki_url)
            
            # 重複を除去
            urls = list(dict.fromkeys(urls))  # 順序を保持しながら重複を除去
            
            logger.info(f"[DEEP RESEARCH] Generated {len(urls)} URLs for keyword: {keyword} (category: {category or 'all'})")
            return urls
            
        except Exception as e:
            logger.error(f"[DEEP RESEARCH] Failed to generate URLs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def _classify_failure_reason(self, error: Exception, url: str) -> str:
        """
        スクレイピング失敗原因を分類
        
        Args:
            error: エラーオブジェクト
            url: 失敗したURL
        
        Returns:
            reason: 失敗原因（'timeout', 'access_denied', 'bot_detection', 'network_error', 'parse_error', 'unknown'）
        """
        error_str = str(error).lower()
        
        if 'timeout' in error_str or 'timed out' in error_str:
            return 'timeout'
        elif '403' in error_str or 'forbidden' in error_str or 'access denied' in error_str:
            return 'access_denied'
        elif 'captcha' in error_str or 'cloudflare' in error_str or 'bot' in error_str:
            return 'bot_detection'
        elif 'network' in error_str or 'connection' in error_str or 'dns' in error_str:
            return 'network_error'
        elif 'parse' in error_str or 'json' in error_str or 'html' in error_str:
            return 'parse_error'
        else:
            return 'unknown'
    
    async def _retry_with_exponential_backoff(
        self,
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        *args,
        **kwargs
    ):
        """
        指数バックオフでリトライ
        
        Args:
            func: 実行する関数（async）
            max_retries: 最大リトライ回数
            base_delay: ベース待機時間（秒）
            max_delay: 最大待機時間（秒）
            *args, **kwargs: 関数に渡す引数
        
        Returns:
            関数の戻り値
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"[RETRY] Attempt {attempt + 1}/{max_retries + 1} failed, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[RETRY] Max retries reached: {e}")
        
        raise last_exception
    
    async def scrape_with_so8t_control(
        self,
        urls: List[str],
        keywords: Optional[List[str]] = None,
        use_deep_research: bool = True
    ) -> List[Dict]:
        """
        SO8T統制でスクレイピング（協調動作対応、DeepResearch統合、失敗処理対応）
        
        Args:
            urls: スクレイピング対象URLのリスト
            keywords: キーワードのリスト（オプション、協調動作時は無視される）
            use_deep_research: DeepResearchを使用するか（Trueの場合はキーワードからURLを生成）
        
        Returns:
            results: スクレイピング結果のリスト
        """
        if not self.status['browsers_started']:
            logger.error("[SCRAPE] Browsers not started")
            return []
        
        if not self.status['tabs_initialized']:
            logger.error("[SCRAPE] Tabs not initialized")
            return []
        
        try:
            # 協調動作が有効な場合、キーワードキューからキーワードを取得
            if self.use_coordination and self.browser_coordinator:
                logger.info("[SCRAPE] Using coordination mode - fetching keywords from queue")
                # 各ブラウザごとにキーワードを取得
                browser_keywords = {}
                for browser_index in range(self.num_browsers):
                    keyword = await self.browser_coordinator.get_next_keyword_with_coordination()
                    if keyword:
                        browser_keywords[browser_index] = keyword
                        logger.info(f"[SCRAPE] Browser {browser_index} assigned keyword: {keyword}")
                
                # キーワードに基づいてURLを生成またはフィルタリング
                # ここでは既存のURLリストを使用し、キーワードを各ブラウザに割り当て
                keywords = [browser_keywords.get(i) for i in range(self.num_browsers)]
            elif self.use_coordination and self.keyword_coordinator:
                # ブラウザコーディネーターがない場合でもキーワードコーディネーターを使用
                logger.info("[SCRAPE] Using keyword coordinator only")
                keywords = []
                for browser_index in range(self.num_browsers):
                    keyword = self.keyword_coordinator.get_next_keyword(browser_index)
                    if keyword:
                        keywords.append(keyword)
                        logger.info(f"[SCRAPE] Browser {browser_index} assigned keyword: {keyword}")
                    else:
                        keywords.append(None)
            
            # DeepResearch統合: キーワードからURLを生成 + 既存サイトリストからのURL追加
            deep_research_config = self.config.get('deep_research', {})
            use_existing_site_lists = deep_research_config.get('use_existing_site_lists', True)
            
            if use_deep_research:
                deep_research_urls = []
                
                # キーワードからURLを生成
                if keywords:
                    logger.info("[DEEP RESEARCH] Generating URLs from keywords...")
                    for keyword in keywords:
                        if keyword:
                            keyword_urls = await self._generate_deep_research_urls(keyword, language='ja')
                            deep_research_urls.extend(keyword_urls)
                
                # 既存サイトリストからのURL追加（キーワードがない場合、または設定で有効な場合）
                if use_existing_site_lists:
                    logger.info("[DEEP RESEARCH] Adding URLs from existing site lists...")
                    site_list_categories = deep_research_config.get('site_list_categories', {})
                    
                    # 有効なカテゴリからURLを取得
                    for category, enabled in site_list_categories.items():
                        if enabled and category in COMPREHENSIVE_SITE_LISTS:
                            category_urls = []
                            for subcat, subcat_urls in COMPREHENSIVE_SITE_LISTS[category].items():
                                category_urls.extend(subcat_urls)
                            
                            if category_urls:
                                deep_research_urls.extend(category_urls)
                                logger.info(f"[DEEP RESEARCH] Added {len(category_urls)} URLs from category: {category}")
                
                # 重複を除去
                deep_research_urls = list(dict.fromkeys(deep_research_urls))
                
                if deep_research_urls:
                    urls.extend(deep_research_urls)
                    logger.info(f"[DEEP RESEARCH] Added {len(deep_research_urls)} URLs from DeepResearch (keywords + site lists)")
            
            logger.info(f"[SCRAPE] Starting SO8T-controlled scraping for {len(urls)} URLs...")
            
            # URLをブラウザ数で分割
            urls_per_browser = len(urls) // self.num_browsers
            url_chunks = [urls[i:i + urls_per_browser] for i in range(0, len(urls), urls_per_browser)]
            
            if keywords:
                # キーワードをブラウザごとに割り当て
                keyword_chunks = []
                for browser_index in range(self.num_browsers):
                    browser_keyword = keywords[browser_index] if browser_index < len(keywords) else None
                    # 各URLチャンクに同じキーワードを割り当て
                    keyword_chunks.append([browser_keyword] * len(url_chunks[browser_index]) if browser_keyword else [None] * len(url_chunks[browser_index]))
            else:
                keyword_chunks = [None] * len(url_chunks)
            
            # 余りのURLを最初のブラウザに追加
            if len(url_chunks) > self.num_browsers:
                url_chunks[self.num_browsers - 1].extend(url_chunks[self.num_browsers:])
                keyword_chunks[self.num_browsers - 1].extend(keyword_chunks[self.num_browsers:])
                url_chunks = url_chunks[:self.num_browsers]
                keyword_chunks = keyword_chunks[:self.num_browsers]
            
            # タブ数よりURLチャンクが少ない場合は空リストで埋める
            while len(url_chunks) < self.num_browsers:
                url_chunks.append([])
                keyword_chunks.append([])
            
            logger.info(f"[SCRAPE] Processing {len(urls)} URLs across {self.num_browsers} browsers")
            logger.info(f"[SCRAPE] URLs per browser: {[len(chunk) for chunk in url_chunks]}")
            
            # キーワード処理開始をマーク（進捗追跡）
            if self.use_coordination and self.keyword_coordinator and keywords:
                for browser_index, browser_keyword in enumerate(keywords):
                    if browser_keyword:
                        self.keyword_coordinator.mark_keyword_processing(browser_keyword, browser_index)
                        logger.info(f"[PROGRESS] Marked keyword '{browser_keyword}' as processing (browser {browser_index})")
            
            # すべてのブラウザを並列処理
            tasks = []
            browser_keyword_map = {}  # ブラウザインデックス -> キーワードのマッピング
            for browser_index, (url_chunk, keyword_chunk) in enumerate(zip(url_chunks, keyword_chunks)):
                if url_chunk and browser_index in self.so8t_scrapers:
                    scraper = self.so8t_scrapers[browser_index]
                    browser_keyword = keywords[browser_index] if browser_index < len(keywords) else None
                    if browser_keyword:
                        browser_keyword_map[browser_index] = browser_keyword
                    task = scraper.scrape_with_so8t_control(url_chunk, keyword_chunk)
                    tasks.append((browser_index, task))
            
            # 並列実行
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # 結果を統合（失敗処理対応）
            all_results = []
            browser_results_map = {}  # ブラウザインデックス -> 結果数のマッピング
            failed_urls = []  # 失敗URLの記録
            
            for idx, (browser_index, _) in enumerate(tasks):
                results = task_results[idx]
                if isinstance(results, Exception):
                    # 失敗原因を分類
                    failure_reason = await self._classify_failure_reason(results, url_chunks[browser_index][0] if url_chunks[browser_index] else 'unknown')
                    
                    # 失敗URLを記録
                    for url in url_chunks[browser_index]:
                        failed_urls.append({
                            'url': url,
                            'browser_id': browser_index,
                            'reason': failure_reason,
                            'error': str(results),
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    logger.error(f"[SCRAPE] Browser {browser_index} scraping exception: {results} (reason: {failure_reason})")
                    browser_results_map[browser_index] = {
                        'samples': 0,
                        'urls_processed': 0,
                        'urls_failed': len(url_chunks[browser_index]),
                        'failure_reason': failure_reason
                    }
                elif isinstance(results, list):
                    all_results.extend(results)
                    failed_count = len(url_chunks[browser_index]) - len(results)
                    
                    # 失敗したURLを記録
                    if failed_count > 0:
                        processed_urls = {r.get('url') for r in results if r.get('url')}
                        for url in url_chunks[browser_index]:
                            if url not in processed_urls:
                                failed_urls.append({
                                    'url': url,
                                    'browser_id': browser_index,
                                    'reason': 'empty_result',
                                    'error': 'No content extracted',
                                    'timestamp': datetime.now().isoformat()
                                })
                    
                    browser_results_map[browser_index] = {
                        'samples': len(results),
                        'urls_processed': len(results),
                        'urls_failed': failed_count
                    }
            
            # 失敗URLを記録（設定が有効な場合）
            failure_config = self.config.get('failure_handling', {})
            if failure_config.get('enabled', True) and failure_config.get('record_failures', True) and failed_urls:
                failure_log_file = Path("D:/webdataset/checkpoints/scraping_failures.json")
                failure_log_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 既存の失敗ログを読み込み
                existing_failures = []
                if failure_log_file.exists():
                    try:
                        with open(failure_log_file, 'r', encoding='utf-8') as f:
                            existing_failures = json.load(f)
                    except Exception:
                        existing_failures = []
                
                # 新しい失敗を追加
                existing_failures.extend(failed_urls)
                
                # 保存
                with open(failure_log_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_failures, f, ensure_ascii=False, indent=2)
                
                logger.info(f"[FAILURE] Recorded {len(failed_urls)} failed URLs to {failure_log_file}")
                
                # 失敗統計レポートを生成（設定が有効な場合）
                if failure_config.get('generate_reports', True):
                    await self._generate_failure_report(failed_urls, browser_results_map)
            
            # 進捗情報を更新
            if self.use_coordination and self.keyword_coordinator:
                import time
                for browser_index, browser_keyword in browser_keyword_map.items():
                    if browser_keyword and browser_index in browser_results_map:
                        results_info = browser_results_map[browser_index]
                        # 進捗情報を更新
                        self.keyword_coordinator.update_keyword_progress(
                            keyword=browser_keyword,
                            browser_id=browser_index,
                            samples_collected=results_info['samples'],
                            urls_processed=results_info['urls_processed'],
                            urls_failed=results_info['urls_failed'],
                            processing_time=None  # 個別URLの処理時間は後で追加可能
                        )
                        logger.info(f"[PROGRESS] Updated progress for keyword '{browser_keyword}': samples={results_info['samples']}, urls={results_info['urls_processed']}, failed={results_info['urls_failed']}")
            
            # ジャンル分類を実行（DataLabeler + SO8T）
            if self.data_labeler:
                logger.info("[GENRE] Classifying scraped data...")
                classified_results = []
                for result in all_results:
                    # DataLabelerによる分類（高速）
                    classified_result = self.data_labeler.label_sample(result)
                    
                    # SO8T分類器による分類（オプション、高精度）
                    if self.so8t_classifier:
                        try:
                            classified_result = self.so8t_classifier.classify_quadruple(classified_result)
                        except Exception as e:
                            logger.warning(f"[GENRE] SO8T classification failed: {e}")
                    
                    classified_results.append(classified_result)
                all_results = classified_results
                logger.info(f"[GENRE] Classified {len(all_results)} samples")
            
            # CUDAでデータ処理（オプション）
            if self.cuda_processor and self.status['cuda_available']:
                logger.info("[CUDA] Processing scraped data with CUDA...")
                processed_results = await self.cuda_processor.process_data_cuda(
                    all_results,
                    processing_type="text"
                )
                all_results = processed_results
            
            # 協調動作: キーワード完了をマーク
            if self.use_coordination and keywords:
                for browser_index, browser_keyword in enumerate(keywords):
                    if browser_keyword and self.keyword_coordinator:
                        # キーワードを完了にマーク
                        self.keyword_coordinator.mark_keyword_completed(browser_keyword, browser_index)
                        # ブラウザコーディネーターに完了を通知
                        if self.browser_coordinator:
                            await self.browser_coordinator.broadcast_keyword_completion(browser_keyword, success=True)
                        logger.info(f"[COORDINATION] Marked keyword '{browser_keyword}' as completed (browser {browser_index})")
            
            logger.info(f"[OK] Scraped {len(all_results)} samples with SO8T control")
            logger.info(f"[FAILURE] Failed URLs: {sum(browser_results_map.get(i, {}).get('urls_failed', 0) for i in range(self.num_browsers))}")
            return all_results
            
        except Exception as e:
            logger.error(f"[SCRAPE] Scraping failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 失敗を記録
            failure_reason = await self._classify_failure_reason(e, 'unknown')
            logger.error(f"[FAILURE] Overall scraping failure reason: {failure_reason}")
            
            return []
    
    async def _generate_failure_report(self, failed_urls: List[Dict], browser_results_map: Dict[int, Dict]):
        """
        失敗統計レポートを生成
        
        Args:
            failed_urls: 失敗URLのリスト
            browser_results_map: ブラウザ別の結果マップ
        """
        try:
            # 失敗原因別の統計
            failure_stats = {}
            for failed_url in failed_urls:
                reason = failed_url.get('reason', 'unknown')
                failure_stats[reason] = failure_stats.get(reason, 0) + 1
            
            # ブラウザ別の失敗統計
            browser_failure_stats = {}
            for browser_index, results_info in browser_results_map.items():
                browser_failure_stats[browser_index] = {
                    'failed': results_info.get('urls_failed', 0),
                    'processed': results_info.get('urls_processed', 0),
                    'failure_reason': results_info.get('failure_reason', 'unknown')
                }
            
            # レポート作成
            total_processed = sum(r.get('urls_processed', 0) for r in browser_results_map.values())
            total_failed = len(failed_urls)
            total_attempted = total_processed + total_failed
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_failed': total_failed,
                'total_processed': total_processed,
                'total_attempted': total_attempted,
                'failure_rate': total_failed / total_attempted if total_attempted > 0 else 0.0,
                'failure_by_reason': failure_stats,
                'failure_by_browser': browser_failure_stats
            }
            
            # レポートを保存
            report_file = Path("D:/webdataset/checkpoints/failure_report.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[FAILURE REPORT] Generated failure report: {report_file}")
            logger.info(f"[FAILURE REPORT] Total failed: {total_failed}, Total processed: {total_processed}, Failure rate: {report['failure_rate']:.2%}")
            logger.info(f"[FAILURE REPORT] Failure by reason: {failure_stats}")
            
        except Exception as e:
            logger.error(f"[FAILURE REPORT] Failed to generate report: {e}")
    
    async def stop_all(self) -> bool:
        """
        すべてのコンポーネントを停止
        
        Returns:
            success: 成功フラグ
        """
        try:
            logger.info("[STOP] Stopping all components...")
            
            # タブを閉じる
            for browser_index, scraper in self.so8t_scrapers.items():
                try:
                    await scraper.close_tabs()
                except Exception as e:
                    logger.warning(f"[BROWSER {browser_index}] Failed to close tabs: {e}")
            
            for browser_index, processor in self.tab_processors.items():
                try:
                    await processor.close_tabs()
                except Exception as e:
                    logger.warning(f"[BROWSER {browser_index}] Failed to close tabs: {e}")
            
            # ブラウザを停止
            if self.use_daemon_mode:
                if self.browser_manager:
                    await self.browser_manager.stop_all_browsers()
            else:
                # ブラウザ直接実行モード: 直接ブラウザを閉じる
                for browser_index, browser in self.direct_browsers.items():
                    try:
                        await browser.close()
                        logger.info(f"[BROWSER {browser_index}] Direct browser closed")
                    except Exception as e:
                        logger.warning(f"[BROWSER {browser_index}] Failed to close browser: {e}")
                
                # Playwrightを停止
                if hasattr(self, 'playwright') and self.playwright:
                    await self.playwright.stop()
                    logger.info("[BROWSER] Playwright stopped")
            
            # Chrome DevToolsを停止
            if self.chrome_devtools_launcher:
                await self.chrome_devtools_launcher.stop_all_instances()
            
            # 協調機能を停止
            if self.browser_coordinator:
                await self.browser_coordinator.stop()
            
            logger.info("[OK] All components stopped")
            return True
            
        except Exception as e:
            logger.error(f"[STOP] Failed to stop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        全体の状態を取得
        
        Returns:
            status: 状態辞書
        """
        status = self.status.copy()
        
        # ブラウザ状態を追加
        if self.use_daemon_mode and self.browser_manager:
            status['browsers'] = self.browser_manager.get_all_browsers_status()
        elif not self.use_daemon_mode:
            status['browsers'] = {
                'mode': 'direct',
                'count': len(self.direct_browsers),
                'contexts': len(self.direct_browser_contexts)
            }
        
        # CUDA状態を追加
        if self.cuda_processor:
            status['cuda'] = self.cuda_processor.get_gpu_memory_info()
        
        # Chrome DevTools状態を追加
        if self.chrome_devtools_launcher:
            status['chrome_devtools'] = self.chrome_devtools_launcher.get_all_instances_status()
        
        return status


async def main():
    """メイン関数（テスト用）"""
    manager = SO8TChromeDevDaemonManager(
        num_browsers=10,
        num_tabs=10,
        base_port=9222
    )
    
    try:
        # すべてのコンポーネントを起動
        success = await manager.start_all()
        if not success:
            logger.error("[ERROR] Failed to start all components")
            return
        
        # テストURL
        test_urls = [
            "https://example.com",
            "https://www.google.com",
            "https://github.com"
        ] * 20  # 60個のURL
        
        test_keywords = ["example", "search", "code"] * 20
        
        # SO8T統制でスクレイピング
        results = await manager.scrape_with_so8t_control(test_urls, test_keywords)
        
        logger.info(f"[OK] Scraped {len(results)} samples")
        
        # 状態を表示
        status = manager.get_status()
        logger.info(f"[STATUS] {json.dumps(status, indent=2, ensure_ascii=False)}")
        
        # すべてのコンポーネントを停止
        await manager.stop_all()
        
    except KeyboardInterrupt:
        logger.warning("[INTERRUPT] Interrupted by user")
        await manager.stop_all()
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main())

