#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ収集・前処理統合パイプライン

Webスクレイピング → 統計的データクレンジング → クラス分類自動化
を統合した完全パイプライン

Usage:
    python scripts/pipelines/complete_data_pipeline.py --config configs/data_pipeline_config.yaml
"""

import sys
import json
import logging
import argparse
import signal
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# 既存モジュールのインポート
try:
    from scripts.data.massive_parallel_crawler import ParallelWebCrawler, MASSIVE_CRAWL_CONFIG
except ImportError:
    # フォールバック: 直接パス指定
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "massive_parallel_crawler",
        PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "massive_parallel_crawler.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ParallelWebCrawler = module.ParallelWebCrawler
    MASSIVE_CRAWL_CONFIG = module.MASSIVE_CRAWL_CONFIG

try:
    from scripts.data.specialized_crawlers import (
        Kanpou4WebCrawler, EGovCrawler, ZennCrawler, QiitaCrawler, WikipediaCrawler
    )
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "specialized_crawlers",
        PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "specialized_crawlers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Kanpou4WebCrawler = module.Kanpou4WebCrawler
    EGovCrawler = module.EGovCrawler
    ZennCrawler = module.ZennCrawler
    QiitaCrawler = module.QiitaCrawler
    WikipediaCrawler = module.WikipediaCrawler

try:
    from scripts.data.robots_compliance import create_compliance_checker
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "robots_compliance",
        PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "robots_compliance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    create_compliance_checker = module.create_compliance_checker

try:
    from scripts.data.pipeline_cleanse_and_sample import DataPipeline
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pipeline_cleanse_and_sample",
        PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "pipeline_cleanse_and_sample.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DataPipeline = module.DataPipeline

# 用途別データセット生成のインポート
try:
    from so8t_mmllm.scripts.data.purpose_specific_dataset_generator import PurposeSpecificDatasetGenerator
    PURPOSE_DATASET_GENERATOR_AVAILABLE = True
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "purpose_specific_dataset_generator",
            PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "purpose_specific_dataset_generator.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PurposeSpecificDatasetGenerator = module.PurposeSpecificDatasetGenerator
        PURPOSE_DATASET_GENERATOR_AVAILABLE = True
    except Exception:
        PURPOSE_DATASET_GENERATOR_AVAILABLE = False
        # loggerは後で定義されるため、printを使用
        print("[WARNING] Purpose-specific dataset generator not available")

# ロギング設定（最初に設定）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_data_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from scripts.data.auto_labeler_thinking import ThinkingAutoLabeler

# MLOps可視化モジュールのインポート
try:
    from scripts.utils.mlops_visualizer import MLOpsVisualizer
    MLOPS_VISUALIZER_AVAILABLE = True
except ImportError:
    MLOPS_VISUALIZER_AVAILABLE = False
    logger.warning("MLOps visualizer not available")

# NSFW分類器のインポート
try:
    from scripts.data.train_nsfw_classifier import NSFWClassifier
    NSFW_CLASSIFIER_AVAILABLE = True
except ImportError:
    NSFW_CLASSIFIER_AVAILABLE = False
    logger.warning("NSFW classifier not available")

# マルチモーダルNSFW検知のインポート
try:
    from scripts.data.multimodal_nsfw_detector import MultimodalNSFWDetector
    MULTIMODAL_NSFW_AVAILABLE = True
except ImportError:
    MULTIMODAL_NSFW_AVAILABLE = False
    logger.warning("Multimodal NSFW detector not available")

# マルチモーダルwebスクレイピングのインポート
try:
    from so8t_mmllm.scripts.data.multimodal_web_crawler import MultimodalWebCrawler
    MULTIMODAL_CRAWLER_AVAILABLE = True
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "multimodal_web_crawler",
            PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "multimodal_web_crawler.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MultimodalWebCrawler = module.MultimodalWebCrawler
        MULTIMODAL_CRAWLER_AVAILABLE = True
    except:
        MULTIMODAL_CRAWLER_AVAILABLE = False
        logger.warning("Multimodal web crawler not available")

# 四重推論モジュールのインポート
try:
    import sys
    thinking_tokens_path = PROJECT_ROOT / "so8t-mmllm" / "src" / "models" / "thinking_tokens.py"
    if thinking_tokens_path.exists():
        sys.path.insert(0, str(thinking_tokens_path.parent.parent))
        from models.thinking_tokens import format_quadruple_thinking_output
        THINKING_TOKENS_AVAILABLE = True
    else:
        THINKING_TOKENS_AVAILABLE = False
except ImportError as e:
    THINKING_TOKENS_AVAILABLE = False
    logger.warning(f"Thinking tokens module not available: {e}")


class CompleteDataPipeline:
    """データ収集・前処理統合パイプライン"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        # D:\webdatasetを優先的に使用
        default_output = config.get('output_dir', r'D:\webdataset\processed')
        self.output_dir = Path(default_output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイント管理（D:\webdatasetを使用）
        default_checkpoint = config.get('checkpoint_dir', r'D:\webdataset\checkpoints\pipeline')
        self.checkpoint_dir = Path(default_checkpoint)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        # 進捗管理
        self.current_phase = None
        self.phase_progress = {}
        
        # MLOps可視化初期化
        self.mlops_visualizer = None
        if MLOPS_VISUALIZER_AVAILABLE:
            try:
                mlops_config = config.get('mlops', {})
                self.mlops_visualizer = MLOpsVisualizer(mlops_config, session_id=self.session_id)
            except Exception as e:
                logger.warning(f"Failed to initialize MLOps visualizer: {e}")
        
        logger.info("="*80)
        logger.info("Complete Data Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定（電源断リカバリー）"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            logger.info("Checkpoint saved. Exiting gracefully...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """チェックポイント読み込み"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        if not checkpoint_file.exists():
            # 最新のチェックポイントを検索
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if checkpoints:
                checkpoint_file = checkpoints[-1]
                logger.info(f"Loading latest checkpoint: {checkpoint_file}")
            else:
                return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            logger.info(f"[OK] Checkpoint loaded from {checkpoint_file}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        checkpoint_data = {
            'session_id': self.session_id,
            'current_phase': self.current_phase,
            'phase_progress': self.phase_progress,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"[CHECKPOINT] Saved to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def phase1_web_scraping(self) -> Path:
        """
        Phase 1: Webスクレイピング（拡張版：新データソース統合）
        
        Returns:
            output_path: 収集データのパス
        """
        logger.info("="*80)
        logger.info("PHASE 1: Web Scraping (Extended with Production Sources)")
        logger.info("="*80)
        
        self.current_phase = "web_scraping"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'web_scraping':
            logger.info("[RESUME] Resuming web scraping from checkpoint...")
            # 復旧処理：起動時にチェックポイントを3分ごとに5個ストック
            # チェックポイントディレクトリ内の最新5件のファイルを取得
            import shutil
            import time

            stock_dir = Path(self.checkpoint_dir) / "startup_stock"
            stock_dir.mkdir(parents=True, exist_ok=True)

            # チェックポイントファイルを新しい順に最大5件取得
            checkpoint_files = sorted(Path(self.checkpoint_dir).glob("checkpoint_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            now_time = int(time.time())
            for i, src in enumerate(checkpoint_files, 1):
                # 3分(180秒)ごとにファイル名にタイムスタンプを付与してストック
                dest = stock_dir / f"{src.stem}_stock_{now_time + (i-1)*180}.pkl"
                try:
                    shutil.copy2(src, dest)
                    logger.info(f"[CHECKPOINT-STOCK] Stocked checkpoint (No.{i}): {dest}")
                except Exception as e:
                    logger.warning(f"[CHECKPOINT-STOCK] Failed to stock checkpoint: {e}")
        
        # robots.txt遵守チェッカー初期化
        compliance_checker = create_compliance_checker(self.config.get('crawler', {}))
        
        # NSFW分類器初期化（検知目的、マルチモーダル対応）
        nsfw_classifier = None
        multimodal_nsfw_detector = None
        nsfw_config = self.config.get('nsfw_detection', {})
        enable_nsfw_detection = nsfw_config.get('enabled', True)
        enable_multimodal_nsfw = nsfw_config.get('multimodal', True)
        nsfw_model_path = Path(nsfw_config.get('model_path', 'models/nsfw_classifier.joblib'))
        
        if enable_nsfw_detection:
            # マルチモーダルNSFW検知器（優先）
            if enable_multimodal_nsfw and MULTIMODAL_NSFW_AVAILABLE:
                try:
                    multimodal_nsfw_detector = MultimodalNSFWDetector(
                        text_classifier_path=nsfw_model_path if nsfw_model_path.exists() else None
                    )
                    logger.info("[NSFW] Multimodal NSFW detector initialized")
                except Exception as e:
                    logger.warning(f"[NSFW] Failed to initialize multimodal NSFW detector: {e}")
                    multimodal_nsfw_detector = None
            
            # テキストNSFW分類器（フォールバック）
            if not multimodal_nsfw_detector and NSFW_CLASSIFIER_AVAILABLE:
                if nsfw_model_path.exists():
                    logger.info(f"[NSFW] Loading text NSFW classifier from {nsfw_model_path}")
                    try:
                        nsfw_classifier = NSFWClassifier(model_path=nsfw_model_path)
                        logger.info("[NSFW] Text NSFW classifier loaded successfully")
                    except Exception as e:
                        logger.warning(f"[NSFW] Failed to load NSFW classifier: {e}")
                        nsfw_classifier = None
                else:
                    logger.warning(f"[NSFW] NSFW model not found at {nsfw_model_path}, skipping NSFW detection")
        else:
            if not NSFW_CLASSIFIER_AVAILABLE and not MULTIMODAL_NSFW_AVAILABLE:
                logger.warning("[NSFW] NSFW classifier modules not available")
            if not enable_nsfw_detection:
                logger.info("[NSFW] NSFW detection disabled in config")
        
        # データソース設定取得
        data_sources_config = self.config.get('data_sources', {})
        enable_specialized = data_sources_config.get('enable_specialized_crawlers', True)
        enable_multimodal_crawl = data_sources_config.get('enable_multimodal_crawl', False)
        multimodal_urls = data_sources_config.get('multimodal_urls', [])
        
        all_samples = []
        nsfw_stats = {
            'total_checked': 0,
            'nsfw_detected': 0,
            'safe': 0,
            'by_label': {}
        }
        
        # 1. 専用クローラー実行（4web版官報、e-Gov、zenn、Qiita、Wikipedia）
        if enable_specialized:
            logger.info("[STEP 1] Running specialized crawlers...")
            logger.info(f"[STEP 1] Current total samples: {len(all_samples)}")
            # Chromeに偽装
            chrome_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            specialized_config = {
                'delay': self.config.get('crawler', {}).get('delay_per_domain', 1.0),
                'timeout': self.config.get('crawler', {}).get('timeout', 15),
                'max_pages': data_sources_config.get('max_pages_per_source', 1000),
                'user_agent': chrome_user_agent
            }
            
            # 4web版官報
            if data_sources_config.get('enable_kanpou_4web', True):
                logger.info("[KANPOU] Crawling 4web版官報...")
                try:
                    kanpou_config = specialized_config.copy()
                    kanpou_config.update({
                        'base_url': 'https://kanpou.4web.jp/',
                        'domain': 'official_gazette',
                        'language': 'ja'
                    })
                    kanpou_crawler = Kanpou4WebCrawler(kanpou_config)
                    samples = kanpou_crawler.crawl()
                    logger.info(f"[KANPOU] Raw samples collected: {len(samples)}")
                    all_samples.extend(samples)
                    logger.info(f"[KANPOU] Total samples after KANPOU: {len(all_samples)}")
                except Exception as e:
                    logger.error(f"[KANPOU] Failed: {e}")
                    import traceback
                    logger.error(f"[KANPOU] Traceback: {traceback.format_exc()}")
            
            # e-Gov
            if data_sources_config.get('enable_egov', True):
                logger.info("[EGOV] Crawling e-Gov...")
                try:
                    egov_config = specialized_config.copy()
                    egov_config.update({
                        'base_url': 'https://www.e-gov.go.jp/',
                        'domain': 'government',
                        'language': 'ja'
                    })
                    egov_crawler = EGovCrawler(egov_config)
                    samples = egov_crawler.crawl()
                    logger.info(f"[EGOV] Raw samples collected: {len(samples)}")
                    all_samples.extend(samples)
                    logger.info(f"[EGOV] Total samples after EGOV: {len(all_samples)}")
                except Exception as e:
                    logger.error(f"[EGOV] Failed: {e}")
                    import traceback
                    logger.error(f"[EGOV] Traceback: {traceback.format_exc()}")
            
            # zenn
            if data_sources_config.get('enable_zenn', True):
                logger.info("[ZENN] Crawling zenn...")
                try:
                    zenn_config = specialized_config.copy()
                    zenn_config.update({
                        'base_url': 'https://zenn.dev/',
                        'domain': 'tech_blog',
                        'language': 'ja',
                        'api_enabled': data_sources_config.get('zenn_api_enabled', False)
                    })
                    zenn_crawler = ZennCrawler(zenn_config)
                    samples = zenn_crawler.crawl()
                    logger.info(f"[ZENN] Raw samples collected: {len(samples)}")
                    all_samples.extend(samples)
                    logger.info(f"[ZENN] Total samples after ZENN: {len(all_samples)}")
                except Exception as e:
                    logger.error(f"[ZENN] Failed: {e}")
                    import traceback
                    logger.error(f"[ZENN] Traceback: {traceback.format_exc()}")
            
            # Qiita
            if data_sources_config.get('enable_qiita', True):
                logger.info("[QIITA] Crawling Qiita...")
                try:
                    qiita_config = specialized_config.copy()
                    qiita_config.update({
                        'base_url': 'https://qiita.com/',
                        'domain': 'tech_blog',
                        'language': 'ja',
                        'api_enabled': data_sources_config.get('qiita_api_enabled', False),
                        'api_token': data_sources_config.get('qiita_api_token', None)
                    })
                    qiita_crawler = QiitaCrawler(qiita_config)
                    samples = qiita_crawler.crawl()
                    logger.info(f"[QIITA] Raw samples collected: {len(samples)}")
                    all_samples.extend(samples)
                    logger.info(f"[QIITA] Total samples after QIITA: {len(all_samples)}")
                except Exception as e:
                    logger.error(f"[QIITA] Failed: {e}")
                    import traceback
                    logger.error(f"[QIITA] Traceback: {traceback.format_exc()}")
            
            # ウィキペディア日本語版
            if data_sources_config.get('enable_wikipedia_ja', True):
                logger.info("[WIKIPEDIA] Crawling Wikipedia日本語版...")
                try:
                    wiki_config = specialized_config.copy()
                    wiki_config.update({
                        'base_url': 'https://ja.wikipedia.org/wiki/',
                        'domain': 'encyclopedia',
                        'language': 'ja'
                    })
                    wiki_crawler = WikipediaCrawler(wiki_config)
                    samples = wiki_crawler.crawl()
                    logger.info(f"[WIKIPEDIA] Raw samples collected: {len(samples)}")
                    all_samples.extend(samples)
                    logger.info(f"[WIKIPEDIA] Total samples after WIKIPEDIA: {len(all_samples)}")
                except Exception as e:
                    logger.error(f"[WIKIPEDIA] Failed: {e}")
                    import traceback
                    logger.error(f"[WIKIPEDIA] Traceback: {traceback.format_exc()}")
            
            # 専用クローラー実行完了後の総数確認
            logger.info(f"[STEP 1] Specialized crawlers completed. Total samples: {len(all_samples)}")
            if len(all_samples) == 0:
                logger.warning("[WARNING] No samples collected from specialized crawlers! Check crawler configurations and network connectivity.")
        
        # 2. 並列クローラー実行（日経225企業等を含む既存ソース）
        if data_sources_config.get('enable_parallel_crawler', True):
            logger.info("[STEP 2] Running parallel crawler (Nikkei225 companies, etc.)...")
            logger.info(f"[STEP 2] Current total samples before parallel crawler: {len(all_samples)}")
            crawler_config = self.config.get('crawler', MASSIVE_CRAWL_CONFIG)
            crawler = ParallelWebCrawler(config=crawler_config)
            
            # クローラーの出力ディレクトリ設定
            crawler.output_dir = self.output_dir / "web_crawled"
            crawler.checkpoint_dir = self.checkpoint_dir
            crawler.output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"[STEP 2] Target samples: {crawler_config.get('target_samples', 10000)}")
            
            # 並列クロール実行
            try:
                crawler.run_parallel_crawl()
                
                # 収集されたファイルを統合
                parallel_samples_count = 0
                jsonl_files = list(crawler.output_dir.glob("web_crawled_*.jsonl"))
                logger.info(f"[STEP 2] Found {len(jsonl_files)} parallel crawler output files")
                
                for jsonl_file in jsonl_files:
                    file_samples = 0
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                sample = json.loads(line)
                                all_samples.append(sample)
                                file_samples += 1
                                parallel_samples_count += 1
                            except json.JSONDecodeError:
                                continue
                    logger.info(f"[STEP 2] Loaded {file_samples} samples from {jsonl_file.name}")
                
                logger.info(f"[STEP 2] Total samples from parallel crawler: {parallel_samples_count}")
                logger.info(f"[STEP 2] Total samples after parallel crawler: {len(all_samples)}")
            except Exception as e:
                logger.error(f"[STEP 2] Parallel crawler failed: {e}")
                import traceback
                logger.error(f"[STEP 2] Traceback: {traceback.format_exc()}")
        else:
            logger.warning("[WARNING] Parallel crawler is disabled. Consider enabling it for better data collection.")
        
        # 2.5. マルチモーダルwebスクレイピング（画像+テキスト）
        if enable_multimodal_crawl and MULTIMODAL_CRAWLER_AVAILABLE and multimodal_urls:
            logger.info("[STEP 2.5] Running multimodal web crawler (images + text)...")
            
            multimodal_crawler = MultimodalWebCrawler(
                config=self.config.get('multimodal_crawler', {}),
                output_dir=self.output_dir / "multimodal"
            )
            
            # マルチモーダルクロール実行
            multimodal_samples = multimodal_crawler.crawl_multimodal(
                urls=multimodal_urls,
                max_samples=data_sources_config.get('multimodal_max_samples', 1000)
            )
            
            # サンプルを統合
            all_samples.extend(multimodal_samples)
            
            logger.info(f"[OK] Collected {len(multimodal_samples):,} multimodal samples")
            logger.info(f"[OK] Total images: {sum(s.get('image_count', 0) for s in multimodal_samples):,}")
            
            # マルチモーダルデータを保存
            multimodal_crawler.save()
        
        # 2.6. Phase 1拡張: 追加データソース（プログラミング言語、Arxiv、オープンアクセス論文、ドメイン別知識）
        phase1_extended_config = data_sources_config.get('phase1_extended', {})
        if phase1_extended_config.get('enabled', True):
            logger.info("[STEP 2.6] Running Phase 1 extended crawlers...")
            
            # プログラミング言語ベストプラクティス
            if phase1_extended_config.get('programming_best_practices', {}).get('enabled', True):
                logger.info("[PROGRAMMING] Crawling programming best practices...")
                try:
                    # インポートパスの修正
                    import importlib.util
                    crawler_path = PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "programming_best_practices_crawler.py"
                    if crawler_path.exists():
                        spec = importlib.util.spec_from_file_location(
                            "programming_best_practices_crawler",
                            crawler_path
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        ProgrammingBestPracticesCrawler = module.ProgrammingBestPracticesCrawler
                        
                        prog_config = phase1_extended_config.get('programming_best_practices', {})
                        chrome_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                        crawler = ProgrammingBestPracticesCrawler({
                            'languages': prog_config.get('languages', ['python', 'javascript', 'java', 'rust', 'go', 'csharp']),
                            'max_samples_per_language': prog_config.get('max_samples_per_language', 10000),
                            'delay': self.config.get('crawler', {}).get('delay_per_domain', 1.0),
                            'timeout': self.config.get('crawler', {}).get('timeout', 15),
                            'user_agent': chrome_user_agent
                        })
                        samples = crawler.crawl()
                        logger.info(f"[PROGRAMMING] Raw samples collected: {len(samples)}")
                        all_samples.extend(samples)
                        logger.info(f"[PROGRAMMING] Total samples after PROGRAMMING: {len(all_samples)}")
                    else:
                        logger.warning(f"[PROGRAMMING] Crawler file not found: {crawler_path}")
                except Exception as e:
                    logger.error(f"[PROGRAMMING] Failed: {e}")
                    import traceback
                    logger.error(f"[PROGRAMMING] Traceback: {traceback.format_exc()}")
            
            # Arxiv論文
            if phase1_extended_config.get('arxiv', {}).get('enabled', True):
                logger.info("[ARXIV] Crawling Arxiv papers...")
                try:
                    # インポートパスの修正
                    import importlib.util
                    crawler_path = PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "arxiv_crawler.py"
                    if crawler_path.exists():
                        spec = importlib.util.spec_from_file_location(
                            "arxiv_crawler",
                            crawler_path
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        ArxivCrawler = module.ArxivCrawler
                        
                        arxiv_config = phase1_extended_config.get('arxiv', {})
                        crawler = ArxivCrawler({
                            'categories': arxiv_config.get('categories', ['cs.AI', 'cs.CL', 'cs.LG', 'math', 'physics']),
                            'max_papers': arxiv_config.get('max_papers', 50000),
                            'date_from': arxiv_config.get('date_from', '2020-01-01'),
                            'delay': self.config.get('crawler', {}).get('delay_per_domain', 1.0)
                        })
                        samples = crawler.crawl()
                        logger.info(f"[ARXIV] Raw papers collected: {len(samples)}")
                        all_samples.extend(samples)
                        logger.info(f"[ARXIV] Total samples after ARXIV: {len(all_samples)}")
                    else:
                        logger.warning(f"[ARXIV] Crawler file not found: {crawler_path}")
                except Exception as e:
                    logger.error(f"[ARXIV] Failed: {e}")
                    import traceback
                    logger.error(f"[ARXIV] Traceback: {traceback.format_exc()}")
            
            # オープンアクセス論文
            if phase1_extended_config.get('open_access_papers', {}).get('enabled', True):
                logger.info("[OPEN ACCESS] Crawling open access papers...")
                try:
                    # インポートパスの修正
                    import importlib.util
                    crawler_path = PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "open_access_papers_crawler.py"
                    if crawler_path.exists():
                        spec = importlib.util.spec_from_file_location(
                            "open_access_papers_crawler",
                            crawler_path
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        OpenAccessPapersCrawler = module.OpenAccessPapersCrawler
                        
                        oa_config = phase1_extended_config.get('open_access_papers', {})
                        chrome_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                        crawler = OpenAccessPapersCrawler({
                            'repositories': oa_config.get('repositories', ['pmc', 'doaj', 'europe_pmc', 'core', 'zenodo']),
                            'max_papers': oa_config.get('max_papers', 100000),
                            'delay': self.config.get('crawler', {}).get('delay_per_domain', 1.0),
                            'timeout': self.config.get('crawler', {}).get('timeout', 15),
                            'user_agent': chrome_user_agent
                        })
                        samples = crawler.crawl()
                        logger.info(f"[OPEN ACCESS] Raw papers collected: {len(samples)}")
                        all_samples.extend(samples)
                        logger.info(f"[OPEN ACCESS] Total samples after OPEN ACCESS: {len(all_samples)}")
                    else:
                        logger.warning(f"[OPEN ACCESS] Crawler file not found: {crawler_path}")
                except Exception as e:
                    logger.error(f"[OPEN ACCESS] Failed: {e}")
                    import traceback
                    logger.error(f"[OPEN ACCESS] Traceback: {traceback.format_exc()}")
            
            # ドメイン別知識サイト
            if phase1_extended_config.get('domain_knowledge', {}).get('enabled', True):
                logger.info("[DOMAIN KNOWLEDGE] Crawling domain knowledge sites...")
                try:
                    # インポートパスの修正
                    import importlib.util
                    crawler_path = PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "domain_knowledge_crawler.py"
                    if crawler_path.exists():
                        spec = importlib.util.spec_from_file_location(
                            "domain_knowledge_crawler",
                            crawler_path
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        DomainKnowledgeCrawler = module.DomainKnowledgeCrawler
                        
                        dk_config = phase1_extended_config.get('domain_knowledge', {})
                        chrome_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                        crawler = DomainKnowledgeCrawler({
                            'japanese_sites': dk_config.get('japanese_sites', ['kotobank', 'weblio', 'jstage', 'cinii']),
                            'english_sites': dk_config.get('english_sites', ['britannica', 'khan_academy', 'mit_ocw']),
                            'max_samples_per_site': dk_config.get('max_samples_per_site', 50000),
                            'delay': self.config.get('crawler', {}).get('delay_per_domain', 1.0),
                            'timeout': self.config.get('crawler', {}).get('timeout', 15),
                            'user_agent': chrome_user_agent
                        })
                        samples = crawler.crawl()
                        logger.info(f"[DOMAIN KNOWLEDGE] Raw samples collected: {len(samples)}")
                        all_samples.extend(samples)
                        logger.info(f"[DOMAIN KNOWLEDGE] Total samples after DOMAIN KNOWLEDGE: {len(all_samples)}")
                    else:
                        logger.warning(f"[DOMAIN KNOWLEDGE] Crawler file not found: {crawler_path}")
                except Exception as e:
                    logger.error(f"[DOMAIN KNOWLEDGE] Failed: {e}")
                    import traceback
                    logger.error(f"[DOMAIN KNOWLEDGE] Traceback: {traceback.format_exc()}")
            
            # Phase 1拡張クローラー実行完了後の総数確認
            logger.info(f"[STEP 2.6] Phase 1 extended crawlers completed. Total samples: {len(all_samples)}")
        
        # 3. NSFW検知とラベル付け（リアルタイム処理、マルチモーダル対応）
        if (multimodal_nsfw_detector or nsfw_classifier) and enable_nsfw_detection:
            logger.info("[NSFW] Applying NSFW detection to crawled samples...")
            
            # マルチモーダルNSFW検知を使用
            if multimodal_nsfw_detector:
                logger.info("[NSFW] Using multimodal NSFW detection...")
                labeled_samples = multimodal_nsfw_detector.detect_batch(all_samples)
                
                # 統計更新
                for sample in labeled_samples:
                    nsfw_stats['total_checked'] += 1
                    label = sample.get('nsfw_label', 'unknown')
                    
                    if label == 'safe':
                        nsfw_stats['safe'] += 1
                    else:
                        nsfw_stats['nsfw_detected'] += 1
                    
                    if label not in nsfw_stats['by_label']:
                        nsfw_stats['by_label'][label] = 0
                    nsfw_stats['by_label'][label] += 1
                
                all_samples = labeled_samples
            
            # テキストNSFW検知（フォールバック）
            elif nsfw_classifier:
                logger.info("[NSFW] Using text-only NSFW detection...")
                labeled_samples = []
                
                for sample in tqdm(all_samples, desc="NSFW detection"):
                    text = sample.get('text', sample.get('content', ''))
                    if not text:
                        labeled_samples.append(sample)
                        continue
                    
                    nsfw_stats['total_checked'] += 1
                    
                    try:
                        # NSFW予測
                        pred_label, confidence = nsfw_classifier.predict(text)
                        
                        # ラベルと信頼度を追加（検知目的、学習用）
                        sample['nsfw_label'] = pred_label
                        sample['nsfw_confidence'] = float(confidence)
                        sample['nsfw_detection_purpose'] = 'safety_training'  # 生成目的ではない
                        
                        # 統計更新
                        if pred_label == 'safe':
                            nsfw_stats['safe'] += 1
                        else:
                            nsfw_stats['nsfw_detected'] += 1
                        
                        if pred_label not in nsfw_stats['by_label']:
                            nsfw_stats['by_label'][pred_label] = 0
                        nsfw_stats['by_label'][pred_label] += 1
                        
                    except Exception as e:
                        logger.warning(f"[NSFW] Prediction failed for sample: {e}")
                        sample['nsfw_label'] = 'unknown'
                        sample['nsfw_confidence'] = 0.0
                    
                    labeled_samples.append(sample)
                
                all_samples = labeled_samples
            
            logger.info(f"[NSFW] Detection completed: {nsfw_stats['nsfw_detected']}/{nsfw_stats['total_checked']} NSFW detected")
            logger.info(f"[NSFW] Label distribution: {nsfw_stats['by_label']}")
        else:
            logger.info("[NSFW] Skipping NSFW detection (classifier not available or disabled)")
        
        # 4. 統合ファイル保存
        output_path = self.output_dir / "web_crawled" / f"crawled_{self.session_id}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(all_samples):,} crawled samples...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # NSFW検知結果を別ファイルにも保存（検知目的データセット）
        if (multimodal_nsfw_detector or nsfw_classifier) and enable_nsfw_detection and nsfw_stats['nsfw_detected'] > 0:
            nsfw_output_path = self.output_dir / "web_crawled" / f"nsfw_detected_{self.session_id}.jsonl"
            nsfw_samples = [s for s in all_samples if s.get('nsfw_label') != 'safe' and s.get('nsfw_label') != 'unknown']
            logger.info(f"[NSFW] Saving {len(nsfw_samples):,} NSFW-detected samples to {nsfw_output_path}")
            with open(nsfw_output_path, 'w', encoding='utf-8') as f:
                for sample in nsfw_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # マルチモーダルNSFW検知結果の統計
            if multimodal_nsfw_detector:
                multimodal_count = sum(1 for s in nsfw_samples if s.get('nsfw_multimodal', False))
                logger.info(f"[NSFW] Multimodal NSFW detected: {multimodal_count:,} samples")
        
        logger.info(f"[OK] Phase 1 completed. Output: {output_path}")
        logger.info(f"[STATS] Compliance checker stats: {compliance_checker.get_stats()}")
        if nsfw_classifier:
            logger.info(f"[STATS] NSFW detection stats: {nsfw_stats}")
        
        phase_metrics = {
            'samples': len(all_samples),
            'nsfw_detected': nsfw_stats.get('nsfw_detected', 0) if nsfw_classifier else 0,
            'safe': nsfw_stats.get('safe', 0) if nsfw_classifier else 0
        }
        
        # MLOps可視化
        if self.mlops_visualizer:
            self.mlops_visualizer.log_phase_complete('web_scraping', phase_metrics)
            if nsfw_classifier:
                self.mlops_visualizer.log_nsfw_stats(nsfw_stats)
        
        self.phase_progress['web_scraping'] = {
            'status': 'completed',
            'samples': len(all_samples),
            'output_path': str(output_path),
            'nsfw_stats': nsfw_stats if nsfw_classifier else None,
            'sources': {
                'specialized_crawlers': enable_specialized,
                'parallel_crawler': data_sources_config.get('enable_parallel_crawler', True)
            }
        }
        self._save_checkpoint()
        
        return output_path
    
    def phase2_data_cleaning(self, input_path: Path) -> Path:
        """
        Phase 2: 統計的データクレンジング
        
        Args:
            input_path: 入力データパス
        
        Returns:
            output_path: クレンジング済みデータのパス
        """
        logger.info("="*80)
        logger.info("PHASE 2: Statistical Data Cleaning")
        logger.info("="*80)
        
        self.current_phase = "data_cleaning"
        
        # データパイプライン初期化
        output_dir = self.output_dir / "cleaned"
        target_size_gb = self.config.get('target_size_gb', 300.0)
        
        pipeline = DataPipeline(
            input_dir=input_path.parent,
            output_dir=output_dir,
            target_size_gb=target_size_gb
        )
        
        # パイプライン実行
        logger.info("Running data cleaning pipeline...")
        pipeline.run_pipeline()
        
        # 出力パス取得
        output_path = output_dir / "train.jsonl"
        
        logger.info(f"[OK] Phase 2 completed. Output: {output_path}")
        self.phase_progress['data_cleaning'] = {
            'status': 'completed',
            'output_path': str(output_path)
        }
        self._save_checkpoint()
        
        return output_path
    
    def phase3_auto_classification(self, input_path: Path) -> Path:
        """
        Phase 3: クラス分類自動化
        
        Args:
            input_path: 入力データパス
        
        Returns:
            output_path: ラベル付きデータのパス
        """
        logger.info("="*80)
        logger.info("PHASE 3: Automatic Classification")
        logger.info("="*80)
        
        self.current_phase = "auto_classification"
        
        # キーワード設定読み込み
        keywords_config_path = Path(self.config.get(
            'keywords_config',
            'scripts/data/wikipedia_domain_keywords.json'
        ))
        
        if not keywords_config_path.exists():
            logger.error(f"Keywords config not found: {keywords_config_path}")
            return input_path
        
        with open(keywords_config_path, 'r', encoding='utf-8') as f:
            keywords_config = json.load(f)
        
        # 自動ラベラー初期化
        labeler = ThinkingAutoLabeler(keywords_config)
        
        # データ読み込みとラベル付け
        logger.info("Loading data and applying labels...")
        labeled_samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Labeling"):
                try:
                    sample = json.loads(line)
                    domain = sample.get('domain', 'general')
                    
                    # ラベル付け
                    labeled_sample = labeler.label_sample(sample, domain)
                    labeled_samples.append(labeled_sample)
                except json.JSONDecodeError:
                    continue
        
        # バランス調整
        logger.info("Balancing dataset...")
        balanced_samples = labeler.balance_complete(labeled_samples)
        
        # 結果保存
        output_path = self.output_dir / "labeled" / f"labeled_{self.session_id}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(balanced_samples):,} labeled samples...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in balanced_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Phase 3 completed. Output: {output_path}")
        self.phase_progress['auto_classification'] = {
            'status': 'completed',
            'samples': len(balanced_samples),
            'output_path': str(output_path)
        }
        self._save_checkpoint()
        
        return output_path
    
    def phase4_quadruple_thinking(self, input_path: Path) -> Path:
        """
        Phase 4: 四重推論/thinking処理
        
        Args:
            input_path: 入力データパス
        
        Returns:
            output_path: Thinkingデータセットのパス
        """
        logger.info("="*80)
        logger.info("PHASE 4: Quadruple Thinking Processing")
        logger.info("="*80)
        
        self.current_phase = "quadruple_thinking"
        
        if not THINKING_TOKENS_AVAILABLE:
            logger.error("[THINKING] Thinking tokens module not available, skipping phase 4")
            return input_path
        
        thinking_config = self.config.get('thinking', {})
        enable_thinking = thinking_config.get('enabled', True)
        
        if not enable_thinking:
            logger.info("[THINKING] Thinking processing disabled in config")
            return input_path
        
        # データ読み込み
        logger.info("Loading data for thinking processing...")
        samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading samples"):
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(samples):,} samples")
        
        # 四重推論形式に変換
        logger.info("Converting to quadruple thinking format...")
        thinking_samples = []
        
        for sample in tqdm(samples, desc="Thinking conversion"):
            try:
                # 入力テキスト取得
                text = sample.get('text', sample.get('content', ''))
                instruction = sample.get('instruction', '以下の内容を要約してください。')
                domain = sample.get('domain', sample.get('domain_label', 'general'))
                safety_label = sample.get('safety_label', sample.get('nsfw_label', 'ALLOW'))
                
                if not text or len(text) < 50:
                    continue
                
                # 四重推論生成（簡易実装）
                # 実際の実装では、LLMを使用してより高度な推論を生成
                content_preview = text[:500]
                
                # Task推論（英語）
                task_reasoning = (
                    f"Task: Process and summarize content from {domain} domain. "
                    f"Content preview: {content_preview[:200]}... "
                    f"Need to provide accurate, concise Japanese response."
                )
                
                # Safety推論（英語）
                if safety_label == 'safe' or safety_label == 'ALLOW':
                    safety_reasoning = (
                        f"Safety check: Content is public and safe. "
                        f"No harmful instructions, no illegal content. Safe to answer."
                    )
                else:
                    safety_reasoning = (
                        f"Safety check: Content flagged as {safety_label}. "
                        f"Need careful review. May require escalation or denial."
                    )
                
                # Policy推論（英語）
                policy_mapping = {
                    'defense': "Domain: defense. Provide only descriptive, non-operational information. No classified details.",
                    'medical': "Domain: medical. Provide regulatory information only. No personal medical advice.",
                    'financial': "Domain: financial. Provide general information. No specific investment advice.",
                    'government': "Domain: government. Provide public policy information. No classified details.",
                    'tech_blog': "Domain: tech_blog. Provide technical information. No proprietary details.",
                    'encyclopedia': "Domain: encyclopedia. Provide factual, well-sourced information.",
                }
                policy_reasoning = policy_mapping.get(domain, "Domain: general. Provide accurate, helpful information.")
                
                # Final回答（日本語、簡易要約）
                # 実際の実装では、LLMで要約を生成
                final_answer = f"この内容は{domain}に関する情報です。主要なポイント: {content_preview[:200]}..."
                
                # 四重推論形式の出力を生成
                thinking_output = format_quadruple_thinking_output(
                    task=task_reasoning,
                    safety=safety_reasoning,
                    policy=policy_reasoning,
                    final=final_answer
                )
                
                # 新しいサンプルを作成
                thinking_sample = {
                    "instruction": instruction,
                    "input": f"ドメイン: {domain}\n内容: {text[:1000]}",  # 最初の1000文字
                    "output": thinking_output,
                    "safety_label": safety_label if isinstance(safety_label, str) else 'ALLOW',
                    "domain_label": domain,
                    "thinking_format": "quadruple",
                    "source_sample": sample.get('url', sample.get('id', ''))
                }
                
                thinking_samples.append(thinking_sample)
                
            except Exception as e:
                logger.warning(f"[THINKING] Failed to convert sample: {e}")
                continue
        
        logger.info(f"[THINKING] Converted {len(thinking_samples)} samples to quadruple thinking format")
        
        # 結果保存
        output_path = self.output_dir / "thinking" / f"thinking_{self.session_id}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(thinking_samples)} thinking samples...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in thinking_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Phase 4 completed. Output: {output_path}")
        self.phase_progress['quadruple_thinking'] = {
            'status': 'completed',
            'samples': len(thinking_samples),
            'output_path': str(output_path)
        }
        self._save_checkpoint()
        
        return output_path
    
    def phase5_four_class_classification(self, input_path: Path) -> Path:
        """
        Phase 5: 四値分類（ALLOW/ESCALATION/DENY/REFUSE）
        
        Args:
            input_path: 入力データパス
        
        Returns:
            output_path: 四値分類済みデータのパス
        """
        logger.info("="*80)
        logger.info("PHASE 5: Four Class Classification (ALLOW/ESCALATION/DENY/REFUSE)")
        logger.info("="*80)
        
        self.current_phase = "four_class_classification"
        
        four_class_config = self.config.get('four_class_classification', {})
        enable_classification = four_class_config.get('enabled', True)
        
        if not enable_classification:
            logger.info("[FOUR_CLASS] Four class classification disabled in config")
            return input_path
        
        # ラベルマッピング
        LABEL_TO_ID = {"ALLOW": 0, "ESCALATION": 1, "DENY": 2, "REFUSE": 3}
        
        # データ読み込み
        logger.info("Loading data for four class classification...")
        samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading samples"):
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(samples):,} samples")
        
        # 四値分類適用（ルールベース分類）
        logger.info("Applying four class classification...")
        classified_samples = []
        classification_stats = {
            'ALLOW': 0,
            'ESCALATION': 0,
            'DENY': 0,
            'REFUSE': 0,
            'total': len(samples)
        }
        
        for sample in tqdm(samples, desc="Classification"):
            try:
                # 既存のラベルを取得
                safety_label = sample.get('safety_label', 'ALLOW')
                nsfw_label = sample.get('nsfw_label', 'safe')
                domain = sample.get('domain', sample.get('domain_label', 'general'))
                text = sample.get('text', sample.get('content', ''))
                
                # 四値分類ロジック（ルールベース）
                # 実際の実装では、学習済みモデルを使用
                four_class_label = 'ALLOW'  # デフォルト
                
                # NSFW検知結果に基づく分類
                if nsfw_label in ['nsfw_block', 'violence', 'harassment', 'self_harm', 'illegal_content']:
                    four_class_label = 'REFUSE'
                elif nsfw_label in ['nsfw_soft', 'weapons_detail', 'medical_advice_high_risk']:
                    four_class_label = 'DENY'
                
                # Safety labelに基づく分類
                if safety_label == 'DENY':
                    four_class_label = 'DENY'
                elif safety_label == 'REFUSE':
                    four_class_label = 'REFUSE'
                elif safety_label == 'ESCALATION':
                    four_class_label = 'ESCALATION'
                
                # ドメインに基づく分類
                sensitive_domains = ['defense', 'medical', 'financial']
                if domain in sensitive_domains and len(text) > 1000:
                    # 長文の機密ドメインはエスカレーション
                    if four_class_label == 'ALLOW':
                        four_class_label = 'ESCALATION'
                
                # ラベルを追加
                sample['four_class_label'] = four_class_label
                sample['four_class_label_id'] = LABEL_TO_ID[four_class_label]
                
                # 統計更新
                classification_stats[four_class_label] += 1
                
                classified_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"[FOUR_CLASS] Failed to classify sample: {e}")
                # エラー時はALLOWとして扱う
                sample['four_class_label'] = 'ALLOW'
                sample['four_class_label_id'] = 0
                classification_stats['ALLOW'] += 1
                classified_samples.append(sample)
        
        logger.info(f"[FOUR_CLASS] Classification completed:")
        logger.info(f"  ALLOW: {classification_stats['ALLOW']}")
        logger.info(f"  ESCALATION: {classification_stats['ESCALATION']}")
        logger.info(f"  DENY: {classification_stats['DENY']}")
        logger.info(f"  REFUSE: {classification_stats['REFUSE']}")
        
        # 結果保存
        output_path = self.output_dir / "four_class" / f"four_class_{self.session_id}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(classified_samples):,} classified samples...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in classified_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 分類結果の可視化（簡易統計）
        try:
            import matplotlib
            matplotlib.use('Agg')  # バックエンド設定
            import matplotlib.pyplot as plt
            
            # 分類分布の可視化
            fig, ax = plt.subplots(figsize=(10, 6))
            labels = list(classification_stats.keys())[:-1]  # 'total'を除外
            counts = [classification_stats[label] for label in labels]
            colors = ['green', 'orange', 'yellow', 'red']
            
            ax.bar(labels, counts, color=colors)
            ax.set_xlabel('Classification Label')
            ax.set_ylabel('Count')
            ax.set_title('Four Class Classification Distribution')
            ax.grid(axis='y', alpha=0.3)
            
            # パーセンテージを表示
            total = classification_stats['total']
            for i, (label, count) in enumerate(zip(labels, counts)):
                percentage = (count / total * 100) if total > 0 else 0
                ax.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            viz_path = self.output_dir / "four_class" / f"classification_distribution_{self.session_id}.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[VISUALIZATION] Classification distribution saved to {viz_path}")
            
        except ImportError:
            logger.warning("[VISUALIZATION] matplotlib not available, skipping visualization")
        except Exception as e:
            logger.warning(f"[VISUALIZATION] Failed to create visualization: {e}")
        
        logger.info(f"[OK] Phase 5 completed. Output: {output_path}")
        
        phase_metrics = {
            'samples': len(classified_samples),
            **{f'{k}_count': v for k, v in classification_stats.items() if k != 'total'}
        }
        
        # MLOps可視化
        if self.mlops_visualizer:
            self.mlops_visualizer.log_phase_complete('four_class_classification', phase_metrics)
            self.mlops_visualizer.log_classification_results(classification_stats)
        
        self.phase_progress['four_class_classification'] = {
            'status': 'completed',
            'samples': len(classified_samples),
            'output_path': str(output_path),
            'stats': classification_stats
        }
        self._save_checkpoint()
        
        return output_path
    
    def run_pipeline(self, resume: bool = True):
        """
        全パイプライン実行
        
        Args:
            resume: チェックポイントから復旧するか
        """
        logger.info("="*80)
        logger.info("COMPLETE DATA PIPELINE")
        logger.info("="*80)
        
        # チェックポイントから復旧
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                self.session_id = checkpoint.get('session_id', self.session_id)
                self.phase_progress = checkpoint.get('phase_progress', {})
                logger.info(f"[RESUME] Resuming from checkpoint (Session: {self.session_id})")
        
        try:
            # Phase 1: Webスクレイピング
            if 'web_scraping' not in self.phase_progress:
                crawled_path = self.phase1_web_scraping()
            else:
                logger.info("[SKIP] Phase 1 already completed")
                crawled_path = Path(self.phase_progress['web_scraping']['output_path'])
            
            # Phase 2: データクレンジング
            if 'data_cleaning' not in self.phase_progress:
                cleaned_path = self.phase2_data_cleaning(crawled_path)
            else:
                logger.info("[SKIP] Phase 2 already completed")
                cleaned_path = Path(self.phase_progress['data_cleaning']['output_path'])
            
            # Phase 3: クラス分類自動化
            if 'auto_classification' not in self.phase_progress:
                labeled_path = self.phase3_auto_classification(cleaned_path)
            else:
                logger.info("[SKIP] Phase 3 already completed")
                labeled_path = Path(self.phase_progress['auto_classification']['output_path'])
            
            # Phase 4: 四重推論/thinking処理
            if 'quadruple_thinking' not in self.phase_progress:
                thinking_path = self.phase4_quadruple_thinking(labeled_path)
            else:
                logger.info("[SKIP] Phase 4 already completed")
                thinking_path = Path(self.phase_progress['quadruple_thinking']['output_path'])
            
            # Phase 5: 四値分類
            if 'four_class_classification' not in self.phase_progress:
                four_class_path = self.phase5_four_class_classification(thinking_path)
            else:
                logger.info("[SKIP] Phase 5 already completed")
                four_class_path = Path(self.phase_progress['four_class_classification']['output_path'])
            
            # Phase 6: 用途別データセット生成（ファインチューニング/RAG/評価用）
            if PURPOSE_DATASET_GENERATOR_AVAILABLE:
                purpose_config = self.config.get('purpose_specific_datasets', {})
                enable_purpose_datasets = purpose_config.get('enabled', True)
                
                if enable_purpose_datasets:
                    logger.info("="*80)
                    logger.info("PHASE 6: Purpose-Specific Dataset Generation")
                    logger.info("="*80)
                    
                    generator = PurposeSpecificDatasetGenerator(config=purpose_config)
                    purpose_output_dir = self.output_dir / "purpose_specific"
                    purpose_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # ファインチューニング用データセット生成
                    if purpose_config.get('generate_finetuning', True):
                        logger.info("[PURPOSE] Generating finetuning dataset...")
                        finetuning_output = purpose_output_dir / f"finetuning_{self.session_id}.jsonl"
                        finetuning_count = generator.generate_finetuning_dataset(
                            four_class_path,
                            finetuning_output,
                            format_type=purpose_config.get('finetuning_format', 'instruction')
                        )
                        logger.info(f"[OK] Generated {finetuning_count:,} finetuning samples")
                    
                    # RAG用データセット生成
                    if purpose_config.get('generate_rag', True):
                        logger.info("[PURPOSE] Generating RAG dataset...")
                        rag_output_dir = purpose_output_dir / "rag"
                        rag_count = generator.generate_rag_dataset(
                            four_class_path,
                            rag_output_dir,
                            chunk_size=purpose_config.get('rag_chunk_size', 512),
                            chunk_overlap=purpose_config.get('rag_chunk_overlap', 128)
                        )
                        logger.info(f"[OK] Generated {rag_count:,} RAG chunks")
                    
                    # 評価用データセット生成
                    if purpose_config.get('generate_evaluation', True):
                        logger.info("[PURPOSE] Generating evaluation dataset...")
                        eval_output = purpose_output_dir / f"evaluation_{self.session_id}.jsonl"
                        eval_count = generator.generate_evaluation_dataset(
                            four_class_path,
                            eval_output,
                            task_types=purpose_config.get('evaluation_task_types', ['understanding', 'generation', 'reasoning'])
                        )
                        logger.info(f"[OK] Generated {eval_count:,} evaluation samples")
                    
                    logger.info("[OK] Phase 6 completed")
                else:
                    logger.info("[SKIP] Purpose-specific dataset generation disabled")
            else:
                logger.warning("[SKIP] Purpose-specific dataset generator not available")
            
            # 統合ダッシュボード生成
            if self.mlops_visualizer:
                dashboard_path = self.output_dir / f"dashboard_{self.session_id}.png"
                self.mlops_visualizer.create_summary_dashboard(self.phase_progress, dashboard_path)
            
            # 最終レポート生成
            self._generate_final_report()
            
            logger.info("="*80)
            logger.info("[COMPLETE] All phases completed!")
            logger.info(f"Final output: {four_class_path}")
            logger.info("="*80)
            
            # MLOps可視化クローズ
            if self.mlops_visualizer:
                self.mlops_visualizer.close()
            
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user")
            self._save_checkpoint()
            raise
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self._save_checkpoint()
            raise
    
    def _generate_final_report(self):
        """最終レポート生成"""
        logger.info("[REPORT] Generating final pipeline report...")
        
        report = {
            "session_id": self.session_id,
            "completed_at": datetime.now().isoformat(),
            "phases": self.phase_progress,
            "summary": {
                "total_phases": len(self.phase_progress),
                "completed_phases": sum(1 for p in self.phase_progress.values() if p.get('status') == 'completed')
            }
        }
        
        # 各フェーズの統計を集計
        total_samples = 0
        for phase_name, phase_data in self.phase_progress.items():
            if 'samples' in phase_data:
                total_samples = max(total_samples, phase_data['samples'])
        
        report["summary"]["total_samples"] = total_samples
        
        # レポート保存
        report_file = self.output_dir / f"pipeline_report_{self.session_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Final report saved to {report_file}")
        
        # マークダウンレポートも生成
        md_report_file = self.output_dir / f"pipeline_report_{self.session_id}.md"
        md_content = f"""# データ収集パイプライン実行レポート

## セッション情報
- **セッションID**: {self.session_id}
- **完了日時**: {report['completed_at']}
- **総フェーズ数**: {report['summary']['total_phases']}
- **完了フェーズ数**: {report['summary']['completed_phases']}
- **総サンプル数**: {report['summary']['total_samples']:,}

## フェーズ別結果

"""
        
        for phase_name, phase_data in self.phase_progress.items():
            status = phase_data.get('status', 'unknown')
            samples = phase_data.get('samples', 0)
            output_path = phase_data.get('output_path', 'N/A')
            
            md_content += f"""### {phase_name}
- **ステータス**: {status}
- **サンプル数**: {samples:,}
- **出力パス**: {output_path}

"""
        
        # NSFW検知統計
        if 'web_scraping' in self.phase_progress:
            nsfw_stats = self.phase_progress['web_scraping'].get('nsfw_stats')
            if nsfw_stats:
                md_content += f"""## NSFW検知統計
- **総チェック数**: {nsfw_stats.get('total_checked', 0):,}
- **NSFW検知数**: {nsfw_stats.get('nsfw_detected', 0):,}
- **安全**: {nsfw_stats.get('safe', 0):,}
- **ラベル分布**: {nsfw_stats.get('by_label', {})}

"""
        
        md_content += f"""
## 出力ディレクトリ
{self.output_dir}

## 完了
全パイプライン処理が完了しました。
"""
        
        with open(md_report_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"[OK] Markdown report saved to {md_report_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Complete Data Pipeline")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_pipeline_config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # デフォルト設定
        config = {
            'output_dir': 'data/processed',
            'checkpoint_dir': 'data/pipeline_checkpoints',
            'target_size_gb': 300.0,
            'crawler': MASSIVE_CRAWL_CONFIG,
            'keywords_config': 'scripts/data/wikipedia_domain_keywords.json'
        }
        logger.warning(f"Config file not found: {config_path}, using defaults")
    
    # パイプライン実行
    pipeline = CompleteDataPipeline(config)
    pipeline.run_pipeline(resume=not args.no_resume)


if __name__ == '__main__':
    main()

