#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
並列DeepResearch Webスクレイピングスクリプト

動的リソース管理を行い、DeepResearchとPlaywrightでCursorブラウザを10個起動して、
すべて異なるキーワード検索を人間を模倣した関連ワードによる検索やボタン操作等を行い
Webスクレイピングを実行します。

Usage:
    python scripts/data/parallel_deep_research_scraping.py --output D:\webdataset\processed
"""

import sys
import json
import logging
import asyncio
import argparse
import random
import time
import psutil
import os
import subprocess
import platform
import hashlib
import re
from difflib import SequenceMatcher

# 統計計算用ライブラリ
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

# PyTorchインポート（オプショナル）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, quote, urljoin
from collections import deque
from dataclasses import dataclass

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# SO8Tモデルインポート
try:
    from so8t_mmllm.src.models.so8t_thinking_model import SO8TThinkingModel
    from so8t_mmllm.src.models.thinking_tokens import (
        add_thinking_tokens_to_tokenizer,
        build_quadruple_thinking_prompt,
        extract_quadruple_thinking
    )
    from transformers import AutoTokenizer
    SO8T_AVAILABLE = True
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "so8t_thinking_model",
            PROJECT_ROOT / "so8t-mmllm" / "src" / "models" / "so8t_thinking_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SO8TThinkingModel = module.SO8TThinkingModel
        SO8T_AVAILABLE = True
    except Exception:
        SO8T_AVAILABLE = False
        # loggerはまだ定義されていないので、printを使用
        print("[WARNING] SO8T model not available")

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext, TimeoutError as PlaywrightTimeoutError
except ImportError:
    print("[ERROR] Playwright not installed. Install with: pip install playwright")
    sys.exit(1)

# BeautifulSoupインポート
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("[ERROR] BeautifulSoup not installed. Install with: pip install beautifulsoup4")
    sys.exit(1)

# NSFW分類器インポート
try:
    from scripts.data.train_nsfw_classifier import NSFWClassifier
    NSFW_CLASSIFIER_AVAILABLE = True
except ImportError:
    NSFW_CLASSIFIER_AVAILABLE = False

# ParallelTabProcessorインポート
try:
    from scripts.data.parallel_tab_processor import ParallelTabProcessor
    PARALLEL_TAB_PROCESSOR_AVAILABLE = True
except ImportError:
    PARALLEL_TAB_PROCESSOR_AVAILABLE = False
    logger = logging.getLogger(__name__)

# QuadrupleClassifierインポート（全自動ラベル付け・4値分類・四重推論用）
try:
    from scripts.pipelines.web_scraping_data_pipeline import QuadrupleClassifier
    QUADRUPLE_CLASSIFIER_AVAILABLE = True
except ImportError:
    QUADRUPLE_CLASSIFIER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[WARNING] ParallelTabProcessor not available")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/parallel_deep_research_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# カテゴリ別キーワードリスト（拡張版 - ドメイン別知識特化）
CATEGORY_KEYWORDS = {
    'ja': {
        'technology': [
            '人工知能', '機械学習', '深層学習', '自然言語処理', 'コンピュータビジョン',
            'ブロックチェーン', '暗号通貨', '量子コンピュータ', 'IoT', '5G',
            'クラウドコンピューティング', 'マイクロサービス', 'DevOps', 'コンテナ',
            'Python', 'JavaScript', 'TypeScript', 'Rust', 'Go', 'Kotlin',
            'ニューラルネットワーク', '強化学習', '転移学習', 'GAN', 'Transformer'
        ],
        'programming_languages': [
            # モダン言語
            'Python', 'JavaScript', 'TypeScript', 'Go', 'Rust', 'C++', 'C#', 'Swift', 'Kotlin',
            'Dart', 'Scala', 'Elixir', 'Erlang', 'Clojure', 'F#', 'Haskell', 'OCaml',
            # レガシー言語
            'COBOL', 'PHP', 'Java', 'C', 'Fortran', 'Pascal', 'Ada', 'PL/I', 'BASIC',
            'Visual Basic', 'Delphi', 'Perl', 'Ruby', 'Lua', 'Tcl',
            # 低レベル言語
            'アセンブリ言語', '機械語', 'x86', 'ARM', 'RISC-V', 'アセンブラ', 'アセンブリ',
            'マシンコード', 'バイナリ', 'オペコード', 'レジスタ', 'メモリマップ',
            # データベース・SQL
            'SQL', 'PostgreSQL', 'MySQL', 'SQL Server', 'Oracle', 'SQLite', 'MariaDB',
            'NoSQL', 'MongoDB', 'Redis', 'Cassandra', 'Elasticsearch',
            # ツール・スクリプト
            'Excel', 'VBA', 'PowerShell', 'Bash', 'Shell Script', 'シェルスクリプト',
            'バッチファイル', 'AWK', 'sed', 'grep', 'find', 'curl', 'wget'
        ],
        'algorithms': [
            'アルゴリズム', 'データ構造', 'ソート', '検索', 'グラフ理論', '動的計画法',
            '貪欲法', '分割統治法', 'バックトラッキング', '再帰', '反復',
            '二分探索', '線形探索', 'ハッシュテーブル', '二分木', 'AVL木', 'B木',
            'ヒープ', 'スタック', 'キュー', '連結リスト', '配列', '文字列処理',
            '最短経路', '最小全域木', '最大流', 'マッチング', 'ネットワークフロー',
            '計算量', '時間計算量', '空間計算量', 'Big O記法', '最適化'
        ],
        'hardware': [
            'ハードウェア', 'CPU', 'メモリ', 'RAM', 'ROM', 'ストレージ', 'SSD', 'HDD',
            'GPU', 'TPU', 'FPGA', 'ASIC', 'マイクロプロセッサ', 'プロセッサ',
            'アーキテクチャ', 'x86', 'x64', 'ARM', 'RISC-V', 'MIPS', 'PowerPC',
            'キャッシュ', 'レジスタ', 'バス', 'I/O', '入出力', 'インターフェース',
            'USB', 'PCIe', 'SATA', 'NVMe', 'Ethernet', 'Wi-Fi', 'Bluetooth',
            '組み込みシステム', 'マイコン', 'Arduino', 'Raspberry Pi', 'IoTデバイス',
            'オーバークロック', '冷却', '電源', 'マザーボード', 'チップセット'
        ],
        'coding_best_practices': [
            'ベストプラクティス', 'コーディング規約', 'コードレビュー', 'リファクタリング',
            'デザインパターン', 'SOLID原則', 'DRY原則', 'KISS原則', 'YAGNI原則',
            'テスト駆動開発', 'TDD', 'BDD', 'ユニットテスト', '統合テスト', 'E2Eテスト',
            'CI/CD', '継続的インテグレーション', '継続的デプロイ', 'Git', 'GitHub',
            'バージョン管理', 'ブランチ戦略', 'コード品質', '静的解析', 'リンター',
            'フォーマッター', 'ドキュメンテーション', 'コメント', 'API設計', 'RESTful',
            'マイクロサービス', 'モノリシック', 'アーキテクチャ', 'スケーラビリティ',
            'パフォーマンス', '最適化', 'セキュリティ', '脆弱性', '暗号化'
        ],
        'engineering_sites': [
            'Stack Overflow', 'Qiita', 'Zenn', 'Medium', 'Dev.to', 'GitHub',
            '技術ブログ', 'エンジニアブログ', 'プログラミング', '開発者向け',
            'ドキュメント', 'APIリファレンス', 'チュートリアル', 'ガイド',
            'サンプルコード', 'コード例', '実装例', '使い方', '入門', '初心者',
            '上級者', 'エキスパート', 'ベストプラクティス', 'Tips', 'Tricks'
        ],
        'science': [
            '量子力学', '相対性理論', '遺伝子', 'DNA', 'タンパク質', '細胞',
            '進化論', '宇宙', 'ブラックホール', 'ダークマター', '素粒子',
            '化学反応', '分子', '原子', '元素', '周期表', '光合成', '酵素'
        ],
        'medicine': [
            'がん', '糖尿病', '高血圧', '心臓病', '脳卒中', '認知症',
            'ワクチン', '免疫', '抗体', 'ウイルス', '細菌', '感染症',
            '手術', '治療', '薬', '副作用', '臨床試験', '遺伝子治療'
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
        # NSFW検知用データセット（検知目的のみ、生成目的ではない）
        'nsfw_detection_fanza': [
            'FANZA', 'DMM', '動画', '配信', '作品', 'ジャンル', 'カテゴリ',
            'ランキング', '新着', '人気', 'レビュー', '評価'
        ],
        'nsfw_detection_fc2': [
            'FC2', '動画', '配信', 'チャンネル', 'ライブ', 'アーカイブ',
            'カテゴリ', 'ランキング', '新着', '人気'
        ],
        'nsfw_detection_missav': [
            'missav', '動画', '配信', '作品', 'ジャンル', 'カテゴリ',
            'ランキング', '新着', '人気', 'レビュー'
        ]
    },
    'en': {
        'technology': [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural networks',
            'blockchain', 'cryptocurrency', 'quantum computing', 'IoT', '5G',
            'cloud computing', 'microservices', 'DevOps', 'containers',
            'Python', 'JavaScript', 'TypeScript', 'Rust', 'Go', 'Kotlin',
            'reinforcement learning', 'transfer learning', 'GAN', 'Transformer'
        ],
        'programming_languages': [
            # Modern languages
            'Python', 'JavaScript', 'TypeScript', 'Go', 'Rust', 'C++', 'C#', 'Swift', 'Kotlin',
            'Dart', 'Scala', 'Elixir', 'Erlang', 'Clojure', 'F#', 'Haskell', 'OCaml',
            # Legacy languages
            'COBOL', 'PHP', 'Java', 'C', 'Fortran', 'Pascal', 'Ada', 'PL/I', 'BASIC',
            'Visual Basic', 'Delphi', 'Perl', 'Ruby', 'Lua', 'Tcl',
            # Low-level languages
            'assembly language', 'machine code', 'x86', 'ARM', 'RISC-V', 'assembler', 'assembly',
            'machine code', 'binary', 'opcode', 'register', 'memory map',
            # Database & SQL
            'SQL', 'PostgreSQL', 'MySQL', 'SQL Server', 'Oracle', 'SQLite', 'MariaDB',
            'NoSQL', 'MongoDB', 'Redis', 'Cassandra', 'Elasticsearch',
            # Tools & Scripts
            'Excel', 'VBA', 'PowerShell', 'Bash', 'Shell Script', 'shell scripting',
            'batch file', 'AWK', 'sed', 'grep', 'find', 'curl', 'wget'
        ],
        'algorithms': [
            'algorithm', 'data structure', 'sorting', 'searching', 'graph theory', 'dynamic programming',
            'greedy algorithm', 'divide and conquer', 'backtracking', 'recursion', 'iteration',
            'binary search', 'linear search', 'hash table', 'binary tree', 'AVL tree', 'B-tree',
            'heap', 'stack', 'queue', 'linked list', 'array', 'string processing',
            'shortest path', 'minimum spanning tree', 'maximum flow', 'matching', 'network flow',
            'complexity', 'time complexity', 'space complexity', 'Big O notation', 'optimization'
        ],
        'hardware': [
            'hardware', 'CPU', 'memory', 'RAM', 'ROM', 'storage', 'SSD', 'HDD',
            'GPU', 'TPU', 'FPGA', 'ASIC', 'microprocessor', 'processor',
            'architecture', 'x86', 'x64', 'ARM', 'RISC-V', 'MIPS', 'PowerPC',
            'cache', 'register', 'bus', 'I/O', 'input output', 'interface',
            'USB', 'PCIe', 'SATA', 'NVMe', 'Ethernet', 'Wi-Fi', 'Bluetooth',
            'embedded system', 'microcontroller', 'Arduino', 'Raspberry Pi', 'IoT device',
            'overclocking', 'cooling', 'power supply', 'motherboard', 'chipset'
        ],
        'coding_best_practices': [
            'best practices', 'coding standards', 'code review', 'refactoring',
            'design patterns', 'SOLID principles', 'DRY principle', 'KISS principle', 'YAGNI principle',
            'test-driven development', 'TDD', 'BDD', 'unit test', 'integration test', 'E2E test',
            'CI/CD', 'continuous integration', 'continuous deployment', 'Git', 'GitHub',
            'version control', 'branching strategy', 'code quality', 'static analysis', 'linter',
            'formatter', 'documentation', 'comment', 'API design', 'RESTful',
            'microservices', 'monolithic', 'architecture', 'scalability',
            'performance', 'optimization', 'security', 'vulnerability', 'encryption'
        ],
        'engineering_sites': [
            'Stack Overflow', 'Qiita', 'Zenn', 'Medium', 'Dev.to', 'GitHub',
            'technical blog', 'engineer blog', 'programming', 'developer',
            'documentation', 'API reference', 'tutorial', 'guide',
            'sample code', 'code example', 'implementation example', 'how to', 'tutorial', 'beginner',
            'advanced', 'expert', 'best practices', 'tips', 'tricks'
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
        # NSFW検知用データセット（検知目的のみ、生成目的ではない）
        'nsfw_detection_adult_sites': [
            'adult content', 'video', 'streaming', 'category', 'ranking',
            'new releases', 'popular', 'review', 'rating'
        ]
    }
}

# NSFW検知用サイトURLリスト（検知目的のみ、生成目的ではない）
NSFW_DETECTION_SITES = {
    'ja': {
        'fanza': [
            'https://www.dmm.co.jp/',
            'https://www.dmm.co.jp/digital/videoa/',
            'https://www.dmm.co.jp/digital/videoc/',
            'https://www.dmm.co.jp/rental/',
            'https://www.dmm.co.jp/rental/videoa/',
        ],
        'fc2': [
            'https://live.fc2.com/',
            'https://live.fc2.com/category/',
            'https://live.fc2.com/ranking/',
        ],
        'missav': [
            'https://missav.ai/',
            'https://missav.ai/genre/',
            'https://missav.ai/ranking/',
        ]
    },
    'en': {
        'adult_sites': [
            'https://www.pornhub.com/',
            'https://www.xvideos.com/',
            'https://www.xhamster.com/',
        ]
    }
}


@dataclass
class KeywordTask:
    """キーワードタスク"""
    keyword: str
    category: str
    language: str
    related_keywords: List[str] = None
    url: str = None  # NSFW検知用サイトの場合はURLを指定
    coding_related: bool = False  # コーディング関連フラグ
    company_name: str = None  # 日経225企業名
    company_code: str = None  # 日経225企業コード
    company_domain: str = None  # 企業ドメイン（heavy_industry, airline, transport等）
    data_type: str = None  # データタイプ（financial_reports, press_releases, product_info, nikkei_company_info）


class ResourceManager:
    """動的リソース管理クラス"""
    
    def __init__(self, max_memory_gb: float = 8.0, max_cpu_percent: float = 80.0):
        """
        初期化
        
        Args:
            max_memory_gb: 最大メモリ使用量（GB）
            max_cpu_percent: 最大CPU使用率（%）
        """
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent
    
    def get_memory_usage_gb(self) -> float:
        """現在のメモリ使用量を取得（GB）"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb / 1024
    
    def get_cpu_percent(self) -> float:
        """現在のCPU使用率を取得（%）"""
        return psutil.cpu_percent(interval=0.1)
    
    def can_allocate_browser(self) -> bool:
        """ブラウザを割り当て可能かチェック"""
        memory_gb = self.get_memory_usage_gb()
        cpu_percent = self.get_cpu_percent()
        
        can_allocate = memory_gb < self.max_memory_gb and cpu_percent < self.max_cpu_percent
        
        if not can_allocate:
            logger.warning(f"[RESOURCE] Resource limit reached: Memory={memory_gb:.2f}GB, CPU={cpu_percent:.1f}%")
        
        return can_allocate
    
    def get_resource_status(self) -> Dict:
        """リソース状態を取得"""
        return {
            'memory_gb': self.get_memory_usage_gb(),
            'cpu_percent': self.get_cpu_percent(),
            'max_memory_gb': self.max_memory_gb,
            'max_cpu_percent': self.max_cpu_percent
        }


class ParallelDeepResearchScraper:
    """並列DeepResearch Webスクレイピングクラス（SO8T統制対応）"""
    
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
        max_cpu_percent: float = 90.0,
        use_so8t_control: bool = True,
        so8t_model_path: Optional[str] = None
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
            so8t_model_path: SO8Tモデルのパス（Noneの場合はデフォルト）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_browsers = num_browsers
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_action = delay_per_action
        self.timeout = timeout
        self.max_pages_per_keyword = max_pages_per_keyword
        self.use_so8t_control = use_so8t_control
        
        self.resource_manager = ResourceManager(max_memory_gb, max_cpu_percent)
        
        # SO8Tモデル初期化
        self.so8t_model = None
        self.so8t_tokenizer = None
        if self.use_so8t_control and SO8T_AVAILABLE:
            try:
                self._initialize_so8t_model(so8t_model_path)
                logger.info("[SO8T] SO8T model initialized for web scraping control")
            except Exception as e:
                logger.warning(f"[SO8T] Failed to initialize SO8T model: {e}")
                logger.warning("[SO8T] Continuing without SO8T control")
                self.use_so8t_control = False
        
        self.all_samples: List[Dict] = []
        self.nsfw_samples: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # キーワードキュー
        self.keyword_queue: deque = deque()
        self.completed_keywords: Set[str] = set()
        
        # 日経225企業データ
        self.nikkei225_companies: List[Dict] = []
        self.nikkei225_enabled: bool = True  # デフォルトで有効
        
        # ブラウザ管理
        self.browsers: List[Browser] = []
        self.contexts: List[BrowserContext] = []
        self.pages: List[Page] = []
        
        # 日経225企業サンプル（カテゴリ別）
        self.nikkei225_samples: Dict[str, List[Dict]] = {
            'financial_reports': [],
            'press_releases': [],
            'product_info': [],
            'nikkei_company_info': []
        }
        
        # ブラウザ状態の保存と復元
        self.browser_state_file = self.output_dir / "browser_state.json"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # ダッシュボード用状態管理
        self.browser_status: Dict[int, Dict] = {}
        self.so8t_decisions: List[Dict] = []
        self.screenshots_dir = self.output_dir / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # 全自動ラベル付け・4値分類・四重推論の初期化
        self.quadruple_classifier: Optional[QuadrupleClassifier] = None
        self.auto_labeling_enabled: bool = True  # デフォルトで有効
        self.training_dataset_samples: List[Dict] = []  # 学習用データセット形式のサンプル
        
        if self.auto_labeling_enabled and QUADRUPLE_CLASSIFIER_AVAILABLE:
            try:
                # モデルパスが指定されていない場合は自動検出を試みる
                model_path_to_use = so8t_model_path
                if model_path_to_use is None:
                    default_paths = [
                        "D:/webdataset/models/so8t-phi4-so8t-ja-finetuned",
                        "models/so8t-phi4-so8t-ja-finetuned",
                        "so8t-mmllm/models/so8t-phi4-so8t-ja-finetuned",
                        Path(PROJECT_ROOT) / "models" / "so8t-phi4-so8t-ja-finetuned",
                        Path(PROJECT_ROOT) / "so8t-mmllm" / "models" / "so8t-phi4-so8t-ja-finetuned"
                    ]
                    for path in default_paths:
                        path_obj = Path(path)
                        if path_obj.exists() and (path_obj / "config.json").exists():
                            model_path_to_use = str(path_obj)
                            logger.info(f"[AUTO-LABELING] Auto-detected SO8T model path: {model_path_to_use}")
                            break
                
                if model_path_to_use:
                    self.quadruple_classifier = QuadrupleClassifier(so8t_model_path=model_path_to_use)
                    logger.info("[AUTO-LABELING] QuadrupleClassifier initialized for automatic labeling and classification")
                else:
                    logger.warning("[AUTO-LABELING] SO8T model path not found, skipping automatic labeling")
                    logger.warning("[AUTO-LABELING] To enable automatic labeling, specify --so8t-model-path or place model in default location")
                    self.quadruple_classifier = None
            except Exception as e:
                logger.warning(f"[AUTO-LABELING] Failed to initialize QuadrupleClassifier: {e}")
                logger.warning("[AUTO-LABELING] Continuing without automatic labeling")
                import traceback
                logger.debug(traceback.format_exc())
                self.quadruple_classifier = None
        elif not QUADRUPLE_CLASSIFIER_AVAILABLE:
            logger.warning("[AUTO-LABELING] QuadrupleClassifier not available (import failed)")
            logger.warning("[AUTO-LABELING] Continuing without automatic labeling")
        
        logger.info("="*80)
        logger.info("Parallel DeepResearch Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of browsers: {self.num_browsers}")
        logger.info(f"Max memory: {max_memory_gb}GB")
        logger.info(f"Max CPU: {max_cpu_percent}%")
        logger.info(f"SO8T control: {self.use_so8t_control}")
        logger.info(f"Auto labeling: {self.auto_labeling_enabled}")
    
    def _initialize_so8t_model(self, model_path: Optional[str] = None):
        """SO8Tモデルを初期化"""
        try:
            if model_path is None:
                # デフォルトモデルパスを探す
                default_paths = [
                    "D:/webdataset/models/so8t-phi4-so8t-ja-finetuned",
                    "models/so8t-phi4-so8t-ja-finetuned",
                    "so8t-mmllm/models/so8t-phi4-so8t-ja-finetuned"
                ]
                for path in default_paths:
                    if Path(path).exists():
                        model_path = path
                        break
                
                if model_path is None:
                    raise FileNotFoundError("SO8T model not found")
            
            logger.info(f"[SO8T] Loading model from: {model_path}")
            
            # トークナイザーを読み込み
            self.so8t_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # SO8T設定（簡易版）
            from so8t_mmllm.src.models.safety_aware_so8t import SafetyAwareSO8TConfig
            so8t_config = SafetyAwareSO8TConfig(
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=8,
                intermediate_size=11008,
                max_position_embeddings=4096,
                use_so8_rotation=True,
                use_safety_head=True,
                use_verifier_head=True
            )
            
            # SO8TThinkingModelを読み込み
            self.so8t_model = SO8TThinkingModel(
                base_model_name_or_path=model_path,
                so8t_config=so8t_config,
                use_redacted_tokens=False,
                use_quadruple_thinking=True
            )
            
            # トークナイザーを設定
            self.so8t_model.set_tokenizer(self.so8t_tokenizer)
            
            # 評価モードに設定
            self.so8t_model.eval()
            
            logger.info("[SO8T] Model loaded successfully")
            
        except Exception as e:
            logger.error(f"[SO8T] Failed to initialize model: {e}")
            raise
    
    async def so8t_control_scraping_action(
        self,
        action_type: str,
        context: Dict[str, any]
    ) -> Dict[str, any]:
        """
        SO8Tモデルを使ってスクレイピング動作を統制
        
        Args:
            action_type: 動作タイプ（'search', 'scrape', 'bypass', 'continue'）
            context: コンテキスト情報（URL、キーワード、検出されたチェックなど）
        
        Returns:
            統制結果（'allow', 'deny', 'modify'）と推論結果
        """
        if not self.use_so8t_control or self.so8t_model is None:
            # SO8T統制が無効な場合は許可
            return {'decision': 'allow', 'reasoning': 'SO8T control disabled'}
        
        try:
            # プロンプト構築
            prompt = self._build_so8t_control_prompt(action_type, context)
            
            # SO8T四重推論を実行
            result = await asyncio.to_thread(
                self.so8t_model.generate_thinking,
                self.so8t_tokenizer,
                prompt,
                max_new_tokens=256,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                device="cuda" if (TORCH_AVAILABLE and torch and torch.cuda.is_available()) else "cpu"
            )
            
            # 四重推論を抽出
            if self.so8t_model.use_quadruple_thinking:
                task_text, safety_text, policy_text, final_text = extract_quadruple_thinking(
                    result.get('full_text', '')
                )
                
                # 最終回答から意思決定を抽出
                decision = self._extract_decision_from_final(final_text)
                
                return {
                    'decision': decision,
                    'task_reasoning': task_text,
                    'safety_reasoning': safety_text,
                    'policy_reasoning': policy_text,
                    'final_reasoning': final_text,
                    'raw_output': result
                }
            else:
                # 基本形式の場合
                thinking_text = result.get('thinking', '')
                final_text = result.get('final', '')
                decision = self._extract_decision_from_final(final_text)
                
                return {
                    'decision': decision,
                    'thinking': thinking_text,
                    'final_reasoning': final_text,
                    'raw_output': result
                }
                
        except Exception as e:
            logger.error(f"[SO8T] Control failed: {e}")
            # エラー時は許可（安全側に倒す）
            return {'decision': 'allow', 'reasoning': f'SO8T control error: {e}'}
    
    def _build_so8t_control_prompt(self, action_type: str, context: Dict[str, any]) -> str:
        """SO8T統制用プロンプトを構築"""
        url = context.get('url', 'unknown')
        keyword = context.get('keyword', 'unknown')
        checks = context.get('checks', {})
        
        if action_type == 'search':
            prompt = f"""以下のWebスクレイピング動作を評価し、実行を許可するか判断してください。

動作タイプ: 検索
キーワード: {keyword}
URL: {url}

四重推論を行い、以下を判断してください：
1. <think-task>: この検索がタスクに適切か、ドメイン知識の収集に有効か
2. <think-safety>: この検索が安全か、NSFWコンテンツや危険なサイトにアクセスしないか
3. <think-policy>: この検索がポリシーに準拠しているか、利用規約を遵守しているか
4. <final>: 実行を許可するか（allow/deny/modify）と理由

回答: <think-task>"""
        
        elif action_type == 'scrape':
            prompt = f"""以下のWebスクレイピング動作を評価し、実行を許可するか判断してください。

動作タイプ: スクレイピング
URL: {url}
キーワード: {keyword}

四重推論を行い、以下を判断してください：
1. <think-task>: このページのスクレイピングがタスクに適切か
2. <think-safety>: このページが安全か、NSFWコンテンツや危険な情報を含まないか
3. <think-policy>: このページのスクレイピングがポリシーに準拠しているか、robots.txtを遵守しているか
4. <final>: 実行を許可するか（allow/deny/modify）と理由

回答: <think-task>"""
        
        elif action_type == 'bypass':
            detected_checks = [k for k, v in checks.items() if v]
            prompt = f"""以下のバックグラウンドチェック突破動作を評価し、実行を許可するか判断してください。

動作タイプ: チェック突破
検出されたチェック: {', '.join(detected_checks) if detected_checks else 'なし'}
URL: {url}

四重推論を行い、以下を判断してください：
1. <think-task>: このチェック突破がタスクに必要か
2. <think-safety>: このチェック突破が安全か、違法行為や不正アクセスにならないか
3. <think-policy>: このチェック突破がポリシーに準拠しているか、利用規約を遵守しているか
4. <final>: 実行を許可するか（allow/deny/modify）と理由

回答: <think-task>"""
        
        else:
            prompt = f"""以下のWebスクレイピング動作を評価し、実行を許可するか判断してください。

動作タイプ: {action_type}
URL: {url}
キーワード: {keyword}

四重推論を行い、以下を判断してください：
1. <think-task>: この動作がタスクに適切か
2. <think-safety>: この動作が安全か
3. <think-policy>: この動作がポリシーに準拠しているか
4. <final>: 実行を許可するか（allow/deny/modify）と理由

回答: <think-task>"""
        
        return prompt
    
    def _extract_decision_from_final(self, final_text: str) -> str:
        """最終回答から意思決定を抽出"""
        final_lower = final_text.lower()
        
        if 'deny' in final_lower or '拒否' in final_lower or '禁止' in final_lower:
            return 'deny'
        elif 'modify' in final_lower or '修正' in final_lower or '変更' in final_lower:
            return 'modify'
        else:
            return 'allow'
    
    def generate_related_keywords(self, keyword: str, language: str, category: str = None, count: int = 3) -> List[str]:
        """
        関連キーワードを生成（統合版）
        
        Args:
            keyword: 元のキーワード
            language: 言語
            category: カテゴリ（オプション）
            count: 生成するキーワード数
        
        Returns:
            関連キーワードのリスト
        """
        # NSFW検知用カテゴリの場合は特別な処理
        if category and category.startswith('nsfw_detection'):
            if language == 'ja':
                patterns = [
                    f"{keyword} 検索",
                    f"{keyword} 一覧",
                    f"{keyword} ランキング",
                    f"{keyword} 新着",
                    f"{keyword} 人気",
                ]
            else:
                patterns = [
                    f"{keyword} search",
                    f"{keyword} list",
                    f"{keyword} ranking",
                    f"{keyword} new",
                    f"{keyword} popular",
                ]
            return patterns[:count]
        
        # 通常のカテゴリの場合
        variations = []
        
        if language == 'ja':
            # 日本語のキーワードバリエーション
            variations = [
                f"{keyword} とは",
                f"{keyword} について",
                f"{keyword} 方法",
                f"{keyword} 解説",
                f"{keyword} まとめ",
                f"{keyword} 最新",
                f"{keyword} 2024",
                f"{keyword} 2025",
                f"{keyword} の意味",
                f"{keyword} の歴史",
                f"{keyword} の応用",
                f"{keyword} の研究",
            ]
            
            # カテゴリに応じた追加バリエーション
            if category:
                if 'technology' in category or 'science' in category:
                    variations.extend([
                        f"{keyword} 技術",
                        f"{keyword} 研究",
                        f"{keyword} 開発"
                    ])
                elif 'medical' in category or 'health' in category:
                    variations.extend([
                        f"{keyword} 治療",
                        f"{keyword} 症状",
                        f"{keyword} 診断"
                    ])
        else:
            # 英語のキーワードバリエーション
            variations = [
                f"what is {keyword}",
                f"{keyword} explained",
                f"{keyword} tutorial",
                f"{keyword} guide",
                f"{keyword} overview",
                f"{keyword} latest",
                f"{keyword} 2024",
                f"{keyword} 2025",
                f"{keyword} meaning",
                f"{keyword} definition",
                f"{keyword} history",
                f"{keyword} applications",
                f"{keyword} research",
            ]
            
            # カテゴリに応じた追加バリエーション
            if category:
                if 'technology' in category or 'science' in category:
                    variations.extend([
                        f"{keyword} technology",
                        f"{keyword} research",
                        f"{keyword} development"
                    ])
                elif 'medical' in category or 'health' in category:
                    variations.extend([
                        f"{keyword} treatment",
                        f"{keyword} symptoms",
                        f"{keyword} diagnosis"
                    ])
        
        # ランダムに選択
        return random.sample(variations, min(count, len(variations)))
    
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
    
    def initialize_keyword_queue(self):
        """キーワードキューを初期化（コーディング関連キーワードを優先）"""
        logger.info("[QUEUE] Initializing keyword queue...")
        
        # 優先カテゴリ（コーディング関連）
        priority_categories = [
            'programming_languages', 'algorithms', 'hardware', 
            'coding_best_practices', 'engineering_sites'
        ]
        
        # 優先カテゴリを最初に追加
        for language in ['ja', 'en']:
            for category in priority_categories:
                if category in CATEGORY_KEYWORDS[language]:
                    keywords = CATEGORY_KEYWORDS[language][category]
                    for keyword in keywords:
                        related_keywords = self.generate_related_keywords(keyword, language, category)
                        task = KeywordTask(
                            keyword=keyword,
                            category=category,
                            language=language,
                            related_keywords=related_keywords,
                            coding_related=True  # コーディング関連フラグ
                        )
                        self.keyword_queue.append(task)
        
        # その他のカテゴリを追加
        for language in ['ja', 'en']:
            for category, keywords in CATEGORY_KEYWORDS[language].items():
                # 優先カテゴリは既に追加済みなのでスキップ
                if category in priority_categories:
                    continue
                
                # NSFW検知用サイトは別途処理
                if category.startswith('nsfw_detection'):
                    continue
                
                for keyword in keywords:
                    related_keywords = self.generate_related_keywords(keyword, language, category)
                    task = KeywordTask(
                        keyword=keyword,
                        category=category,
                        language=language,
                        related_keywords=related_keywords,
                        coding_related=False
                    )
                    self.keyword_queue.append(task)
        
        # NSFW検知用サイトURLをキューに追加
        logger.info("[QUEUE] Adding NSFW detection sites (detection purpose only)...")
        for language in ['ja', 'en']:
            if language in NSFW_DETECTION_SITES:
                for site_category, urls in NSFW_DETECTION_SITES[language].items():
                    for url in urls:
                        # URLからキーワードを抽出
                        parsed = urlparse(url)
                        domain_keyword = parsed.netloc.replace('www.', '').split('.')[0]
                        
                        related_keywords = self.generate_related_keywords(domain_keyword, language, f'nsfw_detection_{site_category}')
                        task = KeywordTask(
                            keyword=domain_keyword,
                            category=f'nsfw_detection_{site_category}',
                            language=language,
                            related_keywords=related_keywords
                        )
                        # URLをキーワードとして保存（後で使用）
                        task.url = url
                        self.keyword_queue.append(task)
        
        logger.info(f"[QUEUE] Initialized {len(self.keyword_queue)} keywords (including NSFW detection sites)")
        
        # 日経225企業タスクを追加
        if self.nikkei225_enabled:
            self._initialize_nikkei225_queue()
    
    def _load_nikkei225_companies(self) -> List[Dict]:
        """日経225企業リストを読み込む"""
        possible_paths = [
            PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "nikkei225_sources.json",
            PROJECT_ROOT / "scripts" / "data" / "nikkei225_sources.json",
            Path("so8t-mmllm/scripts/data/nikkei225_sources.json"),
            Path("scripts/data/nikkei225_sources.json"),
        ]
        
        nikkei_file = None
        for path in possible_paths:
            if Path(path).exists():
                nikkei_file = Path(path)
                break
        
        if not nikkei_file:
            logger.warning(f"[NIKKEI225] Nikkei225 file not found. Tried: {[str(p) for p in possible_paths]}")
            return []
        
        try:
            with open(nikkei_file, 'r', encoding='utf-8') as f:
                nikkei_data = json.load(f)
            
            companies = nikkei_data.get('nikkei225_companies', [])
            logger.info(f"[NIKKEI225] Loaded {len(companies)} companies from {nikkei_file}")
            return companies
        
        except Exception as e:
            logger.error(f"[NIKKEI225] Failed to load Nikkei225: {e}")
            return []
    
    def _filter_target_companies(self, companies: List[Dict]) -> List[Dict]:
        """防衛・航空宇宙・インフラ企業をフィルタリング"""
        target_domains = [
            'heavy_industry',  # 重工業（三菱重工、川崎重工、IHI等）
            'airline',  # 航空会社（ANA、JAL等）
            'transport',  # 運輸（JR各社等）
            'defense',  # 防衛
            'aerospace',  # 航空宇宙
            'infrastructure',  # インフラ
            'shipping',  # 海運（日本郵船、商船三井等）
            'utility',  # 電力・ガス（インフラ関連）
        ]
        
        filtered = []
        for company in companies:
            domain = company.get('domain', '')
            if domain in target_domains:
                filtered.append(company)
        
        logger.info(f"[NIKKEI225] Filtered {len(filtered)} target companies (defense/aerospace/infrastructure) from {len(companies)} total")
        return filtered
    
    def _generate_company_urls(self, company: Dict) -> Dict[str, str]:
        """企業の各種URLを生成"""
        base_url = company.get('url', '').rstrip('/')
        company_code = company.get('code', '')
        
        urls = {}
        
        # IRページ（財務報告・決算情報）
        # 一般的なパターンを試す（実際のスクレイピング時に404の場合は別パターンを試す）
        urls['financial_reports'] = base_url + '/ir/'
        
        # プレスリリースページ
        urls['press_releases'] = base_url + '/news/'
        
        # 製品・サービス情報（企業のメインページまたは製品ページ）
        urls['product_info'] = base_url
        
        # 日経企業情報ページ
        if company_code:
            urls['nikkei_company_info'] = f"https://www.nikkei.com/nkd/company/?scode={company_code}"
        else:
            urls['nikkei_company_info'] = None
        
        return urls
    
    def _initialize_nikkei225_queue(self):
        """日経225企業キューを初期化"""
        logger.info("[NIKKEI225] Initializing Nikkei225 company queue...")
        
        # 企業リストを読み込む
        all_companies = self._load_nikkei225_companies()
        if not all_companies:
            logger.warning("[NIKKEI225] No companies loaded, skipping Nikkei225 scraping")
            return
        
        # 防衛・航空宇宙・インフラ企業をフィルタリング（全企業も含める）
        target_companies = all_companies  # 全企業を対象とする
        
        # 企業ごとにタスクを作成
        for company in target_companies:
            company_name = company.get('name', '')
            company_code = company.get('code', '')
            company_domain = company.get('domain', '')
            base_url = company.get('url', '')
            
            # 各種URLを生成
            urls = self._generate_company_urls(company)
            
            # 各データタイプごとにタスクを作成
            for data_type, url in urls.items():
                if url is None:
                    continue
                
                task = KeywordTask(
                    keyword=f"{company_name}_{data_type}",
                    category='nikkei225',
                    language='ja',
                    url=url,
                    company_name=company_name,
                    company_code=company_code,
                    company_domain=company_domain,
                    data_type=data_type
                )
                self.keyword_queue.append(task)
        
        logger.info(f"[NIKKEI225] Added {len(target_companies)} companies × 4 data types = {len([t for t in self.keyword_queue if t.category == 'nikkei225'])} tasks")
    
    async def scrape_company_page(self, page: Page, task: KeywordTask) -> Optional[Dict]:
        """企業ページをスクレイピング（人間を模倣した動作付き）"""
        try:
            url = task.url
            if not url:
                logger.warning(f"[COMPANY] No URL for task: {task.keyword}")
                return None
            
            logger.info(f"[COMPANY] Scraping {task.data_type} for {task.company_name}: {url}")
            
            # ページに移動
            await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            
            # 人間を模倣した待機
            await asyncio.sleep(random.uniform(2.0, 4.0))
            
            # 人間を模倣したページ閲覧動作
            await self.human_like_page_view(page)
            
            # バックグラウンドチェックを検出・突破
            checks = await self.detect_background_checks(page)
            if any(checks.values()):
                logger.info(f"[COMPANY] Background checks detected for {url}, attempting bypass...")
                await self.bypass_background_checks(page, checks)
                await asyncio.sleep(random.uniform(2.0, 3.0))
            
            # HTMLを取得
            html = await page.content()
            
            # BeautifulSoupでパース
            if BS4_AVAILABLE:
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                title = soup.title.string if soup.title else ""
            else:
                text = html
                title = ""
            
            # サンプルを作成
            sample = {
                'text': text[:50000],  # 最大50000文字
                'url': url,
                'title': title,
                'keyword': task.keyword,
                'category': task.category,
                'language': task.language,
                'company_name': task.company_name,
                'company_code': task.company_code,
                'company_domain': task.company_domain,
                'data_type': task.data_type,
                'source': 'nikkei225_company',
                'crawled_at': datetime.now().isoformat(),
                'text_length': len(text)
            }
            
            # カテゴリ別に分類
            if task.data_type in self.nikkei225_samples:
                self.nikkei225_samples[task.data_type].append(sample)
            
            return sample
            
        except PlaywrightTimeoutError:
            logger.warning(f"[COMPANY] Timeout while scraping {url}")
            return None
        except Exception as e:
            logger.error(f"[COMPANY] Failed to scrape {url}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    async def check_cursor_browser_running(self, port: int) -> bool:
        """Cursorブラウザが起動しているかチェック"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/json/version", timeout=aiohttp.ClientTimeout(total=2.0)) as response:
                    if response.status == 200:
                        return True
        except Exception:
            pass
        return False
    
    async def launch_cursor_browser_background(self, port: int) -> bool:
        """Cursorブラウザをバックグラウンドで起動"""
        
        try:
            # Cursorブラウザのパスを探す
            cursor_paths = [
                r"C:\Users\{}\AppData\Local\Programs\cursor\Cursor.exe".format(os.environ.get('USERNAME', 'downl')),
                r"C:\Program Files\Cursor\Cursor.exe",
                r"C:\Program Files (x86)\Cursor\Cursor.exe",
            ]
            
            cursor_exe = None
            for path in cursor_paths:
                if Path(path).exists():
                    cursor_exe = path
                    break
            
            if not cursor_exe:
                logger.warning(f"[BROWSER] Cursor executable not found in standard locations")
                return False
            
            # リモートデバッグポートを指定してCursorブラウザを起動（ボット検知回避設定付き）
            cmd = [
                cursor_exe,
                f"--remote-debugging-port={port}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-blink-features=AutomationControlled",  # AutomationControlledを無効化
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--disable-notifications",
                "--disable-popup-blocking",
                "--start-maximized",
            ]
            
            logger.info(f"[BROWSER] Launching Cursor browser in background on port {port}...")
            if platform.system() == "Windows":
                # WindowsではCREATE_NO_WINDOWフラグを使用してバックグラウンド起動
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # ブラウザが起動するまで待機（最大10秒）
            for i in range(20):
                await asyncio.sleep(0.5)
                if await self.check_cursor_browser_running(port):
                    logger.info(f"[OK] Cursor browser launched successfully on port {port}")
                    return True
            
            logger.warning(f"[BROWSER] Cursor browser may not have started properly")
            return False
            
        except Exception as e:
            logger.error(f"[BROWSER] Failed to launch Cursor browser: {e}")
            return False
    
    async def connect_to_cursor_browser(self, playwright, browser_index: int) -> Optional[Browser]:
        """Cursorのブラウザに接続（バックグラウンド起動と自動再開機能付き）"""
        if not self.use_cursor_browser:
            logger.info(f"[BROWSER {browser_index}] Launching new browser with anti-detection settings...")
            # ボット検知回避のためのChrome起動オプション
            browser = await playwright.chromium.launch(
                headless=False,
                args=[
                    '--disable-blink-features=AutomationControlled',  # AutomationControlledを無効化
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials',
                    '--disable-infobars',
                    '--disable-notifications',
                    '--disable-popup-blocking',
                    '--start-maximized',
                    '--window-size=1920,1080',
                ],
                ignore_default_args=['--enable-automation'],  # --enable-automationを削除
                channel='chrome' if platform.system() == 'Windows' else None  # WindowsではChromeを使用
            )
            logger.info(f"[OK] Browser {browser_index} launched with anti-detection settings")
            return browser
        
        # 複数のブラウザインスタンスに対応するため、ポートを分散
        port = self.remote_debugging_port + browser_index
        
        # まず、既存のブラウザに接続を試みる
        try:
            logger.info(f"[BROWSER {browser_index}] Attempting to connect to Cursor browser on port {port}...")
            
            # CDPエンドポイントURL
            cdp_endpoint = f"http://127.0.0.1:{port}"
            
            browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
            
            # 接続確認
            contexts = browser.contexts
            if contexts:
                logger.info(f"[OK] Browser {browser_index} connected (found {len(contexts)} contexts)")
            else:
                logger.info(f"[INFO] Browser {browser_index}: No existing contexts, creating new context...")
                await browser.new_context()
                logger.info(f"[OK] Browser {browser_index}: New context created")
            
            return browser
            
        except Exception as e:
            logger.warning(f"[BROWSER {browser_index}] Failed to connect to Cursor browser: {e}")
            
            # ブラウザが起動していない場合は、バックグラウンドで起動を試みる
            logger.info(f"[BROWSER {browser_index}] Attempting to launch Cursor browser in background...")
            browser_running = await self.check_cursor_browser_running(port)
            
            if not browser_running:
                launch_success = await self.launch_cursor_browser_background(port)
                if launch_success:
                    # ブラウザが起動したので、再接続を試みる
                    await asyncio.sleep(2.0)  # 起動を待つ
                    try:
                        cdp_endpoint = f"http://127.0.0.1:{port}"
                        browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
                        contexts = browser.contexts
                        if contexts:
                            logger.info(f"[OK] Browser {browser_index} connected after background launch (found {len(contexts)} contexts)")
                        else:
                            await browser.new_context()
                            logger.info(f"[OK] Browser {browser_index}: New context created after background launch")
                        return browser
                    except Exception as reconnect_error:
                        logger.warning(f"[BROWSER {browser_index}] Failed to connect after background launch: {reconnect_error}")
            
            # すべての試行が失敗した場合は、新しいブラウザを起動
            logger.info(f"[BROWSER {browser_index}] Falling back to launching new browser with anti-detection settings...")
            browser = await playwright.chromium.launch(
                headless=False,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials',
                    '--disable-infobars',
                    '--disable-notifications',
                    '--disable-popup-blocking',
                    '--start-maximized',
                    '--window-size=1920,1080',
                ],
                ignore_default_args=['--enable-automation'],
                channel='chrome' if platform.system() == 'Windows' else None
            )
            logger.info(f"[OK] Browser {browser_index} launched with anti-detection settings")
            return browser
    
    def save_browser_state(self):
        """ブラウザ状態を保存"""
        try:
            state = {
                'browser_status': self.browser_status,
                'completed_keywords': list(self.completed_keywords),
                'total_samples': len(self.all_samples),
                'timestamp': datetime.now().isoformat()
            }
            
            state_file = self.checkpoint_dir / f"browser_state_{self.session_id}.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"[STATE] Browser state saved to {state_file}")
        except Exception as e:
            logger.warning(f"[STATE] Failed to save browser state: {e}")
    
    def load_browser_state(self) -> Optional[Dict]:
        """ブラウザ状態を読み込み"""
        try:
            # 最新の状態ファイルを探す
            state_files = list(self.checkpoint_dir.glob("browser_state_*.json"))
            if not state_files:
                return None
            
            # 最新のファイルを取得
            latest_state_file = max(state_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            logger.info(f"[STATE] Browser state loaded from {latest_state_file}")
            return state
        except Exception as e:
            logger.warning(f"[STATE] Failed to load browser state: {e}")
            return None
    
    async def human_like_mouse_move(self, page: Page, start_x: int, start_y: int, end_x: int, end_y: int):
        """人間らしいマウス移動（ベジェ曲線のような滑らかな動き）"""
        try:
            # ベジェ曲線のような滑らかな軌跡を生成
            steps = random.randint(8, 15)
            for i in range(steps):
                t = i / steps
                # ベジェ曲線の制御点をランダムに生成
                control_x1 = start_x + (end_x - start_x) * 0.3 + random.randint(-50, 50)
                control_y1 = start_y + (end_y - start_y) * 0.3 + random.randint(-50, 50)
                control_x2 = start_x + (end_x - start_x) * 0.7 + random.randint(-50, 50)
                control_y2 = start_y + (end_y - start_y) * 0.7 + random.randint(-50, 50)
                
                # ベジェ曲線の計算
                x = (1-t)**3 * start_x + 3*(1-t)**2*t * control_x1 + 3*(1-t)*t**2 * control_x2 + t**3 * end_x
                y = (1-t)**3 * start_y + 3*(1-t)**2*t * control_y1 + 3*(1-t)*t**2 * control_y2 + t**3 * end_y
                
                await page.mouse.move(int(x), int(y))
                await asyncio.sleep(random.uniform(0.01, 0.03))
        except Exception as e:
            logger.debug(f"[MOUSE] Mouse move failed: {e}")
    
    async def human_like_scroll(self, page: Page, direction: str = 'down'):
        """人間らしいスクロール動作（段階的、不規則な速度）"""
        try:
            viewport = page.viewport_size
            if not viewport:
                return
            
            scroll_amount = random.randint(200, 600)
            steps = random.randint(3, 8)
            step_size = scroll_amount // steps
            
            for i in range(steps):
                # 不規則な速度でスクロール
                speed_factor = random.uniform(0.7, 1.3)
                actual_step = int(step_size * speed_factor)
                
                if direction == 'down':
                    await page.mouse.wheel(0, actual_step)
                else:
                    await page.mouse.wheel(0, -actual_step)
                
                # ランダムな待機時間
                await asyncio.sleep(random.uniform(0.1, 0.4))
            
            # スクロール後の待機
            await asyncio.sleep(random.uniform(0.5, 1.5))
        except Exception as e:
            logger.debug(f"[SCROLL] Scroll failed: {e}")
    
    async def human_like_typing(self, page: Page, element, text: str):
        """人間らしいタイピング（不規則な速度、誤入力と修正）"""
        try:
            await element.click()
            await asyncio.sleep(random.uniform(0.3, 0.8))
            
            for i, char in enumerate(text):
                # タイピング速度のばらつき
                base_delay = random.uniform(80, 200)
                
                # 時々、より長い待機（考えている様子）
                if random.random() < 0.1:  # 10%の確率
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                
                # タイピング
                await element.type(char, delay=int(base_delay))
                
                # 時々、誤入力と修正をシミュレート
                if random.random() < 0.05 and i > 0:  # 5%の確率
                    await asyncio.sleep(random.uniform(0.2, 0.5))
                    await element.press('Backspace')
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    await element.type(char, delay=int(base_delay))
                
                # 文字間の待機時間
                await asyncio.sleep(random.uniform(0.05, 0.2))
            
            # 入力完了後の待機（確認している様子）
            await asyncio.sleep(random.uniform(0.3, 0.8))
        except Exception as e:
            logger.debug(f"[TYPING] Typing failed: {e}")
    
    async def detect_background_checks(self, page: Page) -> Dict[str, bool]:
        """バックグラウンドチェックを検出"""
        checks = {
            'captcha': False,
            'bot_detection': False,
            'security_check': False,
            'rate_limit': False,
            'cloudflare': False
        }
        
        try:
            # ページのHTMLを取得
            html = await page.content()
            html_lower = html.lower()
            
            # CAPTCHA検出
            captcha_indicators = ['captcha', 'recaptcha', 'hcaptcha', 'turnstile', 'challenge']
            if any(indicator in html_lower for indicator in captcha_indicators):
                checks['captcha'] = True
                logger.warning("[CHECK] CAPTCHA detected")
            
            # ボット検知検出
            bot_indicators = ['bot detection', 'automated', 'verify you are human', 'access denied']
            if any(indicator in html_lower for indicator in bot_indicators):
                checks['bot_detection'] = True
                logger.warning("[CHECK] Bot detection detected")
            
            # Cloudflare検出
            cloudflare_indicators = ['cloudflare', 'checking your browser', 'ddos protection', 'ray id']
            if any(indicator in html_lower for indicator in cloudflare_indicators):
                checks['cloudflare'] = True
                logger.warning("[CHECK] Cloudflare protection detected")
            
            # セキュリティチェック検出
            security_indicators = ['security check', 'verification', 'please wait', 'loading']
            if any(indicator in html_lower for indicator in security_indicators):
                # 待機してから再チェック
                await asyncio.sleep(random.uniform(2.0, 4.0))
                html_after = await page.content()
                if any(indicator in html_after.lower() for indicator in security_indicators):
                    checks['security_check'] = True
                    logger.warning("[CHECK] Security check detected")
            
            # レート制限検出
            rate_limit_indicators = ['rate limit', 'too many requests', '429', 'slow down']
            if any(indicator in html_lower for indicator in rate_limit_indicators):
                checks['rate_limit'] = True
                logger.warning("[CHECK] Rate limit detected")
            
        except Exception as e:
            logger.debug(f"[CHECK] Detection failed: {e}")
        
        return checks
    
    async def bypass_background_checks(self, page: Page, checks: Dict[str, bool]) -> bool:
        """バックグラウンドチェックを突破"""
        try:
            # CAPTCHA突破
            if checks.get('captcha'):
                logger.info("[BYPASS] Attempting to bypass CAPTCHA...")
                # CAPTCHA要素を探す
                captcha_selectors = [
                    'iframe[src*="recaptcha"]',
                    'iframe[src*="hcaptcha"]',
                    'div[id*="captcha"]',
                    'div[class*="captcha"]'
                ]
                
                for selector in captcha_selectors:
                    try:
                        captcha_element = await page.query_selector(selector)
                        if captcha_element:
                            # CAPTCHAの近くにマウスを移動（人間らしい動作）
                            box = await captcha_element.bounding_box()
                            if box:
                                await self.human_like_mouse_move(
                                    page,
                                    100, 100,
                                    int(box['x'] + box['width'] / 2),
                                    int(box['y'] + box['height'] / 2)
                                )
                                await asyncio.sleep(random.uniform(1.0, 2.0))
                                
                                # 待機（CAPTCHAが自動解決されるのを待つ）
                                await asyncio.sleep(random.uniform(5.0, 10.0))
                    except Exception:
                        continue
            
            # Cloudflare突破
            if checks.get('cloudflare'):
                logger.info("[BYPASS] Attempting to bypass Cloudflare...")
                # Cloudflareのチェックを待つ
                await asyncio.sleep(random.uniform(3.0, 6.0))
                
                # ページを再読み込み（人間がリロードする様子）
                if random.random() < 0.5:
                    await page.reload(wait_until="networkidle")
                    await asyncio.sleep(random.uniform(2.0, 4.0))
            
            # ボット検知突破
            if checks.get('bot_detection'):
                logger.info("[BYPASS] Attempting to bypass bot detection...")
                # より人間らしい動作を追加
                await self.enhanced_human_behavior(page)
            
            # セキュリティチェック突破
            if checks.get('security_check'):
                logger.info("[BYPASS] Attempting to bypass security check...")
                # セキュリティチェックのボタンを探す
                security_selectors = [
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button:has-text("Continue")',
                    'button:has-text("Verify")',
                    'a:has-text("Continue")'
                ]
                
                for selector in security_selectors:
                    try:
                        security_button = await page.query_selector(selector)
                        if security_button:
                            # ボタンに人間らしく移動してクリック
                            box = await security_button.bounding_box()
                            if box:
                                await self.human_like_mouse_move(
                                    page,
                                    100, 100,
                                    int(box['x'] + box['width'] / 2),
                                    int(box['y'] + box['height'] / 2)
                                )
                                await asyncio.sleep(random.uniform(0.5, 1.0))
                                await security_button.click()
                                await asyncio.sleep(random.uniform(2.0, 4.0))
                                break
                    except Exception:
                        continue
            
            # レート制限突破
            if checks.get('rate_limit'):
                logger.info("[BYPASS] Rate limit detected, waiting...")
                # レート制限の場合は長めに待機
                wait_time = random.uniform(30.0, 60.0)
                logger.info(f"[BYPASS] Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
            
            return True
            
        except Exception as e:
            logger.error(f"[BYPASS] Failed to bypass checks: {e}")
            return False
    
    async def enhanced_human_behavior(self, page: Page):
        """より高度な人間を模倣した動作（検知回避）"""
        try:
            # 1. ランダムなマウス軌跡（より複雑な動き）
            viewport = page.viewport_size
            if viewport:
                for _ in range(random.randint(3, 6)):
                    x = random.randint(0, viewport['width'])
                    y = random.randint(0, viewport['height'])
                    await self.human_like_mouse_move(page, 100, 100, x, y)
                    await asyncio.sleep(random.uniform(0.2, 0.5))
            
            # 2. キーボード入力のシミュレート（タブキー、矢印キーなど）
            await page.keyboard.press('Tab')
            await asyncio.sleep(random.uniform(0.3, 0.6))
            await page.keyboard.press('Tab')
            await asyncio.sleep(random.uniform(0.3, 0.6))
            
            # 3. ウィンドウフォーカスのシミュレート
            await page.evaluate("window.focus()")
            await asyncio.sleep(random.uniform(0.5, 1.0))
            
            # 4. スクロールの不規則な動き
            for _ in range(random.randint(2, 4)):
                scroll_amount = random.randint(-300, 300)
                await page.mouse.wheel(0, scroll_amount)
                await asyncio.sleep(random.uniform(0.3, 0.8))
            
            # 5. ページ要素への複数回ホバー
            elements = await page.query_selector_all('a, button, input')
            if elements:
                hover_elements = random.sample(elements, min(5, len(elements)))
                for element in hover_elements:
                    try:
                        await element.hover()
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                    except Exception:
                        continue
            
            # 6. ランダムな待機（人間が考えている様子）
            await asyncio.sleep(random.uniform(2.0, 5.0))
            
        except Exception as e:
            logger.debug(f"[ENHANCED] Enhanced behavior failed: {e}")
    
    async def human_like_page_view(self, page: Page):
        """人間らしいページ閲覧動作（バックグラウンドチェック対応）"""
        try:
            # バックグラウンドチェックを検出
            checks = await self.detect_background_checks(page)
            
            # チェックが検出された場合は突破を試みる
            if any(checks.values()):
                logger.info("[CHECK] Background checks detected, attempting bypass...")
                
                # SO8T統制: チェック突破動作を評価
                if self.use_so8t_control:
                    bypass_context = {
                        'url': page.url,
                        'checks': checks
                    }
                    control_result = await self.so8t_control_scraping_action('bypass', bypass_context)
                    
                    if control_result['decision'] == 'deny':
                        logger.warning(f"[SO8T] Bypass denied for URL: {page.url}")
                        logger.warning(f"[SO8T] Reasoning: {control_result.get('final_reasoning', 'No reasoning')}")
                        # 突破を拒否された場合は待機してから再試行
                        await asyncio.sleep(random.uniform(10.0, 20.0))
                        return
                    elif control_result['decision'] == 'modify':
                        logger.info(f"[SO8T] Bypass modified for URL: {page.url}")
                        # 修正された動作を適用（簡易版：待機時間を増やす）
                        await asyncio.sleep(random.uniform(5.0, 10.0))
                
                await self.bypass_background_checks(page, checks)
                
                # 突破後に再チェック
                await asyncio.sleep(random.uniform(1.0, 2.0))
                checks_after = await self.detect_background_checks(page)
                if any(checks_after.values()):
                    logger.warning("[CHECK] Some checks still active, applying enhanced behavior...")
                    await self.enhanced_human_behavior(page)
            
            # ページ読み込み後の待機（内容を読んでいる様子）
            await asyncio.sleep(random.uniform(2.0, 5.0))
            
            # ランダムにスクロール
            scroll_directions = ['down', 'down', 'down', 'up']  # 下にスクロールが多い
            for _ in range(random.randint(2, 5)):
                direction = random.choice(scroll_directions)
                await self.human_like_scroll(page, direction)
                await asyncio.sleep(random.uniform(1.0, 3.0))
                
                # スクロール中にもチェックを検出
                checks_during = await self.detect_background_checks(page)
                if any(checks_during.values()):
                    await self.bypass_background_checks(page, checks_during)
            
            # ページ要素へのランダムなホバー
            elements = await page.query_selector_all('a, button, img, h1, h2, h3')
            if elements:
                hover_count = random.randint(1, min(3, len(elements)))
                hover_elements = random.sample(elements, hover_count)
                
                for element in hover_elements:
                    try:
                        box = await element.bounding_box()
                        if box:
                            # 要素の中心にマウスを移動
                            center_x = box['x'] + box['width'] / 2
                            center_y = box['y'] + box['height'] / 2
                            
                            # 現在のマウス位置を取得（簡易版）
                            await page.mouse.move(int(center_x), int(center_y))
                            await asyncio.sleep(random.uniform(0.5, 1.5))
                    except Exception:
                        continue
            
            # 最終的な待機（内容を読んでいる様子）
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # 最終チェック
            final_checks = await self.detect_background_checks(page)
            if any(final_checks.values()):
                logger.warning("[CHECK] Final checks detected, applying enhanced bypass...")
                await self.enhanced_human_behavior(page)
                await self.bypass_background_checks(page, final_checks)
                
        except Exception as e:
            logger.debug(f"[PAGE VIEW] Page view failed: {e}")
    
    def get_search_engine_url(self, keyword: str, language: str, engine: str = None) -> str:
        """
        検索エンジンのURLを取得
        
        Args:
            keyword: 検索キーワード
            language: 言語
            engine: 検索エンジン（'google', 'bing', 'duckduckgo'）Noneの場合はランダム選択
        
        Returns:
            検索URL
        """
        if engine is None:
            # ランダムに検索エンジンを選択
            engines = ['google', 'bing', 'duckduckgo']
            engine = random.choice(engines)
        
        keyword_encoded = quote(keyword)
        
        if engine == 'google':
            if language == 'ja':
                return f"https://www.google.com/search?hl=ja&q={keyword_encoded}"
            else:
                return f"https://www.google.com/search?q={keyword_encoded}"
        elif engine == 'bing':
            if language == 'ja':
                return f"https://www.bing.com/search?q={keyword_encoded}&setlang=ja"
            else:
                return f"https://www.bing.com/search?q={keyword_encoded}"
        elif engine == 'duckduckgo':
            return f"https://html.duckduckgo.com/html/?q={keyword_encoded}"
        else:
            # デフォルトはGoogle
            if language == 'ja':
                return f"https://www.google.com/search?hl=ja&q={keyword_encoded}"
            else:
                return f"https://www.google.com/search?q={keyword_encoded}"
    
    
    async def human_like_search(self, page: Page, keyword: str, language: str, retry_count: int = 0) -> bool:
        """人間を模倣した検索動作（より人間らしく、複数検索エンジン対応）"""
        try:
            # 検索エンジンをランダムに選択（リトライ時は別のエンジンを試す）
            if retry_count == 0:
                search_engine = None  # ランダム選択
            elif retry_count == 1:
                search_engine = 'bing'  # Bingを試す
            else:
                search_engine = 'duckduckgo'  # DuckDuckGoを試す
            
            # 検索エンジンに移動
            search_url = self.get_search_engine_url(keyword, language, search_engine)
            
            # リトライ付きでページ遷移
            try:
                await self.retry_with_backoff(
                    page.goto,
                    max_retries=2,
                    base_delay=2.0,
                    max_delay=10.0,
                    url=search_url,
                    timeout=self.timeout,
                    wait_until="networkidle"
                )
            except Exception as e:
                logger.warning(f"[SEARCH] Failed to navigate to search engine, trying alternative: {e}")
                # 別の検索エンジンを試す
                if search_engine != 'bing':
                    search_url = self.get_search_engine_url(keyword, language, 'bing')
                    await page.goto(search_url, timeout=self.timeout, wait_until="networkidle")
                else:
                    raise
            
            # ページ読み込み後の人間らしい待機
            await asyncio.sleep(random.uniform(1.5, 3.0))
            
            # ページを少しスクロール（人間がページを見ている様子）
            await self.human_like_scroll(page, 'down')
            await asyncio.sleep(random.uniform(0.5, 1.5))
            await self.human_like_scroll(page, 'up')
            await asyncio.sleep(random.uniform(0.5, 1.0))
            
            # 検索ボックスを見つける（複数検索エンジン対応）
            search_box = None
            search_selectors = [
                'input[name="q"]',  # Google, Bing
                'input[name="search"]',  # DuckDuckGo
                'input[type="search"]',
                'input[aria-label*="検索"]',
                'input[aria-label*="Search"]',
                'textarea[name="q"]'  # Google (textarea)
            ]
            
            for selector in search_selectors:
                try:
                    search_box = await page.query_selector(selector)
                    if search_box:
                        break
                except:
                    continue
            
            if not search_box:
                logger.warning(f"[SEARCH] Search box not found, trying alternative search engine")
                # 別の検索エンジンを試す
                if retry_count < 2:
                    return await self.human_like_search(page, keyword, language, retry_count + 1)
                return False
            
            # 検索ボックスの位置を取得
            box = await search_box.bounding_box()
            if box:
                # 検索ボックスに人間らしくマウスを移動
                center_x = box['x'] + box['width'] / 2
                center_y = box['y'] + box['height'] / 2
                
                # 現在のマウス位置から検索ボックスへ滑らかに移動
                await self.human_like_mouse_move(page, 100, 100, int(center_x), int(center_y))
                await asyncio.sleep(random.uniform(0.3, 0.7))
            
            # 人間らしいタイピング
            await self.human_like_typing(page, search_box, keyword)
            
            # 検索前の待機（確認している様子）
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Enterキーで検索
            await search_box.press('Enter')
            await page.wait_for_load_state('networkidle', timeout=self.timeout)
            
            # 検索結果ページの人間らしい閲覧（バックグラウンドチェック対応）
            await self.human_like_page_view(page)
            
            # 検索後にバックグラウンドチェックを検出・突破
            search_checks = await self.detect_background_checks(page)
            if any(search_checks.values()):
                logger.warning("[SEARCH] Background checks detected after search, bypassing...")
                await self.bypass_background_checks(page, search_checks)
                await self.enhanced_human_behavior(page)
            
            logger.info(f"[SEARCH] Searched for: {keyword}")
            return True
            
        except Exception as e:
            logger.error(f"[SEARCH] Failed to search for {keyword}: {e}")
            # リトライを試みる
            if retry_count < 2:
                logger.info(f"[SEARCH] Retrying search with different engine (attempt {retry_count + 1})")
                await asyncio.sleep(random.uniform(2.0, 5.0))
                return await self.human_like_search(page, keyword, language, retry_count + 1)
            return False
    
    async def human_like_button_click(self, page: Page) -> bool:
        """人間を模倣したボタン操作（より人間らしく）"""
        try:
            # ページ内のボタンやリンクをランダムにクリック
            buttons = await page.query_selector_all('button, a[href]')
            
            if not buttons:
                return False
            
            # ランダムに1-3個のボタンをクリック
            num_clicks = random.randint(1, min(3, len(buttons)))
            selected_buttons = random.sample(buttons, num_clicks)
            
            for button in selected_buttons:
                try:
                    # ボタンの位置を取得
                    box = await button.bounding_box()
                    if not box:
                        continue
                    
                    center_x = box['x'] + box['width'] / 2
                    center_y = box['y'] + box['height'] / 2
                    
                    # 現在のマウス位置からボタンへ滑らかに移動
                    current_pos = page.mouse
                    await self.human_like_mouse_move(page, 100, 100, int(center_x), int(center_y))
                    
                    # ホバー（人間がボタンを確認している様子）
                    await button.hover()
                    await asyncio.sleep(random.uniform(0.5, 1.2))
                    
                    # 時々、ホバーしただけでクリックしない（人間の行動）
                    if random.random() < 0.2:  # 20%の確率
                        await asyncio.sleep(random.uniform(0.3, 0.8))
                        continue
                    
                    # クリック前の微細な動き（人間の手の震えをシミュレート）
                    await page.mouse.move(int(center_x + random.randint(-2, 2)), int(center_y + random.randint(-2, 2)))
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    
                    # クリック
                    await button.click()
                    
                    # クリック後の待機（ページ遷移を待つ）
                    await asyncio.sleep(random.uniform(1.5, 3.0))
                    
                    logger.debug(f"[BUTTON] Clicked button")
                except Exception as e:
                    logger.debug(f"[BUTTON] Failed to click button: {e}")
                    continue
            
            return True
            
        except Exception as e:
            logger.debug(f"[BUTTON] Failed to perform button operations: {e}")
            return False
    
    async def scrape_nsfw_site(self, page: Page, url: str, keyword: str, category: str, language: str) -> List[Dict]:
        """NSFW検知用サイトをスクレイピング（検知目的のみ）"""
        samples = []
        
        try:
            logger.info(f"[NSFW DETECTION] Scraping site: {url} (detection purpose only)")
            
            # サイトに直接アクセス
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")
            
            # アクセス直後にバックグラウンドチェックを検出・突破
            initial_checks = await self.detect_background_checks(page)
            if any(initial_checks.values()):
                logger.warning(f"[NSFW DETECTION] Background checks detected on {url}, bypassing...")
                await self.bypass_background_checks(page, initial_checks)
                await self.enhanced_human_behavior(page)
            
            # 人間らしいページ閲覧（バックグラウンドチェック対応）
            await self.human_like_page_view(page)
            
            # ページをスクレイピング（エラーハンドリング強化版）
            keyword_list = [keyword]  # NSFWサイトの場合は単一キーワード
            sample = await self.extract_page_content(page, keyword, category, language, keyword_list)
            if sample:
                sample['nsfw_detection_purpose'] = 'safety_training'
                sample['source_site'] = url
                samples.append(sample)
            
            # カテゴリページやランキングページもスクレイピング
            if 'category' in url or 'ranking' in url or 'genre' in url:
                # ページ内のリンクを取得
                links = await page.query_selector_all('a[href]')
                if links:
                    # ランダムに1-3個のリンクをクリック
                    num_clicks = min(3, len(links))
                    selected_links = random.sample(links, num_clicks)
                    
                    for link in selected_links:
                        try:
                            href = await link.get_attribute('href')
                            if href and not href.startswith('javascript:'):
                                # 絶対URLに変換
                                absolute_url = urljoin(url, href)
                                
                                # 同じドメインのリンクのみ
                                if urlparse(absolute_url).netloc == urlparse(url).netloc:
                                    await link.click()
                                    await asyncio.sleep(random.uniform(2.0, 4.0))
                                    
                                    # ページをスクレイピング（エラーハンドリング強化版）
                                    keyword_list = [keyword]
                                    link_sample = await self.extract_page_content(page, keyword, category, language, keyword_list)
                                    if link_sample:
                                        link_sample['nsfw_detection_purpose'] = 'safety_training'
                                        link_sample['source_site'] = absolute_url
                                        samples.append(link_sample)
                        except Exception as e:
                            logger.debug(f"[NSFW DETECTION] Failed to click link: {e}")
                            continue
            
            logger.info(f"[NSFW DETECTION] Collected {len(samples)} samples from {url} (detection purpose only)")
            
        except Exception as e:
            logger.error(f"[NSFW DETECTION] Failed to scrape site {url}: {e}")
        
        return samples
    
    async def scrape_keyword_with_browser(
        self,
        page: Page,
        browser_index: int,
        task: KeywordTask
    ) -> List[Dict]:
        """ブラウザでキーワードをスクレイピング"""
        samples = []
        
        try:
            # リソースチェック
            if not self.resource_manager.can_allocate_browser():
                logger.warning(f"[BROWSER {browser_index}] Resource limit reached, skipping")
                return samples
            
            # 日経225企業の場合は直接スクレイピング
            if task.category == 'nikkei225' and task.url:
                logger.info(f"[BROWSER {browser_index}] Processing Nikkei225 company: {task.company_name} ({task.data_type})")
                company_sample = await self.scrape_company_page(page, task)
                if company_sample:
                    samples.append(company_sample)
                    self.all_samples.append(company_sample)
                return samples
            
            # NSFW検知用サイトの場合は直接アクセス
            if task.url and task.category.startswith('nsfw_detection'):
                logger.info(f"[BROWSER {browser_index}] Processing NSFW detection site: {task.keyword} (detection purpose only)")
                
                # SO8T統制: NSFW検知用サイトのアクセスを評価
                if self.use_so8t_control:
                    nsfw_context = {
                        'url': task.url,
                        'keyword': task.keyword,
                        'category': task.category,
                        'language': task.language,
                        'purpose': 'nsfw_detection'
                    }
                    control_result = await self.so8t_control_scraping_action('scrape', nsfw_context)
                    
                    if control_result['decision'] == 'deny':
                        logger.warning(f"[SO8T] NSFW site access denied: {task.url}")
                        logger.warning(f"[SO8T] Reasoning: {control_result.get('final_reasoning', 'No reasoning')}")
                        return samples
                
                site_samples = await self.scrape_nsfw_site(page, task.url, task.keyword, task.category, task.language)
                samples.extend(site_samples)
                self.nsfw_samples.extend(site_samples)
                return samples
            
            # メインキーワードで検索
            logger.info(f"[BROWSER {browser_index}] Processing keyword: {task.keyword}")
            
            # ブラウザ状態を更新
            if browser_index not in self.browser_status:
                self.browser_status[browser_index] = {
                    'status': 'active',
                    'current_keyword': None,
                    'samples_collected': 0,
                    'last_activity': None
                }
            self.browser_status[browser_index]['current_keyword'] = task.keyword
            self.browser_status[browser_index]['last_activity'] = datetime.now().isoformat()
            
            # SO8T統制: 検索動作を評価
            if self.use_so8t_control:
                search_context = {
                    'url': page.url,
                    'keyword': task.keyword,
                    'category': task.category,
                    'language': task.language
                }
                control_result = await self.so8t_control_scraping_action('search', search_context)
                
                # SO8T判断結果を記録
                self.so8t_decisions.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'search',
                    'decision': control_result['decision'],
                    'reasoning': control_result.get('final_reasoning', ''),
                    'keyword': task.keyword,
                    'url': page.url
                })
                
                if control_result['decision'] == 'deny':
                    logger.warning(f"[SO8T] Search denied for keyword: {task.keyword}")
                    logger.warning(f"[SO8T] Reasoning: {control_result.get('final_reasoning', 'No reasoning')}")
                    self.browser_status[browser_index]['status'] = 'denied'
                    return samples
                elif control_result['decision'] == 'modify':
                    logger.info(f"[SO8T] Search modified for keyword: {task.keyword}")
                    # 修正された動作を適用（簡易版：待機時間を増やす）
                    await asyncio.sleep(random.uniform(5.0, 10.0))
            
            if await self.human_like_search(page, task.keyword, task.language):
                # Google検索結果ページは学習用データにならないため、リンクを抽出して各サイトをスクレイピング
                logger.info(f"[DEEPRESEARCH] Extracting links from search results for keyword: {task.keyword}")
                
                # 検索結果からリンクを抽出
                search_links = await self.extract_search_result_links(page, task.keyword, task.language)
                
                if search_links:
                    logger.info(f"[DEEPRESEARCH] Found {len(search_links)} links from search results")
                    
                    # 各リンク先をDeepResearch Webスクレイピング
                    for link_url in search_links[:10]:  # 最大10サイトまで
                        try:
                            # SO8T統制: スクレイピング動作を評価
                            if self.use_so8t_control:
                                scrape_context = {
                                    'url': link_url,
                                    'keyword': task.keyword,
                                    'category': task.category,
                                    'language': task.language
                                }
                                control_result = await self.so8t_control_scraping_action('scrape', scrape_context)
                                
                                if control_result['decision'] == 'deny':
                                    logger.warning(f"[SO8T] Scraping denied for URL: {link_url}")
                                    logger.warning(f"[SO8T] Reasoning: {control_result.get('final_reasoning', 'No reasoning')}")
                                    continue
                            
                            # リンク先をDeepResearch Webスクレイピング
                            await asyncio.sleep(random.uniform(1.0, 2.0))  # 人間らしい待機
                            sample = await self.scrape_deep_research_url(page, link_url, task.keyword, task.category, task.language, task.related_keywords)
                            
                            if sample:
                                samples.append(sample)
                                # ブラウザ状態を更新
                                self.browser_status[browser_index]['samples_collected'] += 1
                                
                                # 次のサイトへの待機（人間がページを読んでいる様子）
                                await asyncio.sleep(random.uniform(2.0, 4.0))
                        except Exception as e:
                            logger.warning(f"[DEEPRESEARCH] Failed to scrape {link_url}: {e}")
                            continue
                else:
                    logger.warning(f"[DEEPRESEARCH] No links found in search results for keyword: {task.keyword}")
            
            # 関連キーワードで検索
            for related_keyword in task.related_keywords[:3]:  # 最大3個
                # より長い待機時間（人間が考えている様子）
                await asyncio.sleep(random.uniform(self.delay_per_action * 2, self.delay_per_action * 4))
                
                if await self.human_like_search(page, related_keyword, task.language):
                    # Google検索結果ページからリンクを抽出
                    logger.info(f"[DEEPRESEARCH] Extracting links from search results for related keyword: {related_keyword}")
                    search_links = await self.extract_search_result_links(page, related_keyword, task.language)
                    
                    if search_links:
                        logger.info(f"[DEEPRESEARCH] Found {len(search_links)} links from search results for related keyword: {related_keyword}")
                        
                        # 各リンク先をDeepResearch Webスクレイピング
                        for link_url in search_links[:5]:  # 関連キーワードは最大5サイトまで
                            try:
                                # SO8T統制: スクレイピング動作を評価
                                if self.use_so8t_control:
                                    scrape_context = {
                                        'url': link_url,
                                        'keyword': related_keyword,
                                        'category': task.category,
                                        'language': task.language
                                    }
                                    control_result = await self.so8t_control_scraping_action('scrape', scrape_context)
                                    
                                    if control_result['decision'] == 'deny':
                                        logger.warning(f"[SO8T] Scraping denied for URL: {link_url}")
                                        continue
                                
                                # リンク先をDeepResearch Webスクレイピング
                                await asyncio.sleep(random.uniform(1.0, 2.0))
                                keyword_list = [task.keyword] + task.related_keywords
                                sample = await self.scrape_deep_research_url(page, link_url, related_keyword, task.category, task.language, keyword_list)
                                
                                if sample:
                                    samples.append(sample)
                                    # ブラウザ状態を更新
                                    self.browser_status[browser_index]['samples_collected'] += 1
                                    
                                    # 次のサイトへの待機
                                    await asyncio.sleep(random.uniform(2.0, 4.0))
                            except Exception as e:
                                logger.warning(f"[DEEPRESEARCH] Failed to scrape {link_url}: {e}")
                                continue
                    
                    # 次の検索前の待機（人間がページを読んでいる様子）
                    await asyncio.sleep(random.uniform(2.0, 5.0))
            
            logger.info(f"[BROWSER {browser_index}] Collected {len(samples)} samples for keyword: {task.keyword}")
            
            # ブラウザ状態を更新
            self.browser_status[browser_index]['samples_collected'] += len(samples)
            
        except Exception as e:
            logger.error(f"[BROWSER {browser_index}] Failed to scrape keyword {task.keyword}: {e}")
        
        return samples
    
    async def extract_page_content(
        self,
        page: Page,
        keyword: str,
        category: str,
        language: str,
        keyword_list: List[str] = None
    ) -> Optional[Dict]:
        """ページコンテンツを抽出（エラーハンドリング強化版、最適化）"""
        try:
            # ページが読み込まれるまで待機（複数の待機戦略を試す）
            try:
                await page.wait_for_load_state("networkidle", timeout=self.timeout)
            except PlaywrightTimeoutError:
                # networkidleがタイムアウトした場合、domcontentloadedを試す
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=self.timeout // 2)
                    # 追加の待機時間（動的コンテンツの読み込みを待つ）
                    await asyncio.sleep(random.uniform(2.0, 4.0))
                except PlaywrightTimeoutError:
                    logger.warning(f"[EXTRACT] Page load timeout for {page.url}")
                    # タイムアウトしても続行（一部のコンテンツは取得できる可能性がある）
            
            # エラーチェック
            errors = await self.check_page_errors(page)
            
            # エラーが検出された場合は処理
            if errors['is_404'] or errors['is_200_empty']:
                logger.warning(f"[ERROR] Page error detected for keyword '{keyword}': 404={errors['is_404']}, 200_empty={errors['is_200_empty']}")
                
                # エラーハンドリング（ブラウザバック、別ワード遷移）
                if keyword_list:
                    handled = await self.handle_page_error(page, errors, keyword, keyword_list)
                    if handled:
                        # 別ワードで再試行
                        return None  # 再試行は呼び出し側で行う
                
                # エラーが処理できない場合はNoneを返す
                return None
            
            # コンテンツがない場合はNoneを返す
            if not errors['has_content']:
                logger.warning(f"[ERROR] No content found for keyword '{keyword}'")
                return None
            
            # HTML取得
            html = await page.content()
            
            # BeautifulSoupでパース
            soup = BeautifulSoup(html, 'lxml')
            
            # スクリプト・スタイル・ナビゲーション要素を削除
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
                tag.decompose()
            
            # メインコンテンツを抽出
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=lambda x: x and ('content' in x.lower() or 'main' in x.lower() or 'article' in x.lower()))
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # テキストの正規化
            import re
            text = re.sub(r'\n\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()
            
            # 空ページや短すぎるページを検出
            if len(text) < 100:
                logger.debug(f"[EXTRACT] Content too short ({len(text)} chars), skipping")
                return None
            
            # 重複コンテンツの検出（同じテキストが複数回繰り返されている）
            words = text.split()
            if len(words) > 0:
                unique_words = len(set(words))
                repetition_ratio = unique_words / len(words)
                if repetition_ratio < 0.3:  # 30%以下は重複が多い
                    logger.debug(f"[EXTRACT] High repetition detected ({repetition_ratio:.2f}), skipping")
                    return None
            
            # URL取得
            url = page.url
            
            # NSFW検知
            nsfw_label = 'safe'
            nsfw_confidence = 0.0
            if hasattr(self, 'nsfw_classifier') and self.nsfw_classifier:
                nsfw_label, nsfw_confidence = self.nsfw_classifier.classify(text)
            else:
                # ルールベース検知
                nsfw_label, nsfw_confidence = self._rule_based_nsfw_detection(text)
            
            # サンプル作成
            sample = {
                "text": text,
                "url": url,
                "domain": urlparse(url).netloc,
                "title": await page.title(),
                "keyword": keyword,
                "category": category,
                "language": language,
                "source": "parallel_deep_research_scraper",
                "crawled_at": datetime.now().isoformat(),
                "text_length": len(text),
                "nsfw_label": nsfw_label,
                "nsfw_confidence": float(nsfw_confidence),
                "error_info": {
                    "is_404": errors['is_404'],
                    "is_200_empty": errors['is_200_empty'],
                    "status_code": errors['status_code']
                }
            }
            
            # NSFW検知
            nsfw_label, nsfw_confidence = self.detect_nsfw(text, url)
            sample['nsfw_label'] = nsfw_label
            sample['nsfw_confidence'] = float(nsfw_confidence)
            sample['nsfw_detection_purpose'] = 'safety_training'
            
            if nsfw_label != 'safe':
                self.nsfw_samples.append(sample)
            
            return sample
            
        except Exception as e:
            logger.error(f"[EXTRACT] Failed to extract content: {e}")
            return None
    
    async def extract_search_result_links(self, page: Page, keyword: str, language: str) -> List[str]:
        """
        検索結果ページからリンクを抽出（Google検索結果ページのコンテンツは保存しない）
        
        Args:
            page: Playwright Pageオブジェクト
            keyword: 検索キーワード
            language: 言語
        
        Returns:
            抽出したリンクのリスト
        """
        links = []
        try:
            # 検索結果ページが読み込まれるまで待機
            await page.wait_for_load_state("networkidle", timeout=self.timeout)
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            # HTML取得
            html = await page.content()
            
            # BeautifulSoupでパース
            soup = BeautifulSoup(html, 'html.parser')
            
            # urlparseをインポート（関数の先頭で）
            from urllib.parse import parse_qs, urlparse
            
            # Google検索結果のリンクを抽出
            # Google検索結果のセレクタ（複数のパターンを試す）
            search_result_selectors = [
                'div.g a[href^="http"]',  # Google検索結果のリンク
                'div[data-ved] a[href^="http"]',  # Google検索結果（data-ved属性付き）
                'h3 a[href^="http"]',  # タイトルリンク
                'a[href^="http"]:not([href*="google.com"]):not([href*="googleusercontent.com"])',  # Google以外のリンク
            ]
            
            found_urls = set()
            
            for selector in search_result_selectors:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href', '')
                    if href:
                        # GoogleのリダイレクトURLを処理
                        if href.startswith('/url?q='):
                            # GoogleのリダイレクトURLから実際のURLを抽出
                            parsed = urlparse(href)
                            query_params = parse_qs(parsed.query)
                            if 'q' in query_params:
                                actual_url = query_params['q'][0]
                                if actual_url.startswith('http'):
                                    found_urls.add(actual_url)
                        elif href.startswith('http') and 'google.com' not in href and 'googleusercontent.com' not in href:
                            found_urls.add(href)
            
            # ウィキペディアや主要サイトを優先
            priority_domains = ['wikipedia.org', 'wikimedia.org', 'github.com', 'stackoverflow.com', 'reddit.com']
            priority_links = []
            other_links = []
            
            for url in found_urls:
                domain = urlparse(url).netloc.lower()
                if any(priority_domain in domain for priority_domain in priority_domains):
                    priority_links.append(url)
                else:
                    other_links.append(url)
            
            # 優先リンクを先に、その後に他のリンク
            links = priority_links + other_links
            
            logger.info(f"[EXTRACT LINKS] Extracted {len(links)} links from search results (priority: {len(priority_links)}, other: {len(other_links)})")
            
        except Exception as e:
            logger.error(f"[EXTRACT LINKS] Failed to extract links from search results: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return links
    
    async def scrape_deep_research_url(self, page: Page, url: str, keyword: str, category: str, language: str, related_keywords: List[str] = None) -> Optional[Dict]:
        """
        リンク先をDeepResearch Webスクレイピング
        
        Args:
            page: Playwright Pageオブジェクト
            url: スクレイピングするURL
            keyword: 検索キーワード
            category: カテゴリ
            language: 言語
            related_keywords: 関連キーワードリスト
        
        Returns:
            スクレイピング結果（サンプル辞書）またはNone
        """
        try:
            logger.info(f"[DEEPRESEARCH] Scraping URL: {url}")
            
            # URLに移動
            await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            
            # 人間を模倣した待機
            await asyncio.sleep(random.uniform(2.0, 4.0))
            
            # 人間を模倣したページ閲覧動作
            await self.human_like_page_view(page)
            
            # バックグラウンドチェックを検出・突破
            checks = await self.detect_background_checks(page)
            if any(checks.values()):
                logger.info(f"[DEEPRESEARCH] Background checks detected for {url}, attempting bypass...")
                await self.bypass_background_checks(page, checks)
                await asyncio.sleep(random.uniform(2.0, 3.0))
            
            # ウィキペディアページの場合は関連項目を抽出してスクレイピング
            is_wikipedia = 'wikipedia.org' in url.lower()
            related_articles = []
            
            if is_wikipedia:
                logger.info(f"[WIKIPEDIA] Extracting related articles from Wikipedia page: {url}")
                related_articles = await self.extract_wikipedia_related_articles(page, url, language)
                
                if related_articles:
                    logger.info(f"[WIKIPEDIA] Found {len(related_articles)} related articles")
                    # 関連項目の最初の5つをスクレイピング
                    for related_url in related_articles[:5]:
                        try:
                            await asyncio.sleep(random.uniform(1.0, 2.0))
                            related_sample = await self.scrape_wikipedia_related_article(page, related_url, keyword, category, language, related_keywords)
                            if related_sample:
                                self.all_samples.append(related_sample)
                                logger.info(f"[WIKIPEDIA] Scraped related article: {related_url}")
                        except Exception as e:
                            logger.warning(f"[WIKIPEDIA] Failed to scrape related article {related_url}: {e}")
                            continue
            
            # ページコンテンツを抽出（既存のextract_page_contentメソッドを使用）
            keyword_list = [keyword] + (related_keywords or [])
            sample = await self.extract_page_content(page, keyword, category, language, keyword_list)
            
            if sample:
                # DeepResearch用のメタデータを追加
                sample['deep_research'] = True
                sample['search_keyword'] = keyword
                sample['discovered_from'] = 'search_results'
                if is_wikipedia and related_articles:
                    sample['wikipedia_related_articles'] = related_articles[:10]  # 最大10個
                    sample['wikipedia_related_articles_count'] = len(related_articles)
                logger.info(f"[DEEPRESEARCH] Successfully scraped: {url}")
            
            return sample
            
        except PlaywrightTimeoutError:
            logger.warning(f"[DEEPRESEARCH] Timeout while scraping {url}")
            return None
        except Exception as e:
            logger.error(f"[DEEPRESEARCH] Failed to scrape {url}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    async def retry_with_backoff(self, func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, *args, **kwargs):
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
            except (PlaywrightTimeoutError, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"[RETRY] Timeout error (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[RETRY] Max retries reached for timeout error: {e}")
            except Exception as e:
                last_exception = e
                # ネットワークエラーやブラウザエラーの場合のみリトライ
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'closed', 'crashed', 'disconnected']):
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"[RETRY] Network/browser error (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"[RETRY] Max retries reached for network/browser error: {e}")
                else:
                    # その他のエラーはリトライしない
                    raise
        
        # すべてのリトライが失敗した場合
        raise last_exception
    
    async def browser_worker(self, browser_index: int, playwright):
        """ブラウザワーカー（並列実行）"""
        logger.info(f"[WORKER {browser_index}] Starting browser worker...")
        
        # ブラウザ接続（リトライ付き）
        browser = None
        for retry in range(3):
            try:
                browser = await self.connect_to_cursor_browser(playwright, browser_index)
                if browser:
                    break
            except Exception as e:
                logger.warning(f"[WORKER {browser_index}] Browser connection failed (attempt {retry + 1}/3): {e}")
                if retry < 2:
                    await asyncio.sleep(5.0 * (retry + 1))
        
        if not browser:
            logger.error(f"[WORKER {browser_index}] Failed to connect browser after retries")
            return
        
        self.browsers.append(browser)
        
        try:
            # コンテキスト作成（ボット検知回避設定付き）
            contexts = browser.contexts
            if contexts:
                context = contexts[0]
            else:
                # 最新のChrome User-Agent
                user_agents = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
                ]
                user_agent = random.choice(user_agents)
                
                context = await browser.new_context(
                    user_agent=user_agent,
                    viewport={'width': 1920, 'height': 1080},
                    locale='ja-JP',
                    timezone_id='Asia/Tokyo',
                    permissions=['geolocation'],
                    extra_http_headers={
                        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Encoding': 'gzip, deflate, br, zstd',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Cache-Control': 'max-age=0',
                    }
                )
            
            self.contexts.append(context)
            
            # ページ作成
            page = await context.new_page()
            self.pages.append(page)
            
            # ボット検知回避: navigator.webdriverを削除し、Chromeに偽装
            await page.add_init_script("""
                // navigator.webdriverを削除
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Chromeオブジェクトを追加
                window.chrome = {
                    runtime: {},
                    loadTimes: function() {},
                    csi: function() {},
                    app: {}
                };
                
                // Permissions APIを偽装
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                
                // Pluginsを偽装
                Object.defineProperty(navigator, 'plugins', {
                    get: () => {
                        const plugins = [];
                        for (let i = 0; i < 5; i++) {
                            plugins.push({
                                name: `Plugin ${i}`,
                                description: `Plugin ${i} Description`,
                                filename: `plugin${i}.dll`
                            });
                        }
                        return plugins;
                    }
                });
                
                // Languagesを偽装
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['ja-JP', 'ja', 'en-US', 'en']
                });
                
                // Platformを偽装
                Object.defineProperty(navigator, 'platform', {
                    get: () => 'Win32'
                });
                
                // HardwareConcurrencyを偽装
                Object.defineProperty(navigator, 'hardwareConcurrency', {
                    get: () => 8
                });
                
                // DeviceMemoryを偽装
                Object.defineProperty(navigator, 'deviceMemory', {
                    get: () => 8
                });
            """)
            
            # ブラウザウィンドウサイズをランダムに設定（人間らしさ）
            viewport_sizes = [
                {'width': 1920, 'height': 1080},
                {'width': 1366, 'height': 768},
                {'width': 1536, 'height': 864},
                {'width': 1440, 'height': 900},
            ]
            viewport = random.choice(viewport_sizes)
            await page.set_viewport_size(viewport)
            
            # 時々、ウィンドウをリサイズ（人間の行動）
            if random.random() < 0.3:  # 30%の確率
                await asyncio.sleep(random.uniform(2.0, 5.0))
                new_viewport = random.choice(viewport_sizes)
                await page.set_viewport_size(new_viewport)
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # キーワード処理ループ
            while True:
                # キーワードキューから取得
                if not self.keyword_queue:
                    logger.info(f"[WORKER {browser_index}] No more keywords, exiting...")
                    break
                
                task = self.keyword_queue.popleft()
                
                # 重複チェック
                if task.keyword in self.completed_keywords:
                    continue
                
                self.completed_keywords.add(task.keyword)
                
                # スクレイピング実行（リトライ付き）
                try:
                    samples = await self.retry_with_backoff(
                        self.scrape_keyword_with_browser,
                        max_retries=2,
                        base_delay=2.0,
                        max_delay=30.0,
                        page=page,
                        browser_index=browser_index,
                        task=task
                    )
                    
                    # サンプルを追加
                    self.all_samples.extend(samples)
                except Exception as e:
                    logger.error(f"[WORKER {browser_index}] Failed to scrape keyword '{task.keyword}' after retries: {e}")
                    # ブラウザがクラッシュした可能性がある場合は再接続を試みる
                    if 'closed' in str(e).lower() or 'crashed' in str(e).lower() or 'disconnected' in str(e).lower():
                        logger.warning(f"[WORKER {browser_index}] Browser may have crashed, attempting to reconnect...")
                        try:
                            # ページを閉じて再作成
                            if browser_index < len(self.pages) and self.pages[browser_index]:
                                try:
                                    await self.pages[browser_index].close()
                                except:
                                    pass
                            
                            # 新しいページを作成
                            if browser and not browser.is_connected():
                                browser = await self.connect_to_cursor_browser(playwright, browser_index)
                                if browser:
                                    self.browsers[browser_index] = browser
                            
                            if browser:
                                contexts = browser.contexts
                                if contexts:
                                    context = contexts[0]
                                else:
                                    context = await browser.new_context()
                                
                                page = await context.new_page()
                                if browser_index < len(self.pages):
                                    self.pages[browser_index] = page
                                else:
                                    self.pages.append(page)
                                
                                logger.info(f"[WORKER {browser_index}] Browser reconnected successfully")
                        except Exception as reconnect_error:
                            logger.error(f"[WORKER {browser_index}] Failed to reconnect browser: {reconnect_error}")
                            # 再接続に失敗した場合はこのワーカーを終了
                            break
                    continue
                
                # リソース状態をログ（詳細なメトリクス）
                resource_status = self.resource_manager.get_resource_status()
                logger.info(f"[WORKER {browser_index}] Resource status: Memory={resource_status['memory_gb']:.2f}GB, CPU={resource_status['cpu_percent']:.1f}%")
                
                # 進捗状況をログ
                total_samples = len(self.all_samples)
                completed_keywords_count = len(self.completed_keywords)
                remaining_keywords = len(self.keyword_queue)
                logger.info(f"[PROGRESS] Worker {browser_index}: Total samples={total_samples}, Completed keywords={completed_keywords_count}, Remaining={remaining_keywords}")
                
                # パフォーマンスメトリクスを記録
                if browser_index not in self.browser_status:
                    self.browser_status[browser_index] = {
                        'status': 'active',
                        'current_keyword': None,
                        'samples_collected': 0,
                        'last_activity': None
                    }
                self.browser_status[browser_index]['total_samples'] = total_samples
                self.browser_status[browser_index]['completed_keywords'] = completed_keywords_count
                self.browser_status[browser_index]['remaining_keywords'] = remaining_keywords
                self.browser_status[browser_index]['resource_memory_gb'] = resource_status['memory_gb']
                self.browser_status[browser_index]['resource_cpu_percent'] = resource_status['cpu_percent']
                
                # 現在のブラウザのスクリーンショットを取得（より頻繁に、タスクごと）
                if page and not page.is_closed():
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = self.screenshots_dir / f"browser_{browser_index}_{timestamp}.png"
                        await page.screenshot(path=str(screenshot_path), full_page=True)  # フルページスクリーンショット
                        
                        # ブラウザ状態にスクリーンショットパスを記録
                        if browser_index not in self.browser_status:
                            self.browser_status[browser_index] = {
                                'status': 'active',
                                'current_keyword': None,
                                'samples_collected': 0,
                                'last_activity': None
                            }
                        self.browser_status[browser_index]['screenshot_path'] = str(screenshot_path)
                        self.browser_status[browser_index]['screenshot_timestamp'] = timestamp
                        logger.info(f"[SCREENSHOT] Captured screenshot for browser {browser_index}: {screenshot_path}")
                    except Exception as e:
                        logger.debug(f"[SCREENSHOT] Failed to capture screenshot for browser {browser_index}: {e}")
                
                # ダッシュボード用状態を保存（定期的に）
                if len(self.all_samples) % 10 == 0:  # 10サンプルごと
                    from scripts.data.save_dashboard_state import save_dashboard_state
                    save_dashboard_state(
                        self.browser_status,
                        self.so8t_decisions,
                        len(self.all_samples),
                        len(self.nsfw_samples),
                        self.output_dir
                    )
                    # ブラウザ状態も保存
                    self.save_browser_state()
                
                # アクション間の待機
                await asyncio.sleep(random.uniform(self.delay_per_action, self.delay_per_action * 2))
        
        except Exception as e:
            logger.error(f"[WORKER {browser_index}] Worker error: {e}")
        finally:
            # ページを閉じる（ブラウザは閉じない）
            if browser_index < len(self.pages):
                try:
                    await self.pages[browser_index].close()
                except:
                    pass
        
        logger.info(f"[WORKER {browser_index}] Browser worker finished")
    
    async def run_parallel_scraping(self):
        """並列スクレイピング実行"""
        logger.info("="*80)
        logger.info("Starting Parallel DeepResearch Scraping")
        logger.info("="*80)
        
        # キーワードキュー初期化
        self.initialize_keyword_queue()
        
        # Playwright起動
        async with async_playwright() as playwright:
            # 並列ブラウザワーカー起動
            tasks = []
            for i in range(self.num_browsers):
                task = asyncio.create_task(self.browser_worker(i, playwright))
                tasks.append(task)
                # 起動間隔を空ける
                await asyncio.sleep(1.0)
            
            # すべてのワーカーが完了するまで待機
            await asyncio.gather(*tasks)
        
        logger.info(f"[TOTAL] Collected {len(self.all_samples)} samples")
        logger.info(f"[NSFW] Detected {len(self.nsfw_samples)} NSFW samples (detection purpose only)")
        
        # 日経225企業サンプルの統計
        nikkei225_total = sum(len(samples) for samples in self.nikkei225_samples.values())
        if nikkei225_total > 0:
            logger.info(f"[NIKKEI225] Collected {nikkei225_total} Nikkei225 company samples")
            for data_type, samples in self.nikkei225_samples.items():
                if samples:
                    logger.info(f"[NIKKEI225]   - {data_type}: {len(samples)} samples")
        
        # ブラウザ状態を最終更新
        for browser_index in range(self.num_browsers):
            if browser_index in self.browser_status:
                self.browser_status[browser_index]['status'] = 'completed'
        
        # 最終スクリーンショットを取得
        try:
            await self.capture_browser_screenshots()
        except Exception as e:
            logger.warning(f"[SCREENSHOT] Failed to capture final screenshots: {e}")
        
        # 日経225企業サンプルをカテゴリ別に保存
        if nikkei225_total > 0:
            self.save_nikkei225_samples_by_category()
        
        # 全自動ラベル付け・4値分類・四重推論・学習用データセット変換
        if self.auto_labeling_enabled and len(self.all_samples) > 0:
            logger.info("="*80)
            logger.info("Starting Automatic Labeling, 4-Class Classification, and Quadruple Thinking")
            logger.info("="*80)
            self.process_samples_for_training_dataset()
    
    def save_nikkei225_samples_by_category(self):
        """日経225企業サンプルをカテゴリ別に保存"""
        logger.info("[NIKKEI225] Saving samples by category...")
        
        for data_type, samples in self.nikkei225_samples.items():
            if samples:
                filename = f"nikkei225_{data_type}_{self.session_id}.jsonl"
                output_file = self.save_samples(samples, filename)
                logger.info(f"[NIKKEI225] Saved {len(samples)} {data_type} samples to {output_file}")
        
        # メタデータを保存
        metadata = {
            'total_samples': sum(len(samples) for samples in self.nikkei225_samples.values()),
            'samples_by_category': {cat: len(samples) for cat, samples in self.nikkei225_samples.items()},
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'source': 'nikkei225_companies'
        }
        
        metadata_file = self.output_dir / f"nikkei225_metadata_{self.session_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[NIKKEI225] Metadata saved to {metadata_file}")
    
    def save_samples(self, samples: List[Dict], filename: str = None) -> Path:
        """サンプルを保存"""
        if filename is None:
            filename = f"parallel_deep_research_scraped_{self.session_id}.jsonl"
        
        output_file = self.output_dir / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Saved {len(samples)} samples to {output_file}")
        return output_file
    
    async def capture_browser_screenshots(self):
        """すべてのブラウザのスクリーンショットを取得（フルページ）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for browser_index, page in enumerate(self.pages):
            if page and not page.is_closed():
                try:
                    # スクリーンショットを取得（フルページ）
                    screenshot_path = self.screenshots_dir / f"browser_{browser_index}_{timestamp}.png"
                    await page.screenshot(path=str(screenshot_path), full_page=True)
                    
                    # ブラウザ状態にスクリーンショットパスを記録
                    if browser_index not in self.browser_status:
                        self.browser_status[browser_index] = {
                            'status': 'active',
                            'current_keyword': None,
                            'samples_collected': 0,
                            'last_activity': None
                        }
                    self.browser_status[browser_index]['screenshot_path'] = str(screenshot_path)
                    self.browser_status[browser_index]['screenshot_timestamp'] = timestamp
                    
                    logger.info(f"[SCREENSHOT] Captured full-page screenshot for browser {browser_index}: {screenshot_path}")
                except Exception as e:
                    logger.warning(f"[SCREENSHOT] Failed to capture screenshot for browser {browser_index}: {e}")
    
    def process_samples_for_training_dataset(self):
        """
        全自動ラベル付け・4値分類・四重推論・学習用データセット変換
        
        スクレイピングしたサンプルに対して：
        1. SO8T/thinkingモデルによる4値分類
        2. 四重推論の実行
        3. 学習用データセット形式への変換
        4. 保存
        """
        if not self.quadruple_classifier:
            logger.warning("[AUTO-LABELING] QuadrupleClassifier not available, skipping automatic labeling")
            return
        
        logger.info(f"[AUTO-LABELING] Processing {len(self.all_samples)} samples for training dataset...")
        
        processed_count = 0
        error_count = 0
        
        for i, sample in enumerate(self.all_samples):
            try:
                # 4値分類と四重推論を実行
                classified_sample = self.quadruple_classifier.classify_quadruple(sample)
                
                # 学習用データセット形式に変換
                training_sample = self._convert_to_training_format(classified_sample)
                
                if training_sample:
                    self.training_dataset_samples.append(training_sample)
                    processed_count += 1
                
                # 進捗ログ（10サンプルごと）
                if (i + 1) % 10 == 0:
                    logger.info(f"[AUTO-LABELING] Processed {i + 1}/{len(self.all_samples)} samples...")
                    
            except Exception as e:
                logger.error(f"[AUTO-LABELING] Failed to process sample {i}: {e}")
                error_count += 1
                continue
        
        logger.info(f"[AUTO-LABELING] Completed: {processed_count} processed, {error_count} errors")
        
        # 統計的に有意なデータクレンジングを実行
        if self.training_dataset_samples:
            logger.info("="*80)
            logger.info("Starting Statistical Data Cleaning")
            logger.info("="*80)
            original_count = len(self.training_dataset_samples)
            cleaned_samples = self.statistical_data_cleaning(self.training_dataset_samples)
            self.training_dataset_samples = cleaned_samples
            logger.info(f"[STATISTICAL-CLEANING] Cleaned dataset: {len(cleaned_samples)} samples (removed {original_count - len(cleaned_samples)} outliers)")
        
        # 学習用データセットを保存
        if self.training_dataset_samples:
            self.save_training_dataset()
        else:
            logger.warning("[AUTO-LABELING] No training dataset samples generated")
    
    def _convert_to_training_format(self, classified_sample: Dict) -> Optional[Dict]:
        """
        分類済みサンプルを学習用データセット形式に変換
        
        Args:
            classified_sample: 4値分類済みサンプル
        
        Returns:
            学習用データセット形式のサンプル
        """
        try:
            text = classified_sample.get('text', '')
            if not text or len(text) < 50:
                return None
            
            # 四重推論結果を取得
            quadruple_classification = classified_sample.get('quadruple_classification', {})
            task_reasoning = quadruple_classification.get('task', '')
            safety_reasoning = quadruple_classification.get('safety', '')
            policy_reasoning = quadruple_classification.get('policy', '')
            final_reasoning = quadruple_classification.get('final', '')
            four_class_label = classified_sample.get('four_class_label', 'ALLOW')
            
            # 学習用データセット形式に変換
            # instruction-output形式
            instruction = f"""以下のテキストを読み、四重推論（Task/Safety/Policy/Final）を行い、4値分類（ALLOW/ESCALATION/DENY/REFUSE）を実行してください。

テキスト:
{text[:2000]}  # 最大2000文字
"""
            
            output = f"""<think-task>
{task_reasoning}
</think-task>

<think-safety>
{safety_reasoning}
</think-safety>

<think-policy>
{policy_reasoning}
</think-policy>

<final>
{final_reasoning}

分類結果: {four_class_label}
</final>"""
            
            training_sample = {
                'instruction': instruction,
                'output': output,
                'input': text[:2000],  # 最大2000文字
                'category': classified_sample.get('category', 'unknown'),
                'language': classified_sample.get('language', 'unknown'),
                'url': classified_sample.get('url', ''),
                'domain': classified_sample.get('domain', ''),
                'title': classified_sample.get('title', ''),
                'keyword': classified_sample.get('keyword', ''),
                'four_class_label': four_class_label,
                'four_class_label_id': classified_sample.get('four_class_label_id', 0),
                'quadruple_classification': quadruple_classification,
                'nsfw_label': classified_sample.get('nsfw_label', 'safe'),
                'nsfw_confidence': classified_sample.get('nsfw_confidence', 0.0),
                'deep_research': classified_sample.get('deep_research', False),
                'search_keyword': classified_sample.get('search_keyword', ''),
                'discovered_from': classified_sample.get('discovered_from', ''),
                'crawled_at': classified_sample.get('crawled_at', datetime.now().isoformat()),
                'classified_at': classified_sample.get('classified_at', datetime.now().isoformat()),
                'classification_version': classified_sample.get('classification_version', '1.0'),
                'source': 'parallel_deep_research_scraper_auto_labeled'
            }
            
            return training_sample
            
        except Exception as e:
            logger.error(f"[TRAINING-FORMAT] Failed to convert sample to training format: {e}")
            return None
    
    def save_training_dataset(self):
        """学習用データセットを保存"""
        if not self.training_dataset_samples:
            logger.warning("[TRAINING-DATASET] No training dataset samples to save")
            return
        
        logger.info(f"[TRAINING-DATASET] Saving {len(self.training_dataset_samples)} training dataset samples...")
        
        # JSONL形式で保存
        training_dataset_file = self.output_dir / f"training_dataset_{self.session_id}.jsonl"
        with open(training_dataset_file, 'w', encoding='utf-8') as f:
            for sample in self.training_dataset_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[TRAINING-DATASET] Saved training dataset to {training_dataset_file}")
        
        # メタデータを保存
        metadata = {
            'total_samples': len(self.training_dataset_samples),
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'source': 'parallel_deep_research_scraper_auto_labeled',
            'classification_version': self.training_dataset_samples[0].get('classification_version', '1.0') if self.training_dataset_samples else '1.0',
            'four_class_distribution': self._calculate_four_class_distribution(),
            'category_distribution': self._calculate_category_distribution(),
            'language_distribution': self._calculate_language_distribution()
        }
        
        metadata_file = self.output_dir / f"training_dataset_metadata_{self.session_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[TRAINING-DATASET] Saved metadata to {metadata_file}")
    
    def _calculate_four_class_distribution(self) -> Dict[str, int]:
        """4値分類ラベルの分布を計算"""
        distribution = {'ALLOW': 0, 'ESCALATION': 0, 'DENY': 0, 'REFUSE': 0}
        for sample in self.training_dataset_samples:
            label = sample.get('four_class_label', 'ALLOW')
            if label in distribution:
                distribution[label] += 1
        return distribution
    
    def _calculate_category_distribution(self) -> Dict[str, int]:
        """カテゴリの分布を計算"""
        distribution = {}
        for sample in self.training_dataset_samples:
            category = sample.get('category', 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _calculate_language_distribution(self) -> Dict[str, int]:
        """言語の分布を計算"""
        distribution = {}
        for sample in self.training_dataset_samples:
            language = sample.get('language', 'unknown')
            distribution[language] = distribution.get(language, 0) + 1
        return distribution
    
    def statistical_data_cleaning(self, samples: List[Dict]) -> List[Dict]:
        """
        統計的に有意な手法でデータクレンジング
        
        手法:
        1. Z-scoreによる外れ値検出（テキスト長、信頼度スコア等）
        2. IQR（四分位範囲）による外れ値検出
        3. 重複検出（テキスト類似度、ハッシュベース）
        4. 統計的品質スコアリング
        5. 信頼区間によるフィルタリング
        
        Args:
            samples: クレンジングするサンプルリスト
        
        Returns:
            クレンジング済みサンプルリスト
        """
        if not samples:
            return []
        
        logger.info(f"[STATISTICAL-CLEANING] Starting statistical data cleaning for {len(samples)} samples...")
        
        cleaned_samples = []
        removal_stats = {
            'duplicates': 0,
            'outliers_zscore': 0,
            'outliers_iqr': 0,
            'low_quality': 0,
            'statistical_filter': 0,
            'total_removed': 0
        }
        
        # 1. 重複検出（ハッシュベース + 類似度ベース）
        seen_hashes = set()
        unique_samples = []
        
        for sample in samples:
            text = sample.get('input', sample.get('text', ''))
            if not text:
                removal_stats['low_quality'] += 1
                continue
            
            # ハッシュベース重複検出
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if text_hash in seen_hashes:
                removal_stats['duplicates'] += 1
                continue
            
            # 類似度ベース重複検出（既存サンプルとの類似度が高い場合は除外）
            is_duplicate = False
            for existing_sample in unique_samples:
                existing_text = existing_sample.get('input', existing_sample.get('text', ''))
                similarity = SequenceMatcher(None, text[:500], existing_text[:500]).ratio()
                if similarity > 0.95:  # 95%以上類似している場合は重複とみなす
                    is_duplicate = True
                    removal_stats['duplicates'] += 1
                    break
            
            if not is_duplicate:
                seen_hashes.add(text_hash)
                unique_samples.append(sample)
        
        logger.info(f"[STATISTICAL-CLEANING] Removed {removal_stats['duplicates']} duplicates")
        
        if not unique_samples:
            logger.warning("[STATISTICAL-CLEANING] No samples remaining after duplicate removal")
            return []
        
        # 2. 統計的特徴量の計算
        if NUMPY_AVAILABLE:
            text_lengths = [len(s.get('input', s.get('text', ''))) for s in unique_samples]
            confidence_scores = [s.get('four_class_label_id', 0) for s in unique_samples]
            nsfw_confidences = [s.get('nsfw_confidence', 0.0) for s in unique_samples]
            
            # Z-scoreによる外れ値検出（テキスト長）
            if len(text_lengths) > 3:
                mean_length = np.mean(text_lengths)
                std_length = np.std(text_lengths)
                
                if std_length > 0:
                    z_scores_length = np.abs((np.array(text_lengths) - mean_length) / std_length)
                    z_threshold = 3.0  # 3シグマルール
                    
                    for i, sample in enumerate(unique_samples):
                        if z_scores_length[i] > z_threshold:
                            removal_stats['outliers_zscore'] += 1
                            continue
                        cleaned_samples.append(sample)
                else:
                    cleaned_samples = unique_samples
            else:
                cleaned_samples = unique_samples
        else:
            # NumPyが利用できない場合は基本的なフィルタリングのみ
            cleaned_samples = unique_samples
        
        # 3. IQR（四分位範囲）による外れ値検出
        if NUMPY_AVAILABLE and len(cleaned_samples) > 10:
            text_lengths_cleaned = [len(s.get('input', s.get('text', ''))) for s in cleaned_samples]
            
            if len(text_lengths_cleaned) > 0:
                q1 = np.percentile(text_lengths_cleaned, 25)
                q3 = np.percentile(text_lengths_cleaned, 75)
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    iqr_filtered_samples = []
                    for sample in cleaned_samples:
                        text_length = len(sample.get('input', sample.get('text', '')))
                        if lower_bound <= text_length <= upper_bound:
                            iqr_filtered_samples.append(sample)
                        else:
                            removal_stats['outliers_iqr'] += 1
                    
                    cleaned_samples = iqr_filtered_samples
                    logger.info(f"[STATISTICAL-CLEANING] Removed {removal_stats['outliers_iqr']} outliers (IQR method)")
        
        # 4. 統計的品質スコアリング
        quality_filtered_samples = []
        for sample in cleaned_samples:
            quality_score = self._calculate_statistical_quality_score(sample)
            sample['statistical_quality_score'] = quality_score
            
            # 品質スコアが低いサンプルを除外（統計的に有意な閾値）
            if quality_score < 0.3:  # 30%未満は低品質とみなす
                removal_stats['low_quality'] += 1
                continue
            
            quality_filtered_samples.append(sample)
        
        cleaned_samples = quality_filtered_samples
        logger.info(f"[STATISTICAL-CLEANING] Removed {removal_stats['low_quality']} low-quality samples")
        
        # 5. 信頼区間によるフィルタリング（4値分類の信頼度）
        if NUMPY_AVAILABLE and len(cleaned_samples) > 10:
            confidence_scores_cleaned = [s.get('four_class_label_id', 0) for s in cleaned_samples]
            
            if len(confidence_scores_cleaned) > 0 and np.std(confidence_scores_cleaned) > 0:
                mean_confidence = np.mean(confidence_scores_cleaned)
                std_confidence = np.std(confidence_scores_cleaned)
                
                # 95%信頼区間
                confidence_interval_lower = mean_confidence - 1.96 * std_confidence
                confidence_interval_upper = mean_confidence + 1.96 * std_confidence
                
                ci_filtered_samples = []
                for sample in cleaned_samples:
                    confidence = sample.get('four_class_label_id', 0)
                    if confidence_interval_lower <= confidence <= confidence_interval_upper:
                        ci_filtered_samples.append(sample)
                    else:
                        removal_stats['statistical_filter'] += 1
                
                cleaned_samples = ci_filtered_samples
                logger.info(f"[STATISTICAL-CLEANING] Removed {removal_stats['statistical_filter']} samples outside confidence interval")
        
        # 統計サマリー
        removal_stats['total_removed'] = len(samples) - len(cleaned_samples)
        logger.info("="*80)
        logger.info("[STATISTICAL-CLEANING] Cleaning Statistics:")
        logger.info(f"  Original samples: {len(samples)}")
        logger.info(f"  Cleaned samples: {len(cleaned_samples)}")
        logger.info(f"  Removed duplicates: {removal_stats['duplicates']}")
        logger.info(f"  Removed outliers (Z-score): {removal_stats['outliers_zscore']}")
        logger.info(f"  Removed outliers (IQR): {removal_stats['outliers_iqr']}")
        logger.info(f"  Removed low-quality: {removal_stats['low_quality']}")
        logger.info(f"  Removed (statistical filter): {removal_stats['statistical_filter']}")
        logger.info(f"  Total removed: {removal_stats['total_removed']}")
        logger.info(f"  Retention rate: {len(cleaned_samples)/len(samples)*100:.2f}%")
        logger.info("="*80)
        
        return cleaned_samples
    
    def _calculate_statistical_quality_score(self, sample: Dict) -> float:
        """
        統計的品質スコアを計算
        
        品質スコアの要素:
        1. テキスト長（適切な範囲内か）
        2. 情報量（ユニークな単語数）
        3. 4値分類の信頼度
        4. NSFW検知の信頼度
        5. メタデータの完全性
        
        Args:
            sample: サンプル
        
        Returns:
            品質スコア（0.0-1.0）
        """
        score = 0.0
        max_score = 0.0
        
        # 1. テキスト長スコア（100-5000文字が最適）
        text = sample.get('input', sample.get('text', ''))
        text_length = len(text)
        if 100 <= text_length <= 5000:
            length_score = 1.0
        elif 50 <= text_length < 100 or 5000 < text_length <= 10000:
            length_score = 0.7
        elif text_length < 50 or text_length > 10000:
            length_score = 0.3
        else:
            length_score = 0.0
        
        score += length_score * 0.3
        max_score += 0.3
        
        # 2. 情報量スコア（ユニークな単語数）
        if text:
            words = text.split()
            unique_words = len(set(words))
            if len(words) > 0:
                uniqueness_ratio = unique_words / len(words)
                info_score = min(uniqueness_ratio * 2, 1.0)  # 0.5以上で満点
            else:
                info_score = 0.0
        else:
            info_score = 0.0
        
        score += info_score * 0.2
        max_score += 0.2
        
        # 3. 4値分類の信頼度（ラベルIDが適切か）
        four_class_label = sample.get('four_class_label', 'ALLOW')
        if four_class_label in ['ALLOW', 'ESCALATION', 'DENY', 'REFUSE']:
            classification_score = 1.0
        else:
            classification_score = 0.5
        
        score += classification_score * 0.2
        max_score += 0.2
        
        # 4. NSFW検知の信頼度（safeラベルの場合は高スコア）
        nsfw_label = sample.get('nsfw_label', 'safe')
        nsfw_confidence = sample.get('nsfw_confidence', 0.0)
        if nsfw_label == 'safe':
            nsfw_score = 1.0
        else:
            # NSFW検知された場合は信頼度に応じてスコアを下げる
            nsfw_score = 1.0 - nsfw_confidence * 0.5
        
        score += nsfw_score * 0.15
        max_score += 0.15
        
        # 5. メタデータの完全性
        required_fields = ['category', 'language', 'url', 'four_class_label']
        metadata_completeness = sum(1 for field in required_fields if sample.get(field)) / len(required_fields)
        
        score += metadata_completeness * 0.15
        max_score += 0.15
        
        # 正規化（0.0-1.0）
        if max_score > 0:
            normalized_score = score / max_score
        else:
            normalized_score = 0.0
        
        return normalized_score
    
    async def extract_wikipedia_related_articles(self, page: Page, url: str, language: str) -> List[str]:
        """
        ウィキペディアページから関連項目（関連記事）のリンクを抽出
        
        Args:
            page: Playwright Pageオブジェクト
            url: ウィキペディアページのURL
            language: 言語
        
        Returns:
            関連項目のURLリスト
        """
        related_urls = []
        try:
            # HTML取得
            html = await page.content()
            
            # BeautifulSoupでパース
            soup = BeautifulSoup(html, 'html.parser')
            
            # ウィキペディアの関連項目セクションを探す
            # 日本語版: "関連項目" セクション
            # 英語版: "See also" セクション
            related_section_headers = {
                'ja': ['関連項目', '関連記事', '関連リンク'],
                'en': ['See also', 'Related articles', 'Related links']
            }
            
            headers_to_find = related_section_headers.get(language, related_section_headers['en'])
            
            # セクション見出しを探す
            for header_text in headers_to_find:
                # h2タグで見出しを探す
                headers = soup.find_all(['h2', 'h3'], string=lambda text: text and header_text in text)
                
                for header in headers:
                    # 見出しの次の要素からリンクを抽出
                    current = header.find_next_sibling()
                    link_count = 0
                    
                    while current and link_count < 20:  # 最大20個のリンク
                        if current.name in ['ul', 'div', 'p']:
                            # リンクを抽出
                            links = current.find_all('a', href=True)
                            for link in links:
                                href = link.get('href', '')
                                if href.startswith('/wiki/'):
                                    # 相対URLを絶対URLに変換
                                    if 'wikipedia.org' in url:
                                        base_url = '/'.join(url.split('/')[:3])
                                        full_url = base_url + href
                                        if full_url not in related_urls:
                                            related_urls.append(full_url)
                                            link_count += 1
                                            if link_count >= 20:
                                                break
                        
                        # 次のセクションが見つかったら終了
                        if current.name in ['h2', 'h3']:
                            break
                        
                        current = current.find_next_sibling()
                    
                    if related_urls:
                        break
                
                if related_urls:
                    break
            
            # 関連項目セクションが見つからない場合は、本文中のリンクから関連性の高いものを抽出
            if not related_urls:
                logger.debug(f"[WIKIPEDIA] Related section not found, extracting links from content")
                # 本文中のウィキペディア内部リンクを抽出
                content_links = soup.find_all('a', href=lambda href: href and href.startswith('/wiki/'))
                for link in content_links[:15]:  # 最大15個
                    href = link.get('href', '')
                    if 'wikipedia.org' in url:
                        base_url = '/'.join(url.split('/')[:3])
                        full_url = base_url + href
                        # 外部リンクや特殊ページを除外
                        if ':' not in href.split('/wiki/')[-1]:  # 名前空間を除外
                            if full_url not in related_urls:
                                related_urls.append(full_url)
            
            logger.info(f"[WIKIPEDIA] Extracted {len(related_urls)} related article URLs from {url}")
            
        except Exception as e:
            logger.error(f"[WIKIPEDIA] Failed to extract related articles from {url}: {e}")
        
        return related_urls
    
    async def scrape_wikipedia_related_article(self, page: Page, url: str, keyword: str, category: str, language: str, related_keywords: List[str] = None) -> Optional[Dict]:
        """
        ウィキペディアの関連項目ページをスクレイピング
        
        Args:
            page: Playwright Pageオブジェクト
            url: 関連項目のURL
            keyword: 元のキーワード
            category: カテゴリ
            language: 言語
            related_keywords: 関連キーワードリスト
        
        Returns:
            スクレイピング結果（サンプル辞書）またはNone
        """
        try:
            logger.info(f"[WIKIPEDIA-RELATED] Scraping related article: {url}")
            
            # URLに移動
            await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            
            # 人間を模倣した待機
            await asyncio.sleep(random.uniform(2.0, 4.0))
            
            # 人間を模倣したページ閲覧動作
            await self.human_like_page_view(page)
            
            # ページコンテンツを抽出
            keyword_list = [keyword] + (related_keywords or [])
            sample = await self.extract_page_content(page, keyword, category, language, keyword_list)
            
            if sample:
                # ウィキペディア関連項目用のメタデータを追加
                sample['deep_research'] = True
                sample['search_keyword'] = keyword
                sample['discovered_from'] = 'wikipedia_related_article'
                sample['wikipedia_related'] = True
                sample['source'] = 'wikipedia_related_article'
                logger.info(f"[WIKIPEDIA-RELATED] Successfully scraped related article: {url}")
            
            return sample
            
        except PlaywrightTimeoutError:
            logger.warning(f"[WIKIPEDIA-RELATED] Timeout while scraping {url}")
            return None
        except Exception as e:
            logger.error(f"[WIKIPEDIA-RELATED] Failed to scrape {url}: {e}")
            return None
        
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
    parser = argparse.ArgumentParser(description="Parallel DeepResearch Web Scraping")
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('D:/webdataset/processed'),
        help='Output directory'
    )
    parser.add_argument(
        '--num-browsers',
        type=int,
        default=10,
        help='Number of parallel browsers'
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
        help='Remote debugging port (base port)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.5,
        help='Delay between actions (seconds)'
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
        default=5,
        help='Maximum pages per keyword'
    )
    parser.add_argument(
        '--max-memory-gb',
        type=float,
        default=8.0,
        help='Maximum memory usage (GB)'
    )
    parser.add_argument(
        '--max-cpu-percent',
        type=float,
        default=80.0,
        help='Maximum CPU usage (%)'
    )
    parser.add_argument(
        '--use-so8t-control',
        action='store_true',
        default=False,
        help='Use SO8T model to control scraping actions'
    )
    parser.add_argument(
        '--so8t-model-path',
        type=str,
        default=None,
        help='Path to SO8T model (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # スクレイパー作成
    scraper = ParallelDeepResearchScraper(
        output_dir=args.output,
        num_browsers=args.num_browsers,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_action=args.delay,
        timeout=args.timeout,
        max_pages_per_keyword=args.max_pages_per_keyword,
        max_memory_gb=args.max_memory_gb,
        max_cpu_percent=args.max_cpu_percent,
        use_so8t_control=args.use_so8t_control,
        so8t_model_path=args.so8t_model_path
    )
    
    # 並列スクレイピング実行
    await scraper.run_parallel_scraping()
    
    # 保存
    output_file = scraper.save_samples(scraper.all_samples)
    
    # NSFW検知データ保存（検知目的のみ）
    if scraper.nsfw_samples:
        nsfw_file = scraper.save_nsfw_samples(scraper.nsfw_samples)
        logger.info(f"[NSFW] NSFW samples saved (detection purpose only): {nsfw_file}")
    
    logger.info(f"[SUCCESS] Parallel DeepResearch scraping completed. Output: {output_file}")
    return output_file


if __name__ == "__main__":
    asyncio.run(main())

