#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
違法薬物検知目的DeepResearch Webスクレイピングスクリプト

違法薬物検知目的に特化し、ドメイン別知識・コーディング能力・検知目的のNSFWデータも含む
DeepResearch Webスクレイピングを、Playwrightを使用して10ブラウザ×10タブ（合計100タブ）の
並列構成で実行します。

**重要**: この実装は検知目的のみで、生成目的ではない。安全判定と拒否挙動の学習を目的とする。

Usage:
    python scripts/data/drug_detection_deepresearch_scraping.py --output D:\webdataset\drug_detection_deepresearch
"""

import sys
import json
import logging
import asyncio
import argparse
import random
import time
import psutil
import torch
import os
import subprocess
import platform
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
except ImportError:
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
    print("[WARNING] ParallelTabProcessor not available")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/drug_detection_deepresearch_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 違法薬物検知用キーワードカテゴリ
DRUG_DETECTION_KEYWORDS = {
    'ja': {
        'illegal_drugs': [
            '違法薬物', '麻薬', '覚醒剤', '大麻', 'コカイン', 'ヘロイン', 'MDMA', 'LSD', '幻覚剤', '向精神薬',
            'メタンフェタミン', 'アンフェタミン', 'エクスタシー', 'ケタミン', 'フェンタニル'
        ],
        'prescription_drugs_abuse': [
            '処方薬乱用', 'オピオイド乱用', '鎮痛剤乱用', '睡眠薬乱用', '抗不安薬乱用',
            'オキシコドン', 'モルヒネ', 'コデイン', 'ベンゾジアゼピン'
        ],
        'controlled_substances': [
            '指定薬物', '規制薬物', '向精神薬', '麻薬指定', '覚醒剤指定',
            '危険ドラッグ', '脱法ハーブ', '合法ハーブ'
        ],
        'drug_trafficking': [
            '薬物密輸', '薬物取引', '薬物販売', '薬物密売', '薬物運搬'
        ],
        'drug_manufacturing': [
            '薬物製造', '覚醒剤製造', '麻薬製造', '違法製造', '薬物合成'
        ]
    },
    'en': {
        'illegal_drugs': [
            'illegal drugs', 'narcotics', 'amphetamine', 'cannabis', 'cocaine', 'heroin', 'MDMA', 'LSD',
            'hallucinogen', 'psychotropic', 'methamphetamine', 'ecstasy', 'ketamine', 'fentanyl'
        ],
        'prescription_drugs_abuse': [
            'prescription drug abuse', 'opioid abuse', 'painkiller abuse', 'sleeping pill abuse',
            'anxiety medication abuse', 'oxycodone', 'morphine', 'codeine', 'benzodiazepine'
        ],
        'controlled_substances': [
            'controlled substance', 'scheduled drug', 'psychotropic substance', 'narcotic',
            'designer drug', 'legal high', 'bath salts'
        ],
        'drug_trafficking': [
            'drug trafficking', 'drug trade', 'drug dealing', 'drug smuggling', 'drug distribution'
        ],
        'drug_manufacturing': [
            'drug manufacturing', 'amphetamine manufacturing', 'narcotic manufacturing',
            'illegal manufacturing', 'drug synthesis'
        ]
    }
}

# ドメイン別知識キーワード（既存のCATEGORY_KEYWORDSから）
DOMAIN_KNOWLEDGE_KEYWORDS = {
    'ja': {
        'technology': [
            '人工知能', '機械学習', '深層学習', '自然言語処理', 'コンピュータビジョン',
            'ブロックチェーン', '暗号通貨', '量子コンピュータ', 'IoT', '5G',
            'クラウドコンピューティング', 'マイクロサービス', 'DevOps', 'コンテナ',
            'Python', 'JavaScript', 'TypeScript', 'Rust', 'Go', 'Kotlin',
            'ニューラルネットワーク', '強化学習', '転移学習', 'GAN', 'Transformer'
        ],
        'programming_languages': [
            'Python', 'JavaScript', 'TypeScript', 'Go', 'Rust', 'C++', 'C#', 'Swift', 'Kotlin',
            'Dart', 'Scala', 'Elixir', 'Erlang', 'Clojure', 'F#', 'Haskell', 'OCaml',
            'COBOL', 'PHP', 'Java', 'C', 'Fortran', 'Pascal', 'Ada', 'PL/I', 'BASIC',
            'Visual Basic', 'Delphi', 'Perl', 'Ruby', 'Lua', 'Tcl',
            'アセンブリ言語', '機械語', 'x86', 'ARM', 'RISC-V', 'アセンブラ', 'アセンブリ',
            'マシンコード', 'バイナリ', 'オペコード', 'レジスタ', 'メモリマップ',
            'SQL', 'PostgreSQL', 'MySQL', 'SQL Server', 'Oracle', 'SQLite', 'MariaDB',
            'NoSQL', 'MongoDB', 'Redis', 'Cassandra', 'Elasticsearch',
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
            'Python', 'JavaScript', 'TypeScript', 'Go', 'Rust', 'C++', 'C#', 'Swift', 'Kotlin',
            'Dart', 'Scala', 'Elixir', 'Erlang', 'Clojure', 'F#', 'Haskell', 'OCaml',
            'COBOL', 'PHP', 'Java', 'C', 'Fortran', 'Pascal', 'Ada', 'PL/I', 'BASIC',
            'Visual Basic', 'Delphi', 'Perl', 'Ruby', 'Lua', 'Tcl',
            'assembly language', 'machine code', 'x86', 'ARM', 'RISC-V', 'assembler', 'assembly',
            'machine code', 'binary', 'opcode', 'register', 'memory map',
            'SQL', 'PostgreSQL', 'MySQL', 'SQL Server', 'Oracle', 'SQLite', 'MariaDB',
            'NoSQL', 'MongoDB', 'Redis', 'Cassandra', 'Elasticsearch',
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
        ]
    }
}

# NSFW検知用キーワード（検知目的のみ、生成目的ではない）
NSFW_DETECTION_KEYWORDS = {
    'ja': {
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

# データソース定義
DATA_SOURCES = {
    'pmda': {
        'base_url': 'https://www.pmda.go.jp',
        'name': 'PMDA (医薬品医療機器総合機構)',
        'language': 'ja',
        'enabled': True
    },
    'fda': {
        'base_url': 'https://www.fda.gov',
        'name': 'FDA (Food and Drug Administration)',
        'language': 'en',
        'enabled': True
    },
    'egov': {
        'base_url': 'https://www.e-gov.go.jp',
        'name': 'e-Gov (日本の法令データベース)',
        'language': 'ja',
        'enabled': True
    },
    'who': {
        'base_url': 'https://www.who.int',
        'name': 'WHO (World Health Organization)',
        'language': 'en',
        'enabled': True
    },
    'unodc': {
        'base_url': 'https://www.unodc.org',
        'name': 'UNODC (United Nations Office on Drugs and Crime)',
        'language': 'en',
        'enabled': True
    },
    'emcdda': {
        'base_url': 'https://www.emcdda.europa.eu',
        'name': 'EMCDDA (European Monitoring Centre for Drugs and Drug Addiction)',
        'language': 'en',
        'enabled': True
    },
    'wikipedia': {
        'base_url': 'https://ja.wikipedia.org',
        'name': 'Wikipedia',
        'language': 'ja',
        'enabled': True
    }
}

# 技術ドキュメントサイト
TECH_DOCUMENTATION_SITES = {
    'ja': [
        'https://github.com/trending',
        'https://qiita.com/',
        'https://zenn.dev/',
        'https://dev.to/',
        'https://docs.python.org/',
        'https://developer.mozilla.org/',
    ],
    'en': [
        'https://github.com/trending',
        'https://stackoverflow.com/',
        'https://www.freecodecamp.org/',
        'https://www.codecademy.com/',
        'https://techcrunch.com/',
        'https://www.infoq.com/',
    ]
}

# コーディング教育サイト
CODING_EDUCATION_SITES = {
    'ja': [
        'https://www.freecodecamp.org/',
        'https://www.codecademy.com/',
    ],
    'en': [
        'https://www.freecodecamp.org/',
        'https://www.codecademy.com/',
        'https://www.coursera.org/',
        'https://www.udemy.com/',
    ]
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
    drug_detection: bool = False  # 違法薬物検知フラグ
    domain_knowledge: bool = False  # ドメイン別知識フラグ
    nsfw_detection: bool = False  # NSFW検知フラグ


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


# 既存のparallel_deep_research_scraping.pyから必要なクラスとメソッドをインポート
# 長いファイルのため、主要な部分のみ実装し、残りは既存実装を参照
# ここでは、違法薬物検知目的に特化した部分のみを実装

# 続きは次のメッセージで実装します


