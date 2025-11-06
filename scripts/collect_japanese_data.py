"""
日本語公開データ収集・クリーニングスクリプト（本番環境対応版）

防衛・航空宇宙・運輸向けドメイン特化LLMのための
高品質日本語コーパスを収集・クリーニングする。

データソース:
1. Wikipedia日本語ダンプ
2. CC-100日本語コーパス
3. 青空文庫（技術文書）
4. 公開技術文書・論文

本番環境要件:
- 電源断リカバリー（自動チェックポイント）
- メモリ効率的な処理（ストリーミング）
- 詳細なログ・監査
- プログレス可視化（tqdm）
- エラーハンドリング

Author: SO8T Project Team
Date: 2024-11-06
"""

import os
import sys
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from tqdm import tqdm
import signal
import pickle

# データ処理ライブラリ
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("[WARNING] requests or beautifulsoup4 not installed. Install with: pip install requests beautifulsoup4")
    requests = None
    BeautifulSoup = None

try:
    import datasets
    from datasets import load_dataset
except ImportError:
    print("[WARNING] datasets not installed. Install with: pip install datasets")
    datasets = None

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('collect_japanese_data.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """データサンプルの構造"""
    text: str
    source: str
    domain: str
    quality_score: float
    length: int
    hash: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_text(cls, text: str, source: str, domain: str = "general", 
                  metadata: Optional[Dict] = None):
        """テキストからDataSampleを作成"""
        cleaned_text = clean_text(text)
        return cls(
            text=cleaned_text,
            source=source,
            domain=domain,
            quality_score=estimate_quality(cleaned_text),
            length=len(cleaned_text),
            hash=hashlib.md5(cleaned_text.encode('utf-8')).hexdigest(),
            metadata=metadata or {}
        )


class DataCollectionSession:
    """データ収集セッション（電源断リカバリー対応）"""
    
    def __init__(self, session_dir: str = "data_collection_sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = self.session_dir / f"session_{self.session_id}.pkl"
        
        self.samples: List[DataSample] = []
        self.processed_hashes: set = set()
        self.stats = {
            'total_collected': 0,
            'total_duplicates': 0,
            'total_filtered': 0,
            'sources': {},
            'domains': {},
        }
        
        # シグナルハンドラー設定（Ctrl+C対応）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"[Session] Created session: {self.session_id}")
    
    def _signal_handler(self, signum, frame):
        """シグナル受信時の緊急保存"""
        logger.warning(f"[Session] Received signal {signum}, saving session...")
        self.save()
        logger.info(f"[Session] Session saved. Exiting.")
        sys.exit(0)
    
    def add_sample(self, sample: DataSample) -> bool:
        """サンプルを追加（重複チェック付き）"""
        if sample.hash in self.processed_hashes:
            self.stats['total_duplicates'] += 1
            return False
        
        if sample.quality_score < 0.3:  # 品質フィルタ
            self.stats['total_filtered'] += 1
            return False
        
        self.samples.append(sample)
        self.processed_hashes.add(sample.hash)
        self.stats['total_collected'] += 1
        
        # 統計更新
        self.stats['sources'][sample.source] = self.stats['sources'].get(sample.source, 0) + 1
        self.stats['domains'][sample.domain] = self.stats['domains'].get(sample.domain, 0) + 1
        
        return True
    
    def save(self, path: Optional[Path] = None):
        """セッションを保存"""
        save_path = path or self.session_path
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'session_id': self.session_id,
                'samples': self.samples,
                'processed_hashes': self.processed_hashes,
                'stats': self.stats,
                'timestamp': datetime.now().isoformat(),
            }, f)
        
        logger.info(f"[Session] Saved to {save_path}")
    
    @classmethod
    def load(cls, session_path: str):
        """セッションをロード"""
        session_path = Path(session_path)
        
        with open(session_path, 'rb') as f:
            data = pickle.load(f)
        
        session = cls()
        session.session_id = data['session_id']
        session.samples = data['samples']
        session.processed_hashes = data['processed_hashes']
        session.stats = data['stats']
        
        logger.info(f"[Session] Loaded session: {session.session_id}")
        logger.info(f"[Session] Samples: {len(session.samples)}")
        
        return session
    
    def export_jsonl(self, output_path: str):
        """JSONL形式でエクスポート"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"[Export] Exported {len(self.samples)} samples to {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            'session_id': self.session_id,
            'total_samples': len(self.samples),
            'total_collected': self.stats['total_collected'],
            'total_duplicates': self.stats['total_duplicates'],
            'total_filtered': self.stats['total_filtered'],
            'sources': self.stats['sources'],
            'domains': self.stats['domains'],
            'avg_length': sum(s.length for s in self.samples) / max(len(self.samples), 1),
            'avg_quality': sum(s.quality_score for s in self.samples) / max(len(self.samples), 1),
        }


def clean_text(text: str) -> str:
    """
    テキストをクリーニング
    
    - 不要な空白・改行を削除
    - HTMLタグを除去
    - 制御文字を除去
    - 連続空白を単一化
    """
    if not text:
        return ""
    
    # HTMLタグを除去
    text = re.sub(r'<[^>]+>', '', text)
    
    # 制御文字を除去（改行・タブは保持）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 連続空白を単一化
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 連続改行を2つまでに制限
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 前後の空白を削除
    text = text.strip()
    
    return text


def estimate_quality(text: str) -> float:
    """
    テキストの品質を推定（0.0-1.0）
    
    評価基準:
    - 長さ（短すぎる・長すぎるテキストは低品質）
    - 日本語文字の割合
    - 記号・数字の割合
    - 文の完全性
    """
    if not text or len(text) < 50:
        return 0.0
    
    score = 1.0
    
    # 長さペナルティ
    if len(text) < 100:
        score *= 0.5
    elif len(text) > 50000:
        score *= 0.8
    
    # 日本語文字の割合
    japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))
    japanese_ratio = japanese_chars / len(text)
    
    if japanese_ratio < 0.3:
        score *= 0.3
    elif japanese_ratio > 0.7:
        score *= 1.0
    else:
        score *= japanese_ratio / 0.7
    
    # 記号の割合（多すぎるとノイズ）
    symbol_chars = len(re.findall(r'[^\w\s]', text))
    symbol_ratio = symbol_chars / len(text)
    
    if symbol_ratio > 0.3:
        score *= 0.5
    
    # 改行の頻度（適度な改行は良質）
    newline_ratio = text.count('\n') / max(len(text), 1)
    if 0.01 < newline_ratio < 0.1:
        score *= 1.2
    
    return min(score, 1.0)


def collect_wikipedia_ja(session: DataCollectionSession, max_samples: int = 10000):
    """
    Wikipedia日本語版からデータを収集
    
    Args:
        session: データ収集セッション
        max_samples: 最大サンプル数
    """
    logger.info("[Wikipedia] Starting collection...")
    
    if datasets is None:
        logger.error("[Wikipedia] datasets library not installed")
        return
    
    try:
        # Wikipedia日本語データセットをロード
        logger.info("[Wikipedia] Loading dataset (this may take a while)...")
        ds = load_dataset("wikipedia", "20220301.ja", split="train", streaming=True)
        
        logger.info("[Wikipedia] Processing articles...")
        
        collected = 0
        with tqdm(total=max_samples, desc="Wikipedia", unit="article") as pbar:
            for article in ds:
                if collected >= max_samples:
                    break
                
                text = article.get('text', '')
                title = article.get('title', '')
                
                if len(text) < 200:  # 短すぎる記事はスキップ
                    continue
                
                # ドメイン判定（キーワードベース）
                domain = classify_domain(text, title)
                
                sample = DataSample.from_text(
                    text=text,
                    source="wikipedia_ja",
                    domain=domain,
                    metadata={'title': title}
                )
                
                if session.add_sample(sample):
                    collected += 1
                    pbar.update(1)
                
                # 定期的に保存
                if collected % 1000 == 0:
                    session.save()
        
        logger.info(f"[Wikipedia] Collected {collected} articles")
        
    except Exception as e:
        logger.error(f"[Wikipedia] Error: {e}")


def collect_cc100_ja(session: DataCollectionSession, max_samples: int = 10000):
    """
    CC-100日本語コーパスからデータを収集
    
    Args:
        session: データ収集セッション
        max_samples: 最大サンプル数
    """
    logger.info("[CC-100] Starting collection...")
    
    if datasets is None:
        logger.error("[CC-100] datasets library not installed")
        return
    
    try:
        # CC-100日本語データセットをロード
        logger.info("[CC-100] Loading dataset (this may take a while)...")
        ds = load_dataset("cc100", "ja", split="train", streaming=True)
        
        logger.info("[CC-100] Processing documents...")
        
        collected = 0
        with tqdm(total=max_samples, desc="CC-100", unit="doc") as pbar:
            for doc in ds:
                if collected >= max_samples:
                    break
                
                text = doc.get('text', '')
                
                if len(text) < 200:  # 短すぎる文書はスキップ
                    continue
                
                # ドメイン判定
                domain = classify_domain(text, "")
                
                sample = DataSample.from_text(
                    text=text,
                    source="cc100_ja",
                    domain=domain,
                    metadata={}
                )
                
                if session.add_sample(sample):
                    collected += 1
                    pbar.update(1)
                
                # 定期的に保存
                if collected % 1000 == 0:
                    session.save()
        
        logger.info(f"[CC-100] Collected {collected} documents")
        
    except Exception as e:
        logger.error(f"[CC-100] Error: {e}")


def classify_domain(text: str, title: str = "") -> str:
    """
    テキストのドメインを分類
    
    Args:
        text: テキスト
        title: タイトル（あれば）
        
    Returns:
        ドメイン名
    """
    combined = (title + " " + text).lower()
    
    # 防衛
    defense_keywords = ['防衛', '軍事', '安全保障', 'セキュリティ', '兵器', '戦略']
    if any(kw in combined for kw in defense_keywords):
        return "defense"
    
    # 航空宇宙
    aerospace_keywords = ['航空', '宇宙', 'ロケット', '衛星', '航空機', '飛行']
    if any(kw in combined for kw in aerospace_keywords):
        return "aerospace"
    
    # 運輸
    transport_keywords = ['運輸', '物流', '輸送', '交通', '鉄道', '船舶']
    if any(kw in combined for kw in transport_keywords):
        return "transport"
    
    # 技術
    tech_keywords = ['技術', '工学', 'エンジニアリング', 'システム', '開発']
    if any(kw in combined for kw in tech_keywords):
        return "technology"
    
    return "general"


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Japanese data for SO8T training")
    parser.add_argument("--output", type=str, default="data/japanese_collected.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--max_samples", type=int, default=20000,
                       help="Maximum samples to collect")
    parser.add_argument("--wikipedia", action="store_true",
                       help="Collect from Wikipedia")
    parser.add_argument("--cc100", action="store_true",
                       help="Collect from CC-100")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from session file")
    parser.add_argument("--session_dir", type=str, default="data_collection_sessions",
                       help="Session directory")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SO8T Japanese Data Collection")
    print("=" * 80)
    print(f"\n[Config]")
    print(f"  Output: {args.output}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Wikipedia: {args.wikipedia}")
    print(f"  CC-100: {args.cc100}")
    print(f"  Resume: {args.resume}")
    
    # セッション作成またはロード
    if args.resume:
        logger.info(f"[Main] Resuming from {args.resume}")
        session = DataCollectionSession.load(args.resume)
    else:
        logger.info(f"[Main] Creating new session")
        session = DataCollectionSession(session_dir=args.session_dir)
    
    # データ収集
    if args.wikipedia or (not args.wikipedia and not args.cc100):
        collect_wikipedia_ja(session, max_samples=args.max_samples // 2)
    
    if args.cc100 or (not args.wikipedia and not args.cc100):
        collect_cc100_ja(session, max_samples=args.max_samples // 2)
    
    # 最終保存
    session.save()
    
    # エクスポート
    session.export_jsonl(args.output)
    
    # 統計情報
    stats = session.get_stats()
    print("\n" + "=" * 80)
    print("Collection Statistics")
    print("=" * 80)
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total collected: {stats['total_collected']}")
    print(f"  Total duplicates: {stats['total_duplicates']}")
    print(f"  Total filtered: {stats['total_filtered']}")
    print(f"  Average length: {stats['avg_length']:.1f} chars")
    print(f"  Average quality: {stats['avg_quality']:.3f}")
    print(f"\n  Sources:")
    for source, count in stats['sources'].items():
        print(f"    {source}: {count}")
    print(f"\n  Domains:")
    for domain, count in stats['domains'].items():
        print(f"    {domain}: {count}")
    
    # 統計情報をJSONで保存
    stats_path = Path(args.output).parent / "collection_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Stats saved to: {stats_path}")
    print("\n" + "=" * 80)
    print("[Collection] Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

