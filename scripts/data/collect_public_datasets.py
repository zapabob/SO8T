#!/usr/bin/env python3
"""
日本語公開データセット収集スクリプト
Wikipedia-ja、OSCAR-ja、CC-100から日本語データを収集
"""

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    電源断リカバリー対応データ収集器
    """
    
    def __init__(
        self,
        output_file: Path,
        session_file: Path,
        max_samples: int = 10000,
        min_length: int = 50,
        max_length: int = 2048,
    ):
        """
        Args:
            output_file: 出力JSONLファイルパス
            session_file: セッション管理ファイルパス
            max_samples: 最大サンプル数
            min_length: 最小文字数
            max_length: 最大文字数
        """
        self.output_file = output_file
        self.session_file = session_file
        self.max_samples = max_samples
        self.min_length = min_length
        self.max_length = max_length
        
        self.collected_samples = []
        self.session = self._load_session()
        
        # シグナルハンドラー登録（電源断対応）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_session(self) -> Dict:
        """セッション情報をロード"""
        if self.session_file.exists():
            with open(self.session_file, 'r', encoding='utf-8') as f:
                session = json.load(f)
            logger.info(f"[SESSION] Resuming from {session['collected_count']} samples")
            return session
        else:
            return {
                'collected_count': 0,
                'datasets_processed': {},
            }
    
    def _save_session(self):
        """セッション情報を保存"""
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(self.session, f, indent=2, ensure_ascii=False)
        logger.debug("[SESSION] Saved")
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー（Ctrl+C対応）"""
        logger.warning(f"\n[SIGNAL] Received signal {signum}, saving progress...")
        self._save_all()
        logger.info("[EXIT] Data saved successfully")
        sys.exit(0)
    
    def _save_all(self):
        """全データを保存"""
        # 収集済みデータを追記
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if self.output_file.exists() else 'w'
        with open(self.output_file, mode, encoding='utf-8') as f:
            for sample in self.collected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(self.collected_samples)} samples to {self.output_file}")
        self.collected_samples = []
        
        # セッション保存
        self._save_session()
    
    def _filter_sample(self, text: str) -> bool:
        """サンプルの品質フィルタリング"""
        if not text:
            return False
        
        # 長さチェック
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # 記号比率チェック
        symbol_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        symbol_ratio = symbol_count / len(text)
        if symbol_ratio > 0.3:  # 記号が30%以上
            return False
        
        # URL除外
        if 'http://' in text or 'https://' in text:
            return False
        
        return True
    
    def collect_wikipedia_ja(self, target_samples: int) -> int:
        """
        Wikipedia日本語版から収集
        
        Args:
            target_samples: 目標サンプル数
        
        Returns:
            collected: 収集したサンプル数
        """
        dataset_name = 'wikipedia_ja'
        
        # セッションから進捗を復元
        start_idx = self.session['datasets_processed'].get(dataset_name, 0)
        
        logger.info(f"[WIKIPEDIA] Loading Japanese Wikipedia (starting from {start_idx})...")
        
        try:
            # Wikipediaデータセットをストリーミングでロード
            dataset = load_dataset(
                "wikipedia",
                "20220301.ja",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            collected = 0
            processed = 0
            
            # 進捗をスキップ
            dataset_iter = iter(dataset)
            for _ in range(start_idx):
                try:
                    next(dataset_iter)
                except StopIteration:
                    break
            
            # データ収集
            with tqdm(total=target_samples, desc="Wikipedia-ja") as pbar:
                for sample in dataset_iter:
                    processed += 1
                    
                    text = sample.get('text', '')
                    
                    if self._filter_sample(text):
                        data_sample = {
                            'text': text,
                            'source': 'wikipedia_ja',
                            'domain': 'general',
                        }
                        self.collected_samples.append(data_sample)
                        self.session['collected_count'] += 1
                        collected += 1
                        pbar.update(1)
                        
                        # 定期保存（100サンプルごと）
                        if len(self.collected_samples) >= 100:
                            self._save_all()
                    
                    # 進捗更新
                    self.session['datasets_processed'][dataset_name] = start_idx + processed
                    
                    if collected >= target_samples:
                        break
            
            logger.info(f"[WIKIPEDIA] Collected {collected} samples")
            return collected
        
        except Exception as e:
            logger.error(f"[WIKIPEDIA] Error: {e}")
            return 0
    
    def collect_oscar_ja(self, target_samples: int) -> int:
        """
        OSCAR日本語版から収集
        
        Args:
            target_samples: 目標サンプル数
        
        Returns:
            collected: 収集したサンプル数
        """
        dataset_name = 'oscar_ja'
        
        # セッションから進捗を復元
        start_idx = self.session['datasets_processed'].get(dataset_name, 0)
        
        logger.info(f"[OSCAR] Loading OSCAR Japanese (starting from {start_idx})...")
        
        try:
            # OSCARデータセットをストリーミングでロード
            dataset = load_dataset(
                "oscar-corpus/OSCAR-2301",
                "ja",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            collected = 0
            processed = 0
            
            # 進捗をスキップ
            dataset_iter = iter(dataset)
            for _ in range(start_idx):
                try:
                    next(dataset_iter)
                except StopIteration:
                    break
            
            # データ収集
            with tqdm(total=target_samples, desc="OSCAR-ja") as pbar:
                for sample in dataset_iter:
                    processed += 1
                    
                    text = sample.get('text', '')
                    
                    if self._filter_sample(text):
                        data_sample = {
                            'text': text,
                            'source': 'oscar_ja',
                            'domain': 'general',
                        }
                        self.collected_samples.append(data_sample)
                        self.session['collected_count'] += 1
                        collected += 1
                        pbar.update(1)
                        
                        # 定期保存
                        if len(self.collected_samples) >= 100:
                            self._save_all()
                    
                    # 進捗更新
                    self.session['datasets_processed'][dataset_name] = start_idx + processed
                    
                    if collected >= target_samples:
                        break
            
            logger.info(f"[OSCAR] Collected {collected} samples")
            return collected
        
        except Exception as e:
            logger.error(f"[OSCAR] Error: {e}")
            return 0
    
    def collect_all(self):
        """全データソースから収集"""
        total_collected = self.session['collected_count']
        remaining = self.max_samples - total_collected
        
        if remaining <= 0:
            logger.info(f"[COMPLETE] Already collected {total_collected}/{self.max_samples} samples")
            return
        
        logger.info(f"[START] Collecting {remaining} more samples (total target: {self.max_samples})")
        
        # Wikipedia から 60%
        if remaining > 0:
            wikipedia_target = min(int(remaining * 0.6), remaining)
            collected = self.collect_wikipedia_ja(wikipedia_target)
            total_collected += collected
            remaining = self.max_samples - total_collected
        
        # OSCAR から残り
        if remaining > 0:
            collected = self.collect_oscar_ja(remaining)
            total_collected += collected
        
        # 最終保存
        self._save_all()
        
        logger.info(f"[COMPLETE] Total collected: {total_collected}/{self.max_samples} samples")


def main():
    parser = argparse.ArgumentParser(description="Collect Japanese public datasets")
    parser.add_argument(
        "--output",
        type=str,
        default="data/phi4_japanese_public.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--session",
        type=str,
        default="data/.collection_session.json",
        help="Session file for recovery"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum number of samples to collect"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=50,
        help="Minimum text length"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum text length"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Japanese Public Dataset Collection")
    logger.info("=" * 70)
    logger.info(f"Output file: {args.output}")
    logger.info(f"Session file: {args.session}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Length range: {args.min_length}-{args.max_length}")
    logger.info("=" * 70)
    
    collector = DataCollector(
        output_file=Path(args.output),
        session_file=Path(args.session),
        max_samples=args.max_samples,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    
    collector.collect_all()
    
    logger.info("[SUCCESS] Data collection completed!")


if __name__ == "__main__":
    main()

