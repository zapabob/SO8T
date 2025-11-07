#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語データセットクレンジングスクリプト

重複除去、品質フィルタリング、日本語含有率チェック、長さ正規化を実行

Usage:
    python scripts/clean_japanese_dataset.py --input data/collected --output data/cleaned
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JapaneseDatasetCleaner:
    """日本語データセットクレンジングクラス"""
    
    def __init__(
        self,
        min_quality: float = 0.5,  # 合成データセット用に緩和
        min_japanese_ratio: float = 0.2,  # 合成データセット用に緩和
        min_length: int = 20,  # 合成データセット用に短縮
        max_length: int = 5000
    ):
        """
        Args:
            min_quality: 最小品質スコア
            min_japanese_ratio: 最小日本語含有率
            min_length: 最小文字数
            max_length: 最大文字数
        """
        self.min_quality = min_quality
        self.min_japanese_ratio = min_japanese_ratio
        self.min_length = min_length
        self.max_length = max_length
        
        logger.info("Japanese Dataset Cleaner initialized")
        logger.info(f"  Min quality: {min_quality}")
        logger.info(f"  Min Japanese ratio: {min_japanese_ratio}")
        logger.info(f"  Length range: {min_length}-{max_length}")
    
    def estimate_quality(self, text: str) -> float:
        """テキスト品質スコア計算（0.0-1.0）"""
        if not text or len(text) < 10:
            return 0.0
        
        score = 0.0
        length = len(text)
        
        # 長さスコア（50-500文字が最適）
        if 50 <= length <= 500:
            score += 0.3
        elif 500 < length <= 1000:
            score += 0.2
        elif length > 1000:
            score += 0.1
        
        # 日本語含有率
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf')
        japanese_ratio = japanese_chars / len(text) if len(text) > 0 else 0.0
        score += japanese_ratio * 0.4
        
        # 句読点の適切さ
        punctuation_count = text.count('。') + text.count('、')
        if 2 <= punctuation_count <= length / 50:
            score += 0.2
        
        # 重複文字列チェック
        unique_ratio = len(set(text)) / len(text) if len(text) > 0 else 0.0
        if unique_ratio > 0.3:
            score += 0.1
        
        return min(score, 1.0)
    
    def check_japanese_ratio(self, text: str) -> float:
        """日本語含有率を計算"""
        if not text:
            return 0.0
        
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf')
        return japanese_chars / len(text)
    
    def normalize_length(self, text: str) -> str:
        """長さ正規化（50-5000文字）"""
        if len(text) < self.min_length:
            return ""  # 短すぎる場合は除外
        
        if len(text) > self.max_length:
            # 長すぎる場合は切り詰め（文の境界で）
            text = text[:self.max_length]
            # 最後の文の境界を探す
            last_period = text.rfind('。')
            last_comma = text.rfind('、')
            last_newline = text.rfind('\n')
            
            cut_pos = max(last_period, last_comma, last_newline)
            if cut_pos > self.min_length:
                text = text[:cut_pos + 1]
        
        return text
    
    def compute_hash(self, text: str) -> str:
        """テキストのハッシュ値を計算（重複検出用）"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def clean_dataset(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict[str, int]:
        """データセットクレンジング実行"""
        logger.info("="*80)
        logger.info("Japanese Dataset Cleaning")
        logger.info("="*80)
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 入力ファイル検索
        input_files = list(input_dir.glob("*.jsonl"))
        if not input_files:
            logger.error(f"No JSONL files found in {input_dir}")
            return {}
        
        logger.info(f"Found {len(input_files)} input files")
        
        # 統計
        stats = {
            "total": 0,
            "duplicate": 0,
            "low_quality": 0,
            "low_japanese": 0,
            "invalid_length": 0,
            "passed": 0
        }
        
        seen_hashes: Set[str] = set()
        cleaned_samples: List[Dict] = []
        
        # 全ファイルを処理
        for input_file in input_files:
            logger.info(f"Processing {input_file.name}...")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Cleaning {input_file.name}"):
                    try:
                        sample = json.loads(line.strip())
                        stats["total"] += 1
                        
                        # テキスト抽出（複数フォーマット対応）
                        text = sample.get("text", "")
                        if not text:
                            # 合成データセット形式（outputフィールド）
                            text = sample.get("output", "")
                        if not text:
                            # instruction + output形式
                            instruction = sample.get("instruction", "")
                            output = sample.get("output", "")
                            if instruction and output:
                                text = f"{instruction}\n{output}"
                            elif instruction:
                                text = instruction
                            elif output:
                                text = output
                        if not text:
                            continue
                        
                        # 重複チェック（完全一致のみ、テキストが十分長い場合）
                        # 合成データセットはテンプレートベースなので、完全一致のみを重複とみなす
                        if len(text) >= 50:  # 長いテキストのみ重複チェック
                            text_hash = self.compute_hash(text)
                            if text_hash in seen_hashes:
                                stats["duplicate"] += 1
                                continue
                            seen_hashes.add(text_hash)
                        
                        # 長さ正規化
                        text = self.normalize_length(text)
                        if not text:
                            stats["invalid_length"] += 1
                            continue
                        
                        # 品質チェック
                        quality = self.estimate_quality(text)
                        if quality < self.min_quality:
                            stats["low_quality"] += 1
                            continue
                        
                        # 日本語含有率チェック
                        japanese_ratio = self.check_japanese_ratio(text)
                        if japanese_ratio < self.min_japanese_ratio:
                            stats["low_japanese"] += 1
                            continue
                        
                        # クレンジング済みサンプル
                        cleaned_sample = {
                            "text": text,
                            "quality_score": quality,
                            "japanese_ratio": japanese_ratio,
                            "length": len(text),
                            **{k: v for k, v in sample.items() if k != "text"}
                        }
                        cleaned_samples.append(cleaned_sample)
                        stats["passed"] += 1
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
        
        # 出力ファイルに保存
        output_file = output_dir / "cleaned_japanese_dataset.jsonl"
        logger.info(f"Saving cleaned dataset to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in cleaned_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 統計レポート
        logger.info("="*80)
        logger.info("Cleaning Statistics")
        logger.info("="*80)
        logger.info(f"Total samples: {stats['total']:,}")
        logger.info(f"Duplicates removed: {stats['duplicate']:,}")
        logger.info(f"Low quality removed: {stats['low_quality']:,}")
        logger.info(f"Low Japanese ratio removed: {stats['low_japanese']:,}")
        logger.info(f"Invalid length removed: {stats['invalid_length']:,}")
        logger.info(f"Passed: {stats['passed']:,} ({stats['passed']/stats['total']*100:.1f}%)")
        logger.info("="*80)
        
        return stats


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Clean Japanese dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing JSONL files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for cleaned dataset"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.7,
        help="Minimum quality score (default: 0.7)"
    )
    parser.add_argument(
        "--min-japanese-ratio",
        type=float,
        default=0.3,
        help="Minimum Japanese character ratio (default: 0.3)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum text length (default: 50)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=5000,
        help="Maximum text length (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # クレンジング実行
    cleaner = JapaneseDatasetCleaner(
        min_quality=args.min_quality,
        min_japanese_ratio=args.min_japanese_ratio,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    try:
        stats = cleaner.clean_dataset(
            input_dir=Path(args.input),
            output_dir=Path(args.output)
        )
        
        logger.info("[SUCCESS] Dataset cleaning completed")
        return 0
        
    except Exception as e:
        logger.error(f"[FAILED] Dataset cleaning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
