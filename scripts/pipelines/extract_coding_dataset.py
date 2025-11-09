#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コーディング関連データ抽出スクリプト

既存のデータセットからコーディング関連データを抽出し、
コーディングタスクの種類別に分類します。

Usage:
    python scripts/pipelines/extract_coding_dataset.py --input D:\webdataset\processed --output D:\webdataset\coding_dataset
"""

import sys
import json
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/extract_coding_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CodingDatasetExtractor:
    """コーディング関連データ抽出クラス"""
    
    def __init__(
        self,
        output_dir: Path,
        min_code_length: int = 10,
        min_text_length: int = 50
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            min_code_length: 最小コード長
            min_text_length: 最小テキスト長
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_code_length = min_code_length
        self.min_text_length = min_text_length
        
        # コーディング関連キーワード
        self.coding_keywords = {
            'ja': [
                'コード', 'プログラミング', '関数', '変数', 'クラス', 'メソッド',
                'アルゴリズム', 'データ構造', 'デバッグ', 'エラー', 'バグ',
                'リファクタリング', '最適化', 'パフォーマンス', 'テスト',
                'API', 'ライブラリ', 'フレームワーク', 'モジュール', 'パッケージ',
                'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Rust', 'Go',
                'SQL', 'HTML', 'CSS', 'JSON', 'XML', 'YAML'
            ],
            'en': [
                'code', 'programming', 'function', 'variable', 'class', 'method',
                'algorithm', 'data structure', 'debug', 'error', 'bug',
                'refactoring', 'optimization', 'performance', 'test',
                'API', 'library', 'framework', 'module', 'package',
                'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Rust', 'Go',
                'SQL', 'HTML', 'CSS', 'JSON', 'XML', 'YAML'
            ]
        }
        
        # コーディングタスクの種類
        self.task_types = {
            'code_generation': {
                'keywords': ['generate', 'create', 'write', 'implement', '作成', '生成', '実装', '書く'],
                'patterns': [r'write.*code', r'create.*function', r'implement.*class', r'コード.*書', r'関数.*作成']
            },
            'code_explanation': {
                'keywords': ['explain', 'describe', 'what does', 'how does', '説明', '解説', '何を', 'どのように'],
                'patterns': [r'explain.*code', r'what.*does', r'how.*work', r'コード.*説明', r'動作.*説明']
            },
            'debugging': {
                'keywords': ['debug', 'fix', 'error', 'bug', 'issue', 'デバッグ', '修正', 'エラー', 'バグ', '問題'],
                'patterns': [r'fix.*error', r'debug.*code', r'エラー.*修正', r'バグ.*修正']
            },
            'refactoring': {
                'keywords': ['refactor', 'improve', 'optimize', 'clean', 'リファクタリング', '改善', '最適化', '整理'],
                'patterns': [r'refactor.*code', r'improve.*code', r'コード.*改善', r'コード.*最適化']
            },
            'code_review': {
                'keywords': ['review', 'check', 'analyze', 'evaluate', 'レビュー', '確認', '分析', '評価'],
                'patterns': [r'review.*code', r'check.*code', r'コード.*レビュー', r'コード.*確認']
            },
            'testing': {
                'keywords': ['test', 'unit test', 'integration test', 'テスト', 'ユニットテスト', '統合テスト'],
                'patterns': [r'write.*test', r'create.*test', r'テスト.*書', r'テスト.*作成']
            }
        }
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.extracted_samples: List[Dict] = []
        
        logger.info("="*80)
        logger.info("Coding Dataset Extractor Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Min code length: {self.min_code_length}")
        logger.info(f"Min text length: {self.min_text_length}")
    
    def detect_language(self, text: str) -> str:
        """テキストの言語を検出"""
        ja_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF]', text))
        en_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if ja_chars > en_chars:
            return 'ja'
        else:
            return 'en'
    
    def extract_code_blocks(self, text: str) -> List[Dict]:
        """テキストからコードブロックを抽出"""
        code_blocks = []
        
        # Markdownコードブロック
        markdown_pattern = r'```(\w+)?\n(.*?)```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            language = match.group(1) or 'unknown'
            code = match.group(2).strip()
            if len(code) >= self.min_code_length:
                code_blocks.append({
                    'language': language,
                    'code': code,
                    'type': 'markdown'
                })
        
        # HTMLコードブロック
        html_pattern = r'<code[^>]*>(.*?)</code>'
        for match in re.finditer(html_pattern, text, re.DOTALL):
            code = match.group(1).strip()
            if len(code) >= self.min_code_length:
                code_blocks.append({
                    'language': 'unknown',
                    'code': code,
                    'type': 'html'
                })
        
        # インデントされたコードブロック（Python風）
        indented_pattern = r'^(\s{4,}.*\n)+'
        for match in re.finditer(indented_pattern, text, re.MULTILINE):
            code = match.group(0).strip()
            if len(code) >= self.min_code_length:
                code_blocks.append({
                    'language': 'python',
                    'code': code,
                    'type': 'indented'
                })
        
        return code_blocks
    
    def classify_task_type(self, text: str, code_blocks: List[Dict]) -> str:
        """コーディングタスクの種類を分類"""
        text_lower = text.lower()
        
        scores = {}
        for task_type, config in self.task_types.items():
            score = 0
            
            # キーワードマッチング
            for keyword in config['keywords']:
                if keyword.lower() in text_lower:
                    score += 1
            
            # パターンマッチング
            for pattern in config['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 2
            
            scores[task_type] = score
        
        # 最高スコアのタスクタイプを返す
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                return max(scores.items(), key=lambda x: x[1])[0]
        
        # デフォルトはコード生成
        return 'code_generation'
    
    def is_coding_related(self, sample: Dict) -> bool:
        """サンプルがコーディング関連かチェック"""
        text = sample.get('text', '')
        if len(text) < self.min_text_length:
            return False
        
        # 言語を検出
        language = self.detect_language(text)
        
        # コーディングキーワードをチェック
        keywords = self.coding_keywords.get(language, [])
        text_lower = text.lower()
        
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        # コードブロックを抽出
        code_blocks = self.extract_code_blocks(text)
        
        # キーワードマッチまたはコードブロックがある場合はコーディング関連
        return keyword_matches >= 2 or len(code_blocks) > 0
    
    def extract_sample(self, sample: Dict) -> Optional[Dict]:
        """サンプルからコーディング関連データを抽出"""
        if not self.is_coding_related(sample):
            return None
        
        text = sample.get('text', '')
        code_blocks = self.extract_code_blocks(text)
        task_type = self.classify_task_type(text, code_blocks)
        
        extracted = {
            'original_id': sample.get('id', ''),
            'text': text,
            'code_blocks': code_blocks,
            'task_type': task_type,
            'language': self.detect_language(text),
            'url': sample.get('url', ''),
            'domain': sample.get('domain', ''),
            'category': sample.get('category', 'programming'),
            'metadata': {
                'code_blocks_count': len(code_blocks),
                'text_length': len(text),
                'extraction_timestamp': datetime.now().isoformat()
            }
        }
        
        return extracted
    
    def extract_from_file(self, input_file: Path) -> List[Dict]:
        """ファイルからコーディング関連データを抽出"""
        samples = []
        
        logger.info(f"[EXTRACT] Processing file: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        extracted = self.extract_sample(sample)
                        if extracted:
                            samples.append(extracted)
                    except json.JSONDecodeError as e:
                        logger.debug(f"[EXTRACT] JSON decode error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.debug(f"[EXTRACT] Error processing line {line_num}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"[EXTRACT] Failed to read file {input_file}: {e}")
        
        logger.info(f"[EXTRACT] Extracted {len(samples)} coding-related samples from {input_file}")
        return samples
    
    def extract_from_directory(self, input_dir: Path) -> List[Dict]:
        """ディレクトリ内のすべてのファイルからコーディング関連データを抽出"""
        all_samples = []
        
        # JSONLファイルを検索
        jsonl_files = list(input_dir.glob("*.jsonl"))
        
        logger.info(f"[EXTRACT] Found {len(jsonl_files)} JSONL files in {input_dir}")
        
        for jsonl_file in jsonl_files:
            samples = self.extract_from_file(jsonl_file)
            all_samples.extend(samples)
        
        return all_samples
    
    def save_extracted_samples(self, samples: List[Dict]):
        """抽出されたサンプルを保存"""
        # タスクタイプ別に分類
        task_type_groups = {}
        for sample in samples:
            task_type = sample.get('task_type', 'unknown')
            if task_type not in task_type_groups:
                task_type_groups[task_type] = []
            task_type_groups[task_type].append(sample)
        
        # 各タスクタイプごとにファイルを保存
        for task_type, group_samples in task_type_groups.items():
            output_file = self.output_dir / f"coding_{task_type}_{self.session_id}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in group_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            logger.info(f"[SAVE] Saved {len(group_samples)} samples of type '{task_type}' to {output_file}")
        
        # 統計情報を保存
        stats = {
            'total_samples': len(samples),
            'task_type_distribution': {k: len(v) for k, v in task_type_groups.items()},
            'language_distribution': Counter(s.get('language', 'unknown') for s in samples),
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        stats_file = self.output_dir / f"coding_extraction_stats_{self.session_id}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[STATS] Statistics saved to {stats_file}")
        logger.info(f"[COMPLETE] Extracted {len(samples)} coding-related samples")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Extract Coding Dataset')
    parser.add_argument('--input', type=str, required=True, help='Input directory or file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--min-code-length', type=int, default=10, help='Minimum code block length')
    parser.add_argument('--min-text-length', type=int, default=50, help='Minimum text length')
    
    args = parser.parse_args()
    
    extractor = CodingDatasetExtractor(
        output_dir=args.output,
        min_code_length=args.min_code_length,
        min_text_length=args.min_text_length
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        samples = extractor.extract_from_file(input_path)
    elif input_path.is_dir():
        samples = extractor.extract_from_directory(input_path)
    else:
        logger.error(f"[ERROR] Input path does not exist: {input_path}")
        return
    
    extractor.save_extracted_samples(samples)


if __name__ == '__main__':
    main()

