#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
コーディングタスク用データセット作成スクリプト

コーディングデータをinstruction-output形式に変換し、
コーディングタスクの種類に応じたプロンプトテンプレートを適用します。

Usage:
    python scripts/pipelines/prepare_coding_training_data.py --input D:\webdataset\coding_dataset --output D:\webdataset\coding_training_data
"""

import sys
import json
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prepare_coding_training_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CodingTrainingDataPreparer:
    """コーディングタスク用データセット作成クラス"""
    
    def __init__(self, output_dir: Path):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # プロンプトテンプレート
        self.prompt_templates = {
            'code_generation': {
                'ja': "以下の要件に基づいて、{language}のコードを生成してください。\n\n要件:\n{requirement}\n\nコード:",
                'en': "Generate {language} code based on the following requirements.\n\nRequirements:\n{requirement}\n\nCode:"
            },
            'code_explanation': {
                'ja': "以下のコードの動作を説明してください。\n\nコード:\n{code}\n\n説明:",
                'en': "Explain how the following code works.\n\nCode:\n{code}\n\nExplanation:"
            },
            'debugging': {
                'ja': "以下のコードのエラーを修正してください。\n\nコード:\n{code}\n\nエラー:\n{error}\n\n修正後のコード:",
                'en': "Fix the error in the following code.\n\nCode:\n{code}\n\nError:\n{error}\n\nFixed code:"
            },
            'refactoring': {
                'ja': "以下のコードをリファクタリングして、より読みやすく保守しやすくしてください。\n\nコード:\n{code}\n\nリファクタリング後のコード:",
                'en': "Refactor the following code to make it more readable and maintainable.\n\nCode:\n{code}\n\nRefactored code:"
            },
            'code_review': {
                'ja': "以下のコードをレビューして、改善点を指摘してください。\n\nコード:\n{code}\n\nレビュー:",
                'en': "Review the following code and suggest improvements.\n\nCode:\n{code}\n\nReview:"
            },
            'testing': {
                'ja': "以下のコードのユニットテストを作成してください。\n\nコード:\n{code}\n\nテストコード:",
                'en': "Create unit tests for the following code.\n\nCode:\n{code}\n\nTest code:"
            }
        }
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prepared_samples: List[Dict] = []
        
        logger.info("="*80)
        logger.info("Coding Training Data Preparer Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
    
    def extract_requirement_from_text(self, text: str) -> str:
        """テキストから要件を抽出"""
        # コードブロックを除去
        text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text_without_code = re.sub(r'<code>.*?</code>', '', text_without_code, flags=re.DOTALL)
        
        # 最初の段落を要件として使用
        paragraphs = [p.strip() for p in text_without_code.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[0][:500]  # 最大500文字
        
        return text_without_code[:500]
    
    def extract_code_from_sample(self, sample: Dict) -> str:
        """サンプルからコードを抽出"""
        code_blocks = sample.get('code_blocks', [])
        if code_blocks:
            # 最初のコードブロックを使用
            return code_blocks[0].get('code', '')
        
        # コードブロックがない場合は、テキストからコードを抽出
        text = sample.get('text', '')
        code_match = re.search(r'```(\w+)?\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(2).strip()
        
        return ''
    
    def detect_language_from_code(self, code: str) -> str:
        """コードから言語を検出"""
        # 言語固有のパターン
        language_patterns = {
            'python': [r'def\s+\w+', r'import\s+\w+', r'print\s*\(', r'if\s+__name__'],
            'javascript': [r'function\s+\w+', r'const\s+\w+\s*=', r'console\.log', r'export\s+'],
            'java': [r'public\s+class', r'public\s+static\s+void', r'System\.out\.print'],
            'cpp': [r'#include\s*<', r'using\s+namespace', r'std::'],
            'c': [r'#include\s*<', r'int\s+main\s*\(', r'printf\s*\('],
            'rust': [r'fn\s+\w+', r'let\s+mut\s+\w+', r'println!\s*\('],
            'go': [r'package\s+\w+', r'func\s+\w+', r'fmt\.Print'],
            'sql': [r'SELECT\s+', r'FROM\s+', r'WHERE\s+', r'INSERT\s+INTO'],
            'html': [r'<html', r'<body', r'<div', r'<!DOCTYPE'],
            'css': [r'\{', r'\}', r':\s*', r'@media'],
        }
        
        code_lower = code.lower()
        for lang, patterns in language_patterns.items():
            if any(re.search(pattern, code_lower, re.IGNORECASE) for pattern in patterns):
                return lang
        
        return 'unknown'
    
    def prepare_sample(self, sample: Dict) -> Optional[Dict]:
        """サンプルをinstruction-output形式に変換"""
        task_type = sample.get('task_type', 'code_generation')
        language = sample.get('language', 'en')
        text = sample.get('text', '')
        code = self.extract_code_from_sample(sample)
        
        # プロンプトテンプレートを取得
        if task_type not in self.prompt_templates:
            task_type = 'code_generation'  # デフォルト
        
        template = self.prompt_templates[task_type].get(language, self.prompt_templates[task_type]['en'])
        
        # タスクタイプに応じてプロンプトを構築
        if task_type == 'code_generation':
            requirement = self.extract_requirement_from_text(text)
            code_language = self.detect_language_from_code(code) if code else 'Python'
            instruction = template.format(language=code_language, requirement=requirement)
            output = code if code else text
        elif task_type == 'code_explanation':
            if not code:
                return None
            instruction = template.format(code=code[:1000])  # 最大1000文字
            output = text[:2000]  # 最大2000文字
        elif task_type == 'debugging':
            if not code:
                return None
            # エラーメッセージを抽出
            error_match = re.search(r'error|Error|ERROR|エラー|例外|Exception', text)
            error_text = error_match.group(0) if error_match else 'Unknown error'
            instruction = template.format(code=code[:1000], error=error_text)
            output = code[:2000]  # 修正後のコード（実際には元のコードを使用）
        elif task_type == 'refactoring':
            if not code:
                return None
            instruction = template.format(code=code[:1000])
            output = code[:2000]  # リファクタリング後のコード
        elif task_type == 'code_review':
            if not code:
                return None
            instruction = template.format(code=code[:1000])
            output = text[:2000]  # レビューコメント
        elif task_type == 'testing':
            if not code:
                return None
            instruction = template.format(code=code[:1000])
            output = text[:2000]  # テストコード
        else:
            return None
        
        prepared = {
            'instruction': instruction,
            'output': output,
            'task_type': task_type,
            'language': language,
            'code_language': self.detect_language_from_code(code) if code else 'unknown',
            'metadata': {
                'original_id': sample.get('original_id', ''),
                'url': sample.get('url', ''),
                'domain': sample.get('domain', ''),
                'preparation_timestamp': datetime.now().isoformat()
            }
        }
        
        return prepared
    
    def prepare_from_file(self, input_file: Path) -> List[Dict]:
        """ファイルからデータを準備"""
        samples = []
        
        logger.info(f"[PREPARE] Processing file: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        prepared = self.prepare_sample(sample)
                        if prepared:
                            samples.append(prepared)
                    except json.JSONDecodeError as e:
                        logger.debug(f"[PREPARE] JSON decode error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.debug(f"[PREPARE] Error processing line {line_num}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"[PREPARE] Failed to read file {input_file}: {e}")
        
        logger.info(f"[PREPARE] Prepared {len(samples)} training samples from {input_file}")
        return samples
    
    def prepare_from_directory(self, input_dir: Path) -> List[Dict]:
        """ディレクトリ内のすべてのファイルからデータを準備"""
        all_samples = []
        
        # JSONLファイルを検索
        jsonl_files = list(input_dir.glob("coding_*.jsonl"))
        
        logger.info(f"[PREPARE] Found {len(jsonl_files)} coding JSONL files in {input_dir}")
        
        for jsonl_file in jsonl_files:
            samples = self.prepare_from_file(jsonl_file)
            all_samples.extend(samples)
        
        return all_samples
    
    def save_prepared_samples(self, samples: List[Dict]):
        """準備されたサンプルを保存"""
        # タスクタイプ別に分類
        task_type_groups = {}
        for sample in samples:
            task_type = sample.get('task_type', 'unknown')
            if task_type not in task_type_groups:
                task_type_groups[task_type] = []
            task_type_groups[task_type].append(sample)
        
        # 各タスクタイプごとにファイルを保存
        for task_type, group_samples in task_type_groups.items():
            output_file = self.output_dir / f"coding_training_{task_type}_{self.session_id}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in group_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            logger.info(f"[SAVE] Saved {len(group_samples)} training samples of type '{task_type}' to {output_file}")
        
        # 統計情報を保存
        stats = {
            'total_samples': len(samples),
            'task_type_distribution': {k: len(v) for k, v in task_type_groups.items()},
            'language_distribution': Counter(s.get('language', 'unknown') for s in samples),
            'code_language_distribution': Counter(s.get('code_language', 'unknown') for s in samples),
            'preparation_timestamp': datetime.now().isoformat()
        }
        
        stats_file = self.output_dir / f"coding_training_stats_{self.session_id}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[STATS] Statistics saved to {stats_file}")
        logger.info(f"[COMPLETE] Prepared {len(samples)} coding training samples")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Prepare Coding Training Data')
    parser.add_argument('--input', type=str, required=True, help='Input directory or file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    preparer = CodingTrainingDataPreparer(output_dir=args.output)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        samples = preparer.prepare_from_file(input_path)
    elif input_path.is_dir():
        samples = preparer.prepare_from_directory(input_path)
    else:
        logger.error(f"[ERROR] Input path does not exist: {input_path}")
        return
    
    preparer.save_prepared_samples(samples)


if __name__ == '__main__':
    main()

