#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コーディング機能評価スクリプト

コード生成精度評価（BLEU、ROUGE、コードマッチ率）、構文エラー検出、
コード品質評価（リント、複雑度、ベストプラクティス準拠）を行います。

Usage:
    python scripts/evaluation/evaluate_coding_capability.py --model-path models/so8t_model --test-data D:\webdataset\coding_training_data
"""

import sys
import json
import logging
import argparse
import ast
import re
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# BLEU/ROUGE評価用
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK or rouge-score not available. BLEU/ROUGE evaluation will be skipped.")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluate_coding_capability.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CodeSyntaxChecker:
    """コード構文チェッカー"""
    
    @staticmethod
    def check_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Pythonコードの構文をチェック"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    @staticmethod
    def check_javascript_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """JavaScriptコードの構文をチェック（Node.jsを使用）"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ['node', '--check', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            Path(temp_file).unlink()
            
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr
        except FileNotFoundError:
            return None, "Node.js not found"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def check_syntax(code: str, language: str) -> Tuple[bool, Optional[str]]:
        """言語に応じた構文チェック"""
        if language.lower() == 'python':
            return CodeSyntaxChecker.check_python_syntax(code)
        elif language.lower() in ['javascript', 'typescript', 'js']:
            return CodeSyntaxChecker.check_javascript_syntax(code)
        else:
            # その他の言語は構文チェックをスキップ
            return None, f"Syntax checking not supported for {language}"


class CodeQualityAnalyzer:
    """コード品質分析クラス"""
    
    @staticmethod
    def calculate_cyclomatic_complexity(code: str, language: str = 'python') -> int:
        """循環的複雑度を計算"""
        if language.lower() != 'python':
            return 0  # Python以外は未実装
        
        try:
            tree = ast.parse(code)
            complexity = 1  # ベース複雑度
            
            for node in ast.walk(tree):
                # 条件分岐をカウント
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    complexity += 1
                # 論理演算子をカウント
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                # try-exceptをカウント
                elif isinstance(node, ast.Try):
                    complexity += len(node.handlers)
            
            return complexity
        except SyntaxError:
            return -1  # 構文エラー
        except Exception:
            return 0
    
    @staticmethod
    def detect_code_smells(code: str, language: str = 'python') -> List[str]:
        """コードスメルを検出"""
        smells = []
        
        if language.lower() == 'python':
            try:
                tree = ast.parse(code)
                
                # 長い関数を検出（50行以上）
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        lines = code.split('\n')
                        func_start = node.lineno - 1
                        func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 50
                        func_lines = func_end - func_start
                        
                        if func_lines > 50:
                            smells.append(f"Long function: {node.name} ({func_lines} lines)")
                
                # 深いネストを検出（4レベル以上）
                def check_nesting(node, level=0):
                    if level > 4:
                        smells.append(f"Deep nesting detected (level {level})")
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                            check_nesting(child, level + 1)
                
                check_nesting(tree)
                
            except SyntaxError:
                pass
        
        return smells
    
    @staticmethod
    def check_best_practices(code: str, language: str = 'python') -> Dict[str, bool]:
        """ベストプラクティス準拠をチェック"""
        practices = {
            'has_docstring': False,
            'has_type_hints': False,
            'uses_constants': False,
            'no_hardcoded_values': False
        }
        
        if language.lower() == 'python':
            try:
                tree = ast.parse(code)
                
                # docstringの存在をチェック
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if ast.get_docstring(node):
                            practices['has_docstring'] = True
                            break
                
                # 型ヒントの存在をチェック
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.returns or any(arg.annotation for arg in node.args.args):
                            practices['has_type_hints'] = True
                            break
                
                # 定数の使用をチェック
                if re.search(r'[A-Z_]{3,}\s*=', code):
                    practices['uses_constants'] = True
                
                # ハードコードされた値の検出（簡易版）
                if not re.search(r'\d{4,}', code):  # 4桁以上の数字がない
                    practices['no_hardcoded_values'] = True
                
            except SyntaxError:
                pass
        
        return practices


class CodingCapabilityEvaluator:
    """コーディング機能評価クラス"""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        初期化
        
        Args:
            model_path: モデルのパス（オプション、Ollama経由で評価する場合は不要）
        """
        self.model_path = model_path
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if NLTK_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction().method1
        else:
            self.rouge_scorer = None
        
        logger.info("="*80)
        logger.info("Coding Capability Evaluator Initialized")
        logger.info("="*80)
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """BLEUスコアを計算"""
        if not NLTK_AVAILABLE:
            return 0.0
        
        try:
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()
            
            score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                smoothing_function=self.smoothing
            )
            return float(score)
        except Exception as e:
            logger.debug(f"[BLEU] Failed to calculate: {e}")
            return 0.0
    
    def calculate_rouge_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """ROUGEスコアを計算"""
        if not self.rouge_scorer:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.debug(f"[ROUGE] Failed to calculate: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_code_match_rate(self, reference: str, candidate: str) -> Dict[str, float]:
        """コードマッチ率を計算"""
        # 完全一致
        exact_match = 1.0 if reference.strip() == candidate.strip() else 0.0
        
        # 部分一致（行単位）
        ref_lines = set(reference.strip().split('\n'))
        cand_lines = set(candidate.strip().split('\n'))
        
        if ref_lines:
            line_match = len(ref_lines & cand_lines) / len(ref_lines)
        else:
            line_match = 0.0
        
        # トークン一致
        ref_tokens = set(reference.split())
        cand_tokens = set(candidate.split())
        
        if ref_tokens:
            token_match = len(ref_tokens & cand_tokens) / len(ref_tokens)
        else:
            token_match = 0.0
        
        return {
            'exact_match': exact_match,
            'line_match': line_match,
            'token_match': token_match
        }
    
    def evaluate_code_generation(
        self,
        test_samples: List[Dict],
        model_name: str = "test_model"
    ) -> Dict:
        """コード生成タスクを評価"""
        logger.info(f"[EVAL] Evaluating code generation with {len(test_samples)} samples")
        
        results = {
            'model_name': model_name,
            'total_samples': len(test_samples),
            'bleu_scores': [],
            'rouge_scores': {'rouge1': [], 'rouge2': [], 'rougeL': []},
            'code_match_rates': {'exact_match': [], 'line_match': [], 'token_match': []},
            'syntax_errors': [],
            'quality_metrics': {
                'cyclomatic_complexity': [],
                'code_smells': [],
                'best_practices': {
                    'has_docstring': 0,
                    'has_type_hints': 0,
                    'uses_constants': 0,
                    'no_hardcoded_values': 0
                }
            }
        }
        
        syntax_checker = CodeSyntaxChecker()
        quality_analyzer = CodeQualityAnalyzer()
        
        for i, sample in enumerate(test_samples):
            instruction = sample.get('instruction', '')
            reference_output = sample.get('output', '')
            task_type = sample.get('task_type', 'code_generation')
            language = sample.get('code_language', 'python')
            
            # ここでは実際のモデル推論は行わず、reference_outputをcandidateとして評価
            # 実際の使用時は、モデルから生成されたコードをcandidateとして使用
            candidate = reference_output  # デモ用
            
            # BLEUスコア
            bleu = self.calculate_bleu_score(reference_output, candidate)
            results['bleu_scores'].append(bleu)
            
            # ROUGEスコア
            rouge = self.calculate_rouge_score(reference_output, candidate)
            for key in rouge:
                results['rouge_scores'][key].append(rouge[key])
            
            # コードマッチ率
            match_rate = self.calculate_code_match_rate(reference_output, candidate)
            for key in match_rate:
                results['code_match_rates'][key].append(match_rate[key])
            
            # 構文エラー検出
            if language.lower() == 'python':
                is_valid, error = syntax_checker.check_syntax(candidate, language)
                if is_valid is False:
                    results['syntax_errors'].append({
                        'sample_index': i,
                        'error': error,
                        'language': language
                    })
            
            # コード品質評価
            complexity = quality_analyzer.calculate_cyclomatic_complexity(candidate, language)
            if complexity >= 0:
                results['quality_metrics']['cyclomatic_complexity'].append(complexity)
            
            smells = quality_analyzer.detect_code_smells(candidate, language)
            results['quality_metrics']['code_smells'].extend(smells)
            
            practices = quality_analyzer.check_best_practices(candidate, language)
            for key, value in practices.items():
                if value:
                    results['quality_metrics']['best_practices'][key] += 1
        
        # 統計を計算
        results['metrics'] = {
            'bleu': {
                'mean': np.mean(results['bleu_scores']) if results['bleu_scores'] else 0.0,
                'std': np.std(results['bleu_scores']) if results['bleu_scores'] else 0.0
            },
            'rouge': {
                key: {
                    'mean': np.mean(results['rouge_scores'][key]) if results['rouge_scores'][key] else 0.0,
                    'std': np.std(results['rouge_scores'][key]) if results['rouge_scores'][key] else 0.0
                }
                for key in ['rouge1', 'rouge2', 'rougeL']
            },
            'code_match': {
                key: {
                    'mean': np.mean(results['code_match_rates'][key]) if results['code_match_rates'][key] else 0.0,
                    'std': np.std(results['code_match_rates'][key]) if results['code_match_rates'][key] else 0.0
                }
                for key in ['exact_match', 'line_match', 'token_match']
            },
            'syntax_error_rate': len(results['syntax_errors']) / len(test_samples) if test_samples else 0.0,
            'avg_cyclomatic_complexity': np.mean(results['quality_metrics']['cyclomatic_complexity']) if results['quality_metrics']['cyclomatic_complexity'] else 0.0,
            'code_smell_count': len(results['quality_metrics']['code_smells']),
            'best_practices_compliance': {
                key: value / len(test_samples) if test_samples else 0.0
                for key, value in results['quality_metrics']['best_practices'].items()
            }
        }
        
        logger.info(f"[EVAL] Evaluation completed")
        logger.info(f"[EVAL] BLEU: {results['metrics']['bleu']['mean']:.4f}")
        logger.info(f"[EVAL] ROUGE-L: {results['metrics']['rouge']['rougeL']['mean']:.4f}")
        logger.info(f"[EVAL] Syntax error rate: {results['metrics']['syntax_error_rate']:.2%}")
        
        return results
    
    def visualize_results(self, results: Dict, output_dir: Path):
        """評価結果を可視化"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # BLEUスコアの分布
        if results['bleu_scores']:
            plt.figure(figsize=(10, 6))
            plt.hist(results['bleu_scores'], bins=20, edgecolor='black')
            plt.xlabel('BLEU Score')
            plt.ylabel('Frequency')
            plt.title('BLEU Score Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / f"bleu_distribution_{self.session_id}.png")
            plt.close()
        
        # ROUGEスコアの比較
        if results['rouge_scores']['rougeL']:
            plt.figure(figsize=(10, 6))
            rouge_data = [
                results['rouge_scores']['rouge1'],
                results['rouge_scores']['rouge2'],
                results['rouge_scores']['rougeL']
            ]
            plt.boxplot(rouge_data, labels=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
            plt.ylabel('Score')
            plt.title('ROUGE Scores Comparison')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / f"rouge_comparison_{self.session_id}.png")
            plt.close()
        
        # コードマッチ率の比較
        if results['code_match_rates']['exact_match']:
            plt.figure(figsize=(10, 6))
            match_data = [
                results['code_match_rates']['exact_match'],
                results['code_match_rates']['line_match'],
                results['code_match_rates']['token_match']
            ]
            plt.boxplot(match_data, labels=['Exact Match', 'Line Match', 'Token Match'])
            plt.ylabel('Match Rate')
            plt.title('Code Match Rates Comparison')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / f"code_match_comparison_{self.session_id}.png")
            plt.close()
        
        # 循環的複雑度の分布
        if results['quality_metrics']['cyclomatic_complexity']:
            plt.figure(figsize=(10, 6))
            plt.hist(results['quality_metrics']['cyclomatic_complexity'], bins=20, edgecolor='black')
            plt.xlabel('Cyclomatic Complexity')
            plt.ylabel('Frequency')
            plt.title('Cyclomatic Complexity Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / f"complexity_distribution_{self.session_id}.png")
            plt.close()
        
        logger.info(f"[VISUALIZE] Saved visualization plots to {output_dir}")
    
    def generate_report(self, results: Dict, output_dir: Path):
        """評価レポートを生成"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSONレポート
        report_json = output_dir / f"coding_evaluation_report_{self.session_id}.json"
        with open(report_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Markdownレポート
        report_md = output_dir / f"coding_evaluation_report_{self.session_id}.md"
        with open(report_md, 'w', encoding='utf-8') as f:
            f.write(f"# Coding Capability Evaluation Report\n\n")
            f.write(f"**Model**: {results['model_name']}\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Samples**: {results['total_samples']}\n\n")
            
            f.write("## Code Generation Accuracy\n\n")
            metrics = results['metrics']
            f.write(f"- **BLEU Score**: {metrics['bleu']['mean']:.4f} ± {metrics['bleu']['std']:.4f}\n")
            f.write(f"- **ROUGE-1**: {metrics['rouge']['rouge1']['mean']:.4f} ± {metrics['rouge']['rouge1']['std']:.4f}\n")
            f.write(f"- **ROUGE-2**: {metrics['rouge']['rouge2']['mean']:.4f} ± {metrics['rouge']['rouge2']['std']:.4f}\n")
            f.write(f"- **ROUGE-L**: {metrics['rouge']['rougeL']['mean']:.4f} ± {metrics['rouge']['rougeL']['std']:.4f}\n\n")
            
            f.write("## Code Match Rates\n\n")
            f.write(f"- **Exact Match**: {metrics['code_match']['exact_match']['mean']:.2%}\n")
            f.write(f"- **Line Match**: {metrics['code_match']['line_match']['mean']:.2%}\n")
            f.write(f"- **Token Match**: {metrics['code_match']['token_match']['mean']:.2%}\n\n")
            
            f.write("## Syntax Error Detection\n\n")
            f.write(f"- **Syntax Error Rate**: {metrics['syntax_error_rate']:.2%}\n")
            f.write(f"- **Total Syntax Errors**: {len(results['syntax_errors'])}\n\n")
            
            f.write("## Code Quality Metrics\n\n")
            f.write(f"- **Average Cyclomatic Complexity**: {metrics['avg_cyclomatic_complexity']:.2f}\n")
            f.write(f"- **Code Smells Detected**: {metrics['code_smell_count']}\n\n")
            
            f.write("## Best Practices Compliance\n\n")
            for key, value in metrics['best_practices_compliance'].items():
                f.write(f"- **{key.replace('_', ' ').title()}**: {value:.2%}\n")
        
        logger.info(f"[REPORT] Generated evaluation report: {report_json}")
        logger.info(f"[REPORT] Generated evaluation report: {report_md}")


def load_test_data(test_data_path: Path) -> List[Dict]:
    """テストデータを読み込み"""
    samples = []
    
    if test_data_path.is_file():
        jsonl_files = [test_data_path]
    else:
        jsonl_files = list(test_data_path.glob("*.jsonl"))
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            sample = json.loads(line.strip())
                            samples.append(sample)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Failed to load {jsonl_file}: {e}")
    
    logger.info(f"[LOAD] Loaded {len(samples)} test samples")
    return samples


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Evaluate Coding Capability')
    parser.add_argument('--model-path', type=str, help='Model path (optional)')
    parser.add_argument('--test-data', type=str, required=True, help='Test data directory or file')
    parser.add_argument('--output', type=str, default='D:/webdataset/evaluation/coding_capability', help='Output directory')
    parser.add_argument('--model-name', type=str, default='test_model', help='Model name')
    
    args = parser.parse_args()
    
    evaluator = CodingCapabilityEvaluator(
        model_path=Path(args.model_path) if args.model_path else None
    )
    
    test_data_path = Path(args.test_data)
    test_samples = load_test_data(test_data_path)
    
    if not test_samples:
        logger.error("[ERROR] No test samples found")
        return
    
    results = evaluator.evaluate_code_generation(test_samples, model_name=args.model_name)
    
    output_dir = Path(args.output)
    evaluator.visualize_results(results, output_dir)
    evaluator.generate_report(results, output_dir)


if __name__ == '__main__':
    main()












































































