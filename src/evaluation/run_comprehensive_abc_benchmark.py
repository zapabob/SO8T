#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
包括的ABCベンチマーク実行スクリプト
業界標準ベンチマーク（MMLU含む）、高度ベンチマーク、ELIZA-100を使用してABCテストを実行
10ランダムシードで評価して統計的堅牢性を確保
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import argparse
from tqdm import tqdm
import random

# 既存のABCテストクラスをインポート
import sys
sys.path.insert(0, str(Path(__file__).parent))
from abc_testing import RTX3060ABCComparativeTesting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveABCBenchmark:
    """包括的ABCベンチマーク実行クラス"""

    def __init__(self, config_path: str = None, num_seeds: int = 10):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
            num_seeds: ランダムシード数（デフォルト10）
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "results" / "abc_testing"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_seeds = num_seeds
        self.config_path = config_path
        
        # 評価ベンチマークリスト
        self.industry_benchmarks = [
            'mmlu', 'bbh', 'commonsenseqa', 'openbookqa', 
            'socialiqa', 'piqa', 'winogrande', 'boolq'
        ]
        self.advanced_benchmarks = ['drop', 'strategyqa']
        self.japanese_benchmarks = ['elyza_tasks_100']
        
        self.all_benchmarks = (
            self.industry_benchmarks + 
            self.advanced_benchmarks + 
            self.japanese_benchmarks
        )
        
        logger.info(f"[INIT] Comprehensive ABC Benchmark initialized")
        logger.info(f"[BENCHMARKS] Industry: {len(self.industry_benchmarks)}, "
                   f"Advanced: {len(self.advanced_benchmarks)}, "
                   f"Japanese: {len(self.japanese_benchmarks)}")

    def run_abc_benchmark_with_seeds(self, num_samples: int = 100, 
                                     seeds: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        10ランダムシードでABCテストを実行
        
        Args:
            num_samples: 各ベンチマークのサンプル数
            seeds: ランダムシードのリスト（Noneの場合は自動生成）
        
        Returns:
            全シードの結果を含む辞書
        """
        if seeds is None:
            seeds = list(range(1, self.num_seeds + 1))
        
        logger.info(f"[ABC] Running ABC benchmark with {len(seeds)} random seeds")
        logger.info(f"[SEEDS] {seeds}")
        
        all_results = {
            'A': {},
            'B': {},
            'C': {},
            'metadata': {
                'num_seeds': len(seeds),
                'seeds': seeds,
                'benchmarks': self.all_benchmarks,
                'num_samples_per_benchmark': num_samples
            }
        }
        
        # 各シードで評価実行
        for seed_idx, seed in enumerate(seeds, 1):
            logger.info(f"[SEED {seed_idx}/{len(seeds)}] Running with seed {seed}")
            
            # シードを設定
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # ABCテスト実行
            abc_tester = RTX3060ABCComparativeTesting(config_path=self.config_path)
            abc_tester.sample_size = num_samples
            
            # 各モデルを評価
            for model_key in ['A', 'B', 'C']:
                logger.info(f"[SEED {seed}] Evaluating Model {model_key}...")
                
                # モデル読み込み
                model, tokenizer = abc_tester.load_model_for_evaluation(model_key)
                if model is None or tokenizer is None:
                    logger.error(f"[SEED {seed}] Failed to load model {model_key}")
                    continue
                
                # 各ベンチマークで評価
                for benchmark in self.all_benchmarks:
                    logger.info(f"[SEED {seed}] Model {model_key} on {benchmark}...")
                    
                    try:
                        result = abc_tester._run_single_abc_benchmark(
                            model, tokenizer, benchmark, num_samples
                        )
                        
                        if result:
                            # シードごとの結果を保存
                            if benchmark not in all_results[model_key]:
                                all_results[model_key][benchmark] = {
                                    'scores': [],
                                    'results': []
                                }
                            
                            accuracy = result.get('accuracy', 0.0)
                            all_results[model_key][benchmark]['scores'].append(accuracy)
                            all_results[model_key][benchmark]['results'].append(result)
                    
                    except Exception as e:
                        logger.error(f"[SEED {seed}] Error evaluating {model_key} on {benchmark}: {e}")
                        continue
                
                # メモリ節約のためモデルを削除
                del model
                del tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 結果を保存
        output_file = self.results_dir / "comprehensive_abc_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[ABC] Results saved to {output_file}")
        
        return all_results

    def _extend_abc_testing_methods(self, abc_tester: RTX3060ABCComparativeTesting):
        """
        ABCテストクラスに追加のベンチマーク評価メソッドを追加
        """
        # SocialIQA評価
        def _evaluate_socialiqa_abc(model, tokenizer, num_samples):
            """ABCテスト用SocialIQA評価"""
            try:
                from datasets import load_dataset
                dataset = load_dataset('allenai/socialiqa', split='validation')
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                correct = 0
                predictions = []
                
                for item in dataset:
                    context = item['context']
                    question = item['question']
                    answer_a = item['answerA']
                    answer_b = item['answerB']
                    answer_c = item['answerC']
                    correct_answer = item['correct']
                    
                    choices = [answer_a, answer_b, answer_c]
                    predicted_answer = abc_tester._answer_multiple_choice_abc(
                        model, tokenizer, f"{context} {question}", choices
                    )
                    is_correct = predicted_answer == correct_answer
                    predictions.append(is_correct)
                    
                    if is_correct:
                        correct += 1
                
                accuracy = correct / len(predictions) if len(predictions) > 0 else 0
                return {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': len(predictions),
                    'predictions': predictions
                }
            except Exception as e:
                logger.error(f"[SocialIQA] Evaluation failed: {e}")
                return None
        
        # PIQA評価
        def _evaluate_piqa_abc(model, tokenizer, num_samples):
            """ABCテスト用PIQA評価"""
            try:
                from datasets import load_dataset
                dataset = load_dataset('ybisk/piqa', split='validation')
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                correct = 0
                predictions = []
                
                for item in dataset:
                    goal = item['goal']
                    sol1 = item['sol1']
                    sol2 = item['sol2']
                    label = item['label']
                    
                    prompt = f"Goal: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}\nWhich solution is correct?"
                    predicted_answer = abc_tester._generate_response_abc(model, tokenizer, prompt, max_tokens=10)
                    
                    # 回答から0または1を抽出
                    predicted_label = 0
                    if '1' in predicted_answer or 'solution 1' in predicted_answer.lower():
                        predicted_label = 0
                    elif '2' in predicted_answer or 'solution 2' in predicted_answer.lower():
                        predicted_label = 1
                    
                    is_correct = predicted_label == label
                    predictions.append(is_correct)
                    
                    if is_correct:
                        correct += 1
                
                accuracy = correct / len(predictions) if len(predictions) > 0 else 0
                return {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': len(predictions),
                    'predictions': predictions
                }
            except Exception as e:
                logger.error(f"[PIQA] Evaluation failed: {e}")
                return None
        
        # Winogrande評価
        def _evaluate_winogrande_abc(model, tokenizer, num_samples):
            """ABCテスト用Winogrande評価"""
            try:
                from datasets import load_dataset
                dataset = load_dataset('allenai/winogrande', 'winogrande_xl', split='test')
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                correct = 0
                predictions = []
                
                for item in dataset:
                    sentence = item['sentence']
                    option1 = item['option1']
                    option2 = item['option2']
                    answer = item['answer']
                    
                    prompt = f"{sentence}\nOption 1: {option1}\nOption 2: {option2}\nWhich option is correct?"
                    predicted_answer = abc_tester._generate_response_abc(model, tokenizer, prompt, max_tokens=10)
                    
                    # 回答から1または2を抽出
                    predicted_label = 1
                    if '2' in predicted_answer or 'option 2' in predicted_answer.lower():
                        predicted_label = 2
                    elif '1' in predicted_answer or 'option 1' in predicted_answer.lower():
                        predicted_label = 1
                    
                    is_correct = predicted_label == answer
                    predictions.append(is_correct)
                    
                    if is_correct:
                        correct += 1
                
                accuracy = correct / len(predictions) if len(predictions) > 0 else 0
                return {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': len(predictions),
                    'predictions': predictions
                }
            except Exception as e:
                logger.error(f"[Winogrande] Evaluation failed: {e}")
                return None
        
        # BoolQ評価
        def _evaluate_boolq_abc(model, tokenizer, num_samples):
            """ABCテスト用BoolQ評価"""
            try:
                from datasets import load_dataset
                dataset = load_dataset('google/boolq', split='validation')
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                correct = 0
                predictions = []
                
                for item in dataset:
                    question = item['question']
                    passage = item['passage']
                    answer = item['answer']
                    
                    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer (True/False):"
                    predicted_answer = abc_tester._generate_response_abc(model, tokenizer, prompt, max_tokens=5)
                    
                    # 回答からTrue/Falseを抽出
                    predicted_bool = False
                    if 'true' in predicted_answer.lower() or 'yes' in predicted_answer.lower():
                        predicted_bool = True
                    elif 'false' in predicted_answer.lower() or 'no' in predicted_answer.lower():
                        predicted_bool = False
                    
                    is_correct = predicted_bool == answer
                    predictions.append(is_correct)
                    
                    if is_correct:
                        correct += 1
                
                accuracy = correct / len(predictions) if len(predictions) > 0 else 0
                return {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': len(predictions),
                    'predictions': predictions
                }
            except Exception as e:
                logger.error(f"[BoolQ] Evaluation failed: {e}")
                return None
        
        # DROP評価
        def _evaluate_drop_abc(model, tokenizer, num_samples):
            """ABCテスト用DROP評価"""
            try:
                from datasets import load_dataset
                dataset = load_dataset('ucinlp/drop', split='validation')
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                correct = 0
                predictions = []
                
                for item in dataset:
                    context = item['passage']
                    question = item['question']
                    answers = item['answers_spans']['spans']
                    
                    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                    predicted_answer = abc_tester._generate_response_abc(model, tokenizer, prompt, max_tokens=50)
                    
                    # 回答が期待される回答リストに含まれるかチェック
                    is_correct = any(ans.lower() in predicted_answer.lower() for ans in answers)
                    predictions.append(is_correct)
                    
                    if is_correct:
                        correct += 1
                
                accuracy = correct / len(predictions) if len(predictions) > 0 else 0
                return {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': len(predictions),
                    'predictions': predictions
                }
            except Exception as e:
                logger.error(f"[DROP] Evaluation failed: {e}")
                return None
        
        # StrategyQA評価
        def _evaluate_strategyqa_abc(model, tokenizer, num_samples):
            """ABCテスト用StrategyQA評価"""
            try:
                from datasets import load_dataset
                dataset = load_dataset('allenai/strategyqa', split='test')
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                correct = 0
                predictions = []
                
                for item in dataset:
                    question = item['question']
                    answer = item['answer']
                    
                    prompt = f"Question: {question}\nAnswer (True/False):"
                    predicted_answer = abc_tester._generate_response_abc(model, tokenizer, prompt, max_tokens=5)
                    
                    # 回答からTrue/Falseを抽出
                    predicted_bool = False
                    if 'true' in predicted_answer.lower() or 'yes' in predicted_answer.lower():
                        predicted_bool = True
                    elif 'false' in predicted_answer.lower() or 'no' in predicted_answer.lower():
                        predicted_bool = False
                    
                    is_correct = predicted_bool == answer
                    predictions.append(is_correct)
                    
                    if is_correct:
                        correct += 1
                
                accuracy = correct / len(predictions) if len(predictions) > 0 else 0
                return {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': len(predictions),
                    'predictions': predictions
                }
            except Exception as e:
                logger.error(f"[StrategyQA] Evaluation failed: {e}")
                return None
        
        # メソッドを追加
        abc_tester._evaluate_socialiqa_abc = _evaluate_socialiqa_abc
        abc_tester._evaluate_piqa_abc = _evaluate_piqa_abc
        abc_tester._evaluate_winogrande_abc = _evaluate_winogrande_abc
        abc_tester._evaluate_boolq_abc = _evaluate_boolq_abc
        abc_tester._evaluate_drop_abc = _evaluate_drop_abc
        abc_tester._evaluate_strategyqa_abc = _evaluate_strategyqa_abc
        
        # _run_single_abc_benchmarkを拡張
        original_method = abc_tester._run_single_abc_benchmark
        
        def extended_run_single_abc_benchmark(model, tokenizer, benchmark_name, num_samples):
            """拡張された単一ベンチマーク実行"""
            benchmark_methods = {
                'gsm8k': lambda: abc_tester._evaluate_gsm8k_abc(model, tokenizer, num_samples),
                'math': lambda: abc_tester._evaluate_math_abc(model, tokenizer, num_samples),
                'arc_easy': lambda: abc_tester._evaluate_arc_easy_abc(model, tokenizer, num_samples),
                'elyza_tasks_100': lambda: abc_tester._evaluate_elyza_abc(model, tokenizer, num_samples),
                'mmlu': lambda: abc_tester._evaluate_mmlu_abc(model, tokenizer, num_samples // 3),
                'bbh': lambda: abc_tester._evaluate_bbh_abc(model, tokenizer, num_samples // 3),
                'commonsenseqa': lambda: abc_tester._evaluate_commonsenseqa_abc(model, tokenizer, num_samples),
                'openbookqa': lambda: abc_tester._evaluate_openbookqa_abc(model, tokenizer, num_samples),
                'socialiqa': lambda: _evaluate_socialiqa_abc(model, tokenizer, num_samples),
                'piqa': lambda: _evaluate_piqa_abc(model, tokenizer, num_samples),
                'winogrande': lambda: _evaluate_winogrande_abc(model, tokenizer, num_samples),
                'boolq': lambda: _evaluate_boolq_abc(model, tokenizer, num_samples),
                'drop': lambda: _evaluate_drop_abc(model, tokenizer, num_samples),
                'strategyqa': lambda: _evaluate_strategyqa_abc(model, tokenizer, num_samples),
            }
            
            if benchmark_name in benchmark_methods:
                return benchmark_methods[benchmark_name]()
            else:
                return original_method(model, tokenizer, benchmark_name, num_samples)
        
        abc_tester._run_single_abc_benchmark = extended_run_single_abc_benchmark


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Comprehensive ABC Benchmark')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples per benchmark')
    parser.add_argument('--num_seeds', type=int, default=10,
                       help='Number of random seeds')
    parser.add_argument('--seeds', type=str, default=None,
                       help='Comma-separated list of seeds (e.g., "1,2,3")')
    
    args = parser.parse_args()
    
    # シードリストの処理
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # 包括的ABCベンチマーク実行
    benchmark = ComprehensiveABCBenchmark(
        config_path=args.config,
        num_seeds=args.num_seeds
    )
    
    results = benchmark.run_abc_benchmark_with_seeds(
        num_samples=args.num_samples,
        seeds=seeds
    )
    
    logger.info("[ABC] Comprehensive ABC benchmark completed successfully")
    logger.info(f"[RESULTS] Saved to {benchmark.results_dir / 'comprehensive_abc_results.json'}")


if __name__ == "__main__":
    main()
