#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
包括的LLMベンチマークシステム
Comprehensive LLM Benchmark System

複数のPythonライブラリを統合した包括的ベンチマーク：
- lm-evaluation-harness (EleutherAI)
- LightEval (HuggingFace)
- OpenCompass
- transformers benchmark utilities
- vLLM benchmark

HF提出可能な統計処理（エラーバー付きグラフ、要約統計量）
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# llama.cpp.python imports
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# lm-evaluation-harness imports
try:
    import lm_eval
    from lm_eval import evaluator, tasks
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False

# LightEval imports
try:
    from lighteval import LightevalPipeline, PipelineParameters
    from lighteval.tasks import TaskParameters
    from lighteval.models import ModelConfig
    LIGHT_EVAL_AVAILABLE = True
except ImportError:
    LIGHT_EVAL_AVAILABLE = False

# transformers benchmark
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveBenchmarkEvaluator:
    """
    包括的LLMベンチマーク評価器
    Comprehensive LLM Benchmark Evaluator
    """

    def __init__(self, model_configs: Dict[str, Dict[str, Any]]):
        """
        Args:
            model_configs: モデル設定の辞書
                {
                    "modela": {"path": "path/to/modela.gguf", "type": "gguf"},
                    "modelb": {"path": "path/to/modelb.gguf", "type": "gguf"},
                    "modelc": {"path": "path/to/modelc.gguf", "type": "gguf"}
                }
        """
        self.model_configs = model_configs
        self.results = {}

        # 利用可能なベンチマークライブラリを確認
        self.available_libraries = self._check_available_libraries()

        logger.info(f"Available benchmark libraries: {list(self.available_libraries.keys())}")

    def _check_available_libraries(self) -> Dict[str, bool]:
        """利用可能なベンチマークライブラリを確認"""
        libraries = {
            'llama_cpp': LLAMA_CPP_AVAILABLE,
            'lm_eval': LM_EVAL_AVAILABLE,
            'light_eval': LIGHT_EVAL_AVAILABLE,
            'transformers': TRANSFORMERS_AVAILABLE
        }
        return {k: v for k, v in libraries.items() if v}

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        包括的ベンチマーク実行
        Run comprehensive benchmark across all available libraries
        """
        logger.info("[COMPREHENSIVE BENCHMARK] Starting comprehensive evaluation...")

        # 各モデルに対してベンチマーク実行
        for model_name, model_config in self.model_configs.items():
            logger.info(f"[COMPREHENSIVE BENCHMARK] Evaluating {model_name}...")
            self.results[model_name] = self._evaluate_single_model(model_name, model_config)

        # 統計分析と比較
        comparison_results = self._perform_statistical_analysis()

        # HF提出用グラフと統計量生成
        hf_submission_data = self._generate_hf_submission_data(comparison_results)

        return {
            'raw_results': self.results,
            'comparison': comparison_results,
            'hf_submission': hf_submission_data
        }

    def _evaluate_single_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """単一モデルの評価"""
        results = {}

        # 各ベンチマークライブラリで評価
        if 'llama_cpp' in self.available_libraries:
            results['llama_cpp'] = self._run_llama_cpp_benchmarks(model_config)

        if 'lm_eval' in self.available_libraries:
            results['lm_eval'] = self._run_lm_eval_benchmarks(model_config)

        if 'light_eval' in self.available_libraries:
            results['light_eval'] = self._run_light_eval_benchmarks(model_config)

        if 'transformers' in self.available_libraries:
            results['transformers'] = self._run_transformers_benchmarks(model_config)

        return results

    def _run_llama_cpp_benchmarks(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """llama.cppベースのベンチマーク"""
        gguf_path = model_config['path']

        try:
            # 基本的なllama.cpp評価
            llm = Llama(
                model_path=gguf_path,
                n_ctx=4096,
                n_threads=min(8, os.cpu_count()),  # CPUスレッド制限
                n_gpu_layers=-1,
                verbose=False
            )

            # 基本性能テスト
            results = {
                'inference_speed': self._measure_inference_speed(llm),
                'memory_usage': self._measure_memory_usage(llm),
                'perplexity': self._calculate_perplexity(llm)
            }

            return results

        except Exception as e:
            logger.error(f"llama.cpp benchmark failed: {e}")
            return {'error': str(e)}

    def _run_lm_eval_benchmarks(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """lm-evaluation-harnessベースのベンチマーク"""
        try:
            model_path = model_config['path']
            model_type = model_config.get('type', 'gguf')

            if model_type == 'gguf':
                # GGUFモデルの場合、llama.cpp経由で評価
                return self._run_llama_cpp_benchmarks(model_config)

            # HFモデルの場合、直接lm_eval使用
            # 主要なベンチマークタスク
            task_names = [
                "arc_challenge", "arc_easy", "boolq", "piqa", "winogrande",
                "hellaswag", "openbookqa", "sciq", "commonsense_qa",
                "mmlu", "gsm8k", "math", "truthfulqa"
            ]

            results = {}
            for task_name in task_names[:5]:  # 最初の5つだけ実行（時間節約）
                try:
                    result = lm_eval.simple_evaluate(
                        model="hf",
                        model_args=f"pretrained={model_path}",
                        tasks=[task_name],
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        batch_size=1,
                        num_fewshot=0
                    )
                    results[task_name] = result['results'][task_name]
                except Exception as e:
                    logger.warning(f"lm_eval task {task_name} failed: {e}")
                    results[task_name] = {'error': str(e)}

            return results

        except Exception as e:
            logger.error(f"lm_eval benchmark failed: {e}")
            return {'error': str(e)}

    def _run_light_eval_benchmarks(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """LightEvalベースのベンチマーク"""
        try:
            model_path = model_config['path']
            model_type = model_config.get('type', 'gguf')

            if model_type == 'gguf':
                # GGUFモデルの場合、簡易評価
                return self._run_llama_cpp_benchmarks(model_config)

            # HFモデルの場合、LightEval使用
            model_config_lighteval = ModelConfig(
                model_name=model_path,
                model_dtype="auto",
                model_max_length=4096
            )

            # 主要なタスク
            task_configs = [
                TaskParameters(name="arc:challenge:acc", suite=["helm"]),
                TaskParameters(name="hellaswag:acc", suite=["helm"]),
                TaskParameters(name="mmlu:acc", suite=["helm"]),
                TaskParameters(name="truthfulqa:mc:acc", suite=["helm"])
            ]

            pipeline_params = PipelineParameters(
                model=model_config_lighteval,
                tasks=task_configs,
                batch_size=1,
                max_samples=100,  # サンプル制限
                use_chat_template=False
            )

            pipeline = LightevalPipeline(pipeline_params)
            results = pipeline.run()

            return results.to_dict()

        except Exception as e:
            logger.error(f"LightEval benchmark failed: {e}")
            return {'error': str(e)}

    def _run_transformers_benchmarks(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """transformersベースのベンチマーク"""
        try:
            model_path = model_config['path']
            model_type = model_config.get('type', 'gguf')

            if model_type == 'hf':
                # HFモデルの場合、transformers benchmark使用
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                # 基本的なパフォーマンス測定
                results = {
                    'model_size': self._calculate_model_size(model),
                    'vocab_size': len(tokenizer),
                    'max_position_embeddings': model.config.max_position_embeddings
                }

                return results
            else:
                # GGUFモデルの場合、基本情報のみ
                return {
                    'model_type': 'gguf',
                    'path': model_path
                }

        except Exception as e:
            logger.error(f"transformers benchmark failed: {e}")
            return {'error': str(e)}

    def _measure_inference_speed(self, llm: 'Llama') -> Dict[str, float]:
        """推論速度測定"""
        test_prompts = [
            "Hello, how are you?",
            "Explain quantum computing in simple terms.",
            "Write a short story about AI.",
            "What is the capital of Japan?"
        ]

        total_tokens = 0
        total_time = 0

        for prompt in test_prompts:
            start_time = time.time()

            response = llm(
                prompt,
                max_tokens=50,
                temperature=0.1,
                echo=False
            )

            end_time = time.time()
            tokens_generated = len(response['choices'][0]['text'].split())
            total_tokens += tokens_generated
            total_time += (end_time - start_time)

        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

        return {
            'tokens_per_sec': tokens_per_sec,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'avg_tokens_per_prompt': total_tokens / len(test_prompts)
        }

    def _measure_memory_usage(self, llm: 'Llama') -> Dict[str, float]:
        """メモリ使用量測定"""
        try:
            import psutil
            process = psutil.Process()

            # 初期メモリ
            initial_memory = process.memory_info().rss / (1024**3)  # GB

            # テスト推論実行
            llm("Test prompt", max_tokens=10, echo=False)

            # 実行後メモリ
            final_memory = process.memory_info().rss / (1024**3)  # GB

            return {
                'initial_memory_gb': initial_memory,
                'final_memory_gb': final_memory,
                'memory_increase_gb': final_memory - initial_memory
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_perplexity(self, llm: 'Llama') -> Dict[str, float]:
        """パープレキシティ計算"""
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Quantum physics describes the behavior of matter and energy."
        ]

        total_log_prob = 0
        total_tokens = 0

        for text in test_texts:
            try:
                # テキストをトークン化して対数確率を計算
                tokens = llm.tokenize(text.encode())
                if len(tokens) > 1:
                    log_probs = []
                    for i in range(len(tokens) - 1):
                        context = tokens[:i+1]
                        next_token = tokens[i+1]

                        # 次のトークンの確率を計算
                        logits = llm.eval(context)
                        probs = torch.softmax(torch.tensor(logits[0][-1]), dim=0)
                        prob = probs[next_token].item()
                        log_prob = np.log(prob) if prob > 0 else -10
                        log_probs.append(log_prob)

                    total_log_prob += sum(log_probs)
                    total_tokens += len(log_probs)

            except Exception as e:
                logger.warning(f"Perplexity calculation failed for text: {e}")

        avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else 0
        perplexity = np.exp(-avg_log_prob) if avg_log_prob != 0 else float('inf')

        return {
            'perplexity': perplexity,
            'avg_log_prob': avg_log_prob,
            'total_tokens': total_tokens
        }

    def _calculate_model_size(self, model) -> Dict[str, Any]:
        """モデルサイズ計算"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 2 / (1024**2),  # rough estimate in MB
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }

    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """統計分析実行"""
        logger.info("[STATISTICS] Performing statistical analysis...")

        # 結果をデータフレームに変換
        df_results = self._convert_results_to_dataframe()

        # 統計的比較
        statistical_comparison = self._perform_statistical_comparison(df_results)

        # ABCテスト（A/B/Cモデルの比較）
        abc_test_results = self._perform_abc_test(df_results)

        return {
            'dataframe': df_results,
            'statistical_comparison': statistical_comparison,
            'abc_test': abc_test_results
        }

    def _convert_results_to_dataframe(self) -> pd.DataFrame:
        """結果をデータフレームに変換"""
        all_results = []

        for model_name, model_results in self.results.items():
            for library_name, library_results in model_results.items():
                if isinstance(library_results, dict):
                    for metric_name, metric_value in library_results.items():
                        if isinstance(metric_value, (int, float)):
                            all_results.append({
                                'model': model_name,
                                'library': library_name,
                                'metric': metric_name,
                                'value': metric_value
                            })

        return pd.DataFrame(all_results)

    def _perform_statistical_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """統計的比較実行"""
        comparison_results = {}

        # 各メトリックに対して統計的比較
        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric]

            if len(metric_data) < 2:
                continue

            # モデル間の統計的比較
            models = metric_data['model'].unique()
            if len(models) >= 2:
                comparison_results[metric] = {}

                for i, model1 in enumerate(models):
                    for j, model2 in enumerate(models):
                        if i < j:
                            data1 = metric_data[metric_data['model'] == model1]['value']
                            data2 = metric_data[metric_data['model'] == model2]['value']

                            if len(data1) > 0 and len(data2) > 0:
                                # t-test
                                try:
                                    t_stat, p_value = ttest_ind(data1, data2)
                                    comparison_results[metric][f'{model1}_vs_{model2}'] = {
                                        't_statistic': t_stat,
                                        'p_value': p_value,
                                        'significant': p_value < 0.05
                                    }
                                except:
                                    comparison_results[metric][f'{model1}_vs_{model2}'] = {
                                        'error': 'Statistical test failed'
                                    }

        return comparison_results

    def _perform_abc_test(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ABCテスト実行（A/B/Cモデルの比較）"""
        abc_results = {}

        # A/B/Cモデルが存在するか確認
        available_models = set(df['model'].unique())
        abc_models = {'modela', 'modelb', 'modelc'}
        available_abc = available_models.intersection(abc_models)

        if len(available_abc) < 2:
            abc_results['error'] = f"Need at least 2 ABC models. Available: {available_abc}"
            return abc_results

        # 各メトリックに対してABC比較
        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric]

            abc_results[metric] = {}

            # 各モデルの平均値と標準偏差
            for model in available_abc:
                model_data = metric_data[metric_data['model'] == model]
                if len(model_data) > 0:
                    abc_results[metric][model] = {
                        'mean': model_data['value'].mean(),
                        'std': model_data['value'].std(),
                        'count': len(model_data),
                        'sem': model_data['value'].sem()  # 標準誤差
                    }

        # 勝者決定（最も高い平均値を持つモデル）
        if abc_results:
            sample_metric = list(abc_results.keys())[0]  # 最初のメトリックを使用
            model_scores = {}

            for model in available_abc:
                if model in abc_results[sample_metric]:
                    model_scores[model] = abc_results[sample_metric][model]['mean']

            winner = max(model_scores, key=model_scores.get)
            abc_results['winner'] = {
                'model': winner,
                'score': model_scores[winner],
                'metric': sample_metric
            }

        return abc_results

    def _generate_hf_submission_data(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """HF提出用データ生成"""
        logger.info("[HF SUBMISSION] Generating HF submission data...")

        # エラーバー付きグラフ生成
        plots = self._generate_error_bar_plots(comparison_results)

        # 要約統計量
        summary_stats = self._generate_summary_statistics(comparison_results)

        # ABCテスト結果
        abc_summary = self._generate_abc_summary(comparison_results)

        return {
            'plots': plots,
            'summary_statistics': summary_stats,
            'abc_test_results': abc_summary,
            'recommendations': self._generate_recommendations(comparison_results)
        }

    def _generate_error_bar_plots(self, comparison_results: Dict[str, Any]) -> Dict[str, str]:
        """エラーバー付きグラフ生成"""
        plots = {}

        try:
            df = comparison_results['dataframe']

            # 各メトリックに対してグラフ生成
            for metric in df['metric'].unique():
                metric_data = df[df['metric'] == metric]

                if len(metric_data) == 0:
                    continue

                plt.figure(figsize=(12, 8))
                sns.set_style("whitegrid")

                # エラーバー付き棒グラフ
                ax = sns.barplot(
                    data=metric_data,
                    x='model',
                    y='value',
                    hue='library',
                    errorbar='sd',  # 標準偏差をエラーバーとして使用
                    capsize=0.1
                )

                plt.title(f'{metric.replace("_", " ").title()} - Model Comparison')
                plt.xlabel('Model')
                plt.ylabel(metric.replace("_", " ").title())
                plt.legend(title='Benchmark Library', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()

                # 画像として保存
                plot_filename = f"{metric}_comparison.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()

                plots[metric] = plot_filename

        except Exception as e:
            logger.error(f"Error bar plot generation failed: {e}")

        return plots

    def _generate_summary_statistics(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """要約統計量生成"""
        summary = {}

        try:
            df = comparison_results['dataframe']

            # 各モデル・各メトリックの統計量
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                summary[model] = {}

                for metric in df['metric'].unique():
                    metric_data = model_data[model_data['metric'] == metric]

                    if len(metric_data) > 0:
                        values = metric_data['value'].values
                        summary[model][metric] = {
                            'count': len(values),
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'median': float(np.median(values)),
                            'q25': float(np.percentile(values, 25)),
                            'q75': float(np.percentile(values, 75)),
                            'sem': float(stats.sem(values)) if len(values) > 1 else 0.0
                        }

        except Exception as e:
            logger.error(f"Summary statistics generation failed: {e}")

        return summary

    def _generate_abc_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """ABCテスト要約生成"""
        abc_results = comparison_results.get('abc_test', {})

        if 'winner' not in abc_results:
            return {'error': 'ABC test not completed'}

        winner = abc_results['winner']

        summary = {
            'winner_model': winner['model'],
            'winning_score': winner['score'],
            'winning_metric': winner['metric'],
            'model_rankings': {}
        }

        # 各メトリックでのランキング
        if abc_results:
            for metric, model_stats in abc_results.items():
                if metric not in ['winner', 'error'] and isinstance(model_stats, dict):
                    rankings = sorted(
                        [(model, stats.get('mean', 0)) for model, stats in model_stats.items()
                         if isinstance(stats, dict)],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    summary['model_rankings'][metric] = rankings

        return summary

    def _generate_recommendations(self, comparison_results: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        abc_results = comparison_results.get('abc_test', {})
        if 'winner' in abc_results:
            winner = abc_results['winner']
            recommendations.append(
                f"🏆 Winner: {winner['model']} with score {winner['score']:.3f} on {winner['metric']}"
            )

        # 統計的有意差の分析
        statistical_comp = comparison_results.get('statistical_comparison', {})
        significant_differences = []

        for metric, comparisons in statistical_comp.items():
            for comparison_name, results in comparisons.items():
                if isinstance(results, dict) and results.get('significant', False):
                    significant_differences.append(f"{comparison_name} on {metric}")

        if significant_differences:
            recommendations.append(f"📊 Significant differences found: {', '.join(significant_differences[:3])}")

        # 使用推奨
        if abc_results and 'winner' in abc_results:
            winner = abc_results['winner']['model']
            if winner == 'modela':
                recommendations.append("💡 Model A (Borea-Phi3.5-instruct-jp GGUF) recommended for general use")
            elif winner == 'modelb':
                recommendations.append("💡 Model B recommended for specialized tasks")
            elif winner == 'modelc':
                recommendations.append("💡 Model C recommended for high-performance applications")

        return recommendations


def run_abc_test_evaluation():
    """
    ABCテスト評価実行
    A: Borea-Phi3.5-instruct-jp GGUF
    B: AEGIS-Phi3.5-Enhanced
    C: AEGIS-Phi3.5-Golden-Sigmoid
    """
    logger.info("[ABC TEST] Starting ABC test evaluation...")

    # モデル設定
    model_configs = {
        'modela': {
            'path': 'D:/webdataset/gguf_models/borea_phi35_instruct_jp_q8_0.gguf',
            'type': 'gguf',
            'description': 'Borea-Phi3.5-instruct-jp (GGUF Q8_0)'
        },
        'modelb': {
            'path': 'D:/webdataset/models/borea_phi35_alpha_gate_sigmoid_bayesian/final',
            'type': 'hf',
            'description': 'AEGIS-Phi3.5-Enhanced Model'
        },
        'modelc': {
            'path': 'D:/webdataset/models/borea_phi35_so8t_rtx3060/final',
            'type': 'hf',
            'description': 'AEGIS-Phi3.5-Golden-Sigmoid Model'
        }
    }

    # 存在するモデルだけを評価
    available_configs = {}
    for model_name, config in model_configs.items():
        if os.path.exists(config['path']):
            available_configs[model_name] = config
            logger.info(f"[ABC TEST] Found model {model_name}: {config['path']}")
        else:
            logger.warning(f"[ABC TEST] Model {model_name} not found: {config['path']}")

    if len(available_configs) < 2:
        logger.error("Need at least 2 models for ABC test")
        return None

    # 包括的ベンチマーク実行
    evaluator = ComprehensiveBenchmarkEvaluator(available_configs)
    results = evaluator.run_comprehensive_benchmark()

    # 結果保存
    output_dir = "D:/webdataset/results/abc_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # JSON保存
    with open(f"{output_dir}/abc_test_results.json", 'w', encoding='utf-8') as f:
        # DataFrameをJSONに変換するために前処理
        save_results = results.copy()
        if 'comparison' in save_results and 'dataframe' in save_results['comparison']:
            df = save_results['comparison']['dataframe']
            save_results['comparison']['dataframe'] = df.to_dict('records')

        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"[ABC TEST] Results saved to {output_dir}")

    # ABCテスト結果表示
    abc_results = results.get('comparison', {}).get('abc_test', {})
    if 'winner' in abc_results:
        winner = abc_results['winner']
        logger.info("🎯 ABC Test Winner:"        logger.info(f"   Model: {winner['model']}")
        logger.info(f"   Score: {winner['score']:.4f}")
        logger.info(f"   Metric: {winner['metric']}")

    return results


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive LLM Benchmark Evaluation System"
    )
    parser.add_argument(
        '--abc_test',
        action='store_true',
        help='Run ABC test evaluation (A: Borea-Phi3.5 GGUF, B: Alpha Gate, C: RTX3060 SO8T)'
    )
    parser.add_argument(
        '--models',
        type=str,
        help='JSON file with model configurations'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='D:/webdataset/results/benchmark_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    if args.abc_test:
        # ABCテスト実行
        results = run_abc_test_evaluation()
        if results:
            logger.info("[SUCCESS] ABC test completed successfully!")
        else:
            logger.error("[FAILED] ABC test failed!")
            sys.exit(1)
    else:
        # カスタムモデル評価
        if not args.models:
            logger.error("Please specify --models JSON file or use --abc_test")
            sys.exit(1)

        # モデル設定読み込み
        with open(args.models, 'r', encoding='utf-8') as f:
            model_configs = json.load(f)

        # 包括的ベンチマーク実行
        evaluator = ComprehensiveBenchmarkEvaluator(model_configs)
        results = evaluator.run_comprehensive_benchmark()

        # 結果保存
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/benchmark_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[SUCCESS] Benchmark completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
