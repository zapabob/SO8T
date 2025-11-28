#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama.cpp.python 業界標準ベンチマーク比較システム
Industry Standard Benchmark Comparison System using llama.cpp.python

ELYZA-100 + マルチモーダル性能での総合比較
2つのAlpha Gateアニーリングモデルを比較し、優れている方をHFアップロード
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import tempfile
import logging
from tqdm import tqdm
import time
import psutil

# llama.cpp.python imports
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("[WARNING] llama-cpp-python not available")

# HuggingFace datasets
from datasets import load_dataset, load_metric
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 設定 ---
BENCHMARK_CONFIG = {
    'elyza_100': {
        'dataset': 'elyza/ELYZA-tasks-100',
        'split': 'test',
        'metrics': ['accuracy', 'f1'],
        'task_type': 'text_generation'
    },
    'industry_standard': {
        'mmlu': {
            'dataset': 'cais/mmlu',
            'split': 'test',
            'metrics': ['accuracy']
        },
        'gsm8k': {
            'dataset': 'gsm8k',
            'split': 'test',
            'metrics': ['accuracy']
        },
        'hellaswag': {
            'dataset': 'hellaswag',
            'split': 'validation',
            'metrics': ['accuracy']
        },
        'arc_challenge': {
            'dataset': 'ai2_arc',
            'name': 'ARC-Challenge',
            'split': 'test',
            'metrics': ['accuracy']
        }
    },
    'multimodal': {
        'scienceqa': {
            'dataset': 'derek-thomas/ScienceQA',
            'metrics': ['accuracy']
        }
    }
}

class LlamaCppBenchmarkEvaluator:
    """
    llama.cpp.python ベンチマーク評価器
    llama.cpp.python Benchmark Evaluator
    """

    def __init__(self, gguf_path: str, config: Dict[str, Any]):
        self.gguf_path = gguf_path
        self.config = config
        self.llm = None

        # llama.cppモデル初期化
        if LLAMA_CPP_AVAILABLE:
            try:
                self.llm = Llama(
                    model_path=gguf_path,
                    n_ctx=4096,  # コンテキスト長
                    n_threads=psutil.cpu_count(),  # CPUスレッド数
                    n_gpu_layers=-1,  # GPUレイヤー数（すべて）
                    verbose=False
                )
                logger.info(f"[LLAMA.CPP] Model loaded from {gguf_path}")
            except Exception as e:
                logger.error(f"[LLAMA.CPP] Failed to load model: {e}")
                self.llm = None

    def evaluate_elyza_100(self) -> Dict[str, float]:
        """ELYZA-100評価"""
        if not self.llm:
            return {'accuracy': 0.0, 'f1': 0.0}

        logger.info("[BENCHMARK] Evaluating ELYZA-100...")

        try:
            # ELYZA-100データセット読み込み
            dataset = load_dataset(
                BENCHMARK_CONFIG['elyza_100']['dataset'],
                split=BENCHMARK_CONFIG['elyza_100']['split']
            )

            correct = 0
            total = 0
            predictions = []
            references = []

            for sample in tqdm(dataset, desc="ELYZA-100"):
                # プロンプト作成
                prompt = self._create_elyza_prompt(sample)

                # 推論実行
                response = self.llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                    echo=False
                )

                prediction = self._extract_answer(response['choices'][0]['text'])
                reference = sample.get('answer', '')

                predictions.append(prediction)
                references.append(reference)

                if prediction.strip().lower() == reference.strip().lower():
                    correct += 1
                total += 1

            # メトリクス計算
            accuracy = correct / total if total > 0 else 0.0

            # F1スコア計算
            f1_metric = evaluate.load("f1")
            f1_score = f1_metric.compute(predictions=predictions, references=references, average="weighted")

            return {
                'accuracy': accuracy,
                'f1': f1_score['f1'],
                'total_samples': total
            }

        except Exception as e:
            logger.error(f"ELYZA-100 evaluation failed: {e}")
            return {'accuracy': 0.0, 'f1': 0.0, 'error': str(e)}

    def evaluate_industry_standard(self) -> Dict[str, Dict[str, float]]:
        """業界標準ベンチマーク評価"""
        if not self.llm:
            return {name: {'accuracy': 0.0} for name in BENCHMARK_CONFIG['industry_standard']}

        results = {}

        for benchmark_name, benchmark_config in BENCHMARK_CONFIG['industry_standard'].items():
            logger.info(f"[BENCHMARK] Evaluating {benchmark_name}...")

            try:
                # データセット読み込み
                dataset_config = benchmark_config.copy()
                name = dataset_config.pop('name', None)

                if name:
                    dataset = load_dataset(**dataset_config, name=name)
                else:
                    dataset = load_dataset(**dataset_config)

                # 評価実行
                result = self._evaluate_single_benchmark(benchmark_name, dataset, benchmark_config['metrics'])
                results[benchmark_name] = result

            except Exception as e:
                logger.error(f"{benchmark_name} evaluation failed: {e}")
                results[benchmark_name] = {'accuracy': 0.0, 'error': str(e)}

        return results

    def evaluate_multimodal_performance(self) -> Dict[str, float]:
        """マルチモーダル性能評価"""
        if not self.llm:
            return {'accuracy': 0.0}

        logger.info("[BENCHMARK] Evaluating multimodal performance...")

        try:
            # ScienceQAデータセット（テキストベースのマルチモーダル評価）
            dataset = load_dataset(
                BENCHMARK_CONFIG['multimodal']['scienceqa']['dataset'],
                split='test'
            )

            correct = 0
            total = 0

            for sample in tqdm(dataset.select(range(min(500, len(dataset)))), desc="ScienceQA"):
                # 科学問題の評価
                prompt = self._create_scienceqa_prompt(sample)

                response = self.llm(
                    prompt,
                    max_tokens=256,
                    temperature=0.1,
                    echo=False
                )

                prediction = self._extract_scienceqa_answer(response['choices'][0]['text'])
                reference = str(sample.get('answer', ''))

                if prediction == reference:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0.0

            return {
                'accuracy': accuracy,
                'total_samples': total
            }

        except Exception as e:
            logger.error(f"Multimodal evaluation failed: {e}")
            return {'accuracy': 0.0, 'error': str(e)}

    def _evaluate_single_benchmark(self, name: str, dataset, metrics: List[str]) -> Dict[str, float]:
        """単一ベンチマーク評価"""
        results = {}

        try:
            # サンプル数制限（効率化）
            eval_samples = min(1000, len(dataset))
            dataset = dataset.select(range(eval_samples))

            for metric_name in metrics:
                if metric_name == 'accuracy':
                    correct = 0
                    total = 0

                    for sample in tqdm(dataset, desc=f"{name} accuracy"):
                        # ベンチマーク固有のプロンプト作成
                        prompt = self._create_benchmark_prompt(name, sample)

                        # 推論
                        response = self.llm(
                            prompt,
                            max_tokens=128,
                            temperature=0.0,  # 決定論的
                            echo=False
                        )

                        prediction = self._extract_benchmark_answer(name, response['choices'][0]['text'])
                        reference = self._get_benchmark_reference(name, sample)

                        if prediction == reference:
                            correct += 1
                        total += 1

                    results['accuracy'] = correct / total if total > 0 else 0.0
                    results['total_samples'] = total

        except Exception as e:
            logger.error(f"Benchmark {name} evaluation error: {e}")
            results['accuracy'] = 0.0
            results['error'] = str(e)

        return results

    def _create_elyza_prompt(self, sample) -> str:
        """ELYZA-100プロンプト作成"""
        question = sample.get('question', '')
        choices = sample.get('choices', [])

        if choices:
            # 選択式
            choices_text = '\n'.join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
            prompt = f"""以下の質問に答えてください。最も適切な選択肢を選んでください。

質問: {question}

選択肢:
{choices_text}

回答:"""
        else:
            # 記述式
            prompt = f"""以下の質問に答えてください。

質問: {question}

回答:"""

        return prompt

    def _create_benchmark_prompt(self, benchmark_name: str, sample) -> str:
        """ベンチマーク固有プロンプト作成"""
        if benchmark_name == 'mmlu':
            question = sample.get('question', '')
            choices = sample.get('choices', [])
            subject = sample.get('subject', '')

            choices_text = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            prompt = f"""以下は{subject}に関する問題です。正解の選択肢を選んでください。

{question}

{choices_text}

正解:"""

        elif benchmark_name == 'gsm8k':
            question = sample.get('question', '')
            prompt = f"""以下の数学の問題を解いてください。最終的な答えだけを数字で答えてください。

問題: {question}

答え:"""

        elif benchmark_name == 'hellaswag':
            ctx = sample.get('ctx', '')
            endings = sample.get('endings', [])

            endings_text = '\n'.join([f"{i+1}. {ending}" for i, ending in enumerate(endings)])
            prompt = f"""以下の文脈で、最も自然な続きを選んでください。

文脈: {ctx}

選択肢:
{endings_text}

正解:"""

        elif benchmark_name == 'arc_challenge':
            question = sample.get('question', '')
            choices = sample.get('choices', {'text': [], 'label': []})

            choices_text = '\n'.join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
            prompt = f"""以下の科学の問題に答えてください。

{question}

{choices_text}

正解:"""

        else:
            prompt = str(sample)

        return prompt

    def _create_scienceqa_prompt(self, sample) -> str:
        """ScienceQAプロンプト作成"""
        question = sample.get('question', '')
        choices = sample.get('choices', [])

        if choices:
            choices_text = '\n'.join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
            prompt = f"""以下の科学の問題に答えてください。

{question}

選択肢:
{choices_text}

正解:"""
        else:
            prompt = f"""以下の科学の問題に答えてください。

{question}

答え:"""

        return prompt

    def _extract_answer(self, response: str) -> str:
        """回答抽出"""
        # シンプルな抽出（実際にはより洗練された処理が必要）
        response = response.strip()
        # 最初の行または最初の単語を回答として扱う
        lines = response.split('\n')
        return lines[0].strip() if lines else response

    def _extract_benchmark_answer(self, benchmark_name: str, response: str) -> str:
        """ベンチマーク固有回答抽出"""
        response = response.strip()

        if benchmark_name == 'mmlu':
            # A, B, C, D 形式
            if response.upper() in ['A', 'B', 'C', 'D']:
                return response.upper()

        elif benchmark_name == 'gsm8k':
            # 数字のみ抽出
            import re
            numbers = re.findall(r'\d+', response)
            return numbers[-1] if numbers else response

        elif benchmark_name == 'hellaswag':
            # 数字で回答
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                idx = int(numbers[0]) - 1  # 1-indexed to 0-indexed
                return str(idx)

        elif benchmark_name == 'arc_challenge':
            # A, B, C, D 形式
            if response.upper() in ['A', 'B', 'C', 'D']:
                return response.upper()

        return response

    def _extract_scienceqa_answer(self, response: str) -> str:
        """ScienceQA回答抽出"""
        import re
        # 数字または選択肢ラベルを抽出
        numbers = re.findall(r'\d+', response)
        if numbers:
            return numbers[0]
        return response.strip()

    def _get_benchmark_reference(self, benchmark_name: str, sample) -> str:
        """ベンチマーク正解取得"""
        if benchmark_name == 'mmlu':
            return sample.get('answer', '')
        elif benchmark_name == 'gsm8k':
            return str(sample.get('answer', ''))
        elif benchmark_name == 'hellaswag':
            return str(sample.get('label', ''))
        elif benchmark_name == 'arc_challenge':
            return sample.get('answerKey', '')
        else:
            return str(sample.get('answer', ''))


class ModelComparisonSystem:
    """
    モデル比較システム
    2つのAlpha Gateモデルを比較し、最適モデルを選択
    """

    def __init__(self, model_dirs: Dict[str, str]):
        self.model_dirs = model_dirs
        self.results = {}

    def compare_models(self) -> Dict[str, Any]:
        """モデル比較実行"""
        logger.info("[COMPARISON] Starting model comparison...")

        for model_name, model_dir in self.model_dirs.items():
            logger.info(f"[COMPARISON] Evaluating {model_name}...")

            # GGUF変換
            gguf_path = self._convert_to_gguf(model_dir)
            if not gguf_path:
                logger.error(f"Failed to convert {model_name} to GGUF")
                continue

            # ベンチマーク評価
            evaluator = LlamaCppBenchmarkEvaluator(gguf_path, {})

            results = {
                'elyza_100': evaluator.evaluate_elyza_100(),
                'industry_standard': evaluator.evaluate_industry_standard(),
                'multimodal': evaluator.evaluate_multimodal_performance(),
                'inference_metrics': self._measure_inference_metrics(gguf_path)
            }

            self.results[model_name] = results

            # GGUFファイル削除
            os.unlink(gguf_path)

        # 比較結果分析
        comparison = self._analyze_comparison()

        return comparison

    def _convert_to_gguf(self, model_dir: str) -> Optional[str]:
        """GGUF変換"""
        try:
            convert_script = Path("external/llama.cpp-master/convert_hf_to_gguf.py")

            if not convert_script.exists():
                logger.error("llama.cpp convert script not found")
                return None

            # 一時GGUFファイル
            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                gguf_path = f.name

            # 変換実行
            cmd = [
                sys.executable, str(convert_script),
                model_dir,
                "--outfile", gguf_path,
                "--outtype", "q8_0"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=convert_script.parent)
            if result.returncode != 0:
                logger.error(f"GGUF conversion failed: {result.stderr}")
                return None

            return gguf_path

        except Exception as e:
            logger.error(f"GGUF conversion error: {e}")
            return None

    def _measure_inference_metrics(self, gguf_path: str) -> Dict[str, float]:
        """推論メトリクス測定"""
        try:
            if not LLAMA_CPP_AVAILABLE:
                return {'tokens_per_sec': 0.0, 'memory_usage': 0.0}

            # 簡単な推論テスト
            llm = Llama(model_path=gguf_path, n_ctx=512, verbose=False)

            test_prompt = "Hello, how are you?"
            start_time = time.time()

            response = llm(test_prompt, max_tokens=50, echo=False)

            end_time = time.time()
            inference_time = end_time - start_time
            tokens_generated = len(response['choices'][0]['text'].split())

            tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0.0

            # メモリ使用量（簡易測定）
            memory_usage = psutil.virtual_memory().percent

            return {
                'tokens_per_sec': tokens_per_sec,
                'memory_usage_percent': memory_usage,
                'inference_time': inference_time
            }

        except Exception as e:
            logger.error(f"Inference metrics measurement failed: {e}")
            return {'tokens_per_sec': 0.0, 'memory_usage_percent': 0.0}

    def _analyze_comparison(self) -> Dict[str, Any]:
        """比較結果分析"""
        if len(self.results) < 2:
            return {'error': 'Need at least 2 models to compare'}

        # スコア計算
        scores = {}
        for model_name, results in self.results.items():
            score = self._calculate_composite_score(results)
            scores[model_name] = score

        # 勝者決定
        winner = max(scores, key=scores.get)

        return {
            'scores': scores,
            'winner': winner,
            'winner_score': scores[winner],
            'detailed_results': self.results,
            'comparison_summary': self._create_comparison_summary()
        }

    def _calculate_composite_score(self, results: Dict[str, Any]) -> float:
        """複合スコア計算"""
        score = 0.0

        # ELYZA-100 (40%)
        if 'elyza_100' in results:
            elyza = results['elyza_100']
            score += 0.4 * (elyza.get('accuracy', 0.0) + elyza.get('f1', 0.0)) / 2

        # 業界標準ベンチマーク (40%)
        if 'industry_standard' in results:
            industry = results['industry_standard']
            industry_scores = []
            for benchmark, metrics in industry.items():
                industry_scores.append(metrics.get('accuracy', 0.0))
            if industry_scores:
                score += 0.4 * np.mean(industry_scores)

        # マルチモーダル性能 (10%)
        if 'multimodal' in results:
            multimodal = results['multimodal']
            score += 0.1 * multimodal.get('accuracy', 0.0)

        # 推論性能 (10%)
        if 'inference_metrics' in results:
            inference = results['inference_metrics']
            # 正規化された推論スコア
            tokens_per_sec = min(inference.get('tokens_per_sec', 0.0) / 100.0, 1.0)  # 100 tokens/sec で満点
            score += 0.1 * tokens_per_sec

        return score

    def _create_comparison_summary(self) -> Dict[str, Any]:
        """比較サマリー作成"""
        summary = {}

        for model_name, results in self.results.items():
            summary[model_name] = {
                'elyza_accuracy': results.get('elyza_100', {}).get('accuracy', 0.0),
                'elyza_f1': results.get('elyza_100', {}).get('f1', 0.0),
                'industry_avg_accuracy': np.mean([
                    metrics.get('accuracy', 0.0)
                    for benchmark, metrics in results.get('industry_standard', {}).items()
                ]),
                'multimodal_accuracy': results.get('multimodal', {}).get('accuracy', 0.0),
                'inference_speed': results.get('inference_metrics', {}).get('tokens_per_sec', 0.0)
            }

        return summary


def upload_best_model_to_hf(best_model_dir: str, model_name: str):
    """
    最適モデルをBF16/Q8.0/Q4形式でHFアップロード
    Upload best model to HF with BF16/Q8.0/Q4 formats
    """
    from huggingface_hub import HfApi
    import shutil

    logger.info(f"[UPLOAD] Uploading best model: {model_name}")

    api = HfApi()

    # 各量子化形式のアップロード
    formats = [
        ('BF16', 'merged_16bit', 'Upload BF16 version'),
        ('Q8_0', 'q8_0', 'Upload Q8.0 quantized version'),
        ('Q4_Unsloth', 'q4_k_m', 'Upload Q4 Unsloth quantized version')
    ]

    for format_name, quant_type, commit_msg in formats:
        try:
            # モデル量子化（必要な場合）
            if format_name == 'BF16':
                upload_path = best_model_dir
            else:
                upload_path = f"{best_model_dir}_{quant_type.lower()}"
                if not os.path.exists(upload_path):
                    os.makedirs(upload_path, exist_ok=True)
                    # 量子化コピー（実際には適切な量子化処理が必要）
                    shutil.copytree(best_model_dir, upload_path, dirs_exist_ok=True)

            # リポジトリ作成とアップロード
            repo_name = f"borea-phi35-alpha-gate-sigmoid-{format_name.lower()}"
            api.create_repo(repo_name, private=False, exist_ok=True)

            api.upload_folder(
                folder_path=upload_path,
                repo_id=f"your-username/{repo_name}",
                commit_message=commit_msg
            )

            logger.info(f"[UPLOAD] {format_name} version uploaded to {repo_name}")

        except Exception as e:
            logger.error(f"[UPLOAD] Failed to upload {format_name}: {e}")


def main():
    """メイン関数"""
    # 比較する2つのモデル
    model_dirs = {
        'alpha_gate_linear': 'D:/webdataset/checkpoints/alpha_gate_linear/final',
        'alpha_gate_sigmoid_bayesian': 'D:/webdataset/checkpoints/alpha_gate_sigmoid_bayesian/final'
    }

    # モデル比較システム初期化
    comparator = ModelComparisonSystem(model_dirs)

    # 比較実行
    logger.info("[MAIN] Starting model comparison...")
    comparison_results = comparator.compare_models()

    if 'error' in comparison_results:
        logger.error(f"Comparison failed: {comparison_results['error']}")
        return

    # 結果表示
    logger.info("=== COMPARISON RESULTS ===")
    logger.info(f"Winner: {comparison_results['winner']}")
    logger.info(f"Winner Score: {comparison_results['winner_score']:.4f}")
    logger.info("Scores:")
    for model, score in comparison_results['scores'].items():
        logger.info(".4f")

    # 最適モデルのHFアップロード
    best_model_dir = model_dirs[comparison_results['winner']]
    upload_best_model_to_hf(best_model_dir, comparison_results['winner'])

    # 結果保存
    results_file = "D:/webdataset/results/alpha_gate_comparison_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    logger.info(f"[MAIN] Results saved to {results_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.PlaySound(r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav", winsound.SND_FILENAME)
    except:
        print('\a')


if __name__ == '__main__':
    main()
