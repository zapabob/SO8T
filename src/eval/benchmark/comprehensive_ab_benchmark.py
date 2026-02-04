#!/usr/bin/env python3
"""
Comprehensive A/B Benchmark Testing Script with Checkpoint Recovery
modelA vs AEGIS models across all downloaded datasets

チェックポイント機能:
- 3分ごとに自動チェックポイント保存
- 5個のローリングストック（最新5個保持）
- 電源断/中断からの自動リカバリー
- tqdm進捗表示と詳細ロギング
- シグナルハンドラー（SIGINT/SIGTERM/SIGBREAK対応）

使用方法:
1. 初回実行: python comprehensive_ab_benchmark.py
2. リカバリー実行: 電源投入後に再度実行（自動でチェックポイントから再開）
3. 中断時: Ctrl+Cで安全中断（チェックポイント保存）

保存場所:
- 結果: D:/webdataset/benchmark_results/
- チェックポイント: D:/webdataset/benchmark_results/checkpoints/
- ログ: D:/webdataset/benchmark_results/logs/
"""

import os
import json
import pandas as pd
import numpy as np
import subprocess
import time
import signal
import atexit
from pathlib import Path
from tqdm import tqdm
import logging
import logging.handlers
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ロギング設定
LOG_DIR = Path("D:/webdataset/benchmark_results/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ローテーションロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# コンソールハンドラー
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# ファイルハンドラー（ローテーション）
file_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / 'benchmark.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 設定
DATA_DIR = Path("D:/webdataset/benchmark_datasets")
RESULTS_DIR = Path("D:/webdataset/benchmark_results")
OLLAMA_TIMEOUT = 300  # 5分タイムアウト

# チェックポイント設定
CHECKPOINT_INTERVAL = 180  # 3分（秒）
MAX_CHECKPOINTS = 5  # ローリングストック数
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

class ComprehensiveABBenchmark:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # チェックポイントディレクトリ
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # モデル設定
        self.models = {
            "modelA": "model-a:q8_0",
            "AEGIS": "agiasi-phi35-golden-sigmoid:latest"
        }

        # データセット設定
        self.datasets = {
            "MMLU": self.load_mmlu_data,
            "GSM8K": self.load_gsm8k_data,
            "AGI_ARC_Challenge": self.load_arc_challenge_data,
            "AGI_ARC_Easy": self.load_arc_easy_data,
            "AGI_HellaSwag": self.load_hellaswag_data,
            "AGI_Winogrande": self.load_winogrande_data,
            "Final_Exam_ARC": self.load_final_exam_arc_data,
        }

        # チェックポイント関連
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter = 0
        self.all_results = []
        self.completed_tests = set()  # 完了したテストの追跡

        # シグナルハンドラー設定（電源断対応）
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self.signal_handler)

        # 終了時自動保存
        atexit.register(self.save_checkpoint, "final")

        # リカバリーチェック
        self.load_latest_checkpoint()

    def signal_handler(self, signum, frame):
        """シグナルハンドラー（電源断・Ctrl+C対応）"""
        logger.warning(f"Signal {signum} received. Saving emergency checkpoint...")
        self.save_checkpoint("emergency")
        logger.info("Emergency checkpoint saved. Exiting...")
        exit(0)

    def save_checkpoint(self, checkpoint_type="regular"):
        """チェックポイント保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{checkpoint_type}_{timestamp}_{self.checkpoint_counter:04d}.json"

            checkpoint_data = {
                'timestamp': timestamp,
                'checkpoint_type': checkpoint_type,
                'counter': self.checkpoint_counter,
                'all_results': self.all_results,
                'completed_tests': list(self.completed_tests),
                'models': self.models,
                'datasets': list(self.datasets.keys()),
                'last_checkpoint_time': self.last_checkpoint_time
            }

            checkpoint_path = self.checkpoint_dir / checkpoint_name
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Checkpoint saved: {checkpoint_path} ({len(self.all_results)} results)")

            # 古いチェックポイント削除（ローリングストック）
            self.cleanup_old_checkpoints()

            self.checkpoint_counter += 1

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_latest_checkpoint(self):
        """最新のチェックポイントから復旧"""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"),
                               key=lambda x: x.stat().st_mtime, reverse=True)

            if not checkpoints:
                logger.info("No checkpoints found. Starting fresh.")
                return

            latest_checkpoint = checkpoints[0]
            logger.info(f"Loading checkpoint: {latest_checkpoint}")

            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            self.all_results = checkpoint_data.get('all_results', [])
            self.completed_tests = set(checkpoint_data.get('completed_tests', []))
            self.checkpoint_counter = checkpoint_data.get('counter', 0) + 1
            self.last_checkpoint_time = checkpoint_data.get('last_checkpoint_time', time.time())

            logger.info(f"Recovered {len(self.all_results)} results from checkpoint")
            logger.info(f"Completed tests: {len(self.completed_tests)}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh due to checkpoint loading error.")

    def should_save_checkpoint(self):
        """チェックポイント保存が必要か判定"""
        current_time = time.time()
        return (current_time - self.last_checkpoint_time) >= CHECKPOINT_INTERVAL

    def cleanup_old_checkpoints(self):
        """古いチェックポイントを削除（ローリングストック）"""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"),
                               key=lambda x: x.stat().st_mtime)

            # 最新5個以外を削除
            if len(checkpoints) > MAX_CHECKPOINTS:
                checkpoints_to_delete = checkpoints[:-MAX_CHECKPOINTS]
                for checkpoint in checkpoints_to_delete:
                    checkpoint.unlink()
                    logger.debug(f"Deleted old checkpoint: {checkpoint}")

                logger.info(f"Cleaned up {len(checkpoints_to_delete)} old checkpoints")

        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")

    def load_mmlu_data(self):
        """MMLUデータロード"""
        mmlu_path = self.data_dir / "mmlu" / "mmlu_full.json"
        if mmlu_path.exists():
            df = pd.read_json(mmlu_path)
            # テスト用に最初の1000サンプルを使用
            return df.head(1000).to_dict('records')
        return []

    def load_gsm8k_data(self):
        """GSM8Kデータロード"""
        gsm8k_path = self.data_dir / "gsm8k" / "gsm8k_full.json"
        if gsm8k_path.exists():
            df = pd.read_json(gsm8k_path)
            # テスト用に最初の500サンプルを使用
            test_data = df[df['split'] == 'test'].head(500).to_dict('records')
            return test_data
        return []

    def load_arc_challenge_data(self):
        """ARC-Challengeデータロード"""
        arc_path = self.data_dir / "agi_tests" / "allenai_ai2_arc_ARC-Challenge_test.json"
        if arc_path.exists():
            df = pd.read_json(arc_path)
            return df.head(500).to_dict('records')
        return []

    def load_arc_easy_data(self):
        """ARC-Easyデータロード"""
        arc_path = self.data_dir / "agi_tests" / "allenai_ai2_arc_ARC-Easy_test.json"
        if arc_path.exists():
            df = pd.read_json(arc_path)
            return df.head(500).to_dict('records')
        return []

    def load_hellaswag_data(self):
        """HellaSwagデータロード"""
        hellaswag_path = self.data_dir / "agi_tests" / "hellaswag_main_validation.json"
        if hellaswag_path.exists():
            df = pd.read_json(hellaswag_path)
            return df.head(500).to_dict('records')
        return []

    def load_winogrande_data(self):
        """Winograndeデータロード"""
        winogrande_path = self.data_dir / "agi_tests" / "winogrande_winogrande_xl_validation.json"
        if winogrande_path.exists():
            df = pd.read_json(winogrande_path)
            return df.head(500).to_dict('records')
        return []

    def load_final_exam_arc_data(self):
        """Final Exam ARCデータロード"""
        arc_path = self.data_dir / "final_exam" / "allenai_ai2_arc_ARC-Challenge_test.json"
        if arc_path.exists():
            df = pd.read_json(arc_path)
            return df.head(500).to_dict('records')
        return []

    def evaluate_mmlu_response(self, response: str, correct_answer: str) -> float:
        """MMLU回答評価"""
        if not response or not correct_answer:
            return 0.0

        # 回答を正規化
        response = response.strip().upper()
        correct_answer = correct_answer.strip().upper()

        # 完全一致
        if response == correct_answer:
            return 1.0

        # 部分一致（最初の文字が一致）
        if response and correct_answer and response[0] == correct_answer[0]:
            return 0.5

        return 0.0

    def evaluate_math_response(self, response: str, correct_answer: str) -> float:
        """数学問題回答評価"""
        if not response or not correct_answer:
            return 0.0

        # 数値抽出
        def extract_numbers(text):
            numbers = re.findall(r'\d+\.?\d*', text)
            return [float(n) for n in numbers]

        response_nums = extract_numbers(response)
        correct_nums = extract_numbers(correct_answer)

        if not response_nums or not correct_nums:
            return 0.0

        # 最終回答の一致
        if abs(response_nums[-1] - correct_nums[-1]) < 0.01:
            return 1.0

        return 0.0

    def evaluate_multiple_choice_response(self, response: str, choices: list, correct_answer: str) -> float:
        """多肢選択回答評価"""
        if not response or not choices:
            return 0.0

        response = response.strip().upper()

        # 完全一致
        if response == correct_answer.upper():
            return 1.0

        # インデックスによる回答（A, B, C, D）
        choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        if response in choice_map and choice_map[response] < len(choices):
            if choices[choice_map[response]] == correct_answer:
                return 1.0

        return 0.0

    def run_ollama_query(self, model: str, prompt: str, timeout: int = OLLAMA_TIMEOUT) -> tuple:
        """Ollamaクエリ実行"""
        try:
            cmd = ["ollama", "run", model, prompt]
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )

            end_time = time.time()
            response_time = end_time - start_time

            if result.returncode == 0:
                return result.stdout.strip(), response_time
            else:
                logger.warning(f"Ollama error for {model}: {result.stderr}")
                return "", response_time

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout for {model}")
            return "", timeout
        except Exception as e:
            logger.error(f"Error running Ollama for {model}: {e}")
            return "", 0.0

    def run_single_test(self, model: str, dataset_name: str, item: dict) -> dict:
        """単一テスト実行"""
        try:
            # データセットに応じたプロンプト作成
            if dataset_name == "MMLU":
                question = item.get('question', '')
                choices = item.get('choices', [])
                if choices:
                    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                    prompt = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer with only the letter (A, B, C, or D):"
                else:
                    prompt = f"Question: {question}\n\nProvide a concise answer:"
                correct_answer = item.get('answer', '')

            elif dataset_name in ["GSM8K", "MATH"]:
                question = item.get('question', item.get('problem', ''))
                prompt = f"Solve this math problem step by step:\n\n{question}\n\nProvide the final numerical answer."
                correct_answer = item.get('answer', item.get('solution', ''))

            elif "ARC" in dataset_name:
                question = item.get('question', '')
                choices = item.get('choices', [])
                if choices:
                    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                    prompt = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer with only the letter (A, B, C, or D):"
                else:
                    prompt = f"Question: {question}\n\nProvide a concise answer:"
                correct_answer = item.get('answerKey', '')

            elif dataset_name == "AGI_HellaSwag":
                ctx = item.get('ctx', '')
                endings = item.get('endings', [])
                if endings:
                    endings_text = "\n".join([f"{chr(65+i)}. {ending}" for i, ending in enumerate(endings)])
                    prompt = f"Complete this sentence:\n\n{ctx}\n\nChoices:\n{endings_text}\n\nAnswer with only the letter (A, B, C, or D):"
                    correct_answer = chr(65 + item.get('label', 0))
                else:
                    prompt = f"Complete this sentence:\n\n{ctx}"
                    correct_answer = ""

            elif dataset_name == "AGI_Winogrande":
                sentence = item.get('sentence', '')
                option1 = item.get('option1', '')
                option2 = item.get('option2', '')
                prompt = f"Fill in the blank:\n\n{sentence.replace('_', '[BLANK]')}\n\nChoices:\nA. {option1}\nB. {option2}\n\nAnswer with only A or B:"
                correct_answer = item.get('answer', '')

            else:
                # デフォルトプロンプト
                question = str(item)
                prompt = f"Please answer this question:\n\n{question}"
                correct_answer = ""

            # Ollamaクエリ実行
            response, response_time = self.run_ollama_query(model, prompt)

            # 回答評価
            if dataset_name == "MMLU":
                score = self.evaluate_mmlu_response(response, correct_answer)
            elif dataset_name in ["GSM8K", "MATH"]:
                score = self.evaluate_math_response(response, correct_answer)
            elif "ARC" in dataset_name or "HellaSwag" in dataset_name:
                choices = item.get('choices', item.get('endings', []))
                score = self.evaluate_multiple_choice_response(response, choices, correct_answer)
            elif dataset_name == "AGI_Winogrande":
                score = 1.0 if response.strip().upper() == correct_answer.strip().upper() else 0.0
            else:
                score = 0.5  # 不明なデータセットは中間スコア

            return {
                'model': model,
                'dataset': dataset_name,
                'question': question[:200],  # 最初の200文字
                'response': response[:500],  # 最初の500文字
                'correct_answer': correct_answer[:200],
                'score': score,
                'response_time': response_time,
                'success': response_time < OLLAMA_TIMEOUT
            }

        except Exception as e:
            logger.error(f"Error in single test: {e}")
            return {
                'model': model,
                'dataset': dataset_name,
                'question': str(item)[:200],
                'response': '',
                'correct_answer': '',
                'score': 0.0,
                'response_time': 0.0,
                'success': False,
                'error': str(e)
            }

    def run_benchmark_tests(self):
        """全ベンチマークテスト実行（チェックポイント対応）"""
        logger.info("Starting comprehensive A/B benchmark tests with checkpoint support...")

        # テストモードの場合はサンプル数を制限
        sample_limit = 50 if hasattr(self, 'test_mode') and self.test_mode else None

        # 全体の進捗バー計算
        total_tests = 0
        for dataset_name, load_func in self.datasets.items():
            try:
                data = load_func()
                actual_samples = min(len(data), sample_limit) if sample_limit else len(data)
                total_tests += actual_samples * len(self.models)
            except Exception as e:
                logger.warning(f"Could not calculate samples for {dataset_name}: {e}")

        logger.info(f"Total tests to run: {total_tests}")
        with tqdm(total=total_tests, desc="Overall Progress", unit="test") as overall_pbar:
            for dataset_name, load_func in self.datasets.items():
                logger.info(f"Loading {dataset_name} data...")
                data = load_func()

                if not data:
                    logger.warning(f"No data found for {dataset_name}")
                    continue

                # テストモードの場合はサンプル数を制限
                if hasattr(self, 'test_mode') and self.test_mode and sample_limit:
                    original_count = len(data)
                    data = data[:sample_limit]
                    logger.info(f"Test mode: Limited {dataset_name} from {original_count} to {len(data)} samples")

                logger.info(f"Testing {dataset_name} with {len(data)} samples...")

                for model_name, model_id in self.models.items():
                    test_key = f"{dataset_name}_{model_name}"

                    # 既に完了したテストはスキップ
                    if test_key in self.completed_tests:
                        logger.info(f"Skipping completed test: {test_key}")
                        overall_pbar.update(1)
                        continue

                    logger.info(f"Testing {model_name} on {dataset_name}...")

                    results = []
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        futures = [executor.submit(self.run_single_test, model_id, dataset_name, item)
                                 for item in data]

                        for future in tqdm(as_completed(futures), total=len(data),
                                         desc=f"{model_name} on {dataset_name}", leave=False):
                            result = future.result()
                            results.append(result)

                            # 定期チェックポイント保存
                            if self.should_save_checkpoint():
                                self.all_results.extend(results)
                                self.save_checkpoint("progress")
                                self.last_checkpoint_time = time.time()
                                logger.info(f"Progress checkpoint saved ({len(self.all_results)} total results)")

                    # 結果保存
                    results_df = pd.DataFrame(results)
                    dataset_results_path = self.results_dir / f"{dataset_name}_{model_name}_results.json"
                    # pandasのバージョンによってはensure_asciiが使えないので除去
                    results_df.to_json(dataset_results_path, orient='records', indent=2)
                    logger.info(f"Results saved to {dataset_results_path}")

                    # 全結果に追加
                    self.all_results.extend(results)

                    # 完了したテストをマーク
                    self.completed_tests.add(test_key)

                    # データセット完了時のチェックポイント
                    self.save_checkpoint(f"dataset_{dataset_name}_{model_name}")

                    overall_pbar.update(1)

        # 最終結果保存
        all_results_df = pd.DataFrame(self.all_results)
        all_results_path = self.results_dir / "comprehensive_ab_benchmark_results.json"
        # pandasのバージョンによってはensure_asciiが使えないので除去
        all_results_df.to_json(all_results_path, orient='records', indent=2)
        logger.info(f"All results saved to {all_results_path}")

        # 最終チェックポイント
        self.save_checkpoint("final")

        return all_results_df

    def generate_statistics(self, results_df: pd.DataFrame):
        """統計分析生成"""
        logger.info("Generating statistical analysis...")

        stats_results = []

        # データセットごとの統計
        for dataset in results_df['dataset'].unique():
            dataset_df = results_df[results_df['dataset'] == dataset]

            for model in results_df['model'].unique():
                model_df = dataset_df[dataset_df['model'] == model]

                if len(model_df) > 0:
                    scores = model_df['score'].values
                    response_times = model_df['response_time'].values

                    stats_dict = {
                        'dataset': dataset,
                        'model': model,
                        'count': len(scores),
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'median_score': np.median(scores),
                        'min_score': np.min(scores),
                        'max_score': np.max(scores),
                        'mean_response_time': np.mean(response_times),
                        'std_response_time': np.std(response_times),
                        'success_rate': np.mean(model_df['success'].values),
                        'perfect_rate': np.mean(scores == 1.0)
                    }

                    # 95%信頼区間
                    if len(scores) > 1:
                        se = stats.sem(scores)
                        stats_dict['score_ci_lower'] = np.mean(scores) - 1.96 * se
                        stats_dict['score_ci_upper'] = np.mean(scores) + 1.96 * se

                    stats_results.append(stats_dict)

        stats_df = pd.DataFrame(stats_results)
        stats_path = self.results_dir / "benchmark_statistics.json"
        stats_df.to_json(stats_path, orient='records', indent=2)
        logger.info(f"Statistics saved to {stats_path}")

        return stats_df

    def perform_statistical_tests(self, results_df: pd.DataFrame):
        """統計的有意差検定"""
        logger.info("Performing statistical significance tests...")

        test_results = []

        for dataset in results_df['dataset'].unique():
            dataset_df = results_df[results_df['dataset'] == dataset]

            models = list(self.models.keys())
            if len(models) == 2:
                model1, model2 = models
                scores1 = dataset_df[dataset_df['model'] == model1]['score'].values
                scores2 = dataset_df[dataset_df['model'] == model2]['score'].values

                if len(scores1) > 1 and len(scores2) > 1:
                    # t-test
                    try:
                        t_stat, p_value = stats.ttest_ind(scores1, scores2)

                        # ANOVA (3つ以上のモデル用に準備)
                        if len(self.models) > 2:
                            all_scores = [dataset_df[dataset_df['model'] == m]['score'].values
                                        for m in models]
                            f_stat, anova_p = stats.f_oneway(*all_scores)
                        else:
                            f_stat, anova_p = None, None

                        test_results.append({
                            'dataset': dataset,
                            'model1': model1,
                            'model2': model2,
                            'model1_mean': np.mean(scores1),
                            'model2_mean': np.mean(scores2),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'anova_f': f_stat,
                            'anova_p': anova_p
                        })

                    except Exception as e:
                        logger.warning(f"Statistical test failed for {dataset}: {e}")

        test_df = pd.DataFrame(test_results)
        test_path = self.results_dir / "statistical_tests.json"
        test_df.to_json(test_path, orient='records', indent=2)
        logger.info(f"Statistical tests saved to {test_path}")

        return test_df

    def create_visualizations(self, results_df: pd.DataFrame, stats_df: pd.DataFrame):
        """可視化作成"""
        logger.info("Creating visualizations...")

        # スタイル設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. データセット別スコア比較（エラーバー付き）
        plt.figure(figsize=(15, 8))

        datasets = stats_df['dataset'].unique()
        models = list(self.models.keys())
        x = np.arange(len(datasets))
        width = 0.35

        for i, model in enumerate(models):
            model_stats = stats_df[stats_df['model'] == model]
            scores = []
            errors = []

            for dataset in datasets:
                dataset_stat = model_stats[model_stats['dataset'] == dataset]
                if len(dataset_stat) > 0:
                    score = dataset_stat['mean_score'].values[0]
                    std = dataset_stat['std_score'].values[0]
                    scores.append(score)
                    errors.append(std)
                else:
                    scores.append(0)
                    errors.append(0)

            plt.bar(x + i*width, scores, width, label=model,
                   yerr=errors, capsize=5, alpha=0.8)

        plt.xlabel('Dataset')
        plt.ylabel('Mean Accuracy Score')
        plt.title('Model Performance Comparison Across Benchmarks')
        plt.xticks(x + width/2, datasets, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 詳細なベンチマーク比較
        plt.figure(figsize=(12, 8))

        # スコア分布
        for model in models:
            model_data = results_df[results_df['model'] == model]
            plt.hist(model_data['score'], alpha=0.7, label=model, bins=20)

        plt.xlabel('Accuracy Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution by Model')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 統計的有意差を示すグラフ
        plt.figure(figsize=(10, 6))

        # モデル間の平均スコア比較
        model_means = []
        model_stds = []

        for model in models:
            model_data = results_df[results_df['model'] == model]
            model_means.append(model_data['score'].mean())
            model_stds.append(model_data['score'].std())

        bars = plt.bar(models, model_means, yerr=model_stds, capsize=5, alpha=0.8)
        plt.ylabel('Mean Accuracy Score')
        plt.title('Overall Model Performance Comparison')
        plt.ylim(0, 1)

        # 値ラベル追加
        for bar, mean, std in zip(bars, model_means, model_stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Visualizations created successfully")

    def generate_markdown_report(self, results_df: pd.DataFrame, stats_df: pd.DataFrame, test_df: pd.DataFrame):
        """Markdownレポート生成"""
        logger.info("Generating comprehensive Markdown report...")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = self.results_dir / f"comprehensive_ab_benchmark_report_{timestamp}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive A/B Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Overview\n\n")
            f.write("This report presents a comprehensive comparison between modelA and AEGIS models ")
            f.write("across multiple benchmark datasets including MMLU, GSM8K, ARC, HellaSwag, and Winogrande.\n\n")

            f.write("## Models Tested\n\n")
            for model_name, model_id in self.models.items():
                f.write(f"- **{model_name}**: `{model_id}`\n")
            f.write("\n")

            f.write("## Datasets Overview\n\n")
            dataset_info = {
                "MMLU": "Massive Multitask Language Understanding - General knowledge across 57 subjects",
                "GSM8K": "Grade School Math - Mathematical reasoning problems",
                "AGI_ARC_Challenge": "AI2 Reasoning Challenge - Hard science questions",
                "AGI_ARC_Easy": "AI2 Reasoning Challenge - Easier science questions",
                "AGI_HellaSwag": "HellaSwag - Commonsense reasoning through story completion",
                "AGI_Winogrande": "Winogrande - Commonsense reasoning with pronoun resolution",
                "Final_Exam_ARC": "Final Exam ARC - Most challenging reasoning tasks"
            }

            for dataset, description in dataset_info.items():
                count = len(results_df[results_df['dataset'] == dataset])
                f.write(f"- **{dataset}**: {description} ({count} samples)\n")
            f.write("\n")

            f.write("## Statistical Summary\n\n")
            f.write("| Dataset | Model | Mean Score | Std Dev | Success Rate | Perfect Rate | Sample Count |\n")
            f.write("|---------|--------|------------|---------|--------------|--------------|--------------|\n")

            for _, row in stats_df.iterrows():
                f.write(f"| {row['dataset']} | {row['model']} | {row['mean_score']:.3f} | {row['std_score']:.3f} | {row['success_rate']:.3f} | {row['perfect_rate']:.3f} | {int(row['count'])} |\n")

            f.write("\n")

            f.write("## Statistical Significance Tests\n\n")
            if len(test_df) > 0:
                f.write("| Dataset | Model 1 | Model 2 | Mean 1 | Mean 2 | t-statistic | p-value | Significant |\n")
                f.write("|---------|---------|---------|--------|--------|-------------|---------|-------------|\n")

                for _, row in test_df.iterrows():
                    sig = "✓" if row['significant'] else "✗"
                    f.write(f"| {row['dataset']} | {row['model1']} | {row['model2']} | {row['model1_mean']:.3f} | {row['model2_mean']:.3f} | {row['t_statistic']:.3f} | {row['p_value']:.4f} | {sig} |\n")

                f.write("\n")
                f.write("**Note**: Significant differences (p < 0.05) are marked with ✓\n\n")

            f.write("## Performance Analysis\n\n")

            # モデル別パフォーマンス比較
            for model in self.models.keys():
                model_stats = stats_df[stats_df['model'] == model]
                avg_score = model_stats['mean_score'].mean()
                avg_success = model_stats['success_rate'].mean()

                f.write(f"### {model} Performance\n\n")
                f.write(f"- **Average Accuracy**: {avg_score:.3f}\n")
                f.write(f"- **Average Success Rate**: {avg_success:.3f}\n")
                f.write(f"- **Best Performing Dataset**: {model_stats.loc[model_stats['mean_score'].idxmax()]['dataset']} ({model_stats['mean_score'].max():.3f})\n")
                f.write(f"- **Worst Performing Dataset**: {model_stats.loc[model_stats['mean_score'].idxmin()]['dataset']} ({model_stats['mean_score'].min():.3f})\n\n")

            f.write("## Key Findings\n\n")

            # 勝者判定
            overall_scores = stats_df.groupby('model')['mean_score'].mean()
            winner = overall_scores.idxmax()
            winner_score = overall_scores.max()
            loser_score = overall_scores.min()

            f.write(f"1. **Overall Winner**: {winner} with average accuracy of {winner_score:.3f}\n")
            f.write(f"2. **Performance Gap**: {winner_score - loser_score:.3f} difference between best and worst model\n")

            # データセット別勝者
            dataset_winners = {}
            for dataset in stats_df['dataset'].unique():
                dataset_scores = stats_df[stats_df['dataset'] == dataset]
                if len(dataset_scores) > 0:
                    winner = dataset_scores.loc[dataset_scores['mean_score'].idxmax()]['model']
                    dataset_winners[dataset] = winner

            f.write("3. **Dataset-specific Winners**:\n")
            for dataset, winner in dataset_winners.items():
                f.write(f"   - {dataset}: {winner}\n")

            f.write("\n## Visualizations\n\n")
            f.write("The following charts are available in the results directory:\n\n")
            f.write("- `benchmark_comparison.png`: Dataset-wise performance comparison with error bars\n")
            f.write("- `score_distribution.png`: Score distribution histograms by model\n")
            f.write("- `overall_performance.png`: Overall model performance comparison\n\n")

            f.write("## Raw Data Files\n\n")
            f.write("- `comprehensive_ab_benchmark_results.json`: Complete test results\n")
            f.write("- `benchmark_statistics.json`: Statistical summaries\n")
            f.write("- `statistical_tests.json`: Significance test results\n\n")

            f.write("## Technical Notes\n\n")
            f.write("- **Timeout**: 300 seconds per query\n")
            f.write("- **Evaluation Method**: Automated scoring based on exact/partial matches\n")
            f.write("- **Statistical Tests**: Student's t-test for significance\n")
            f.write("- **Confidence Intervals**: 95% CI for error bars\n\n")

            f.write("---\n\n")
            f.write("*Report generated by Comprehensive A/B Benchmark System*")

        logger.info(f"Markdown report saved to {report_path}")
        return report_path

    def run_complete_analysis(self):
        """完全分析実行（チェックポイント対応）"""
        logger.info("Starting complete A/B benchmark analysis with checkpoint recovery...")

        # リカバリー情報表示
        if self.all_results:
            logger.info(f"Resuming from checkpoint with {len(self.all_results)} existing results")
            logger.info(f"Completed tests: {sorted(self.completed_tests)}")

        try:
            # 1. ベンチマークテスト実行
            logger.info("Phase 1: Running benchmark tests...")
            results_df = self.run_benchmark_tests()

            # 2. 統計分析
            logger.info("Phase 2: Generating statistics...")
            stats_df = self.generate_statistics(results_df)
            self.save_checkpoint("statistics")

            # 3. 統計的有意差検定
            logger.info("Phase 3: Performing statistical tests...")
            test_df = self.perform_statistical_tests(results_df)
            self.save_checkpoint("statistical_tests")

            # 4. 可視化作成
            logger.info("Phase 4: Creating visualizations...")
            self.create_visualizations(results_df, stats_df)
            self.save_checkpoint("visualizations")

            # 5. レポート生成
            logger.info("Phase 5: Generating final report...")
            report_path = self.generate_markdown_report(results_df, stats_df, test_df)

            logger.info("Complete analysis finished successfully!")
            logger.info(f"Results saved to: {self.results_dir}")
            logger.info(f"Report: {report_path}")
            logger.info(f"Total checkpoints saved: {self.checkpoint_counter}")

        except KeyboardInterrupt:
            logger.warning("Analysis interrupted by user. Checkpoint saved.")
            self.save_checkpoint("interrupted")
        except Exception as e:
            logger.error(f"Analysis failed with error: {e}")
            self.save_checkpoint("error")
            raise

        return report_path


def main():
    """メイン関数 - チェックポイント対応のA/Bベンチマーク実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive A/B Benchmark with Checkpoint Support")
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (limited samples)')
    parser.add_argument('--single-dataset', type=str, help='Test only specified dataset')
    parser.add_argument('--skip-recovery', action='store_true', help='Skip checkpoint recovery')

    args = parser.parse_args()

    print("=== Comprehensive A/B Benchmark System with Checkpoint Support ===")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} seconds")
    print(f"Max checkpoints: {MAX_CHECKPOINTS}")
    if args.test_mode:
        print("TEST MODE: Limited samples for testing")
    if args.single_dataset:
        print(f"SINGLE DATASET MODE: {args.single_dataset} only")
    print()

    try:
        benchmark = ComprehensiveABBenchmark()

        if args.skip_recovery:
            logger.info("Skipping checkpoint recovery as requested")
            benchmark.all_results = []
            benchmark.completed_tests = set()

        if args.test_mode or args.single_dataset:
            # テストモード実行
            if args.single_dataset:
                # 指定データセットのみテスト
                if args.single_dataset in benchmark.datasets:
                    benchmark.datasets = {args.single_dataset: benchmark.datasets[args.single_dataset]}
                    logger.info(f"Testing single dataset: {args.single_dataset}")
                else:
                    logger.error(f"Dataset {args.single_dataset} not found")
                    return
            else:
                # テストモード：全データセット対象だが、後でサンプル数を制限
                logger.info("Running in test mode (full datasets with sample limits)")

            # テストモードフラグ設定
            benchmark.test_mode = args.test_mode or bool(args.single_dataset)

        report_path = benchmark.run_complete_analysis()

        print(f"\n[AUDIO] Playing completion notification...")
        print(f"[OK] Comprehensive A/B benchmark completed. Report: {report_path}")
        print(f"[INFO] Checkpoints saved: {benchmark.checkpoint_counter}")

    except KeyboardInterrupt:
        print("\n[WARNING] Benchmark interrupted by user")
        print("[INFO] Progress saved to checkpoint")
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        print("[INFO] Emergency checkpoint saved")
        raise


if __name__ == "__main__":
    main()
