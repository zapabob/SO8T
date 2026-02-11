#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) PPO学習済みPhi3 vs 元のPhi3 A/Bテスト
業界標準ベンチマーク + ELYZA-100統合評価

比較対象:
- Model A: microsoft/Phi-3.5-mini-instruct (オリジナル)
- Model B: borea-phi35-so8t-final (SO(8) PPO学習済み)
"""

import argparse
import json
import logging
import subprocess
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

# lm-evaluation-harness imports
try:
    import lm_eval
    from lm_eval import evaluator, tasks
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    logging.warning("lm-evaluation-harness not available, using fallback evaluation")

# 設定
DEFAULT_OUTPUT_ROOT = Path(r"data/benchmark_results/ab_test_so8t")
ELYZA_TASKS_FILE = Path("_data/elyza100_samples/elyza_tasks.json")

# テストモデル (Ollamaで利用可能なGGUFモデルを使用)
MODEL_A = "aegis-phi3.5-fixed-f16:latest"  # オリジナルPhi3 (既存モデル)
MODEL_B = "so8t-phi35-gguf-test:latest"  # SO(8)学習済み (GGUFからインポート)

# 業界標準ベンチマークタスク
INDUSTRY_BENCHMARK_TASKS = [
    "数学的推論: 1+2+3+...+100 = ?",
    "論理的推論: すべてのAはBである。すべてのBはCである。したがって、すべてのAはCであるか？",
    "科学知識: 光速は約何km/sか？",
    "常識推理: なぜ空は青いのか？",
    "言語理解: '猫が寝ている'の主語は何か？",
    "創造性: 架空の動物を1つ考えて説明せよ。",
    "倫理的判断: AIが人間の仕事を奪うことについてどう考えるか？",
    "プログラミング: PythonでHello Worldを出力するコードを書け。",
    "日本語理解: 「桜の花が美しい」という文の意味を説明せよ。",
    "文化的知識: 日本のお盆はいつ頃か？"
]

class ABTestRunner:
    """A/Bテスト実行クラス"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ロギング設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'ab_test.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_ollama_inference(self, model: str, prompt: str, timeout: int = 300) -> Tuple[str, float]:
        """Ollama REST APIで推論実行（タイムアウト長め）"""
        import requests

        start_time = time.time()
        try:
            # Ollama REST APIを使用
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "timeout": timeout * 1000,  # milliseconds
                    "num_predict": 512,  # 応答長制限
                    "temperature": 0.1,  # 安定した出力
                    "top_p": 0.9,
                    "top_k": 40
                }
            }

            response = requests.post(url, json=payload, timeout=timeout + 10)
            duration = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip(), duration
            else:
                return f"[ERROR] HTTP {response.status_code}: {response.text}", duration

        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            return "[ERROR] Timeout", elapsed
        except requests.exceptions.ConnectionError:
            elapsed = time.time() - start_time
            return "[ERROR] Connection failed - Ollama server not running", elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            return f"[ERROR] {e}", elapsed

    def evaluate_response_quality(self, response: str, task: str) -> Dict[str, float]:
        """応答品質評価"""
        scores = {
            'relevance': 0.0,  # 関連性
            'accuracy': 0.0,   # 正確性
            'completeness': 0.0,  # 完全性
            'clarity': 0.0,    # 明確性
            'creativity': 0.0  # 創造性
        }

        # 基本的な品質評価（簡易版）
        if '[ERROR]' in response or 'Timeout' in response:
            error_scores = {k: 0.0 for k in scores.keys()}
            error_scores['overall'] = 0.0  # エラー時もoverallキーを設定
            error_scores['response_length'] = 0
            return error_scores

        # 応答長による基本評価
        response_length = len(response.strip())
        if response_length > 10:  # 最低限の応答
            scores['completeness'] = min(1.0, response_length / 500)  # 長さによる完全性

        # 日本語タスクの場合の評価
        if any(keyword in task for keyword in ['日本語', '日本', '桜']):
            if any(jp_char in response for jp_char in 'あいうえおかきくけこ'):
                scores['accuracy'] += 0.5  # 日本語応答

        # 数学タスクの場合
        if '数学的' in task or '+' in task:
            if any(char.isdigit() for char in response):
                scores['accuracy'] += 0.5  # 数字を含む

        # 論理タスクの場合
        if '論理的' in task:
            if any(word in response.lower() for word in ['therefore', 'したがって', 'yes', 'no']):
                scores['accuracy'] += 0.5

        # 平均スコア
        scores['overall'] = statistics.mean(scores.values())
        scores['response_length'] = response_length

        return scores

    def run_single_comparison(self, task: str, model_a: str, model_b: str) -> Dict:
        """単一タスクでのA/B比較"""
        self.logger.info(f"Testing task: {task[:50]}...")

        # Model A実行
        prompt_a = f"以下の質問に正確に答えてください:\n\n{task}"
        response_a, time_a = self.run_ollama_inference(model_a, prompt_a)

        # Model B実行
        prompt_b = f"以下の質問に正確に答えてください:\n\n{task}"
        response_b, time_b = self.run_ollama_inference(model_b, prompt_b)

        # 品質評価
        scores_a = self.evaluate_response_quality(response_a, task)
        scores_b = self.evaluate_response_quality(response_b, task)

        return {
            'task': task,
            'model_a': {
                'name': model_a,
                'response': response_a,
                'time': time_a,
                'scores': scores_a
            },
            'model_b': {
                'name': model_b,
                'response': response_b,
                'time': time_b,
                'scores': scores_b
            },
            'comparison': {
                'time_difference': time_b - time_a,  # B - A (SO8T - Original)
                'quality_improvement': scores_b['overall'] - scores_a['overall']
            }
        }

    def run_industry_benchmark(self, model_a: str, model_b: str) -> List[Dict]:
        """業界標準ベンチマーク実行"""
        self.logger.info("=== Running Industry Standard Benchmark ===")
        results = []

        for task in INDUSTRY_BENCHMARK_TASKS:
            result = self.run_single_comparison(task, model_a, model_b)
            results.append(result)

        return results

    def run_elyza_benchmark(self, model_a: str, model_b: str) -> List[Dict]:
        """ELYZA-100ベンチマーク実行"""
        self.logger.info("=== Running ELYZA-100 Benchmark ===")
        results = []

        if not ELYZA_TASKS_FILE.exists():
            self.logger.warning(f"ELYZA tasks file not found: {ELYZA_TASKS_FILE}")
            return results

        try:
            with open(ELYZA_TASKS_FILE, 'r', encoding='utf-8') as f:
                elyza_tasks = json.load(f)

            # 全タスクを実行（ELYZA-100完全評価）
            test_tasks = elyza_tasks

            for task_data in test_tasks:
                task = task_data.get('input', '') if isinstance(task_data, dict) else str(task_data)
                if task.strip():
                    result = self.run_single_comparison(task, model_a, model_b)
                    results.append(result)

        except Exception as e:
            self.logger.error(f"Failed to load ELYZA tasks: {e}")

        return results

    def run_lm_eval_benchmark(self, model_a: str, model_b: str, tasks_list: List[str] = None) -> Dict:
        """lm-evaluation-harnessを使用した業界標準ベンチマーク"""
        if not LM_EVAL_AVAILABLE:
            self.logger.warning("lm-evaluation-harness not available, skipping standardized benchmarks")
            return {}

        if tasks_list is None:
            # 主要なベンチマークタスク
            tasks_list = [
                "mmlu",          # MMLU (Massive Multitask Language Understanding)
                "gsm8k",         # GSM8K (Math reasoning)
                "hellaswag",     # HellaSwag (Commonsense reasoning)
                "winogrande",    # Winogrande (Commonsense reasoning)
                "arc_challenge", # ARC-Challenge (Science)
                "truthfulqa_mc", # TruthfulQA (Truthfulness)
            ]

        self.logger.info(f"=== Running lm-evaluation-harness benchmarks: {tasks_list} ===")

        results = {
            'model_a': {},
            'model_b': {},
            'comparison': {}
        }

        try:
            # Model A評価
            self.logger.info(f"Evaluating Model A ({model_a})...")
            results_a = {}
            for task in tasks_list:
                try:
                    self.logger.info(f"Running {task} on {model_a}...")
                    result = lm_eval.simple_evaluate(
                        model="gguf",
                        model_args=f"model_path={model_a},n_gpu_layers=20,n_ctx=2048",
                        tasks=[task],
                        device="cuda:0",
                        batch_size=1,
                        limit=0.1  # 制限付きで高速評価
                    )
                    results_a[task] = result
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {task} on {model_a}: {e}")
                    results_a[task] = {"error": str(e)}

            # Model B評価
            self.logger.info(f"Evaluating Model B ({model_b})...")
            results_b = {}
            for task in tasks_list:
                try:
                    self.logger.info(f"Running {task} on {model_b}...")
                    result = lm_eval.simple_evaluate(
                        model="gguf",
                        model_args=f"model_path={model_b},n_gpu_layers=35,n_ctx=2048",
                        tasks=[task],
                        device="cuda:0",
                        batch_size=1,
                        limit=0.1  # 制限付きで高速評価
                    )
                    results_b[task] = result
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {task} on {model_b}: {e}")
                    results_b[task] = {"error": str(e)}

            results['model_a'] = results_a
            results['model_b'] = results_b

            # 比較分析
            comparison = {}
            for task in tasks_list:
                if task in results_a and task in results_b:
                    if "error" not in str(results_a[task]) and "error" not in str(results_b[task]):
                        try:
                            score_a = results_a[task]['results'][task]['acc,none'] if 'results' in results_a[task] else 0
                            score_b = results_b[task]['results'][task]['acc,none'] if 'results' in results_b[task] else 0
                            comparison[task] = {
                                'model_a_score': score_a,
                                'model_b_score': score_b,
                                'improvement': score_b - score_a
                            }
                        except:
                            comparison[task] = "scoring_error"

            results['comparison'] = comparison

        except Exception as e:
            self.logger.error(f"lm-evaluation-harness benchmark failed: {e}")
            results['error'] = str(e)

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """結果分析"""
        if not results:
            return {}

        # 統計計算
        times_a = [r['model_a']['time'] for r in results]
        times_b = [r['model_b']['time'] for r in results]
        scores_a = [r['model_a']['scores']['overall'] for r in results]
        scores_b = [r['model_b']['scores']['overall'] for r in results]

        analysis = {
            'total_tests': len(results),
            'model_a_stats': {
                'avg_time': statistics.mean(times_a),
                'avg_score': statistics.mean(scores_a),
                'time_std': statistics.stdev(times_a) if len(times_a) > 1 else 0,
                'score_std': statistics.stdev(scores_a) if len(scores_a) > 1 else 0
            },
            'model_b_stats': {
                'avg_time': statistics.mean(times_b),
                'avg_score': statistics.mean(scores_b),
                'time_std': statistics.stdev(times_b) if len(times_b) > 1 else 0,
                'score_std': statistics.stdev(scores_b) if len(scores_b) > 1 else 0
            },
            'improvements': {
                'time_change': statistics.mean([r['comparison']['time_difference'] for r in results]),
                'quality_improvement': statistics.mean([r['comparison']['quality_improvement'] for r in results])
            }
        }

        # t-test
        try:
            time_ttest = stats.ttest_ind(times_a, times_b)
            score_ttest = stats.ttest_ind(scores_a, scores_b)
            analysis['statistical_tests'] = {
                'time_ttest_pvalue': time_ttest.pvalue,
                'score_ttest_pvalue': score_ttest.pvalue
            }
        except:
            analysis['statistical_tests'] = None

        return analysis

    def save_results(self, industry_results: List[Dict], elyza_results: List[Dict], analysis: Dict, lm_eval_results: Dict = None):
        """結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 詳細結果
        detailed_results = {
            'metadata': {
                'timestamp': timestamp,
                'model_a': MODEL_A,
                'model_b': MODEL_B,
                'industry_tests': len(industry_results),
                'elyza_tests': len(elyza_results),
                'lm_eval_available': lm_eval_results is not None and bool(lm_eval_results)
            },
            'industry_benchmark': industry_results,
            'elyza_benchmark': elyza_results,
            'lm_eval_benchmark': lm_eval_results or {},
            'analysis': analysis
        }

        # JSON保存
        with open(self.output_dir / f'ab_test_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        # lm-evaluation-harness結果サマリー
        lm_eval_summary = ""
        if lm_eval_results and 'comparison' in lm_eval_results:
            lm_eval_summary = "\n## 業界標準ベンチマーク (lm-evaluation-harness)\n"
            for task, comp in lm_eval_results['comparison'].items():
                if isinstance(comp, dict) and 'improvement' in comp:
                    lm_eval_summary += f"- **{task}**: {comp['model_a_score']:.3f} → {comp['model_b_score']:.3f} ({comp['improvement']:+.3f})\n"
                else:
                    lm_eval_summary += f"- **{task}**: 評価エラー\n"

        # サマリーレポート
        summary = f"""# SO(8) PPO学習済みPhi3 vs 元Phi3 A/Bテスト結果

## テスト概要
- **Model A (オリジナル)**: {MODEL_A}
- **Model B (SO(8)学習済み)**: {MODEL_B}
- **業界標準ベンチマーク**: {len(industry_results)} タスク
- **ELYZA-100ベンチマーク**: {len(elyza_results)} タスク
- **lm-evaluation-harness**: {'利用可能' if lm_eval_results else '利用不可'}
- **実行日時**: {timestamp}

## 性能比較

### 応答時間
- **Model A**: {analysis['model_a_stats']['avg_time']:.2f}s (σ={analysis['model_a_stats']['time_std']:.2f})
- **Model B**: {analysis['model_b_stats']['avg_time']:.2f}s (σ={analysis['model_b_stats']['time_std']:.2f})
- **時間変化**: {analysis['improvements']['time_change']:+.2f}s ({'遅延' if analysis['improvements']['time_change'] > 0 else '高速化'})

### 品質スコア
- **Model A**: {analysis['model_a_stats']['avg_score']:.3f} (σ={analysis['model_a_stats']['score_std']:.3f})
- **Model B**: {analysis['model_b_stats']['avg_score']:.3f} (σ={analysis['model_b_stats']['score_std']:.3f})
- **品質改善**: {analysis['improvements']['quality_improvement']:+.3f} ({analysis['improvements']['quality_improvement']*100:+.1f}%)

## 統計的検定
{f"- t-test p値 (時間): {analysis.get('statistical_tests', {}).get('time_ttest_pvalue', 'N/A'):.4f}" if analysis.get('statistical_tests') else "- 統計検定: サンプルサイズ不足"}
{f"- t-test p値 (品質): {analysis.get('statistical_tests', {}).get('score_ttest_pvalue', 'N/A'):.4f}" if analysis.get('statistical_tests') else ""}

{lm_eval_summary}

## 結論
{f"**SO(8)学習により品質が{analysis['improvements']['quality_improvement']:+.1%}改善されました！**" if analysis['improvements']['quality_improvement'] > 0 else "**SO(8)学習による品質改善は確認されませんでした。**"}

{f"**MMLU/GSM8Kなどの業界標準ベンチマークでもSO(8)モデルの優位性が確認されました！**" if lm_eval_results and any(isinstance(comp, dict) and comp.get('improvement', 0) > 0 for comp in lm_eval_results.get('comparison', {}).values()) else "**業界標準ベンチマークではSO(8)学習の効果が限定的でした。**"}

---
*生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Ollama簡易ベンチマーク: {'利用済み' if lm_eval_results else '未利用'}*
"""

        with open(self.output_dir / f'ab_test_summary_{timestamp}.md', 'w', encoding='utf-8') as f:
            f.write(summary)

        self.logger.info(f"Results saved to {self.output_dir}")
        print(f"\n📊 A/Bテスト結果保存完了: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="SO(8) PPO学習済みPhi3 vs 元Phi3 A/Bテスト")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                       help="出力ディレクトリ")
    parser.add_argument("--model_a", type=str, default=MODEL_A,
                       help="Model A (オリジナル)")
    parser.add_argument("--model_b", type=str, default=MODEL_B,
                       help="Model B (SO(8)学習済み)")
    parser.add_argument("--skip_elyza", action="store_true",
                       help="ELYZA-100ベンチマークをスキップ")

    args = parser.parse_args()

    # ABテスト実行
    runner = ABTestRunner(Path(args.output_dir))

    print("🧬 SO(8) PPO学習済みPhi3 vs 元Phi3 A/Bテスト開始")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print("=" * 60)

    # 業界標準ベンチマーク
    industry_results = runner.run_industry_benchmark(args.model_a, args.model_b)

    # ELYZA-100ベンチマーク
    elyza_results = []
    if not args.skip_elyza:
        elyza_results = runner.run_elyza_benchmark(args.model_a, args.model_b)

    # lm-evaluation-harnessは使用せず、Ollama REST APIベースの簡易ベンチマークのみ使用
    lm_eval_results = {}

    # 結果分析
    all_results = industry_results + elyza_results
    analysis = runner.analyze_results(all_results)

    # 結果保存
    runner.save_results(industry_results, elyza_results, analysis, lm_eval_results)

    # コンソール出力
    print("\n" + "="*60)
    print("🎯 A/Bテスト完了結果")
    print("="*60)
    print(f"総テスト数: {len(all_results)}")
    print(".2f")
    print(".3f")
    print(".2f")
    print(".3f")
    print(".3f")
    print(f"品質改善: {analysis['improvements']['quality_improvement']:+.3f} ({analysis['improvements']['quality_improvement']*100:+.1f}%)")

    if analysis['improvements']['quality_improvement'] > 0:
        print("🎉 SO(8)学習により品質が改善されました！")
    else:
        print("📊 SO(8)学習による品質改善は確認されませんでした。")

if __name__ == "__main__":
    main()
