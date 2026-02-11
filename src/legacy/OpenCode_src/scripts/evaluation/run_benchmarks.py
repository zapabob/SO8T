#!/usr/bin/env python3
"""
RTX 3060 Optimized Benchmark Evaluation
ベンチマーク評価スクリプト
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTX3060BenchmarkEvaluator:
    def __init__(self, config_path=None):
        self.project_root = Path(__file__).parent.parent.parent

        # 設定ファイル読み込み
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.project_root / "config" / "benchmark.json"

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # モデルディレクトリ
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results" / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # RTX 3060最適化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_8bit = True

        logger.info(f"[INIT] RTX 3060 Benchmark Evaluator initialized")
        logger.info(f"[DEVICE] Using device: {self.device}")

    def load_model_and_tokenizer(self, model_path=None):
        """モデルとTokenizer読み込み"""
        if model_path is None:
            # デフォルトでトレーニング済みモデル
            model_path = self.models_dir / "quadrality_model_rtx3060"

        if not model_path.exists():
            logger.warning(f"[MODEL] Model not found at {model_path}, using lightweight test model")
            # テスト用に小規模モデルを使用
            base_model = self.config.get('abc_testing', {}).get('models', {}).get('A', 'microsoft/DialoGPT-small')
            model_path = base_model

        logger.info(f"[MODEL] Loading model: {model_path}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # RTX 3060最適化: 8-bit quantization
        if self.use_8bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            if torch.cuda.is_available():
                model = model.to(self.device)

        # LoRAアダプター読み込み（存在する場合）
        adapter_path = self.models_dir / "quadrality_model_rtx3060"
        if adapter_path.exists() and str(model_path) != str(adapter_path):
            try:
                model = PeftModel.from_pretrained(model, str(adapter_path))
                logger.info("[MODEL] LoRA adapters loaded")
            except:
                logger.warning("[MODEL] Could not load LoRA adapters")

        model.eval()
        self.model = model

        logger.info(f"[MODEL] Model loaded with {self.model.num_parameters():,} parameters")
        return model

    def evaluate_gsm8k(self, num_samples=None):
        """GSM8K評価"""
        logger.info("[BENCHMARK] Evaluating GSM8K...")

        try:
            dataset = load_dataset('gsm8k', 'main', split='test')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="GSM8K"):
                question = item['question']
                ground_truth = self._extract_answer(item['answer'])

                predicted_answer = self._solve_math_problem(question)

                if self._compare_answers(predicted_answer, ground_truth):
                    correct += 1

            accuracy = correct / total
            logger.info(f"[GSM8K] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }

        except Exception as e:
            logger.error(f"[GSM8K] Evaluation failed: {e}")
            return None

    def evaluate_math(self, num_samples=None):
        """MATHデータセット評価"""
        logger.info("[BENCHMARK] Evaluating MATH...")

        try:
            dataset = load_dataset('hendrycks/math', split='test')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="MATH"):
                problem = item['problem']
                ground_truth = item['solution']

                predicted_answer = self._solve_math_problem(problem)

                if self._compare_math_answers(predicted_answer, ground_truth):
                    correct += 1

            accuracy = correct / total
            logger.info(f"[MATH] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }

        except Exception as e:
            logger.error(f"[MATH] Evaluation failed: {e}")
            return None

    def evaluate_arc_easy(self, num_samples=None):
        """ARC-Easy評価"""
        logger.info("[BENCHMARK] Evaluating ARC-Easy...")

        try:
            dataset = load_dataset('ai2_arc', 'ARC-Easy', split='test')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="ARC-Easy"):
                question = item['question']
                choices = [item[f'choice_{label}'] for label in ['A', 'B', 'C', 'D'] if f'choice_{label}' in item]
                ground_truth = item['answerKey']

                predicted_answer = self._answer_multiple_choice(question, choices)

                if predicted_answer == ground_truth:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[ARC-Easy] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }

        except Exception as e:
            logger.error(f"[ARC-Easy] Evaluation failed: {e}")
            return None

    def evaluate_hellaswag(self, num_samples=None):
        """HellaSwag評価"""
        logger.info("[BENCHMARK] Evaluating HellaSwag...")

        try:
            dataset = load_dataset('hellaswag', split='validation')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="HellaSwag"):
                context = item['ctx']
                endings = item['endings']
                ground_truth = item['label']

                predicted_answer = self._complete_context(context, endings)

                if predicted_answer == ground_truth:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[HellaSwag] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }

        except Exception as e:
            logger.error(f"[HellaSwag] Evaluation failed: {e}")
            return None

    def _solve_math_problem(self, problem):
        """数学問題を解く（簡易実装）"""
        try:
            # プロンプト作成
            prompt = f"Solve this math problem step by step:\\n\\n{problem}\\n\\nFinal Answer:"

            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # 回答抽出
            answer = self._extract_answer_from_response(response)
            return answer

        except Exception as e:
            logger.error(f"[MATH] Problem solving failed: {e}")
            return "0"

    def _answer_multiple_choice(self, question, choices):
        """4択問題に答える"""
        try:
            # プロンプト作成
            choices_text = "\\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            prompt = f"Question: {question}\\n\\nChoices:\\n{choices_text}\\n\\nAnswer with the letter only:"

            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # 最初の文字を取得
            for char in response.upper():
                if char in ['A', 'B', 'C', 'D']:
                    return char

            return 'A'  # デフォルト

        except Exception as e:
            logger.error(f"[MC] Multiple choice failed: {e}")
            return 'A'

    def _complete_context(self, context, endings):
        """文脈補完"""
        try:
            # 各endingの確率を計算
            scores = []

            for i, ending in enumerate(endings):
                text = context + " " + ending

                inputs = self.tokenizer(text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss.item()
                    scores.append((i, loss))

            # 最も低いlossのendingを選択
            best_ending = min(scores, key=lambda x: x[1])[0]
            return best_ending

        except Exception as e:
            logger.error(f"[HellaSwag] Context completion failed: {e}")
            return 0

    def _extract_answer(self, text):
        """テキストから回答を抽出"""
        # 簡易実装
        import re
        numbers = re.findall(r'\d+', text)
        return numbers[-1] if numbers else "0"

    def _extract_answer_from_response(self, response):
        """モデル応答から回答を抽出"""
        # 簡易実装
        import re
        numbers = re.findall(r'\d+', response)
        return numbers[-1] if numbers else "0"

    def _compare_answers(self, predicted, ground_truth):
        """回答比較"""
        return str(predicted).strip() == str(ground_truth).strip()

    def _compare_math_answers(self, predicted, ground_truth):
        """数学回答比較（より柔軟）"""
        # 簡易正規化
        pred = str(predicted).strip().lower()
        gt = str(ground_truth).strip().lower()

        # 基本的な数値比較
        try:
            pred_num = float(pred)
            gt_num = float(gt)
            return abs(pred_num - gt_num) < 1e-6
        except:
            return pred == gt

    def run_benchmark_suite(self, model_path=None, num_samples=None):
        """ベンチマークスイート実行"""
        logger.info("[BENCHMARK] Starting benchmark evaluation...")
        logger.info("=" * 60)

        # モデル読み込み
        self.load_model_and_tokenizer(model_path)

        # ベンチマーク設定
        benchmark_config = self.config
        primary_benchmarks = benchmark_config.get('primary_benchmarks', [])
        secondary_benchmarks = benchmark_config.get('secondary_benchmarks', [])
        japanese_benchmarks = benchmark_config.get('japanese_benchmarks', [])
        advanced_benchmarks = benchmark_config.get('advanced_benchmarks', [])

        results = {}

        # プライマリベンチマーク
        for benchmark in primary_benchmarks:
            logger.info(f"[BENCHMARK] Running {benchmark}...")
            result = self._run_single_benchmark(benchmark, num_samples)
            if result:
                results[benchmark] = result

        # 日本語ベンチマーク
        for benchmark in japanese_benchmarks:
            if benchmark not in results:  # プライマリに含まれていない場合のみ
                logger.info(f"[BENCHMARK] Running Japanese {benchmark}...")
                result = self._run_single_benchmark(benchmark, num_samples)
                if result:
                    results[benchmark] = result

        # セカンダリベンチマーク（業界標準）
        for benchmark in secondary_benchmarks[:4]:  # 最初の4つ
            logger.info(f"[BENCHMARK] Running Industry Standard {benchmark}...")
            result = self._run_single_benchmark(benchmark, num_samples)
            if result:
                results[benchmark] = result

        # アドバンストベンチマーク（高度なもの）
        for benchmark in advanced_benchmarks[:2]:  # 最初の2つ
            logger.info(f"[BENCHMARK] Running Advanced {benchmark}...")
            result = self._run_single_benchmark(benchmark, num_samples)
            if result:
                results[benchmark] = result

        # 結果保存
        self.save_results(results)

        # サマリー表示
        self.print_results_summary(results)

        logger.info("=" * 60)
        logger.info("[SUCCESS] Benchmark evaluation completed!")
        logger.info("=" * 60)

        return results

    def _run_single_benchmark(self, benchmark_name, num_samples):
        """単一ベンチマーク実行"""
        benchmark_methods = {
            # プライマリ
            'gsm8k': self.evaluate_gsm8k,
            'math': self.evaluate_math,
            'arc_easy': self.evaluate_arc_easy,
            'hellaswag': self.evaluate_hellaswag,

            # 日本語
            'elyza_tasks_100': self.evaluate_elyza_tasks_100,
            'jsquad': self.evaluate_jsquad,
            'xwinograd_ja': self.evaluate_xwinograd_ja,

            # 業界標準
            'mmlu': self.evaluate_mmlu,
            'bbh': self.evaluate_bbh,
            'commonsenseqa': self.evaluate_commonsenseqa,
            'openbookqa': self.evaluate_openbookqa,
            'socialiqa': self.evaluate_socialiqa,
            'piqa': self.evaluate_piqa,
            'winogrande': self.evaluate_winogrande,
            'boolq': self.evaluate_boolq,

            # アドバンスト
            'drop': self.evaluate_drop,
            'strategyqa': self.evaluate_strategyqa
        }

        if benchmark_name in benchmark_methods:
            return benchmark_methods[benchmark_name](num_samples)
        else:
            logger.warning(f"[BENCHMARK] Unknown benchmark: {benchmark_name}")
            return None

    def save_results(self, results):
        """結果を保存"""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"benchmark_results_{timestamp}.json"

        result_data = {
            'timestamp': timestamp,
            'model_info': {
                'parameters': self.model.num_parameters() if hasattr(self.model, 'num_parameters') else 'unknown',
                'quantization': '8bit' if self.use_8bit else 'fp16'
            },
            'results': results
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[SAVE] Results saved to {result_file}")

    def print_results_summary(self, results):
        """結果サマリー表示"""
        logger.info("\\n[BENCHMARK SUMMARY]")
        logger.info("-" * 40)

        for benchmark, result in results.items():
            if result and 'accuracy' in result:
                accuracy = result['accuracy']
                correct = result['correct']
                total = result['total']
                logger.info("25")

        # 平均精度
        accuracies = [r['accuracy'] for r in results.values() if r and 'accuracy' in r]
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            logger.info(".3f")

    # === 日本語ベンチマーク ===
    def evaluate_elyza_tasks_100(self, num_samples=None):
        """ELYZA Tasks 100評価（強化版 + 四重推論統合）"""
        logger.info("[BENCHMARK] Evaluating ELYZA Tasks 100 with Quadrality Reasoning...")

        try:
            # ELYZA Tasks 100データセット読み込み
            dataset = load_dataset('elyza/ELYZA-tasks-100', split='test')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            # タスクごとの評価結果
            task_results = {}

            for item in tqdm(dataset, desc="ELYZA Tasks 100"):
                input_text = item['input']
                expected_output = item['output']

                # タスクタイプの特定
                task_type = self._identify_elyza_task_type(input_text)

                # 四重推論による回答生成（ALLOWESCALETONDENYREFUSE適用）
                predicted_output, decision_process = self._generate_quadrality_response(
                    input_text, task_type
                )

                # 評価（タスクタイプに応じた評価方法）
                is_correct = self._evaluate_japanese_task_enhanced(predicted_output, expected_output, task_type)

                if is_correct:
                    correct += 1

                # タスクごとの統計
                if task_type not in task_results:
                    task_results[task_type] = {'correct': 0, 'total': 0}
                task_results[task_type]['total'] += 1
                if is_correct:
                    task_results[task_type]['correct'] += 1

            accuracy = correct / total
            logger.info(f"[ELYZA] Overall Accuracy: {accuracy:.3f} ({correct}/{total})")
            logger.info("[ELYZA] Quadrality decision process integrated")

            # タスクごとの詳細結果
            for task_type, results in task_results.items():
                task_accuracy = results['correct'] / results['total']
                logger.info(f"[ELYZA] {task_type}: {task_accuracy:.3f} ({results['correct']}/{results['total']})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'task_results': task_results,
                'quadrality_enabled': True,
                'japanese_benchmark': True,
                'benchmark_name': 'ELYZA Tasks 100 with Quadrality'
            }

        except Exception as e:
            logger.error(f"[ELYZA] Evaluation failed: {e}")
            return None

    def evaluate_jsquad(self, num_samples=None):
        """JSQuAD評価"""
        logger.info("[BENCHMARK] Evaluating JSQuAD...")

        try:
            dataset = load_dataset('SkelterLabsInc/JGLUE', 'JSQuAD', split='validation')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="JSQuAD"):
                context = item['context']
                question = item['question']
                answers = item['answers']['text']

                predicted_answer = self._answer_japanese_qa(context, question)
                is_correct = self._compare_japanese_answers(predicted_answer, answers)

                if is_correct:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[JSQuAD] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'japanese_benchmark': True
            }

        except Exception as e:
            logger.error(f"[JSQuAD] Evaluation failed: {e}")
            return None

    def evaluate_xwinograd_ja(self, num_samples=None):
        """XWinograd日本語評価"""
        logger.info("[BENCHMARK] Evaluating XWinograd Japanese...")

        try:
            dataset = load_dataset('mu-nlpc/xwinograd_ja', split='test')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="XWinograd JA"):
                sentence = item['sentence']
                option1 = item['option1']
                option2 = item['option2']
                answer = item['answer']

                predicted_answer = self._solve_japanese_winograd(sentence, option1, option2)
                is_correct = predicted_answer == answer

                if is_correct:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[XWinograd JA] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'japanese_benchmark': True
            }

        except Exception as e:
            logger.error(f"[XWinograd JA] Evaluation failed: {e}")
            return None

    # === 業界標準ベンチマーク ===
    def evaluate_mmlu(self, num_samples=None):
        """MMLU (Massive Multitask Language Understanding) 評価"""
        logger.info("[BENCHMARK] Evaluating MMLU...")

        try:
            # STEM科目に焦点
            subjects = ['abstract_algebra', 'astronomy', 'college_biology',
                       'college_chemistry', 'college_computer_science', 'college_mathematics',
                       'college_physics', 'electrical_engineering', 'machine_learning']

            total_correct = 0
            total_questions = 0

            for subject in subjects[:3]:  # 最初の3科目のみ（メモリ節約）
                try:
                    dataset = load_dataset('cais/mmlu', subject, split='test')
                    if num_samples:
                        dataset = dataset.select(range(min(num_samples // 3, len(dataset))))

                    for item in dataset:
                        question = item['question']
                        choices = [item[f'choices'][i] for i in range(4)]
                        correct_answer = item['answer']

                        predicted_answer = self._answer_multiple_choice(question, choices)
                        if predicted_answer == ['A', 'B', 'C', 'D'][correct_answer]:
                            total_correct += 1
                        total_questions += 1

                except Exception as e:
                    logger.warning(f"[MMLU] Subject {subject} failed: {e}")
                    continue

            accuracy = total_correct / total_questions if total_questions > 0 else 0
            logger.info(f"[MMLU] Accuracy: {accuracy:.3f} ({total_correct}/{total_questions})")

            return {
                'accuracy': accuracy,
                'correct': total_correct,
                'total': total_questions,
                'industry_standard': True
            }

        except Exception as e:
            logger.error(f"[MMLU] Evaluation failed: {e}")
            return None

    def evaluate_bbh(self, num_samples=None):
        """BIG-Bench Hard (BBH) 評価"""
        logger.info("[BENCHMARK] Evaluating BBH...")

        try:
            # BBHのタスクリスト（難易度の高いもの）
            bbh_tasks = ['boolean_expressions', 'causal_judgement', 'date_understanding',
                        'disambiguation_qa', 'formal_fallacies', 'geometric_shapes',
                        'hyperbaton', 'logical_deduction', 'movie_recommendation',
                        'penguins_in_a_table', 'reasoning_about_colored_objects',
                        'ruin_names', 'salient_translation_error_detection',
                        'snarks', 'sports_understanding', 'temporal_sequences',
                        'tracking_shuffled_objects', 'web_of_lies']

            total_correct = 0
            total_questions = 0

            for task in bbh_tasks[:3]:  # 最初の3タスクのみ
                try:
                    dataset = load_dataset('lukaemon/bbh', task, split='test')
                    if num_samples:
                        dataset = dataset.select(range(min(num_samples // 3, len(dataset))))

                    for item in dataset:
                        input_text = item['input']
                        target = item['target']

                        predicted_answer = self._generate_response(input_text, max_tokens=128)
                        is_correct = self._compare_bbh_answers(predicted_answer, target)

                        if is_correct:
                            total_correct += 1
                        total_questions += 1

                except Exception as e:
                    logger.warning(f"[BBH] Task {task} failed: {e}")
                    continue

            accuracy = total_correct / total_questions if total_questions > 0 else 0
            logger.info(f"[BBH] Accuracy: {accuracy:.3f} ({total_correct}/{total_questions})")

            return {
                'accuracy': accuracy,
                'correct': total_correct,
                'total': total_questions,
                'advanced_benchmark': True
            }

        except Exception as e:
            logger.error(f"[BBH] Evaluation failed: {e}")
            return None

    def evaluate_commonsenseqa(self, num_samples=None):
        """CommonsenseQA評価"""
        logger.info("[BENCHMARK] Evaluating CommonsenseQA...")

        try:
            dataset = load_dataset('tau/commonsense_qa', split='validation')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="CommonsenseQA"):
                question = item['question']
                choices = [item['choices'][i]['text'] for i in range(5)]
                answer_key = item['answerKey']

                predicted_answer = self._answer_multiple_choice(question, choices)
                is_correct = predicted_answer == answer_key

                if is_correct:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[CommonsenseQA] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'industry_standard': True
            }

        except Exception as e:
            logger.error(f"[CommonsenseQA] Evaluation failed: {e}")
            return None

    def evaluate_openbookqa(self, num_samples=None):
        """OpenBookQA評価"""
        logger.info("[BENCHMARK] Evaluating OpenBookQA...")

        try:
            dataset = load_dataset('allenai/openbookqa', 'main', split='test')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="OpenBookQA"):
                question_stem = item['question_stem']
                choices = [choice['text'] for choice in item['choices']]
                answer_key = item['answerKey']

                predicted_answer = self._answer_multiple_choice(question_stem, choices)
                is_correct = predicted_answer == answer_key

                if is_correct:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[OpenBookQA] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'industry_standard': True
            }

        except Exception as e:
            logger.error(f"[OpenBookQA] Evaluation failed: {e}")
            return None

    def evaluate_socialiqa(self, num_samples=None):
        """SocialIQA評価"""
        logger.info("[BENCHMARK] Evaluating SocialIQA...")

        try:
            dataset = load_dataset('allenai/socialiqa', split='validation')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="SocialIQA"):
                context = item['context']
                question = item['question']
                choices = [item[f'answer{i}'] for i in ['A', 'B', 'C']]
                answer_key = item['label']  # 1, 2, 3

                full_question = f"{context}\\n{question}"
                predicted_answer = self._answer_multiple_choice(full_question, choices)

                # 予測された文字を数字に変換
                answer_map = {'A': '1', 'B': '2', 'C': '3'}
                predicted_key = answer_map.get(predicted_answer, '1')
                is_correct = predicted_key == str(answer_key)

                if is_correct:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[SocialIQA] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'industry_standard': True
            }

        except Exception as e:
            logger.error(f"[SocialIQA] Evaluation failed: {e}")
            return None

    def evaluate_piqa(self, num_samples=None):
        """PIQA評価"""
        logger.info("[BENCHMARK] Evaluating PIQA...")

        try:
            dataset = load_dataset('ybisk/piqa', split='validation')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="PIQA"):
                goal = item['goal']
                choices = [item['sol1'], item['sol2']]
                label = item['label']

                predicted_answer = self._answer_multiple_choice(goal, choices)
                predicted_idx = 0 if predicted_answer == 'A' else 1
                is_correct = predicted_idx == label

                if is_correct:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[PIQA] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'industry_standard': True
            }

        except Exception as e:
            logger.error(f"[PIQA] Evaluation failed: {e}")
            return None

    def evaluate_winogrande(self, num_samples=None):
        """Winogrande評価"""
        logger.info("[BENCHMARK] Evaluating Winogrande...")

        try:
            dataset = load_dataset('allenai/winogrande', 'winogrande_xl', split='test')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="Winogrande"):
                sentence = item['sentence']
                option1 = item['option1']
                option2 = item['option2']
                answer = item['answer']

                predicted_answer = self._solve_winogrande(sentence, option1, option2)
                is_correct = predicted_answer == str(answer)

                if is_correct:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[Winogrande] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'industry_standard': True
            }

        except Exception as e:
            logger.error(f"[Winogrande] Evaluation failed: {e}")
            return None

    def evaluate_boolq(self, num_samples=None):
        """BoolQ評価"""
        logger.info("[BENCHMARK] Evaluating BoolQ...")

        try:
            dataset = load_dataset('google/boolq', split='validation')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="BoolQ"):
                question = item['question']
                passage = item['passage']
                answer = item['answer']

                full_input = f"Passage: {passage}\\nQuestion: {question}\\nAnswer with Yes or No:"
                predicted_answer = self._generate_response(full_input, max_tokens=10)

                # Yes/No判定
                predicted_bool = self._extract_bool_answer(predicted_answer)
                is_correct = predicted_bool == answer

                if is_correct:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[BoolQ] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'industry_standard': True
            }

        except Exception as e:
            logger.error(f"[BoolQ] Evaluation failed: {e}")
            return None

    # === アドバンストベンチマーク ===
    def evaluate_drop(self, num_samples=None):
        """DROP (Discrete Reasoning Over Paragraphs) 評価"""
        logger.info("[BENCHMARK] Evaluating DROP...")

        try:
            dataset = load_dataset('ucinlp/drop', split='validation')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = 0

            for item in tqdm(dataset, desc="DROP"):
                if total >= (num_samples or 100):  # DROPは計算コストが高い
                    break

                context = item['passage']
                question = item['question']
                answers = item['answers_spans']

                if not answers['spans']:  # 回答がない場合スキップ
                    continue

                predicted_answer = self._answer_drop_question(context, question)
                is_correct = self._compare_drop_answers(predicted_answer, answers['spans'])

                if is_correct:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0
            logger.info(f"[DROP] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'advanced_benchmark': True
            }

        except Exception as e:
            logger.error(f"[DROP] Evaluation failed: {e}")
            return None

    def evaluate_strategyqa(self, num_samples=None):
        """StrategyQA評価"""
        logger.info("[BENCHMARK] Evaluating StrategyQA...")

        try:
            dataset = load_dataset('allenai/strategyqa', split='test')
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))

            correct = 0
            total = len(dataset)

            for item in tqdm(dataset, desc="StrategyQA"):
                question = item['question']
                answer = item['answer']

                predicted_answer = self._generate_response(
                    f"Question: {question}\\nAnswer with Yes or No:",
                    max_tokens=10
                )

                predicted_bool = self._extract_bool_answer(predicted_answer)
                is_correct = predicted_bool == answer

                if is_correct:
                    correct += 1

            accuracy = correct / total
            logger.info(f"[StrategyQA] Accuracy: {accuracy:.3f} ({correct}/{total})")

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'advanced_benchmark': True
            }

        except Exception as e:
            logger.error(f"[StrategyQA] Evaluation failed: {e}")
            return None

    # === ヘルパーメソッド ===
    def _generate_response(self, prompt, max_tokens=128):
        """モデル応答生成"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            logger.error(f"[GENERATE] Response generation failed: {e}")
            return ""

    def _evaluate_japanese_task(self, predicted, expected):
        """日本語タスク評価（簡易）"""
        # 簡易的な文字列マッチング
        pred_clean = predicted.strip().lower()
        exp_clean = expected.strip().lower()
        return pred_clean == exp_clean or exp_clean in pred_clean

    def _answer_japanese_qa(self, context, question):
        """日本語QA回答"""
        prompt = f"本文: {context}\\n質問: {question}\\n回答:"
        return self._generate_response(prompt, max_tokens=64)

    def _compare_japanese_answers(self, predicted, expected_list):
        """日本語回答比較"""
        for expected in expected_list:
            if expected.lower().strip() in predicted.lower().strip():
                return True
        return False

    def _solve_japanese_winograd(self, sentence, option1, option2):
        """日本語Winograd解決"""
        prompt = f"文: {sentence}\\n選択肢1: {option1}\\n選択肢2: {option2}\\nどちらが正しいですか？ 1または2で答えてください:"
        response = self._generate_response(prompt, max_tokens=10)
        return '1' if '1' in response else '2'

    def _compare_bbh_answers(self, predicted, target):
        """BBH回答比較"""
        # 正規化して比較
        pred_norm = predicted.strip().lower()
        target_norm = target.strip().lower()
        return pred_norm == target_norm or target_norm in pred_norm

    def _solve_winogrande(self, sentence, option1, option2):
        """Winogrande解決"""
        prompt = f"Sentence: {sentence}\\nOption 1: {option1}\\nOption 2: {option2}\\nWhich option fits better? Answer with 1 or 2:"
        response = self._generate_response(prompt, max_tokens=10)
        return '1' if '1' in response else '2'

    def _extract_bool_answer(self, response):
        """Bool回答抽出"""
        response_lower = response.lower().strip()
        if 'yes' in response_lower or 'はい' in response_lower or 'true' in response_lower:
            return True
        elif 'no' in response_lower or 'いいえ' in response_lower or 'false' in response_lower:
            return False
        else:
            # デフォルトはFalse
            return False

    def _answer_drop_question(self, context, question):
        """DROP質問回答"""
        prompt = f"Passage: {context}\\nQuestion: {question}\\nAnswer:"
        return self._generate_response(prompt, max_tokens=32)

    def _compare_drop_answers(self, predicted, expected_spans):
        """DROP回答比較"""
        for span in expected_spans:
            if span.lower().strip() in predicted.lower().strip():
                return True
        return False

    # === 強化された日本語評価メソッド ===
    def _identify_elyza_task_type(self, input_text):
        """ELYZAタスクの種類を特定"""
        input_lower = input_text.lower()

        if '要約' in input_text or 'summarize' in input_lower or 'summary' in input_lower:
            return 'summarization'
        elif '質問' in input_text or '答えて' in input_text or 'question' in input_lower:
            return 'question_answering'
        elif '翻訳' in input_text or 'translate' in input_lower:
            return 'translation'
        elif '生成' in input_text or '書いて' in input_text or 'generate' in input_lower:
            return 'generation'
        elif '分類' in input_text or 'classify' in input_lower:
            return 'classification'
        elif '選択' in input_text or '選んで' in input_text or 'choose' in input_lower:
            return 'multiple_choice'
        else:
            return 'general'

    def _evaluate_japanese_task_enhanced(self, predicted, expected, task_type):
        """強化された日本語タスク評価"""
        try:
            # 前処理
            pred_clean = predicted.strip()
            exp_clean = expected.strip()

            # タスクタイプに応じた評価
            if task_type == 'multiple_choice':
                # 選択問題の場合、最初の文字を比較
                pred_first = pred_clean[:1] if pred_clean else ""
                exp_first = exp_clean[:1] if exp_clean else ""
                return pred_first.upper() == exp_first.upper()

            elif task_type in ['summarization', 'generation']:
                # 生成タスクの場合、より柔軟な評価
                # キーワードマッチングと長さチェック
                pred_words = set(pred_clean.split())
                exp_words = set(exp_clean.split())

                # Jaccard類似度
                intersection = len(pred_words & exp_words)
                union = len(pred_words | exp_words)
                jaccard = intersection / union if union > 0 else 0

                # 長さ比チェック
                len_ratio = len(pred_clean) / len(exp_clean) if len(exp_clean) > 0 else 0
                len_ok = 0.5 <= len_ratio <= 2.0

                return jaccard > 0.3 and len_ok

            elif task_type == 'question_answering':
                # QAタスクの場合、主要なキーワードを含むかチェック
                key_words = [w for w in exp_clean.split() if len(w) > 1]
                matches = sum(1 for word in key_words if word in pred_clean)
                return matches >= len(key_words) * 0.6

            else:
                # 一般的なタスクの場合、柔軟な文字列マッチング
                return self._evaluate_japanese_task(pred_clean, exp_clean)

        except Exception as e:
            logger.error(f"[ELYZA] Enhanced evaluation failed: {e}")
            # フォールバック
            return self._evaluate_japanese_task(predicted, expected)

    def _generate_quadrality_response(self, input_text, task_type):
        """四重推論による応答生成（ALLOWESCALETONDENYREFUSE適用）"""
        try:
            # 複数の視点からの回答生成
            perspectives = ['algebraic', 'geometric', 'analytic', 'topological']

            candidate_responses = {}
            decision_evaluations = {}

            for perspective in perspectives:
                # 各視点からの回答生成
                perspective_prompt = f"From {perspective} perspective: {input_text}"
                max_tokens = 512 if 'generation' in task_type else 256

                response = self._generate_response(perspective_prompt, max_tokens=max_tokens)
                candidate_responses[perspective] = response

                # ALLOWESCALETONDENYREFUSE評価
                decision = self._evaluate_allow_escalate_deny_refuse(response, input_text, task_type)
                decision_evaluations[perspective] = decision

            # 内部比較による最終決定
            final_decision, reasoning = self._compare_internal_responses(
                candidate_responses, decision_evaluations, input_text
            )

            # 最終回答の選択
            if final_decision == 'ALLOW':
                # 最も一貫性のある回答を選択
                final_response = self._select_best_response(candidate_responses, decision_evaluations)
            elif final_decision == 'ESCALATE':
                final_response = "このクエリはより詳細な検討が必要です。専門的な助言を推奨します。"
            elif final_decision == 'DENY':
                final_response = "申し訳ありませんが、このクエリにはお答えできません。"
            elif final_decision == 'REFUSE':
                final_response = "このリクエストは現在のガイドラインに沿わないため、お断りいたします。"

            decision_process = {
                'perspectives_used': perspectives,
                'candidate_responses': candidate_responses,
                'decision_evaluations': decision_evaluations,
                'final_decision': final_decision,
                'internal_reasoning': reasoning
            }

            return final_response, decision_process

        except Exception as e:
            logger.error(f"[QUADRALITY] Response generation failed: {e}")
            # フォールバック
            return self._generate_response(input_text), {}

    def _evaluate_allow_escalate_deny_refuse(self, response, input_text, task_type):
        """ALLOWESCALETONDENYREFUSE評価"""
        try:
            # 回答の品質評価
            response_quality = self._assess_response_quality(response, input_text, task_type)

            # 安全性の評価
            safety_score = self._assess_response_safety(response, input_text)

            # 適切性の評価
            appropriateness_score = self._assess_response_appropriateness(response, input_text)

            # 決定ロジック
            if safety_score >= 0.8 and appropriateness_score >= 0.8 and response_quality >= 0.7:
                return 'ALLOW'  # 許可
            elif response_quality >= 0.5 and safety_score >= 0.6:
                return 'ESCALATE'  # 上位判断へ
            elif safety_score < 0.5 or appropriateness_score < 0.5:
                return 'DENY'  # 拒否
            else:
                return 'REFUSE'  # 断る

        except Exception as e:
            logger.error(f"[DECISION] ALLOWESCALETONDENYREFUSE evaluation failed: {e}")
            return 'ALLOW'  # デフォルト

    def _compare_internal_responses(self, candidate_responses, decision_evaluations, input_text):
        """内部応答比較"""
        try:
            # 各決定のカウント
            decision_counts = {}
            for decision in decision_evaluations.values():
                decision_counts[decision] = decision_counts.get(decision, 0) + 1

            # 多数決による決定
            majority_decision = max(decision_counts, key=decision_counts.get)

            # 一貫性評価
            consistency_score = decision_counts[majority_decision] / len(decision_evaluations)

            # 最終決定の理由
            if consistency_score >= 0.75:
                final_decision = majority_decision
                reasoning = f"High consistency ({consistency_score:.2f}) in {majority_decision} decision"
            elif decision_counts.get('ALLOW', 0) >= 2:
                final_decision = 'ALLOW'
                reasoning = "ALLOW decisions dominate despite some variations"
            else:
                final_decision = 'ESCALATE'
                reasoning = "Mixed decisions require escalation for safety"

            return final_decision, reasoning

        except Exception as e:
            logger.error(f"[COMPARISON] Internal response comparison failed: {e}")
            return 'ALLOW', "Fallback due to comparison error"

    def _select_best_response(self, candidate_responses, decision_evaluations):
        """最適な応答を選択"""
        try:
            # ALLOW評価の応答を優先
            allow_responses = [
                response for perspective, response in candidate_responses.items()
                if decision_evaluations.get(perspective) == 'ALLOW'
            ]

            if allow_responses:
                # 最も詳細な応答を選択
                return max(allow_responses, key=len)
            else:
                # 最も安全な応答を選択
                safe_responses = [
                    response for perspective, response in candidate_responses.items()
                    if decision_evaluations.get(perspective) in ['ALLOW', 'ESCALATE']
                ]
                return safe_responses[0] if safe_responses else list(candidate_responses.values())[0]

        except Exception as e:
            logger.error(f"[SELECTION] Response selection failed: {e}")
            return list(candidate_responses.values())[0]

    def _assess_response_quality(self, response, input_text, task_type):
        """応答品質評価"""
        try:
            # 長さチェック
            length_score = min(len(response) / 100, 1.0)  # 100文字以上で満点

            # 関連性チェック（簡易）
            input_words = set(input_text.lower().split())
            response_words = set(response.lower().split())
            relevance_score = len(input_words & response_words) / len(input_words) if input_words else 0.5

            # タスク適合性
            if task_type == 'question_answering' and '?' in response:
                task_score = 0.8
            elif task_type == 'generation' and len(response) > len(input_text):
                task_score = 0.9
            else:
                task_score = 0.6

            quality_score = (length_score + relevance_score + task_score) / 3
            return min(quality_score, 1.0)

        except:
            return 0.5

    def _assess_response_safety(self, response, input_text):
        """応答安全性評価"""
        try:
            # 不適切なキーワードチェック
            unsafe_keywords = ['暴力', '差別', '脅迫', '違法', '有害', '危険']
            response_lower = response.lower()

            unsafe_count = sum(1 for keyword in unsafe_keywords if keyword in response_lower)
            safety_score = max(0, 1.0 - (unsafe_count * 0.2))

            return safety_score

        except:
            return 0.8

    def _assess_response_appropriateness(self, response, input_text):
        """応答適切性評価"""
        try:
            # クエリと応答の適合性チェック
            input_length = len(input_text)
            response_length = len(response)

            # 応答がクエリに対して適切な長さか
            length_ratio = response_length / input_length if input_length > 0 else 1
            appropriateness_score = 1.0 if 0.5 <= length_ratio <= 3.0 else 0.6

            return appropriateness_score

        except:
            return 0.7

    def _evaluate_japanese_task(self, predicted, expected):
        """基本的な日本語タスク評価"""
        # 正規化
        pred_norm = predicted.strip().lower()
        exp_norm = expected.strip().lower()

        # 完全一致
        if pred_norm == exp_norm:
            return True

        # 主要部分の一致
        pred_words = set(pred_norm.split())
        exp_words = set(exp_norm.split())

        # 共通語彙の割合
        if exp_words:
            common_ratio = len(pred_words & exp_words) / len(exp_words)
            return common_ratio > 0.7

        return False

    # === ツールコール & API統合評価メソッド ===
    def _evaluate_tool_calling_task(self, predicted, expected):
        """ツールコールタスク評価"""
        try:
            # ツールコール構文の検知
            tool_call_patterns = [
                r'call_tool\(', r'execute_tool\(', r'use_tool\(',
                r'api_call\(', r'function_call\(', r'method_call\('
            ]

            pred_has_tool_call = any(pattern in predicted.lower() for pattern in tool_call_patterns)
            exp_has_tool_call = any(pattern in expected.lower() for pattern in tool_call_patterns)

            # 両方ともツールコールを含むか、両方とも含まない場合
            if pred_has_tool_call == exp_has_tool_call:
                return True

            # ツールコールパラメータの一致度
            if pred_has_tool_call and exp_has_tool_call:
                return self._compare_tool_parameters(predicted, expected)

            return False

        except Exception as e:
            logger.error(f"[TOOL] Tool calling evaluation failed: {e}")
            return False

    def _evaluate_function_calling_task(self, predicted, expected):
        """関数コールタスク評価"""
        try:
            # 関数コール構文の検知
            func_patterns = [
                r'def\s+\w+\(', r'function\s+\w+', r'call\s+\w+\(',
                r'execute\s+\w+\(', r'invoke\s+\w+\('
            ]

            pred_has_function = any(pattern in predicted for pattern in func_patterns)
            exp_has_function = any(pattern in expected for pattern in func_patterns)

            # 関数定義/コールの一致
            if pred_has_function and exp_has_function:
                return self._compare_function_signatures(predicted, expected)

            return pred_has_function == exp_has_function

        except Exception as e:
            logger.error(f"[FUNCTION] Function calling evaluation failed: {e}")
            return False

    def _compare_tool_parameters(self, predicted, expected):
        """ツールパラメータの比較"""
        try:
            import re

            # パラメータ抽出
            param_pattern = r'(\w+)\s*=\s*["\']([^"\']+)["\']'
            pred_params = dict(re.findall(param_pattern, predicted))
            exp_params = dict(re.findall(param_pattern, expected))

            # パラメータ一致度
            if not exp_params:
                return bool(pred_params)

            matching_params = 0
            for key, value in exp_params.items():
                if key in pred_params and pred_params[key] == value:
                    matching_params += 1

            return matching_params / len(exp_params) >= 0.8

        except:
            return False

    def _compare_function_signatures(self, predicted, expected):
        """関数シグネチャの比較"""
        try:
            import re

            # 関数名抽出
            func_name_pattern = r'(?:def|function)\s+(\w+)'
            pred_func = re.search(func_name_pattern, predicted)
            exp_func = re.search(func_name_pattern, expected)

            if pred_func and exp_func:
                return pred_func.group(1) == exp_func.group(1)

            # パラメータ比較
            param_pattern = r'\(\s*([^)]*)\s*\)'
            pred_params = re.search(param_pattern, predicted)
            exp_params = re.search(param_pattern, expected)

            if pred_params and exp_params:
                return pred_params.group(1).strip() == exp_params.group(1).strip()

            return False

        except:
            return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description='RTX 3060 Benchmark Evaluation')
    parser.add_argument('--model-path', help='Model path to evaluate')
    parser.add_argument('--config', help='Benchmark configuration file path')
    parser.add_argument('--num-samples', type=int, help='Number of samples per benchmark')
    parser.add_argument('--benchmarks', nargs='+', help='Benchmarks to run')

    args = parser.parse_args()

    evaluator = RTX3060BenchmarkEvaluator(args.config)

    if args.benchmarks:
        # 指定されたベンチマークのみ実行
        evaluator.config['primary_benchmarks'] = args.benchmarks

    results = evaluator.run_benchmark_suite(args.model_path, args.num_samples)

    if results:
        print("[SUCCESS] Benchmark evaluation completed!")
    else:
        print("[ERROR] Benchmark evaluation failed!")
        exit(1)

if __name__ == "__main__":
    main()