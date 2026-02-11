#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUF Model Benchmark Evaluator

Evaluates GGUF format models using llama-cpp-python for memory-efficient evaluation.
Supports FP16 and other quantization levels with official benchmark protocols.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from llama_cpp import Llama
import logging
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GGUFStandardizedBenchmarkEvaluator:
    """
    GGUF model evaluator using llama-cpp-python for memory-efficient evaluation.
    """

    def __init__(self, model_path: str, model_name: str = "GGUF-AEGIS", n_gpu_layers: int = -1):
        """
        Initialize GGUF evaluator with llama-cpp-python.

        Args:
            model_path: Path to GGUF model file
            model_name: Name for logging purposes
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        """
        self.model_name = model_name
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        logger.info(f"Loading GGUF model: {model_path}")

        # Initialize llama-cpp-python model
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,  # -1 for all layers on GPU
            n_ctx=4096,  # Context window (Phi-3.5 standard)
            n_threads=8,  # CPU threads
            verbose=False,  # Reduce logging
            seed=42  # For reproducibility
        )

        logger.info(f"GGUF model loaded successfully: {model_name}")
        logger.info(f"Model info: {self.llm.model_path}")

    def evaluate_gsm8k(self, num_samples: int = None, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate on GSM8K with official protocol: 8-shot, CoT.
        """
        logger.info("Starting GSM8K evaluation (8-shot, CoT) with GGUF model")

        gsm8k_data = self._load_gsm8k_data()
        if num_samples:
            gsm8k_data = gsm8k_data[:num_samples]

        # 8-shot examples (GSM8K official protocol)
        few_shot_examples = [
            {
                "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "reasoning": "Natalia sold 48 clips in April. In May, she sold half as many, so 48 ÷ 2 = 24. Total: 48 + 24 = 72.",
                "answer": "72"
            },
            {
                "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                "reasoning": "Weng earns $12 per hour. 50 minutes is 50/60 = 5/6 hours. So she earned 12 × (5/6) = $10.",
                "answer": "10"
            },
            {
                "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
                "reasoning": "Betty needs $100. She has half, so she has $50. Parents give $15, grandparents give 2 × $15 = $30. Total she has: 50 + 15 + 30 = $95. She needs 100 - 95 = $5 more.",
                "answer": "5"
            },
            {
                "question": "Julie is reading a 120-page book. Yesterday, she was on page 12. Today, she read twice as many pages as yesterday. How many pages does she have left to read?",
                "reasoning": "Yesterday she read 12 pages. Today she read 2 × 12 = 24 pages. Total read: 12 + 24 = 36 pages. Left: 120 - 36 = 84 pages.",
                "answer": "84"
            },
            {
                "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
                "reasoning": "James writes 3 pages × 2 friends = 6 pages per letter session. Twice a week × 52 weeks = 104 sessions. Total: 6 × 104 = 624 pages.",
                "answer": "624"
            },
            {
                "question": "Mark has a garden with flowers. He planted plants around the perimeter and in each of the 20 rows. There are 75 plants in each row and 24 plants around the perimeter. How many plants did Mark plant?",
                "reasoning": "Plants in rows: 20 rows × 75 plants = 1500. Plus perimeter: 24. Total: 1500 + 24 = 1524 plants.",
                "answer": "1524"
            },
            {
                "question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many slices does he eat that day?",
                "reasoning": "Large pizzas: 2 × 16 = 32 slices. Small pizzas: 2 × 8 = 16 slices. Total: 32 + 16 = 48 slices.",
                "answer": "48"
            },
            {
                "question": "Ken created a care package to send to his brother for his birthday. Ken placed a teddy bear that weighs 200 grams and chocolates that weigh 300 grams in a box. The box itself weighs 300 grams. If the care package is to weigh 3 kilograms, how many 200-gram teddy bears should Ken add?",
                "reasoning": "Current weight: 200 + 300 + 300 = 800 grams = 0.8 kg. Target: 3 kg. Need: 3 - 0.8 = 2.2 kg = 2200 grams. Each additional bear: 200g. Number: 2200 ÷ 200 = 11 bears.",
                "answer": "11"
            }
        ]

        correct = 0
        results = []

        for item in gsm8k_data:
            # Build 8-shot prompt
            prompt = ""
            for example in few_shot_examples:
                prompt += f"Question: {example['question']}\n"
                prompt += f"Reasoning: {example['reasoning']}\n"
                prompt += f"Answer: {example['answer']}\n\n"

            # Add current question
            prompt += f"Question: {item['question']}\n"
            prompt += "Reasoning: Let's solve this step by step.\n"
            prompt += "Answer:"

            start_time = time.time()

            # Generate response
            try:
                response = self.llm(
                    prompt,
                    max_tokens=512,
                    temperature=temperature,
                    stop=["\n\n", "Question:"],  # Stop at next question or double newline
                    echo=False
                )

                generated_text = response['choices'][0]['text'].strip()
                elapsed_time = time.time() - start_time

                # Extract answer
                predicted_answer = self._extract_gsm8k_answer(generated_text)
                correct_answer = item['answer']

                is_correct = predicted_answer == correct_answer

                results.append({
                    'question': item['question'],
                    'predicted': predicted_answer,
                    'correct': correct_answer,
                    'is_correct': is_correct,
                    'response': generated_text,
                    'inference_time': elapsed_time
                })

                if is_correct:
                    correct += 1

            except Exception as e:
                logger.error(f"Error evaluating GSM8K item: {e}")
                results.append({
                    'question': item['question'],
                    'predicted': '',
                    'correct': item['answer'],
                    'is_correct': False,
                    'response': f"Error: {e}",
                    'inference_time': time.time() - start_time
                })

        accuracy = correct / len(results) if results else 0.0

        logger.info(".4f")

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(results),
            'results': results,
            'metadata': {
                'benchmark': 'GSM8K',
                'protocol': '8-shot CoT',
                'temperature': temperature,
                'model': self.model_name,
                'format': 'GGUF'
            }
        }

    def evaluate_math(self, num_samples: int = None, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate on MATH with official protocol: 0-shot, CoT.
        """
        logger.info("Starting MATH evaluation (0-shot, CoT) with GGUF model")

        math_data = self._load_math_data()
        if num_samples:
            math_data = math_data[:num_samples]

        correct = 0
        results = []

        for item in math_data:
            # 0-shot CoT prompt
            prompt = f"Problem: {item['problem']}\n\n"
            prompt += "Solve this step by step, showing your work clearly.\n"
            prompt += "Final answer:"

            start_time = time.time()

            try:
                response = self.llm(
                    prompt,
                    max_tokens=1024,  # MATH needs longer responses
                    temperature=temperature,
                    stop=["\n\n", "Problem:"],  # Stop at next problem
                    echo=False
                )

                generated_text = response['choices'][0]['text'].strip()
                elapsed_time = time.time() - start_time

                # Extract final answer
                predicted_answer = self._extract_math_answer(generated_text)
                correct_answer = item['answer']

                is_correct = self._compare_math_answers(predicted_answer, correct_answer)

                results.append({
                    'problem': item['problem'],
                    'predicted': predicted_answer,
                    'correct': correct_answer,
                    'is_correct': is_correct,
                    'response': generated_text,
                    'inference_time': elapsed_time
                })

                if is_correct:
                    correct += 1

            except Exception as e:
                logger.error(f"Error evaluating MATH item: {e}")
                results.append({
                    'problem': item['problem'],
                    'predicted': '',
                    'correct': item['answer'],
                    'is_correct': False,
                    'response': f"Error: {e}",
                    'inference_time': time.time() - start_time
                })

        accuracy = correct / len(results) if results else 0.0

        logger.info(".4f")

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(results),
            'results': results,
            'metadata': {
                'benchmark': 'MATH',
                'protocol': '0-shot CoT',
                'temperature': temperature,
                'model': self.model_name,
                'format': 'GGUF'
            }
        }

    def evaluate_arc_challenge(self, num_samples: int = None, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate on ARC-Challenge with official protocol: 10-shot.
        """
        logger.info("Starting ARC-Challenge evaluation (10-shot) with GGUF model")

        arc_data = self._load_arc_challenge_data()
        if num_samples:
            arc_data = arc_data[:num_samples]

        # 10-shot examples
        few_shot_examples = self._get_arc_few_shot_examples()

        correct = 0
        results = []

        for item in arc_data:
            # Build 10-shot prompt
            prompt = ""
            for example in few_shot_examples[:10]:  # Use first 10 examples
                prompt += f"Question: {example['question']}\n"
                prompt += f"Choices:\n"
                for i, choice in enumerate(example['choices']):
                    prompt += f"{chr(65+i)}) {choice}\n"
                prompt += f"Answer: {example['answer']}\n\n"

            # Add current question
            prompt += f"Question: {item['question']}\n"
            prompt += f"Choices:\n"
            for i, choice in enumerate(item['choices']):
                prompt += f"{chr(65+i)}) {choice}\n"
            prompt += "Answer:"

            start_time = time.time()

            try:
                response = self.llm(
                    prompt,
                    max_tokens=256,
                    temperature=temperature,
                    stop=["\n\n", "Question:"],
                    echo=False
                )

                generated_text = response['choices'][0]['text'].strip()
                elapsed_time = time.time() - start_time

                # Extract answer choice
                predicted_answer = self._extract_arc_answer(generated_text)
                correct_answer = item['answer']

                is_correct = predicted_answer == correct_answer

                results.append({
                    'question': item['question'],
                    'choices': item['choices'],
                    'predicted': predicted_answer,
                    'correct': correct_answer,
                    'is_correct': is_correct,
                    'response': generated_text,
                    'inference_time': elapsed_time
                })

                if is_correct:
                    correct += 1

            except Exception as e:
                logger.error(f"Error evaluating ARC item: {e}")
                results.append({
                    'question': item['question'],
                    'choices': item['choices'],
                    'predicted': '',
                    'correct': item['answer'],
                    'is_correct': False,
                    'response': f"Error: {e}",
                    'inference_time': time.time() - start_time
                })

        accuracy = correct / len(results) if results else 0.0

        logger.info(".4f")

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(results),
            'results': results,
            'metadata': {
                'benchmark': 'ARC-Challenge',
                'protocol': '10-shot',
                'temperature': temperature,
                'model': self.model_name,
                'format': 'GGUF'
            }
        }

    def run_comprehensive_evaluation(self, num_samples: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on all supported benchmarks.
        """
        if num_samples is None:
            num_samples = {'gsm8k': None, 'math': None, 'arc_challenge': None}

        results = {
            'model': self.model_name,
            'format': 'GGUF',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info("Starting comprehensive GGUF model evaluation")

        # GSM8K evaluation
        if 'gsm8k' in num_samples:
            results['gsm8k'] = self.evaluate_gsm8k(num_samples['gsm8k'])

        # MATH evaluation
        if 'math' in num_samples:
            results['math'] = self.evaluate_math(num_samples['math'])

        # ARC-Challenge evaluation
        if 'arc_challenge' in num_samples:
            results['arc_challenge'] = self.evaluate_arc_challenge(num_samples['arc_challenge'])

        logger.info("Comprehensive GGUF evaluation completed")
        return results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to JSON file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"GGUF evaluation results saved to: {output_file}")

    # Helper methods
    def _load_gsm8k_data(self) -> List[Dict]:
        """Load GSM8K test data (placeholder)"""
        return [
            {"question": "If you have 7 apples and you give away 3, how many do you have left?", "answer": "4"},
            # Add more examples...
        ]

    def _load_math_data(self) -> List[Dict]:
        """Load MATH test data (placeholder)"""
        return [
            {"problem": "Solve for x: 2x + 5 = 17", "answer": "6"},
            # Add more examples...
        ]

    def _load_arc_challenge_data(self) -> List[Dict]:
        """Load ARC-Challenge test data (placeholder)"""
        return [
            {"question": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "answer": "C"},
            # Add more examples...
        ]

    def _get_arc_few_shot_examples(self) -> List[Dict]:
        """Get ARC-Challenge few-shot examples"""
        return [
            # Add proper ARC examples...
        ]

    def _extract_gsm8k_answer(self, response: str) -> str:
        """Extract final answer from GSM8K response"""
        match = re.search(r'(\d+)(?:\.\d+)?(?=\s*$|\s*[^0-9])', response.strip())
        return match.group(1) if match else ""

    def _extract_math_answer(self, response: str) -> str:
        """Extract boxed answer from MATH response"""
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            return match.group(1).strip()
        return response.strip().split()[-1] if response.strip() else ""

    def _extract_arc_answer(self, response: str) -> str:
        """Extract answer choice with robust ARC-Challenge specific logic"""
        import re
        from typing import Optional

        CHOICE_RE = re.compile(r"\b([A-E])\b")

        def extract_arc_choice(text: str) -> Optional[str]:
            """
            ARC-Challenge用の頑健な選択肢抽出。
            - 'Answer: C', '答え：C', '(C)', 'C.' などを拾う
            - 最後の出現を優先（長文の最後に結論を書く癖対策）
            """
            if text is None:
                return None

            # タグやコードブロックの残骸を軽く掃除
            cleaned = text.strip()

            # よくある「答え」行を優先的に拾う
            patterns = [
                r"(?:Answer|答え|解答|最終回答)\s*[:：]?\s*([A-E])\b",
                r"(?:Answer|答え|解答|最終回答)\s*[:：]?\s*\(?\s*([A-E])\s*\)?",
            ]
            for pat in patterns:
                m = re.search(pat, cleaned, flags=re.IGNORECASE)
                if m:
                    return m.group(1).upper()

            # 単独文字A-Eの出現を全部拾って最後を返す
            matches = CHOICE_RE.findall(cleaned.upper())
            if matches:
                return matches[-1]

            return None

        # Apply the robust extraction
        result = extract_arc_choice(response)
        return result if result else ""

    def _compare_math_answers(self, predicted: str, correct: str) -> bool:
        """Compare MATH answers with tolerance"""
        try:
            pred_num = float(predicted.replace('$', '').replace('\\', '').strip())
            corr_num = float(correct.replace('$', '').replace('\\', '').strip())
            return abs(pred_num - corr_num) < 1e-6
        except:
            return predicted.strip() == correct.strip()


def main():
    """Main GGUF evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='GGUF Model Benchmark Evaluation')
    parser.add_argument('--model_path', required=True, help='Path to GGUF model file')
    parser.add_argument('--model_name', default='GGUF-Model', help='Model name for logging')
    parser.add_argument('--output_path', default='evaluation_results/gguf_benchmark_results.json')
    parser.add_argument('--gsm8k_samples', type=int, help='Number of GSM8K samples')
    parser.add_argument('--math_samples', type=int, help='Number of MATH samples')
    parser.add_argument('--arc_samples', type=int, help='Number of ARC-Challenge samples')
    parser.add_argument('--n_gpu_layers', type=int, default=-1, help='GPU layers (-1 for all)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Generation temperature')

    args = parser.parse_args()

    # Initialize GGUF evaluator
    evaluator = GGUFStandardizedBenchmarkEvaluator(
        model_path=args.model_path,
        model_name=args.model_name,
        n_gpu_layers=args.n_gpu_layers
    )

    # Setup sample counts
    num_samples = {}
    if args.gsm8k_samples:
        num_samples['gsm8k'] = args.gsm8k_samples
    if args.math_samples:
        num_samples['math'] = args.math_samples
    if args.arc_samples:
        num_samples['arc_challenge'] = args.arc_samples

    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(num_samples)

    # Save results
    evaluator.save_results(results, args.output_path)

    # Print summary
    print("\n" + "="*80)
    print("GGUF MODEL BENCHMARK EVALUATION RESULTS")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Format: GGUF")

    for benchmark in ['gsm8k', 'math', 'arc_challenge']:
        if benchmark in results:
            data = results[benchmark]
            print(f"\n{benchmark.upper()}:")
            print(".4f")


if __name__ == "__main__":
    main()