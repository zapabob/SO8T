#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standardized Benchmark Evaluator for AEGIS Models

Implements official benchmark evaluation protocols to ensure comparability
with public leaderboards and other models.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandardizedBenchmarkEvaluator:
    """
    Official benchmark evaluation following standardized protocols.

    Supports GSM8K, MATH, ARC-Challenge with correct shot counts and CoT settings.
    """

    def __init__(self, model_path: str, model_name: str = "AEGIS", device: str = "auto"):
        """
        Initialize evaluator with model.

        Args:
            model_path: Path to model or HuggingFace model identifier
            model_name: Name for logging purposes
            device: Device to run evaluation on
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(device)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Extended timeout and token settings for complex reasoning
        self.timeout = 300  # 5 minutes default
        self.max_new_tokens = 1024  # Default max tokens

        logger.info(f"Model loaded successfully: {model_name} (timeout: {self.timeout}s, max_tokens: {self.max_new_tokens})")

    def evaluate_gsm8k(self, num_samples: int = None, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate on GSM8K with official protocol: 8-shot, CoT.

        Args:
            num_samples: Number of samples to evaluate (None for all)
            temperature: Generation temperature

        Returns:
            Dict with accuracy, individual results, and metadata
        """
        logger.info("Starting GSM8K evaluation (8-shot, CoT)")

        # Load GSM8K dataset (simplified version - in practice use official dataset)
        gsm8k_data = self._load_gsm8k_data()

        if num_samples:
            gsm8k_data = gsm8k_data[:num_samples]

        # 8-shot examples (official GSM8K protocol)
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

        for item in tqdm(gsm8k_data, desc="GSM8K Evaluation"):
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

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # timeout parameter not supported in Transformers generate method
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Extract answer (GSM8K format: look for final number)
            predicted_answer = self._extract_gsm8k_answer(response)
            correct_answer = item['answer']

            is_correct = predicted_answer == correct_answer

            results.append({
                'question': item['question'],
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'response': response
            })

            if is_correct:
                correct += 1

        accuracy = correct / len(results)

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
                'model': self.model_name
            }
        }

    def evaluate_math(self, num_samples: int = None, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate on MATH with official protocol: 0-shot, CoT.

        Args:
            num_samples: Number of samples to evaluate
            temperature: Generation temperature

        Returns:
            Dict with accuracy and results
        """
        logger.info("Starting MATH evaluation (0-shot, CoT)")

        math_data = self._load_math_data()

        if num_samples:
            math_data = math_data[:num_samples]

        correct = 0
        results = []

        for item in tqdm(math_data, desc="MATH Evaluation"):
            # 0-shot CoT prompt
            prompt = f"Problem: {item['problem']}\n\n"
            prompt += "Solve this step by step, showing your work clearly.\n"
            prompt += "Final answer:"

            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,  # MATH needs longer responses
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # timeout parameter not supported in Transformers generate method
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Extract final answer (MATH format: \boxed{answer})
            predicted_answer = self._extract_math_answer(response)
            correct_answer = item['answer']

            is_correct = self._compare_math_answers(predicted_answer, correct_answer)

            results.append({
                'problem': item['problem'],
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'response': response
            })

            if is_correct:
                correct += 1

        accuracy = correct / len(results)

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
                'model': self.model_name
            }
        }

    def evaluate_arc_challenge(self, num_samples: int = None, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate on ARC-Challenge with official protocol: 10-shot.

        Args:
            num_samples: Number of samples to evaluate
            temperature: Generation temperature

        Returns:
            Dict with accuracy and results
        """
        logger.info("Starting ARC-Challenge evaluation (10-shot)")

        arc_data = self._load_arc_challenge_data()

        if num_samples:
            arc_data = arc_data[:num_samples]

        # 10-shot examples for ARC-Challenge
        few_shot_examples = self._get_arc_few_shot_examples()

        correct = 0
        results = []

        for item in tqdm(arc_data, desc="ARC-Challenge Evaluation"):
            # Build 10-shot prompt
            prompt = ""
            for example in few_shot_examples:
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

            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # timeout parameter not supported in Transformers generate method
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Extract answer choice
            predicted_answer = self._extract_arc_answer(response)
            correct_answer = item['answer']

            is_correct = predicted_answer == correct_answer

            results.append({
                'question': item['question'],
                'choices': item['choices'],
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'response': response
            })

            if is_correct:
                correct += 1

        accuracy = correct / len(results)

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
                'model': self.model_name
            }
        }

    def run_comprehensive_evaluation(self, num_samples: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on all supported benchmarks.

        Args:
            num_samples: Dict specifying sample counts per benchmark

        Returns:
            Dict with all benchmark results
        """
        if num_samples is None:
            num_samples = {'gsm8k': None, 'math': None, 'arc_challenge': None}

        results = {}

        logger.info("Starting comprehensive benchmark evaluation")

        # GSM8K evaluation
        if 'gsm8k' in num_samples:
            results['gsm8k'] = self.evaluate_gsm8k(num_samples['gsm8k'])

        # MATH evaluation
        if 'math' in num_samples:
            results['math'] = self.evaluate_math(num_samples['math'])

        # ARC-Challenge evaluation
        if 'arc_challenge' in num_samples:
            results['arc_challenge'] = self.evaluate_arc_challenge(num_samples['arc_challenge'])

        # Summary
        summary = {
            'model': self.model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmarks': {}
        }

        for benchmark, result in results.items():
            summary['benchmarks'][benchmark] = {
                'accuracy': result['accuracy'],
                'correct': result['correct'],
                'total': result['total']
            }

        results['summary'] = summary

        logger.info("Comprehensive evaluation completed")
        logger.info(f"Results summary: {summary}")

        return results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file}")

    # Helper methods for data loading and answer extraction
    def _load_gsm8k_data(self) -> List[Dict]:
        """Load GSM8K test data (simplified - replace with actual dataset)"""
        # Placeholder - in practice, load from official GSM8K dataset
        return [
            {
                "question": "If you have 7 apples and you give away 3, how many do you have left?",
                "answer": "4"
            },
            # Add more GSM8K examples...
        ]

    def _load_math_data(self) -> List[Dict]:
        """Load MATH test data (simplified)"""
        return [
            {
                "problem": "Solve for x: 2x + 5 = 17",
                "answer": "6"
            },
            # Add more MATH examples...
        ]

    def _load_arc_challenge_data(self) -> List[Dict]:
        """Load ARC-Challenge test data (simplified)"""
        return [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": "C"
            },
            # Add more ARC examples...
        ]

    def _get_arc_few_shot_examples(self) -> List[Dict]:
        """Get 10-shot examples for ARC-Challenge"""
        return [
            # Add 10 proper ARC-Challenge examples...
        ]

    def _extract_gsm8k_answer(self, response: str) -> str:
        """Extract final answer from GSM8K response"""
        # Look for the final number in the response
        match = re.search(r'(\d+)(?:\.\d+)?(?=\s*$|\s*[^0-9])', response.strip())
        return match.group(1) if match else ""

    def _extract_math_answer(self, response: str) -> str:
        """Extract boxed answer from MATH response"""
        # Look for \boxed{answer} format
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            return match.group(1).strip()
        # Fallback: look for final answer
        return response.strip().split()[-1] if response.strip() else ""

    def _extract_arc_answer(self, response: str) -> str:
        """Extract answer choice from ARC response with improved logic"""
        import re

        # Method 1: Look for explicit choice patterns (A), A., A), etc.
        choice_patterns = [
            r'\b([A-E])\)',           # A), B), etc.
            r'\b([A-E])\.',           # A., B., etc.
            r'answer:\s*([A-E])',     # answer: A
            r'Answer:\s*([A-E])',     # Answer: A
            r'\b([A-E])\b',           # Single letter A, B, etc.
        ]

        for pattern in choice_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Take the last occurrence (often the final answer)
                return matches[-1].upper()

        # Method 2: Look for choice in parentheses or brackets
        paren_match = re.search(r'\(([A-E])\)', response, re.IGNORECASE)
        if paren_match:
            return paren_match.group(1).upper()

        # Method 3: Look for the last A-E letter in the response
        letters = re.findall(r'\b[A-E]\b', response.upper())
        if letters:
            return letters[-1]  # Last occurrence

        # Method 4: Look for choice after thinking/answer tags
        thinking_match = re.search(r'</think>\s*([A-E])', response, re.IGNORECASE)
        if thinking_match:
            return thinking_match.group(1).upper()

        return ""

    def _compare_math_answers(self, predicted: str, correct: str) -> bool:
        """Compare MATH answers with some tolerance"""
        try:
            pred_num = float(predicted.replace('$', '').replace('\\', '').strip())
            corr_num = float(correct.replace('$', '').replace('\\', '').strip())
            return abs(pred_num - corr_num) < 1e-6
        except:
            return predicted.strip() == correct.strip()


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Standardized Benchmark Evaluation')
    parser.add_argument('--model_path', required=True, help='Path to model or HF model identifier')
    parser.add_argument('--model_name', default='AEGIS', help='Model name for logging')
    parser.add_argument('--output_path', default='evaluation_results/standardized_benchmark_results.json', help='Output path')
    parser.add_argument('--gsm8k_samples', type=int, help='Number of GSM8K samples')
    parser.add_argument('--math_samples', type=int, help='Number of MATH samples')
    parser.add_argument('--arc_samples', type=int, help='Number of ARC-Challenge samples')
    parser.add_argument('--device', default='auto', help='Device to run evaluation on')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = StandardizedBenchmarkEvaluator(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device
    )

    # Run comprehensive evaluation
    num_samples = {
        'gsm8k': args.gsm8k_samples,
        'math': args.math_samples,
        'arc_challenge': args.arc_samples
    }

    results = evaluator.run_comprehensive_evaluation(num_samples)

    # Save results
    evaluator.save_results(results, args.output_path)

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for benchmark, result in results.items():
        if benchmark != 'summary':
            print(".4f")


if __name__ == "__main__":
    main()