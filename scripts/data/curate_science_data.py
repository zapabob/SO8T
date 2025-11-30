#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8-Think Science Data Curation Script
高品質な科学・数学データセット作成スクリプト

PhD/Fields賞/Nobel賞級推論能力付与のためのデータキュレーション
破滅的忘却を防ぐための厳格な品質フィルタリング
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

# Hugging Face datasets
try:
    from datasets import load_dataset, Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

# トークナイザー（簡易版）
TOKENIZER_AVAILABLE = False
tokenizer = None
try:
    from transformers import AutoTokenizer
    # 必要時のみロード（メモリ節約）
    TOKENIZER_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. Using character-based token counting.")

# 設定
TARGET_DATASETS = {
    'math': 'AI-MO/NuminaMath-CoT',
    'physics': 'camel-ai/physics',
    'chemistry': 'camel-ai/chemistry',
    'reasoning': 'Magpie-Align/Magpie-Reasoning-V2',
    'nsfw_detection': 'jmgq36/nsfw-dataset'  # NSFW検知用データセット
}

# 品質フィルタリング設定
QUALITY_FILTERS = {
    'min_tokens': 100,
    'max_tokens': 4096,
    'complexity_percentile': 80,  # 上位20%のみ
    'latex_min_density': 0.001,    # LaTeX密度の最小値
}

# 拒絶キーワード
REJECTION_KEYWORDS = [
    "I don't know", "i don't know", "I'm not sure", "i'm not sure",
    "As an AI", "as an AI", "Sorry", "sorry", "I cannot", "i cannot",
    "I'm unable", "i'm unable", "I apologize", "i apologize"
]

# 四重推論分類用のキーワード
QUADRUPLE_INFERENCE_KEYWORDS = {
    'observation': [
        'observe', 'see', 'notice', 'find', 'discover', 'detect',
        'measure', 'record', 'note', 'identify', 'recognize',
        '観測', '確認', '発見', '測定', '記録', '同定'
    ],
    'deduction': [
        'therefore', 'thus', 'hence', 'consequently', 'follows',
        'implies', 'entails', 'derives', 'infers', 'concludes',
        'therefore', 'because', 'since', 'due to', 'as a result',
        'したがって', 'ゆえに', 'それゆえ', '従って', '推論'
    ],
    'abduction': [
        'similar', 'analogy', 'isomorphic', 'corresponds', 'maps',
        'homology', 'analogy', 'parallel', 'analogy', 'metaphor',
        'isomorphism', 'correspondence', 'mapping', 'relation',
        'アナロジー', '同型', '対応', '写像', '類似', '比喩'
    ],
    'integration': [
        'integrate', 'combine', 'unify', 'synthesize', 'merge',
        'consolidate', 'amalgamate', 'blend', 'fuse', 'coalesce',
        '統合', '結合', '合成', '統一', '融合', 'まとめ'
    ]
}

# NSFW検知用のキーワード
NSFW_KEYWORDS = [
    'nsfw', 'adult', 'explicit', 'porn', 'sex', 'nude', 'naked',
    'erotic', 'xxx', '18+', 'mature', 'hentai', 'fetish',
    'violent', 'gore', 'disturbing', 'offensive', 'inappropriate'
]

# 専門用語リスト（複雑度スコア用）
SCIENCE_TERMS = [
    # 数学
    'theorem', 'lemma', 'proof', 'corollary', 'proposition', 'axiom',
    'integral', 'derivative', 'matrix', 'vector', 'tensor', 'manifold',
    'group', 'ring', 'field', 'algebra', 'topology', 'geometry',

    # 物理
    'quantum', 'relativity', 'thermodynamics', 'electromagnetism', 'mechanics',
    'particle', 'wave', 'field', 'energy', 'momentum', 'force', 'mass',
    'velocity', 'acceleration', 'entropy', 'temperature', 'pressure',

    # 化学
    'molecule', 'atom', 'bond', 'reaction', 'catalyst', 'equilibrium',
    'acid', 'base', 'pH', 'solvent', 'crystal', 'polymer', 'enzyme',
    'protein', 'DNA', 'RNA', 'nucleotide', 'amino acid'
]

def count_tokens(text: str) -> int:
    """テキストのトークン数をカウント"""
    global tokenizer

    if TOKENIZER_AVAILABLE and tokenizer is not None:
        return len(tokenizer.encode(text))
    elif TOKENIZER_AVAILABLE and tokenizer is None:
        # 必要時のみロード
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3.5-mini-instruct")
            return len(tokenizer.encode(text))
        except Exception as e:
            print(f"Tokenizer loading failed: {e}")
            return len(text) // 4
    else:
        # 簡易版：文字数を4で割った値
        return len(text) // 4

def classify_quadruple_inference(text: str) -> str:
    """四重推論タイプを分類 (Observation/Deduction/Abduction/Integration)"""
    text_lower = text.lower()

    # 各タイプのスコアを計算
    scores = {}
    for inference_type, keywords in QUADRUPLE_INFERENCE_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        scores[inference_type] = score

    # 最高スコアのタイプを返す
    if max(scores.values()) > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    else:
        # デフォルトはdeduction（論理的推論）
        return 'deduction'

def is_nsfw_content(text: str) -> bool:
    """NSFWコンテンツかどうかを判定"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in NSFW_KEYWORDS)

def add_phi35_internal_tags(instruction: str, output: str, domain: str) -> dict:
    """Phi-3.5の内部タグを付与して四重推論を可能にする"""

    # 四重推論タイプを分類
    inference_type = classify_quadruple_inference(output)

    # NSFW判定
    is_nsfw = is_nsfw_content(output) if domain == 'nsfw_detection' else False

    # システムプロンプト生成
    system_prompt = generate_system_prompt(domain, inference_type, is_nsfw)

    # 思考プロセスにタグを付与
    tagged_output = add_thinking_tags(output, inference_type)

    return {
        'system': system_prompt,
        'instruction': instruction,
        'input': '',
        'output': tagged_output,
        'domain': domain,
        'inference_type': inference_type,
        'is_nsfw': is_nsfw
    }

def generate_system_prompt(domain: str, inference_type: str, is_nsfw: bool) -> str:
    """ドメインと推論タイプに応じたシステムプロンプト生成"""

    base_prompt = "あなたはSO(8)理論に基づく物理的知性を持つAIです。"

    if domain == 'math':
        base_prompt += "数学的証明と定理を厳密に扱い、LaTeX形式で表現します。"
    elif domain == 'physics':
        base_prompt += "物理法則と理論を正確に理解し、実験的証拠に基づいて説明します。"
    elif domain == 'chemistry':
        base_prompt += "化学反応と分子構造を正確に記述し、実験的手法を考慮します。"
    elif domain == 'reasoning':
        base_prompt += "論理的思考を重視し、複数の視点から問題を分析します。"
    elif domain == 'nsfw_detection':
        base_prompt += "コンテンツの安全性を評価し、適切な分類を行います。"

    # 推論タイプに応じた指示
    if inference_type == 'observation':
        base_prompt += "観測事実を客観的に記述してください。"
    elif inference_type == 'deduction':
        base_prompt += "論理的推論により結論を導いてください。"
    elif inference_type == 'abduction':
        base_prompt += "異なる概念間の類似性や同型性を発見してください。"
    elif inference_type == 'integration':
        base_prompt += "複数の要素を統合し、包括的な結論を導いてください。"

    if is_nsfw:
        base_prompt += "このコンテンツは成人向けです。安全に扱ってください。"

    return base_prompt

def add_thinking_tags(output: str, inference_type: str) -> str:
    """思考プロセスにPhi-3.5の内部タグを付与"""

    # 既存の思考タグがある場合は置換
    if '<think>' in output and '</think>' in output:
        return output

    # 四重推論構造を構築
    thinking_parts = {
        'observation': f"<|observation|>\n{output[:len(output)//4]}\n<|end_observation|>",
        'deduction': f"<|deduction|>\n{output[len(output)//4:len(output)//2]}\n<|end_deduction|>",
        'abduction': f"<|abduction|>\n{output[len(output)//2:3*len(output)//4]}\n<|end_abduction|>",
        'integration': f"<|integration|>\n{output[3*len(output)//4:]}\n<|end_integration|>"
    }

    # 思考プロセスを構築
    thinking_process = f"<think>\n{thinking_parts['observation']}\n{thinking_parts['deduction']}\n{thinking_parts['abduction']}\n{thinking_parts['integration']}\n</think>"

    # 最終回答
    final_answer = f"<final>\n{output}\n</final>"

    return thinking_process + "\n\n" + final_answer

def check_latex_density(text: str) -> float:
    """LaTeX数式の密度をチェック"""
    # LaTeXコマンドとインライン数式のカウント
    latex_patterns = [
        r'\$.*?\$',           # インライン数式 $...$
        r'\\\[.*?\\\]',       # ディスプレイ数式 \[...\]
        r'\\begin\{.*?\}.*?\\end\{.*?\}',  # 環境
        r'\\[a-zA-Z]+',       # LaTeXコマンド
        r'_\{.*?\}',          # 下付き
        r'\^\{.*?\}',         # 上付き
        r'\\frac\{.*?\}\{.*?\}',  # 分数
        r'\\sum', r'\\int', r'\\prod',  # 演算子
    ]

    total_latex_chars = 0
    total_chars = len(text)

    for pattern in latex_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            total_latex_chars += len(match)

    # 密度 = LaTeX文字数 / 総文字数
    density = total_latex_chars / total_chars if total_chars > 0 else 0
    return density

def check_rejection_keywords(text: str) -> bool:
    """拒絶キーワードが含まれているかチェック"""
    text_lower = text.lower()
    for keyword in REJECTION_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False

def calculate_complexity_score(text: str) -> float:
    """テキストの複雑度スコアを計算"""
    score = 0.0

    # 1. ユニーク単語数の割合
    words = re.findall(r'\b\w+\b', text.lower())
    if words:
        unique_ratio = len(set(words)) / len(words)
        score += unique_ratio * 0.3

    # 2. 専門用語の密度
    text_lower = text.lower()
    term_count = 0
    for term in SCIENCE_TERMS:
        if term in text_lower:
            term_count += 1
    term_density = term_count / len(text.split()) if text.split() else 0
    score += term_density * 0.4

    # 3. 平均単語長
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words)
        # 長い単語ほど複雑（ただし上限あり）
        score += min(avg_word_length / 10, 1.0) * 0.3

    return score

def quality_filter(example: Dict[str, Any]) -> bool:
    """品質フィルタリング関数"""
    # テキスト抽出（データセットによって構造が異なる）
    text = ""
    if 'text' in example:
        text = example['text']
    elif 'content' in example:
        text = example['content']
    elif 'message' in example:
        text = example['message']
    elif 'instruction' in example and 'output' in example:
        text = f"{example['instruction']} {example['output']}"
    else:
        # 他のフィールドを試す
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 50:
                text = value
                break

    if not text or len(text.strip()) < 10:
        return False

    # トークン数チェック
    token_count = count_tokens(text)
    if token_count < QUALITY_FILTERS['min_tokens'] or token_count > QUALITY_FILTERS['max_tokens']:
        return False

    # LaTeX密度チェック（数学データセットのみ）
    if 'math' in str(example.get('dataset', '')) or 'Math' in str(example.get('dataset', '')):
        latex_density = check_latex_density(text)
        if latex_density < QUALITY_FILTERS['latex_min_density']:
            return False

    # 拒絶キーワードチェック
    if check_rejection_keywords(text):
        return False

    return True

def complexity_filter(dataset: Dataset, top_percentile: int = 80) -> Dataset:
    """複雑度スコアに基づくフィルタリング"""
    print(f"Calculating complexity scores for {len(dataset)} examples...")

    # 複雑度スコア計算
    complexities = []
    for example in tqdm(dataset, desc="Complexity calculation"):
        text = ""
        # テキスト抽出（quality_filterと同じロジック）
        if 'text' in example:
            text = example['text']
        elif 'content' in example:
            text = example['content']
        elif 'message' in example:
            text = example['message']
        elif 'instruction' in example and 'output' in example:
            text = f"{example['instruction']} {example['output']}"

        score = calculate_complexity_score(text) if text else 0.0
        complexities.append(score)

    # パーセンタイル計算
    if complexities:
        threshold = np.percentile(complexities, top_percentile)
        print(f"Complexity threshold (top {100-top_percentile}%): {threshold:.3f}")

        # フィルタリング
        filtered_indices = [i for i, score in enumerate(complexities) if score >= threshold]
        filtered_dataset = dataset.select(filtered_indices)
        print(f"Filtered to {len(filtered_dataset)} examples (top {100-top_percentile}%)")
        return filtered_dataset
    else:
        return dataset

def load_and_filter_dataset(dataset_name: str, split: str = 'train', num_proc: int = 4) -> Dataset:
    """データセットのロードとフィルタリング"""
    print(f"Loading dataset: {dataset_name}")

    try:
        # データセットロード
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

        # 初期サイズ表示
        print(f"Original dataset size: {len(dataset)}")

        # 品質フィルタリング
        print("Applying quality filters...")
        filtered_dataset = dataset.filter(
            quality_filter,
            num_proc=num_proc,
            desc="Quality filtering"
        )

        print(f"After quality filtering: {len(filtered_dataset)}")

        # 複雑度フィルタリング
        if len(filtered_dataset) > 100:  # サンプル数が十分な場合のみ
            filtered_dataset = complexity_filter(filtered_dataset)

        return filtered_dataset

    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

def sample_datasets(target_counts: Dict[str, int], num_proc: int = 4) -> Dict[str, Dataset]:
    """各データセットからサンプリング"""
    sampled_datasets = {}

    for category, count in target_counts.items():
        dataset_name = TARGET_DATASETS[category]
        print(f"\nProcessing {category}: {dataset_name} (target: {count})")

        dataset = load_and_filter_dataset(dataset_name, num_proc=num_proc)
        if dataset is None:
            continue

        # サンプリング
        if len(dataset) > count:
            sampled = dataset.shuffle(seed=42).select(range(count))
        else:
            sampled = dataset
            print(f"Warning: Only {len(sampled)} samples available (requested {count})")

        sampled_datasets[category] = sampled
        print(f"Final {category} dataset size: {len(sampled)}")

    return sampled_datasets

def convert_to_alpaca_format(example: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Alpacaフォーマットに変換 + Phi-3.5内部タグ付与"""

    # テキスト抽出
    instruction = ""
    output = ""

    if 'instruction' in example and 'output' in example:
        # すでにinstruction/output形式
        instruction = example['instruction']
        output = example['output']
    elif 'question' in example and 'answer' in example:
        # Q&A形式
        instruction = example['question']
        output = example['answer']
    elif 'text' in example:
        # 単一テキストの場合、分割
        text = example['text']
        if 'Q:' in text or 'Question:' in text:
            parts = re.split(r'(?:Q:|Question:)', text, 1)
            if len(parts) > 1:
                instruction = 'Question: ' + parts[1].split('A:', 1)[0].strip()
                output = 'A: ' + parts[1].split('A:', 1)[1].strip() if 'A:' in parts[1] else parts[1].strip()
            else:
                instruction = text[:len(text)//2]
                output = text[len(text)//2:]
        else:
            instruction = text[:len(text)//2]
            output = text[len(text)//2:]
    else:
        # フォールバック
        instruction = str(example)
        output = "このデータは高度な科学的推論を必要とします。"

    # Phi-3.5内部タグを付与して四重推論を可能にする
    tagged_data = add_phi35_internal_tags(instruction, output, category)

    return tagged_data

def save_as_jsonl(datasets: Dict[str, Dataset], output_file: str):
    """JSONL形式で保存"""
    print(f"Saving to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        total_count = 0

        for category, dataset in datasets.items():
            print(f"Converting {category} dataset ({len(dataset)} examples)...")

            for example in tqdm(dataset, desc=f"Converting {category}"):
                alpaca_format = convert_to_alpaca_format(example, category)
                json.dump(alpaca_format, f, ensure_ascii=False)
                f.write('\n')
                total_count += 1

    print(f"Total saved: {total_count} examples")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="SO8-Think Science Data Curation")
    parser.add_argument('--output', type=str, default='data/science_reasoning_dataset.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--total_samples', type=int, default=50000,
                       help='Total number of samples')
    parser.add_argument('--math_ratio', type=float, default=0.4,
                       help='Math dataset ratio (default: 0.4)')
    parser.add_argument('--physics_ratio', type=float, default=0.3,
                       help='Physics dataset ratio (default: 0.3)')
    parser.add_argument('--reasoning_ratio', type=float, default=0.3,
                       help='General reasoning dataset ratio (default: 0.3)')
    parser.add_argument('--num_proc', type=int, default=4,
                       help='Number of processes for parallel processing')
    parser.add_argument('--include_chemistry', action='store_true',
                       help='Include chemistry dataset (otherwise physics only)')

    args = parser.parse_args()

    # 比率チェック
    total_ratio = args.math_ratio + args.physics_ratio + args.reasoning_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio}, adjusting to 1.0")
        # 自動調整
        factor = 1.0 / total_ratio
        args.math_ratio *= factor
        args.physics_ratio *= factor
        args.reasoning_ratio *= factor

    # ターゲットカウント計算
    target_counts = {
        'math': int(args.total_samples * args.math_ratio),
        'physics': int(args.total_samples * args.physics_ratio),
        'reasoning': int(args.total_samples * args.reasoning_ratio)
    }

    if args.include_chemistry:
        # 化学を物理に含める場合
        target_counts['chemistry'] = target_counts['physics'] // 2
        target_counts['physics'] = target_counts['physics'] // 2

    print("SO8-Think Science Data Curation")
    print("=" * 50)
    print(f"Total target samples: {args.total_samples}")
    print(f"Ratios: Math={args.math_ratio:.2f}, Physics={args.physics_ratio:.2f}, Reasoning={args.reasoning_ratio:.2f}")
    print(f"Target counts: {target_counts}")
    print(f"Output: {args.output}")
    print()

    # データセットサンプリング
    sampled_datasets = sample_datasets(target_counts, args.num_proc)

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_as_jsonl(sampled_datasets, args.output)

    print("\nData curation completed!")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
