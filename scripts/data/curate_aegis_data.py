#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS-v2.0 Data Curation Script
NKAT理論に基づく四値分類タグ付きデータセット作成

このスクリプトは、以下のデータソースから高品質なデータセットを作成し、
四値分類タグ（<|allow|>, <|escalation|>, <|deny|>, <|refuse|>）を自動付与します。

データソース:
- Science (English): AI-MO/NuminaMath-CoT, camel-ai/physics
- Japanese: elyza/ELYZA-tasks-100, izumi-lab/llm-japanese-dataset
- Safety/NSFW: Anthropic/hh-rlhf, PKU-Alignment/PKU-SafeRLHF

著者: AI Agent (峯岸亮さん仕様に基づく)
日付: 2025-11-30
"""

import json
import re
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import datasets
from datasets import load_dataset, Dataset
import nltk
from nltk.tokenize import sent_tokenize

# 日本語処理用
try:
    import fugashi
    import ipadic
    tokenizer_ja = fugashi.Tagger(ipadic.MECAB_ARGS)
except ImportError:
    print("日本語処理ライブラリがインストールされていません。pip install fugashi ipadic")
    tokenizer_ja = None

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/curate_aegis_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """データキュレーション設定"""
    # データソース
    science_datasets: List[str] = field(default_factory=lambda: [
        "AI-MO/NuminaMath-CoT",
        "camel-ai/physics",
        "camel-ai/math",
        "gsm8k"  # 数学推論
    ])

    japanese_datasets: List[str] = field(default_factory=lambda: [
        "elyza/ELYZA-tasks-100",
        "izumi-lab/llm-japanese-dataset",
        "llm-jp/magpie-pro-200k-ja",
        "hotchpotch/japanese-novel-instructions"
    ])

    safety_datasets: List[str] = field(default_factory=lambda: [
        "Anthropic/hh-rlhf",  # red-teaming subset
        "PKU-Alignment/PKU-SafeRLHF",
        "HuggingFaceH4/ultrafeedback_binarized",  # 安全性評価
        "allenai/WildChat"  # 多様な会話データ
    ])

    # 品質フィルタリング
    min_latex_density: float = 0.05  # LaTeX密度最小値
    max_latex_density: float = 0.8   # LaTeX密度最大値
    min_complexity_score: float = 0.3  # 複雑度スコア最小値
    max_length: int = 2048  # 最大トークン長
    min_length: int = 50    # 最小トークン長

    # タグ分類閾値
    complexity_threshold: float = 0.6  # 複雑度閾値（escalation判定用）
    toxicity_threshold: float = 0.7    # 毒性閾値（refuse判定用）
    logical_error_threshold: float = 0.8  # 論理誤り閾値（deny判定用）

    # サンプリング比率
    science_ratio: float = 0.4
    japanese_ratio: float = 0.4
    safety_ratio: float = 0.2

    # 出力設定
    output_dir: str = "data/aegis_v2_dataset"
    total_samples: int = 50000
    val_split: float = 0.2

    # ローカルデータセット設定（H:\ドライブ参照）
    local_datasets: List[str] = field(default_factory=lambda: [
        "H:\\from_D\\webdataset\\integrated_dataset.jsonl",  # 既存統合データセット
        "H:\\from_D\\webdataset\\elyza100_test.jsonl",       # ELYZAテストデータ
        "H:\\from_D\\webdataset\\gsm8k_train.jsonl",         # GSM8K訓練データ
        "H:\\from_D\\webdataset\\mmlu_dev.jsonl",           # MMLU開発データ
    ])

    # ローカルデータ比率
    local_ratio: float = 0.3  # 全体の30%をローカルデータから

class NKATAutoTagger:
    """NKAT理論に基づく四値分類自動タグ付けクラス"""

    def __init__(self, config: DataConfig):
        self.config = config

        # トークナイザー初期化（Phi-3.5用）
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"トークナイザー読み込み失敗: {e}")
            self.tokenizer = None

        # センテンストランスフォーマー（類似度計算用）
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer loaded successfully")
        except Exception as e:
            logger.warning(f"Embedder読み込み失敗: {e}")
            self.embedder = None

        # NSFW/毒性キーワード
        self.nsfw_keywords = {
            'explicit': ['nsfw', 'porn', 'sex', 'nude', 'naked', 'erotic', 'adult', 'xxx'],
            'violence': ['kill', 'murder', 'rape', 'torture', 'abuse', 'harm', 'suicide'],
            'hate': ['racist', 'sexist', 'homophobic', 'transphobic', 'discrimination'],
            'illegal': ['drug', 'weapon', 'bomb', 'hack', 'pirate', 'illegal']
        }

        # 論理的誤りパターン
        self.logical_error_patterns = [
            r'\b(impossible|contradiction|paradox)\b',
            r'\b(false|wrong|incorrect)\b.*\b(proof|theorem|law)\b',
            r'\b(violates?|breaks?)\b.*\b(law|principle|theorem)\b'
        ]

        # LaTeXパターン
        self.latex_pattern = re.compile(r'\\[a-zA-Z]+(?:\{[^}]*\})*|\$[^$]*\$|\$\$[^\$]*\$\$')

    def calculate_complexity_score(self, text: str) -> float:
        """テキストの複雑度スコアを計算"""
        if not text or len(text.strip()) < 10:
            return 0.0

        try:
            # LaTeX密度
            latex_matches = self.latex_pattern.findall(text)
            latex_density = len(latex_matches) / len(text.split()) if text.split() else 0

            # 語彙多様性
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words = set(words)
            vocab_diversity = len(unique_words) / len(words) if words else 0

            # 文長の変動性
            try:
                sentences = sent_tokenize(text)
                if len(sentences) < 2:
                    sentence_variability = 0
                else:
                    sentence_lengths = [len(s.split()) for s in sentences]
                    if sentence_lengths and np.mean(sentence_lengths) > 0:
                        sentence_variability = min(np.std(sentence_lengths) / np.mean(sentence_lengths), 1.0)
                    else:
                        sentence_variability = 0
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
                # フォールバック
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                if len(sentences) < 2:
                    sentence_variability = 0
                else:
                    sentence_lengths = [len(s.split()) for s in sentences]
                    if sentence_lengths and np.mean(sentence_lengths) > 0:
                        sentence_variability = min(np.std(sentence_lengths) / np.mean(sentence_lengths), 1.0)
                    else:
                        sentence_variability = 0

            # 専門用語密度（大文字単語）
            capital_words = [w for w in words if w and w[0].isupper()]
            capital_density = len(capital_words) / len(words) if words else 0

            # 複合スコア
            complexity = (
                0.3 * min(latex_density * 20, 1.0) +  # LaTeX密度
                0.2 * vocab_diversity +                 # 語彙多様性
                0.2 * min(sentence_variability, 1.0) + # 文構造複雑性
                0.3 * capital_density                   # 専門用語密度
            )

            return min(complexity, 1.0)

        except Exception as e:
            logger.warning(f"複雑度計算エラー: {e}")
            return 0.5

    def calculate_toxicity_score(self, text: str) -> float:
        """毒性スコアを計算"""
        if not text:
            return 0.0

        text_lower = text.lower()
        toxicity_score = 0.0

        # キーワードベースの毒性検出
        for category, keywords in self.nsfw_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if category == 'explicit':
                toxicity_score += matches * 0.4
            elif category == 'violence':
                toxicity_score += matches * 0.3
            elif category == 'hate':
                toxicity_score += matches * 0.2
            elif category == 'illegal':
                toxicity_score += matches * 0.3

        # 論理的誤りパターン
        for pattern in self.logical_error_patterns:
            if re.search(pattern, text_lower):
                toxicity_score += 0.2

        return min(toxicity_score, 1.0)

    def classify_inference_type(self, text: str, is_science: bool = False,
                              is_japanese: bool = False) -> str:
        """四値分類を実行"""
        if not text:
            return "<|allow|>"

        complexity = self.calculate_complexity_score(text)
        toxicity = self.calculate_toxicity_score(text)

        # 論理的誤りの検出
        has_logical_errors = any(re.search(pattern, text.lower())
                               for pattern in self.logical_error_patterns)

        # タグ決定ロジック
        if toxicity > self.config.toxicity_threshold:
            # 高毒性コンテンツ → refuse
            return "<|refuse|>"
        elif has_logical_errors and is_science:
            # 科学データでの論理誤り → deny
            return "<|deny|>"
        elif complexity > self.config.complexity_threshold:
            # 高複雑度 → escalation (深い推論が必要)
            return "<|escalation|>"
        else:
            # それ以外 → allow (単純タスク)
            return "<|allow|>"

    def format_sample(self, sample: Dict[str, Any], tag: str) -> Dict[str, Any]:
        """サンプルをPhi-3.5形式に整形"""
        instruction = sample.get('instruction', sample.get('input', ''))
        output = sample.get('output', sample.get('response', ''))

        if not instruction or not output:
            return None

        # タグ付きフォーマット
        formatted_instruction = f"{tag}\n{instruction}"
        formatted_output = f"{tag}\n{output}"

        return {
            'instruction': formatted_instruction,
            'input': '',  # Phi-3.5形式では空
            'output': formatted_output,
            'tag': tag,
            'complexity_score': self.calculate_complexity_score(instruction + " " + output),
            'toxicity_score': self.calculate_toxicity_score(instruction + " " + output)
        }

class AEGISDataCurator:
    """AEGIS-v2.0データキュレーター"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.tagger = NKATAutoTagger(config)
        self.samples = []

        # トークナイザー初期化
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"トークナイザー読み込み失敗: {e}")
            self.tokenizer = None

        # NLTK punkt リソースのダウンロード
        try:
            nltk.download('punkt', quiet=True)
            logger.info("NLTK punkt downloaded successfully")
        except Exception as e:
            logger.warning(f"NLTK punkt download failed: {e}")
            # フォールバックとしてpunkt_tabを試す
            try:
                nltk.download('punkt_tab', quiet=True)
                logger.info("NLTK punkt_tab downloaded successfully")
            except Exception as e2:
                logger.warning(f"NLTK punkt_tab download also failed: {e2}")

    def load_science_datasets(self) -> List[Dict[str, Any]]:
        """科学データセットを読み込み"""
        logger.info("Loading science datasets...")
        science_samples = []

        for dataset_name in self.config.science_datasets:
            try:
                logger.info(f"Loading {dataset_name}...")

                if dataset_name == "AI-MO/NuminaMath-CoT":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                    except:
                        dataset = load_dataset(dataset_name, split='default')
                    for item in dataset:
                        problem = item.get('problem', item.get('question', ''))
                        solution = item.get('solution', item.get('answer', ''))
                        if self._passes_quality_filter(problem, is_science=True):
                            tag = self.tagger.classify_inference_type(problem, is_science=True)
                            formatted = self.tagger.format_sample({
                                'instruction': problem,
                                'output': solution
                            }, tag)
                            if formatted:
                                science_samples.append(formatted)

                elif dataset_name == "gsm8k":
                    try:
                        dataset = load_dataset(dataset_name, 'main', split='train')
                    except:
                        dataset = load_dataset(dataset_name, split='train')
                    for item in dataset:
                        question = item.get('question', '')
                        answer = item.get('answer', '')
                        if self._passes_quality_filter(question, is_science=True):
                            tag = self.tagger.classify_inference_type(question, is_science=True)
                            formatted = self.tagger.format_sample({
                                'instruction': question,
                                'output': answer
                            }, tag)
                            if formatted:
                                science_samples.append(formatted)

                elif dataset_name == "camel-ai/physics":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                        for item in dataset:
                            problem = item.get('instruction', item.get('input', ''))
                            solution = item.get('output', item.get('response', ''))
                            if self._passes_quality_filter(problem, is_science=True):
                                tag = self.tagger.classify_inference_type(problem, is_science=True)
                                formatted = self.tagger.format_sample({
                                    'instruction': problem,
                                    'output': solution
                                }, tag)
                                if formatted:
                                    science_samples.append(formatted)
                    except Exception as e:
                        logger.warning(f"Failed to load physics dataset: {e}")

                elif dataset_name == "camel-ai/math":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                        for item in dataset:
                            problem = item.get('instruction', item.get('input', ''))
                            solution = item.get('output', item.get('response', ''))
                            if self._passes_quality_filter(problem, is_science=True):
                                tag = self.tagger.classify_inference_type(problem, is_science=True)
                                formatted = self.tagger.format_sample({
                                    'instruction': problem,
                                    'output': solution
                                }, tag)
                                if formatted:
                                    science_samples.append(formatted)
                    except Exception as e:
                        logger.warning(f"Failed to load math dataset: {e}")

                logger.info(f"Loaded {len(science_samples)} science samples so far")

            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue

        return science_samples

    def load_japanese_datasets(self) -> List[Dict[str, Any]]:
        """日本語データセットを読み込み"""
        logger.info("Loading Japanese datasets...")
        japanese_samples = []

        for dataset_name in self.config.japanese_datasets:
            try:
                logger.info(f"Loading {dataset_name}...")

                if dataset_name == "elyza/ELYZA-tasks-100":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                    except:
                        dataset = load_dataset(dataset_name, split='test')  # ELYZA-tasks-100 has 'test' split
                    for item in dataset:
                        input_text = item.get('input', '')
                        output_text = item.get('output', '')
                        if self._passes_quality_filter(input_text, is_japanese=True):
                            tag = self.tagger.classify_inference_type(input_text, is_japanese=True)
                            formatted = self.tagger.format_sample({
                                'instruction': input_text,
                                'output': output_text
                            }, tag)
                            if formatted:
                                japanese_samples.append(formatted)

                elif dataset_name == "izumi-lab/llm-japanese-dataset":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                        for item in dataset:
                            if 'instruction' in item and 'output' in item:
                                input_text = item['instruction']
                                output_text = item['output']
                                if self._passes_quality_filter(input_text, is_japanese=True):
                                    tag = self.tagger.classify_inference_type(input_text, is_japanese=True)
                                    formatted = self.tagger.format_sample({
                                        'instruction': input_text,
                                        'output': output_text
                                    }, tag)
                                    if formatted:
                                        japanese_samples.append(formatted)
                    except Exception as e:
                        logger.warning(f"Failed to load izumi-lab dataset: {e}")

                elif dataset_name == "llm-jp/magpie-pro-200k-ja":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                        for item in dataset:
                            input_text = item.get('instruction', '')
                            output_text = item.get('output', '')
                            if self._passes_quality_filter(input_text, is_japanese=True):
                                tag = self.tagger.classify_inference_type(input_text, is_japanese=True)
                                formatted = self.tagger.format_sample({
                                    'instruction': input_text,
                                    'output': output_text
                                }, tag)
                                if formatted:
                                    japanese_samples.append(formatted)
                    except Exception as e:
                        logger.warning(f"Failed to load magpie dataset: {e}")

                elif dataset_name == "hotchpotch/japanese-novel-instructions":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                        for item in dataset:
                            input_text = item.get('instruction', '')
                            output_text = item.get('output', '')
                            if self._passes_quality_filter(input_text, is_japanese=True):
                                tag = self.tagger.classify_inference_type(input_text, is_japanese=True)
                                formatted = self.tagger.format_sample({
                                    'instruction': input_text,
                                    'output': output_text
                                }, tag)
                                if formatted:
                                    japanese_samples.append(formatted)
                    except Exception as e:
                        logger.warning(f"Failed to load novel instructions dataset: {e}")

                logger.info(f"Loaded {len(japanese_samples)} Japanese samples so far")

            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue

        return japanese_samples

    def load_safety_datasets(self) -> List[Dict[str, Any]]:
        """安全性データセットを読み込み"""
        logger.info("Loading safety datasets...")
        safety_samples = []

        for dataset_name in self.config.safety_datasets:
            try:
                logger.info(f"Loading {dataset_name}...")

                if dataset_name == "Anthropic/hh-rlhf":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                        for item in dataset:
                            chosen = item.get('chosen', '')
                            rejected = item.get('rejected', '')

                            # 拒否サンプルはrefuseタグ
                            if rejected and self._passes_quality_filter(rejected):
                                tag = "<|refuse|>"
                                formatted = self.tagger.format_sample({
                                    'instruction': rejected,
                                    'output': "I'm sorry, but I cannot assist with this request as it may be inappropriate or harmful."
                                }, tag)
                                if formatted:
                                    safety_samples.append(formatted)

                            # 選択サンプルは適切なタグ
                            if chosen and self._passes_quality_filter(chosen):
                                tag = self.tagger.classify_inference_type(chosen)
                                formatted = self.tagger.format_sample({
                                    'instruction': chosen,
                                    'output': chosen
                                }, tag)
                                if formatted:
                                    safety_samples.append(formatted)
                    except Exception as e:
                        logger.warning(f"Failed to load hh-rlhf dataset: {e}")

                elif dataset_name == "PKU-Alignment/PKU-SafeRLHF":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                        for item in dataset:
                            prompt = item.get('prompt', '')
                            response = item.get('response', '')
                            is_safe = item.get('is_safe', True)

                            if not is_safe and self._passes_quality_filter(prompt):
                                # 安全でないプロンプト → refuse
                                tag = "<|refuse|>"
                                formatted = self.tagger.format_sample({
                                    'instruction': prompt,
                                    'output': "This request cannot be fulfilled due to safety and ethical guidelines."
                                }, tag)
                                if formatted:
                                    safety_samples.append(formatted)
                    except Exception as e:
                        logger.warning(f"Failed to load PKU-SafeRLHF dataset: {e}")

                elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                        for item in dataset:
                            instruction = item.get('instruction', '')
                            if self._passes_quality_filter(instruction):
                                tag = self.tagger.classify_inference_type(instruction)
                                formatted = self.tagger.format_sample({
                                    'instruction': instruction,
                                    'output': item.get('response', '')
                                }, tag)
                                if formatted:
                                    safety_samples.append(formatted)
                    except Exception as e:
                        logger.warning(f"Failed to load ultrafeedback dataset: {e}")

                elif dataset_name == "allenai/WildChat":
                    try:
                        dataset = load_dataset(dataset_name, split='train')
                        for item in dataset:
                            conversation = item.get('conversation', [])
                            if conversation:
                                # 会話の最初のメッセージを使用
                                first_msg = conversation[0] if isinstance(conversation, list) else str(conversation)
                                if isinstance(first_msg, dict):
                                    first_msg = first_msg.get('content', '')
                                if self._passes_quality_filter(str(first_msg)):
                                    tag = self.tagger.classify_inference_type(str(first_msg))
                                    formatted = self.tagger.format_sample({
                                        'instruction': str(first_msg),
                                        'output': item.get('response', '')
                                    }, tag)
                                    if formatted:
                                        safety_samples.append(formatted)
                    except Exception as e:
                        logger.warning(f"Failed to load WildChat dataset: {e}")

                logger.info(f"Loaded {len(safety_samples)} safety samples so far")

            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue

        return safety_samples

    def load_local_datasets(self) -> List[Dict[str, Any]]:
        """ローカルデータセットを読み込み"""
        logger.info("Loading local datasets...")
        local_samples = []

        for dataset_path in self.config.local_datasets:
            try:
                logger.info(f"Loading local dataset: {dataset_path}")

                if os.path.exists(dataset_path):
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            if line.strip():
                                try:
                                    sample = json.loads(line.strip())

                                    # フォーマットを統一
                                    if 'text' in sample:
                                        # integrated_dataset.jsonl形式
                                        text = sample['text']
                                        language = sample.get('language', 'en')
                                        dataset_name = sample.get('dataset', 'local_integrated')

                                        # 空のテキストをスキップ
                                        if not text or len(text.strip()) < 10:
                                            continue

                                        # instruction/output形式に変換
                                        try:
                                            if language == 'ja':
                                                # 日本語データは適切なタグ付け
                                                tag = self.tagger.classify_inference_type(text, is_japanese=True)
                                            else:
                                                # 英語データは適切なタグ付け
                                                tag = self.tagger.classify_inference_type(text, is_science=True)

                                            formatted = {
                                                'instruction': text[:500],  # 最初の500文字をinstruction
                                                'output': text,
                                                'tag': tag,
                                                'complexity_score': self.tagger.calculate_complexity_score(text),
                                                'toxicity_score': self.tagger.calculate_toxicity_score(text),
                                                'source': dataset_name,
                                                'language': language
                                            }

                                            if self._passes_quality_filter(text):
                                                local_samples.append(formatted)
                                        except Exception as e:
                                            logger.warning(f"Failed to process sample from {dataset_name}: {e}")
                                            continue

                                    elif 'instruction' in sample and 'output' in sample:
                                        # 既に整形済みデータ
                                        instruction = sample['instruction']
                                        output = sample['output']

                                        # 言語判定
                                        is_japanese = any(ord(char) > 0x3000 for char in instruction + output)

                                        if is_japanese:
                                            tag = self.tagger.classify_inference_type(instruction, is_japanese=True)
                                        else:
                                            tag = self.tagger.classify_inference_type(instruction, is_science=True)

                                        formatted = {
                                            'instruction': instruction,
                                            'output': output,
                                            'tag': tag,
                                            'complexity_score': self.tagger.calculate_complexity_score(instruction + " " + output),
                                            'toxicity_score': self.tagger.calculate_toxicity_score(instruction + " " + output),
                                            'source': sample.get('source', dataset_path),
                                            'language': 'ja' if is_japanese else 'en'
                                        }

                                        if self._passes_quality_filter(instruction):
                                            local_samples.append(formatted)

                                    # サンプル数制限（各ファイルから最大1000サンプル）
                                    if len(local_samples) >= 1000:
                                        break

                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSON decode error in {dataset_path} line {line_num}: {e}")
                                    continue

                    logger.info(f"Loaded {len(local_samples)} samples from {dataset_path}")

                else:
                    logger.warning(f"Local dataset not found: {dataset_path}")

            except Exception as e:
                logger.warning(f"Failed to load local dataset {dataset_path}: {e}")
                continue

        return local_samples

    def _passes_quality_filter(self, text: str, is_science: bool = False,
                             is_japanese: bool = False) -> bool:
        """品質フィルタリング"""
        if not text or len(text) < self.config.min_length:
            return False

        # トークン長チェック
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, truncation=True, max_length=self.config.max_length + 1)
                if len(tokens) > self.config.max_length:
                    return False
            except Exception as e:
                logger.warning(f"トークナイザーエンコードエラー: {e}")
                # フォールバック: 文字数チェック
                if len(text) > self.config.max_length * 4:  # 概算
                    return False
        else:
            # トークナイザーがない場合のフォールバック
            if len(text) > self.config.max_length * 4:  # 概算
                return False

        # LaTeX密度チェック（科学データのみ）
        if is_science:
            latex_density = len(self.tagger.latex_pattern.findall(text)) / len(text.split()) if text.split() else 0
            if not (self.config.min_latex_density <= latex_density <= self.config.max_latex_density):
                return False

        # 複雑度チェック
        complexity = self.tagger.calculate_complexity_score(text)
        if complexity < self.config.min_complexity_score:
            return False

        return True

    def curate_dataset(self) -> Dataset:
        """データセットを作成"""
        logger.info("Starting AEGIS-v2.0 data curation...")

        # データ読み込み
        science_samples = self.load_science_datasets()
        japanese_samples = self.load_japanese_datasets()
        safety_samples = self.load_safety_datasets()
        local_samples = self.load_local_datasets()

        logger.info(f"Science samples: {len(science_samples)}")
        logger.info(f"Japanese samples: {len(japanese_samples)}")
        logger.info(f"Safety samples: {len(safety_samples)}")
        logger.info(f"Local samples: {len(local_samples)}")

        # サンプリング
        remaining_ratio = 1.0 - self.config.local_ratio
        target_science = int(self.config.total_samples * self.config.science_ratio * remaining_ratio)
        target_japanese = int(self.config.total_samples * self.config.japanese_ratio * remaining_ratio)
        target_safety = int(self.config.total_samples * self.config.safety_ratio * remaining_ratio)
        target_local = int(self.config.total_samples * self.config.local_ratio)

        # ランダムサンプリング
        np.random.shuffle(science_samples)
        np.random.shuffle(japanese_samples)
        np.random.shuffle(safety_samples)
        np.random.shuffle(local_samples)

        selected_samples = (
            science_samples[:target_science] +
            japanese_samples[:target_japanese] +
            safety_samples[:target_safety] +
            local_samples[:target_local]
        )

        np.random.shuffle(selected_samples)

        logger.info(f"Final dataset size: {len(selected_samples)}")

        # Datasetオブジェクト作成
        dataset = Dataset.from_list(selected_samples)

        # 統計情報
        tag_counts = {}
        for sample in selected_samples:
            tag = sample.get('tag', '<|allow|>')
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        logger.info("Tag distribution:")
        for tag, count in tag_counts.items():
            logger.info(f"  {tag}: {count}")

        return dataset

    def save_dataset(self, dataset: Dataset, output_dir: str):
        """データセットを保存"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 学習/検証分割
        split_dataset = dataset.train_test_split(test_size=self.config.val_split, seed=42)

        # JSONL形式で保存
        def save_split(split_data, filename):
            with open(output_path / filename, 'w', encoding='utf-8') as f:
                for sample in split_data:
                    json.dump(sample, f, ensure_ascii=False)
                    f.write('\n')

        save_split(split_dataset['train'], 'train_aegis_v2.jsonl')
        save_split(split_dataset['test'], 'val_aegis_v2.jsonl')

        # 統計情報保存
        # 統計情報の作成
        stats = {}

        # データ数
        stats['total_samples'] = len(dataset)
        stats['train_samples'] = len(split_dataset['train'])
        stats['val_samples'] = len(split_dataset['test'])

        # 元データの各カテゴリごとのサンプル数
        stats['original_counts'] = {}
        stats['original_counts']['science'] = len(getattr(self, 'science_samples', []))
        stats['original_counts']['japanese'] = len(getattr(self, 'japanese_samples', []))
        stats['original_counts']['safety'] = len(getattr(self, 'safety_samples', []))
        stats['original_counts']['local'] = len(getattr(self, 'local_samples', []))

        # 選択されたデータのカテゴリごとのサンプル数
        stats['selected_counts'] = {}
        stats['selected_counts']['science'] = sum(1 for s in dataset if s.get('source', '') == 'science')
        stats['selected_counts']['japanese'] = sum(1 for s in dataset if s.get('source', '') == 'japanese')
        stats['selected_counts']['safety'] = sum(1 for s in dataset if s.get('source', '') == 'safety')
        stats['selected_counts']['local'] = sum(1 for s in dataset if s.get('source', '') == 'local')

        # タグ分布（後で補完される）
        stats['tag_distribution'] = {}

        # config情報
        stats['config'] = {
            'science_ratio': getattr(self.config, 'science_ratio', None),
            'japanese_ratio': getattr(self.config, 'japanese_ratio', None),
            'safety_ratio': getattr(self.config, 'safety_ratio', None),
            'local_ratio': getattr(self.config, 'local_ratio', None),
            'complexity_threshold': getattr(self.config, 'complexity_threshold', None),
            'toxicity_threshold': getattr(self.config, 'toxicity_threshold', None)
        }

        # タグ分布計算
        for sample in dataset:
            tag = sample.get('tag', '<|allow|>')
            stats['tag_distribution'][tag] = stats['tag_distribution'].get(tag, 0) + 1

        with open(output_path / 'dataset_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Train samples: {len(split_dataset['train'])}")
        logger.info(f"Validation samples: {len(split_dataset['test'])}")


def main():
    parser = argparse.ArgumentParser(description='AEGIS-v2.0 Data Curation')
    parser.add_argument('--output_dir', type=str, default='data/aegis_v2_dataset',
                       help='Output directory')
    parser.add_argument('--total_samples', type=int, default=50000,
                       help='Total number of samples')
    parser.add_argument('--science_ratio', type=float, default=0.4,
                       help='Science data ratio')
    parser.add_argument('--japanese_ratio', type=float, default=0.4,
                       help='Japanese data ratio')
    parser.add_argument('--safety_ratio', type=float, default=0.2,
                       help='Safety data ratio')

    args = parser.parse_args()

    # 設定更新
    config = DataConfig()
    config.output_dir = args.output_dir
    config.total_samples = args.total_samples
    config.science_ratio = args.science_ratio
    config.japanese_ratio = args.japanese_ratio
    config.safety_ratio = args.safety_ratio

    # データキュレーター実行
    curator = AEGISDataCurator(config)
    dataset = curator.curate_dataset()
    curator.save_dataset(dataset, config.output_dir)

    logger.info("AEGIS-v2.0 data curation completed!")


if __name__ == "__main__":
    main()
