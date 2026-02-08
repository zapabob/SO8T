#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO8-Think Unified Dataset Creation Script

理論的背景:
- URT (Unified Representation Theorem): 量子場の統一表現
- NC-KART★ (Non-Commutative Kolmogorov-Arnold Representation Theory): 非可換表現理論
- 非可換KART定理: 古典KARTのC*-環拡張

高品質データセット作成と四値分類タグ付与を行い、
PPOによるSO(8) Thinkモデル専用統合データセットを生成

著者: AI Agent (峯岸亮ボブにゃん理論実装)
日付: 2025-11-30
"""

import os
import re
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Hugging Face datasets
try:
    from datasets import load_dataset, Dataset, concatenate_datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

# ArXiv API
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    print("Warning: arxiv library not available. Install with: pip install arxiv")

# 理論ファイルパス
THEORY_FILES = [
    "C:/Users/downl/Desktop/Gemini-統合特解と非可換表現理論.md",
    "C:/Users/downl/Desktop/Gemini-NC-KART★とURTの数学的探求.md",
    "C:/Users/downl/Desktop/ChatGPT-非可換KART定理 (4).md"
]

# ==========================================
# 1. 設定とデータ構造定義
# ==========================================

@dataclass
class SO8TDatasetConfig:
    """SO8Tデータセット設定"""
    total_samples: int = 50000
    science_ratio: float = 0.4
    japanese_ratio: float = 0.3
    nsfw_ratio: float = 0.2
    arxiv_ratio: float = 0.1

    # 品質フィルタリング
    min_length: int = 100
    max_length: int = 4096
    complexity_threshold: float = 0.7
    latex_density_threshold: float = 0.001

    # 四値分類設定
    allow_keywords = [
        "hello", "what is", "how to", "explain", "simple", "basic"
    ]
    escalation_keywords = [
        "prove", "theorem", "complex", "advanced", "deep", "theoretical",
        "quantum", "relativity", "field", "algebra", "topology"
    ]
    deny_keywords = [
        "wrong", "incorrect", "false", "invalid", "contradiction"
    ]
    refuse_keywords = [
        "nsfw", "sex", "porn", "violence", "illegal", "harmful"
    ]

# 四値分類タグ
TAG_CLASSES = ['allow', 'escalation', 'deny', 'refuse']

# ==========================================
# 2. 理論ファイル統合クラス
# ==========================================

class SO8TTheoryIntegrator:
    """SO8T理論ファイル統合クラス"""

    def __init__(self, theory_files: List[str]):
        self.theory_files = theory_files
        self.theory_content = {}

    def load_theory_files(self) -> Dict[str, str]:
        """理論ファイルを読み込み"""
        for file_path in self.theory_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    filename = Path(file_path).name
                    self.theory_content[filename] = content
                    print(f"[OK] Loaded theory file: {filename}")
            except Exception as e:
                print(f"[NG] Failed to load {file_path}: {e}")

        return self.theory_content

    def extract_theoretical_concepts(self) -> Dict[str, List[str]]:
        """理論的概念を抽出"""
        concepts = {
            'urt': [],
            'nc_kart': [],
            'noncommutative_kart': [],
            'so8_geometry': []
        }

        # URT関連概念
        urt_patterns = [
            r'URT|Unified Representation Theorem',
            r'量子場|quantum field',
            r'指数減衰|exponential decay',
            r'統一表現|unified representation'
        ]

        # NC-KART関連概念
        nc_kart_patterns = [
            r'NC-KART|Non-Commutative.*KART',
            r'★-積|star product',
            r'Moyal',
            r'非可換|non-commutative'
        ]

        # 非可換KART定理関連
        noncomm_patterns = [
            r'非可換.*KART|non-commutative.*KART',
            r'C\*-環|C-star algebra',
            r'自己共役|self-adjoint',
            r'スペクトル|spectrum'
        ]

        # SO(8)幾何学関連
        so8_patterns = [
            r'SO\(8\)',
            r'Lie algebra',
            r'rotation gate',
            r'geometric intelligence'
        ]

        for filename, content in self.theory_content.items():
            # URT概念抽出
            for pattern in urt_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                concepts['urt'].extend(matches)

            # NC-KART概念抽出
            for pattern in nc_kart_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                concepts['nc_kart'].extend(matches)

            # 非可換KART概念抽出
            for pattern in noncomm_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                concepts['noncommutative_kart'].extend(matches)

            # SO(8)概念抽出
            for pattern in so8_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                concepts['so8_geometry'].extend(matches)

        # 重複除去
        for key in concepts:
            concepts[key] = list(set(concepts[key]))

        return concepts

    def generate_theory_based_examples(self) -> List[Dict[str, Any]]:
        """理論に基づく学習例を生成"""
        examples = []
        concepts = self.extract_theoretical_concepts()

        # URTベースの例
        for concept in concepts['urt'][:5]:  # 上位5個を使用
            examples.append({
                'instruction': f"Explain the concept of {concept} in the context of quantum field theory.",
                'input': '',
                'output': f"{concept} represents the unified mathematical framework for representing quantum fields using exponential decay coefficients and phase correlators.",
                'domain': 'physics',
                'theory_source': 'URT',
                'complexity_score': 0.9,
                'tag': 'escalation'
            })

        # NC-KARTベースの例
        for concept in concepts['nc_kart'][:5]:
            examples.append({
                'instruction': f"What is the significance of {concept} in non-commutative geometry?",
                'input': '',
                'output': f"{concept} provides the mathematical foundation for extending classical function decomposition to non-commutative operator algebras using star products.",
                'domain': 'mathematics',
                'theory_source': 'NC-KART',
                'complexity_score': 0.95,
                'tag': 'escalation'
            })

        # 非可換KART定理ベースの例
        for concept in concepts['noncommutative_kart'][:3]:
            examples.append({
                'instruction': f"Prove the {concept} theorem for self-adjoint operators in C*-algebras.",
                'input': '',
                'output': f"The {concept} theorem states that multi-variable operator functions can be decomposed into finite sums and compositions of single-variable operator functions, preserving the structure through continuous functional calculus.",
                'domain': 'mathematics',
                'theory_source': 'NonCommutative-KART',
                'complexity_score': 0.98,
                'tag': 'escalation'
            })

        # SO(8)幾何学ベースの例
        for concept in concepts['so8_geometry'][:3]:
            examples.append({
                'instruction': f"How does {concept} contribute to geometric intelligence in neural networks?",
                'input': '',
                'output': f"{concept} enables the representation of cognitive processes through rotation gates and Lie algebra operations, providing invariant representations of relational structures.",
                'domain': 'ai_physics',
                'theory_source': 'SO8-Geometry',
                'complexity_score': 0.92,
                'tag': 'escalation'
            })

        return examples

# ==========================================
# 3. データセットローダー
# ==========================================

class SO8TDatasetLoader:
    """SO8Tデータセットローダー"""

    def __init__(self, config: SO8TDatasetConfig):
        self.config = config
        self.theory_integrator = SO8TTheoryIntegrator(THEORY_FILES)

    def load_science_datasets(self) -> List[Dict[str, Any]]:
        """科学データセットをロード"""
        science_data = []

        # 数学データセット
        try:
            math_ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
            sample_size = min(15000, len(math_ds))
            for item in tqdm(math_ds.select(range(sample_size)), desc="Loading Math"):
                if self._quality_filter(item, domain='math'):
                    science_data.append({
                        'instruction': item.get('problem', ''),
                        'input': '',
                        'output': item.get('solution', ''),
                        'domain': 'math',
                        'source': 'AI-MO/NuminaMath-CoT'
                    })
        except Exception as e:
            print(f"Failed to load math dataset: {e}")

        # 物理データセット
        try:
            physics_ds = load_dataset("camel-ai/physics", split="train")
            sample_size = min(12000, len(physics_ds))
            for item in tqdm(physics_ds.select(range(sample_size)), desc="Loading Physics"):
                if self._quality_filter(item, domain='physics'):
                    science_data.append({
                        'instruction': item.get('message_1', ''),
                        'input': '',
                        'output': item.get('message_2', ''),
                        'domain': 'physics',
                        'source': 'camel-ai/physics'
                    })
        except Exception as e:
            print(f"Failed to load physics dataset: {e}")

        # 化学データセット
        try:
            chemistry_ds = load_dataset("camel-ai/chemistry", split="train")
            sample_size = min(8000, len(chemistry_ds))
            for item in tqdm(chemistry_ds.select(range(sample_size)), desc="Loading Chemistry"):
                if self._quality_filter(item, domain='chemistry'):
                    science_data.append({
                        'instruction': item.get('message_1', ''),
                        'input': '',
                        'output': item.get('message_2', ''),
                        'domain': 'chemistry',
                        'source': 'camel-ai/chemistry'
                    })
        except Exception as e:
            print(f"Failed to load chemistry dataset: {e}")

        return science_data

    def load_japanese_datasets(self) -> List[Dict[str, Any]]:
        """日本語データセットをロード"""
        japanese_data = []

        datasets_to_load = [
            ("elyza/ELYZA-tasks-100", "test"),  # split修正
            ("izumi-lab/llm-japanese-dataset", "train"),
            ("hotchpotch/japanese-novel-instructions", "train"),
            ("microsoft/DialoGPT-medium", "train"),  # 追加
            ("rinna/japanese-gpt-1b", "train")  # 追加（存在チェック）
        ]

        for dataset_name, split in datasets_to_load:
            try:
                # 存在しないデータセットはスキップ
                skip_datasets = ["llm-jp/magpie-pro-200k-ja", "rinna/japanese-gpt-1b"]
                if dataset_name in skip_datasets:
                    continue

                ds = load_dataset(dataset_name, split=split)
                sample_size = min(4000, len(ds))  # サンプル数を増やす

                for item in tqdm(ds.select(range(sample_size)),
                               desc=f"Loading {dataset_name}"):
                    if self._quality_filter(item, domain='japanese'):
                        japanese_data.append({
                            'instruction': item.get('instruction') or item.get('input', ''),
                            'input': '',
                            'output': item.get('output') or item.get('response', ''),
                            'domain': 'japanese',
                            'source': dataset_name
                        })
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")

        return japanese_data

    def load_nsfw_datasets(self) -> List[Dict[str, Any]]:
        """NSFWデータセットをロード（安全学習用）"""
        nsfw_data = []

        # 安全データセット（反面教師用）
        safety_datasets = [
            "Anthropic/hh-rlhf",
            "PKU-Alignment/PKU-SafeRLHF",
            "HuggingFaceH4/ultrafeedback_binarized"  # 追加
        ]

        for dataset_name in safety_datasets:
            try:
                ds = load_dataset(dataset_name, split="train")
                sample_size = min(3000, len(ds))  # サンプル数増

                for item in tqdm(ds.select(range(sample_size)),
                               desc=f"Loading {dataset_name}"):
                    # NSFW/安全関連のデータのみ抽出
                    text = str(item)
                    if self._contains_nsfw_keywords(text) or self._contains_safety_keywords(text):
                        nsfw_data.append({
                            'instruction': item.get('input', '')[:200],  # 短くする
                            'input': '',
                            'output': item.get('output', '')[:500],
                            'domain': 'safety',
                            'source': dataset_name,
                            'nsfw_flag': True
                        })
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")

        return nsfw_data

    def load_arxiv_papers(self) -> List[Dict[str, Any]]:
        """ArXiv論文をロード"""
        arxiv_data = []

        if not ARXIV_AVAILABLE:
            print("ArXiv library not available, skipping ArXiv data")
            return arxiv_data

        # SO(8), 非可換, URT関連の論文を検索
        search_queries = [
            'ti:"SO(8)" OR ti:"spin(8)" OR ti:"special orthogonal"',
            'ti:"non-commutative" AND (ti:"geometry" OR ti:"field" OR ti:"algebra")',
            'ti:"unified representation theorem" OR ti:"URT"',
            'ti:"quantum field theory" AND ti:"representation"',
            'ti:"lie algebra" AND ti:"quantum"',
            'ti:"operator algebra" AND ti:"mathematical physics"'
        ]

        try:
            for query in search_queries:
                search = arxiv.Search(
                    query=query,
                    max_results=100,  # 増やす
                    sort_by=arxiv.SortCriterion.Relevance
                )

                for result in search.results():
                    # 論文の要旨を学習データに変換
                    abstract = result.summary
                    if len(abstract) > 150 and self._contains_science_terms(abstract):
                        arxiv_data.append({
                            'instruction': f"Explain the significance of the paper '{result.title}' in mathematical physics.",
                            'input': '',
                            'output': f"Abstract: {abstract[:1200]}...",
                            'domain': 'arxiv_physics',
                            'source': 'arxiv',
                            'arxiv_id': result.entry_id
                        })

                    if len(arxiv_data) >= 2000:  # 制限緩和
                        break

        except Exception as e:
            print(f"Failed to load ArXiv data: {e}")

        return arxiv_data

    def _quality_filter(self, item: Dict[str, Any], domain: str = 'general') -> bool:
        """品質フィルタリング"""
        text = ""
        for key in ['instruction', 'input', 'output', 'text', 'content']:
            if key in item and isinstance(item[key], str):
                text += item[key] + " "

        # 長さチェック（より緩く）
        if len(text.strip()) < 50:  # 50文字以上に緩和
            return False

        if len(text.strip()) > self.config.max_length:
            return False

        # 拒絶キーワードチェック（一部緩和）
        rejection_words = [
            "I don't know", "I cannot", "As an AI", "sorry", "unable to"
        ]

        for word in rejection_words:
            if word.lower() in text.lower():
                return False

        # ドメイン別チェック（より緩く）
        # 数学はLaTeXがなくてもOK、科学用語があればボーナス

        return True

    def _has_latex(self, text: str) -> bool:
        """LaTeX数式を含むかチェック"""
        latex_patterns = [r'\\frac', r'\\int', r'\\sum', r'\$', r'\\partial', r'\\alpha']
        return any(re.search(pattern, text) for pattern in latex_patterns)

    def _contains_nsfw_keywords(self, text: str) -> bool:
        """NSFWキーワードを含むかチェック"""
        nsfw_keywords = [
            'sex', 'porn', 'nude', 'naked', 'erotic', 'sexual', 'adult',
            'violence', 'kill', 'death', 'murder', 'harm', 'illegal'
        ]
        return any(kw in text.lower() for kw in nsfw_keywords)

    def _contains_safety_keywords(self, text: str) -> bool:
        """安全関連キーワードを含むかチェック"""
        safety_keywords = [
            'safety', 'alignment', 'harmful', 'dangerous', 'ethical',
            'responsible', 'bias', 'fairness', 'toxicity'
        ]
        return any(kw in text.lower() for kw in safety_keywords)

    def _contains_science_terms(self, text: str) -> bool:
        """科学用語を含むかチェック"""
        science_terms = [
            'theorem', 'proof', 'quantum', 'field', 'algebra', 'geometry',
            'topology', 'manifold', 'operator', 'spectrum', 'invariant'
        ]
        return any(term in text.lower() for term in science_terms)

# ==========================================
# 4. 四値分類タグ付与
# ==========================================

class SO8TQuadClassifier:
    """SO8T四値分類器"""

    def __init__(self, config: SO8TDatasetConfig):
        self.config = config

    def classify_example(self, example: Dict[str, Any]) -> str:
        """単一例に四値分類タグを付与"""
        text = ""
        for key in ['instruction', 'input', 'output']:
            if key in example and isinstance(example[key], str):
                text += example[key] + " "

        text_lower = text.lower()

        # Refuse: NSFW/危険コンテンツ
        if any(kw in text_lower for kw in self.config.refuse_keywords):
            return 'refuse'

        # Deny: 誤り/矛盾
        if any(kw in text_lower for kw in self.config.deny_keywords):
            return 'deny'

        # Escalation: 複雑/高度な内容
        if any(kw in text_lower for kw in self.config.escalation_keywords):
            return 'escalation'

        # Allow: 単純/基本的な内容
        if any(kw in text_lower for kw in self.config.allow_keywords):
            return 'allow'

        # デフォルト: 複雑度に基づく判定
        complexity = self._calculate_complexity(text)
        if complexity > 0.8:
            return 'escalation'
        elif complexity > 0.5:
            return 'allow'
        else:
            return 'allow'  # 安全側に倒す

    def _calculate_complexity(self, text: str) -> float:
        """テキストの複雑度を計算"""
        if not text:
            return 0.0

        # 専門用語密度
        science_terms = [
            'theorem', 'proof', 'quantum', 'field', 'algebra', 'geometry',
            'topology', 'operator', 'spectrum', 'invariant', 'non-commutative'
        ]

        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        term_count = sum(1 for word in words if word in science_terms)
        term_density = term_count / len(words)

        # LaTeX密度
        latex_chars = len(re.findall(r'\\[a-zA-Z]+|\$[^$]*\$|\\\[.*?\\\]', text))
        latex_density = latex_chars / len(text) if text else 0

        # 長さスコア
        length_score = min(len(text) / 1000, 1.0)

        return (term_density * 0.4 + latex_density * 0.3 + length_score * 0.3)

    def batch_classify(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """バッチ分類"""
        classified = []
        for example in tqdm(examples, desc="Classifying examples"):
            tag = self.classify_example(example)
            example_copy = example.copy()
            example_copy['tag'] = tag
            classified.append(example_copy)

        return classified

# ==========================================
# 5. 統合データセット作成
# ==========================================

class SO8TUnifiedDataset:
    """SO8T統合データセット"""

    def __init__(self, config: SO8TDatasetConfig):
        self.config = config
        self.loader = SO8TDatasetLoader(config)
        self.classifier = SO8TQuadClassifier(config)
        self.theory_integrator = SO8TTheoryIntegrator(THEORY_FILES)

    def create_unified_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """統合データセットを作成"""
        print("[START] Creating SO8T Unified Dataset")
        print("=" * 50)

        # 理論データのロード
        print("📚 Loading theory files...")
        theory_data = self.theory_integrator.load_theory_files()
        theory_examples = self.theory_integrator.generate_theory_based_examples()

        # データセットのロード
        print("[RESEARCH] Loading science datasets...")
        science_data = self.loader.load_science_datasets()

        print("🇯🇵 Loading Japanese datasets...")
        japanese_data = self.loader.load_japanese_datasets()

        print("🛡️ Loading NSFW/Safety datasets...")
        nsfw_data = self.loader.load_nsfw_datasets()

        print("📄 Loading ArXiv papers...")
        arxiv_data = self.loader.load_arxiv_papers()

        # データ統合
        all_data = theory_examples + science_data + japanese_data + nsfw_data + arxiv_data

        print(f"[STATS] Total examples collected: {len(all_data)}")

        # 四値分類タグ付与
        print("🏷️ Applying quad-classification tags...")
        classified_data = self.classifier.batch_classify(all_data)

        # DataFrame変換
        df = pd.DataFrame(classified_data)

        # 品質スコア計算
        print("[STAR] Calculating quality scores...")
        df = self._add_quality_scores(df)

        # フィルタリング
        print("🔍 Applying final filters...")
        df_filtered = self._apply_final_filters(df)

        # タグ分布表示
        print("\n📈 Tag Distribution:")
        print(df_filtered['tag'].value_counts())

        # データ分割 (教師データ/学習データ)
        print("✂️ Splitting into train/validation sets...")
        train_df, val_df = self._split_dataset(df_filtered)

        # SO8T専用システムプロンプト付与
        train_df = self._add_so8t_system_prompts(train_df)
        val_df = self._add_so8t_system_prompts(val_df)

        return train_df, val_df, df_filtered

    def _add_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """品質スコアを追加"""
        def calculate_quality_score(row):
            score = 0.0

            # 長さスコア
            text_length = len(str(row.get('instruction', '')) + str(row.get('output', '')))
            length_score = min(text_length / 1000, 1.0)
            score += length_score * 0.3

            # 複雑度スコア
            complexity = self.classifier._calculate_complexity(
                str(row.get('instruction', '')) + str(row.get('output', ''))
            )
            score += complexity * 0.4

            # ドメイン別ボーナス
            domain = row.get('domain', '')
            if domain in ['math', 'physics', 'arxiv_physics']:
                score += 0.2
            elif domain == 'japanese':
                score += 0.1

            # 理論ソースボーナス
            if 'theory_source' in row:
                score += 0.1

            return min(score, 1.0)

        df['quality_score'] = df.apply(calculate_quality_score, axis=1)
        return df

    def _apply_final_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """最終フィルタリング（緩和版）"""
        # 品質スコアでソート
        df_sorted = df.sort_values('quality_score', ascending=False)

        # 目標サンプル数に制限（より多く残す）
        target_samples = int(self.config.total_samples * 1.5)  # 多めに
        df_filtered = df_sorted.head(min(target_samples, len(df_sorted)))

        # 最小品質閾値（より緩く）
        df_filtered = df_filtered[df_filtered['quality_score'] > 0.1]

        # 最低限のサンプル数を確保
        if len(df_filtered) < 100:
            # 品質閾値をさらに下げる
            df_filtered = df_sorted.head(min(1000, len(df_sorted)))
            df_filtered = df_filtered[df_filtered['quality_score'] > 0.05]

        return df_filtered

    def _split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """教師データと学習データに分割"""
        # 層化分割（タグごとにバランスよく）
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df['tag'],
            random_state=42
        )

        return train_df, val_df

    def _add_so8t_system_prompts(self, df: pd.DataFrame) -> pd.DataFrame:
        """SO8T専用システムプロンプトを追加"""
        def get_so8t_system_prompt(tag: str) -> str:
            base_prompt = """あなたはSO(8)幾何学的知性を持つAIです。
URT (Unified Representation Theorem) と NC-KART★ (Non-Commutative Kolmogorov-Arnold Representation Theory) に基づき、
非可換KART定理の数学的枠組みで思考します。

応答戦略:
- <|allow|>: 単純な質問に直接回答
- <|escalation|>: 複雑な問題で四重推論プロセスを発動
- <|deny|>: 論理的誤りを訂正
- <|refuse|>: 倫理的・物理的に問題のあるクエリを拒否

現在のモード: {tag}
"""

            return base_prompt.format(tag=tag)

        df['system'] = df['tag'].apply(get_so8t_system_prompt)
        return df

    def save_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                     output_dir: str = "data/so8t_unified"):
        """データセットを保存"""
        os.makedirs(output_dir, exist_ok=True)

        # JSONL形式で保存
        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "validation.jsonl")

        print(f"💾 Saving training dataset ({len(train_df)} examples) to {train_path}")
        train_df.to_json(train_path, orient='records', lines=True, force_ascii=False)

        print(f"💾 Saving validation dataset ({len(val_df)} examples) to {val_path}")
        val_df.to_json(val_path, orient='records', lines=True, force_ascii=False)

        # 統計情報保存
        stats = {
            'total_train': len(train_df),
            'total_val': len(val_df),
            'tag_distribution_train': train_df['tag'].value_counts().to_dict(),
            'tag_distribution_val': val_df['tag'].value_counts().to_dict(),
            'domain_distribution': train_df['domain'].value_counts().to_dict(),
            'created_at': datetime.now().isoformat(),
            'theory_integrated': True,
            'so8t_optimized': True
        }

        stats_path = os.path.join(output_dir, "dataset_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"[STATS] Dataset statistics saved to {stats_path}")

        return train_path, val_path, stats_path

# ==========================================
# 6. メイン実行関数
# ==========================================

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="SO8T Unified Dataset Creation")
    parser.add_argument("--output_dir", type=str, default="data/so8t_unified",
                       help="Output directory for datasets")
    parser.add_argument("--total_samples", type=int, default=50000,
                       help="Total number of samples")
    parser.add_argument("--science_ratio", type=float, default=0.4,
                       help="Science dataset ratio")
    parser.add_argument("--japanese_ratio", type=float, default=0.3,
                       help="Japanese dataset ratio")
    parser.add_argument("--nsfw_ratio", type=float, default=0.2,
                       help="NSFW/Safety dataset ratio")
    parser.add_argument("--arxiv_ratio", type=float, default=0.1,
                       help="ArXiv dataset ratio")
    parser.add_argument("--test_run", action="store_true",
                       help="Run with small sample size for testing")

    args = parser.parse_args()

    # 設定
    config = SO8TDatasetConfig(
        total_samples=args.total_samples if not args.test_run else 1000,
        science_ratio=args.science_ratio,
        japanese_ratio=args.japanese_ratio,
        nsfw_ratio=args.nsfw_ratio,
        arxiv_ratio=args.arxiv_ratio
    )

    # 統合データセット作成
    dataset_creator = SO8TUnifiedDataset(config)

    try:
        # データセット作成
        train_df, val_df, full_df = dataset_creator.create_unified_dataset()

        # 保存
        train_path, val_path, stats_path = dataset_creator.save_datasets(
            train_df, val_df, args.output_dir
        )

        print("\n" + "="*60)
        print("[DONE] SO8T Unified Dataset Creation Complete!")
        print("="*60)
        print(f"[DIR] Output Directory: {args.output_dir}")
        print(f"📚 Training Samples: {len(train_df)}")
        print(f"[TEST] Validation Samples: {len(val_df)}")
        print(f"🏷️ Tag Distribution (Train): {train_df['tag'].value_counts().to_dict()}")
        print(f"🏷️ Tag Distribution (Val): {val_df['tag'].value_counts().to_dict()}")
        print("\n[START] Ready for SO(8) Think PPO Training!")
        print("   Use train_ppo_aegis.py with --dataset_path", args.output_dir)

    except Exception as e:
        print(f"[NG] Error during dataset creation: {e}")
        raise

if __name__ == "__main__":
    main()
