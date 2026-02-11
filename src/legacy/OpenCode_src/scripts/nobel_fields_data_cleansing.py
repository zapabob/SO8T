#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Nobel Fields Data Cleansing and Quad Classification
データクレンジングと四値分類システム

機能:
- 重複データ除去
- 品質スコアベースのフィルタリング
- 四値分類の精度検証
- データ正規化と標準化
- 統計分析とレポート生成
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter
import logging
from datetime import datetime
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nobel_fields_cleansing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NobelFieldsDataCleanser:
    """ノーベル賞・フィールズ賞級データクレンジング器"""

    def __init__(self, data_dir: str = "data/nobel_fields_cot"):
        self.data_dir = Path(data_dir)
        self.cleansed_dir = self.data_dir / "cleansed"
        self.cleansed_dir.mkdir(parents=True, exist_ok=True)

        # 品質閾値
        self.min_quality_score = 0.85
        self.max_similarity_threshold = 0.95  # 重複判定閾値

        # 四値分類の検証用キーワード
        self.category_keywords = {
            'mathematics': [
                'theorem', 'proof', 'conjecture', 'lemma', 'corollary', 'axiom',
                'algebra', 'topology', 'geometry', 'analysis', 'number theory',
                'combinatorics', 'graph theory', 'category theory', 'logic'
            ],
            'physics': [
                'quantum', 'relativity', 'thermodynamics', 'electromagnetism',
                'nuclear', 'particle', 'condensed matter', 'optics', 'mechanics',
                'field theory', 'symmetry', 'gauge theory', 'string theory'
            ],
            'chemistry': [
                'organic', 'inorganic', 'physical chemistry', 'quantum chemistry',
                'biochemistry', 'catalysis', 'polymer', 'materials', 'spectroscopy',
                'reaction', 'molecule', 'bond', 'crystal', 'solution'
            ],
            'biology': [
                'molecular biology', 'genetics', 'neuroscience', 'ecology',
                'evolution', 'cell biology', 'developmental biology', 'immunology',
                'microbiology', 'biochemistry', 'physiology', 'ecosystem',
                'dna', 'rna', 'protein', 'enzyme', 'gene'
            ]
        }

        logger.info(f"Initialized NobelFieldsDataCleanser with data directory: {data_dir}")

    def load_dataset(self, file_path: str) -> List[Dict]:
        """データセットの読み込み"""
        problems = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        problems.append(json.loads(line.strip()))
            logger.info(f"Loaded {len(problems)} problems from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load dataset {file_path}: {e}")
        return problems

    def save_dataset(self, problems: List[Dict], file_path: str):
        """データセットの保存"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for problem in problems:
                    json.dump(problem, f, ensure_ascii=False, indent=None)
                    f.write('\n')
            logger.info(f"Saved {len(problems)} problems to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset {file_path}: {e}")

    def remove_duplicates(self, problems: List[Dict]) -> List[Dict]:
        """重複データの除去"""
        logger.info("Removing duplicates...")

        # IDベースの重複除去
        seen_ids = set()
        unique_problems = []

        for problem in problems:
            problem_id = problem.get('id', '')
            if problem_id and problem_id not in seen_ids:
                seen_ids.add(problem_id)
                unique_problems.append(problem)

        logger.info(f"Removed {len(problems) - len(unique_problems)} duplicates by ID")

        # 内容ベースの類似度チェック
        if len(unique_problems) > 1:
            unique_problems = self._remove_similar_problems(unique_problems)

        return unique_problems

    def _remove_similar_problems(self, problems: List[Dict]) -> List[Dict]:
        """内容ベースの類似問題除去"""
        logger.info("Checking content similarity...")

        # タイトルと要約を結合したテキストを作成
        texts = []
        for problem in problems:
            text = f"{problem.get('title', '')} {problem.get('problem_statement', '')}"
            texts.append(text.lower())

        if not texts:
            return problems

        # TF-IDFベクトル化
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)

            # コサイン類似度の計算
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # 類似度の高いペアを検出して除去
            to_remove = set()
            n = len(problems)

            for i in range(n):
                if i in to_remove:
                    continue
                for j in range(i + 1, n):
                    if j in to_remove:
                        continue
                    if similarity_matrix[i, j] > self.max_similarity_threshold:
                        # 品質スコアが低い方を除去
                        score_i = problems[i].get('quality_score', 0)
                        score_j = problems[j].get('quality_score', 0)
                        if score_i >= score_j:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break

            unique_problems = [p for i, p in enumerate(problems) if i not in to_remove]
            logger.info(f"Removed {len(to_remove)} similar problems by content")

            return unique_problems

        except Exception as e:
            logger.warning(f"Content similarity check failed: {e}")
            return problems

    def filter_by_quality(self, problems: List[Dict]) -> List[Dict]:
        """品質スコアによるフィルタリング"""
        logger.info(f"Filtering by quality score >= {self.min_quality_score}...")

        filtered_problems = []
        for problem in problems:
            quality_score = problem.get('quality_score', 0)
            if quality_score >= self.min_quality_score:
                filtered_problems.append(problem)

        logger.info(f"Filtered {len(problems) - len(filtered_problems)} low-quality problems")
        return filtered_problems

    def validate_quad_classification(self, problems: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """四値分類の検証と修正"""
        logger.info("Validating quad classification...")

        corrected_problems = []
        validation_stats = {
            'total_problems': len(problems),
            'corrected_count': 0,
            'category_distribution': Counter(),
            'validation_details': []
        }

        for problem in problems:
            original_category = problem.get('category', '')
            validated_category = self._validate_category(problem)

            problem['category'] = validated_category
            corrected_problems.append(problem)

            validation_stats['category_distribution'][validated_category] += 1

            if original_category != validated_category:
                validation_stats['corrected_count'] += 1
                validation_stats['validation_details'].append({
                    'id': problem.get('id', ''),
                    'title': problem.get('title', '')[:100],
                    'original_category': original_category,
                    'corrected_category': validated_category,
                    'confidence': self._calculate_category_confidence(problem, validated_category)
                })

        logger.info(f"Classification validation completed. Corrected {validation_stats['corrected_count']} problems")
        return corrected_problems, validation_stats

    def _validate_category(self, problem: Dict) -> str:
        """個別問題のカテゴリ検証"""
        title = problem.get('title', '').lower()
        abstract = problem.get('summary', '') if 'summary' in problem else problem.get('problem_statement', '').lower()
        text = f"{title} {abstract}"

        # 各カテゴリのスコアを計算
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            # 専門用語の重み付け
            if category == 'mathematics' and any(term in text for term in ['theorem', 'proof', 'conjecture']):
                score += 2
            elif category == 'physics' and any(term in text for term in ['quantum', 'relativity', 'field']):
                score += 2
            elif category == 'chemistry' and any(term in text for term in ['molecule', 'reaction', 'bond']):
                score += 2
            elif category == 'biology' and any(term in text for term in ['dna', 'gene', 'protein', 'cell']):
                score += 2

            category_scores[category] = score

        # 最高スコアのカテゴリを返す
        if max(category_scores.values()) > 0:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            # デフォルトは元のカテゴリ
            return problem.get('category', 'mathematics')

    def _calculate_category_confidence(self, problem: Dict, category: str) -> float:
        """カテゴリ分類の信頼度計算"""
        title = problem.get('title', '').lower()
        abstract = problem.get('summary', '') if 'summary' in problem else problem.get('problem_statement', '').lower()
        text = f"{title} {abstract}"

        keywords = self.category_keywords.get(category, [])
        if not keywords:
            return 0.0

        matches = sum(1 for keyword in keywords if keyword in text)
        confidence = min(matches / len(keywords), 1.0)

        # 専門用語がある場合は信頼度を上げる
        if category == 'mathematics' and any(term in text for term in ['theorem', 'proof']):
            confidence += 0.2
        elif category == 'physics' and any(term in text for term in ['quantum', 'relativity']):
            confidence += 0.2
        elif category == 'chemistry' and any(term in text for term in ['molecule', 'reaction']):
            confidence += 0.2
        elif category == 'biology' and any(term in text for term in ['dna', 'gene']):
            confidence += 0.2

        return min(confidence, 1.0)

    def normalize_data(self, problems: List[Dict]) -> List[Dict]:
        """データの正規化"""
        logger.info("Normalizing data...")

        normalized_problems = []
        for problem in problems:
            # テキストの正規化
            if 'title' in problem:
                problem['title'] = self._normalize_text(problem['title'])
            if 'problem_statement' in problem:
                problem['problem_statement'] = self._normalize_text(problem['problem_statement'])
            if 'solution' in problem:
                problem['solution'] = self._normalize_text(problem['solution'])

            # 四重推論チェーンの正規化
            if 'quad_inference_chain' in problem:
                problem['quad_inference_chain'] = self._normalize_quad_chain(problem['quad_inference_chain'])

            # 数値フィールドの正規化
            if 'quality_score' in problem:
                problem['quality_score'] = max(0.0, min(1.0, problem['quality_score']))

            normalized_problems.append(problem)

        return normalized_problems

    def _normalize_text(self, text: str) -> str:
        """テキストの正規化"""
        if not text:
            return ""

        # Unicode正規化
        text = text.strip()

        # 連続する空白の除去
        text = re.sub(r'\s+', ' ', text)

        # 数式の前後の空白調整
        text = re.sub(r'\s*(\$[^\$]*\$)\s*', r' \1 ', text)

        return text.strip()

    def _normalize_quad_chain(self, quad_chain: List[Dict]) -> List[Dict]:
        """四重推論チェーンの正規化"""
        normalized_chain = []
        for step in quad_chain:
            normalized_step = step.copy()
            if 'content' in normalized_step:
                normalized_step['content'] = self._normalize_text(normalized_step['content'])
            if 'reasoning' in normalized_step:
                normalized_step['reasoning'] = self._normalize_text(normalized_step['reasoning'])
            if 'mathematical_formalism' in normalized_step and normalized_step['mathematical_formalism']:
                normalized_step['mathematical_formalism'] = normalized_step['mathematical_formalism'].strip()
            normalized_chain.append(normalized_step)
        return normalized_chain

    def generate_statistics_report(self, problems: List[Dict], validation_stats: Dict) -> Dict[str, Any]:
        """統計レポートの生成"""
        logger.info("Generating statistics report...")

        # 基本統計
        stats = {
            'total_problems': len(problems),
            'category_distribution': dict(Counter(p.get('category', '') for p in problems)),
            'difficulty_distribution': dict(Counter(p.get('difficulty', '') for p in problems)),
            'average_quality_score': np.mean([p.get('quality_score', 0) for p in problems]),
            'quality_score_std': np.std([p.get('quality_score', 0) for p in problems]),
            'theoretical_depth_distribution': dict(Counter(p.get('theoretical_depth', '') for p in problems)),
            'validation_stats': validation_stats
        }

        # カテゴリ別詳細統計
        category_stats = {}
        for category in ['mathematics', 'physics', 'chemistry', 'biology']:
            category_problems = [p for p in problems if p.get('category') == category]
            if category_problems:
                category_stats[category] = {
                    'count': len(category_problems),
                    'avg_quality': np.mean([p.get('quality_score', 0) for p in category_problems]),
                    'difficulty_dist': dict(Counter(p.get('difficulty', '') for p in category_problems)),
                    'key_concepts_top10': self._get_top_concepts(category_problems)
                }

        stats['category_detailed_stats'] = category_stats

        # 四重推論チェーンの分析
        quad_stats = self._analyze_quad_chains(problems)
        stats['quad_inference_stats'] = quad_stats

        return stats

    def _get_top_concepts(self, problems: List[Dict], top_n: int = 10) -> List[Tuple[str, int]]:
        """主要概念のトップNを取得"""
        concept_counts = Counter()
        for problem in problems:
            concepts = problem.get('key_concepts', [])
            concept_counts.update(concepts)

        return concept_counts.most_common(top_n)

    def _analyze_quad_chains(self, problems: List[Dict]) -> Dict[str, Any]:
        """四重推論チェーンの分析"""
        chain_stats = {
            'total_chains': 0,
            'complete_chains': 0,  # 4ステップ全てがある
            'avg_steps_per_chain': 0,
            'step_completeness': {
                'problem_formulation': 0,
                'theoretical_approach': 0,
                'computational_verification': 0,
                'insightful_conclusion': 0
            },
            'mathematical_formalism_count': 0,
            'computational_result_count': 0
        }

        total_steps = 0
        for problem in problems:
            chain = problem.get('quad_inference_chain', [])
            if chain:
                chain_stats['total_chains'] += 1
                total_steps += len(chain)

                if len(chain) == 4:
                    chain_stats['complete_chains'] += 1

                # ステップの完全性をチェック
                step_types = {step.get('step_type', '') for step in chain}
                for step_type in chain_stats['step_completeness'].keys():
                    if step_type in step_types:
                        chain_stats['step_completeness'][step_type] += 1

                # 形式的表現のカウント
                for step in chain:
                    if step.get('mathematical_formalism'):
                        chain_stats['mathematical_formalism_count'] += 1
                    if step.get('computational_result'):
                        chain_stats['computational_result_count'] += 1

        if chain_stats['total_chains'] > 0:
            chain_stats['avg_steps_per_chain'] = total_steps / chain_stats['total_chains']

        return chain_stats

    def cleanse_dataset(self, input_file: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """データセットの完全クレンジング処理"""
        logger.info(f"Starting data cleansing for {input_file}")

        # 1. データ読み込み
        problems = self.load_dataset(input_file)
        if not problems:
            return [], {}

        logger.info(f"Original dataset: {len(problems)} problems")

        # 2. 重複除去
        problems = self.remove_duplicates(problems)
        logger.info(f"After deduplication: {len(problems)} problems")

        # 3. 品質フィルタリング
        problems = self.filter_by_quality(problems)
        logger.info(f"After quality filtering: {len(problems)} problems")

        # 4. 四値分類検証
        problems, validation_stats = self.validate_quad_classification(problems)

        # 5. データ正規化
        problems = self.normalize_data(problems)

        # 6. 統計レポート生成
        stats_report = self.generate_statistics_report(problems, validation_stats)

        logger.info(f"Cleansing completed: {len(problems)} final problems")
        return problems, stats_report

    def process_all_datasets(self):
        """全データセットの処理"""
        logger.info("Processing all Nobel Fields datasets...")

        # メイン統合データセット
        main_file = self.data_dir / "nobel_fields_cot_dataset.jsonl"
        if main_file.exists():
            logger.info("Processing main dataset...")
            cleansed_problems, stats = self.cleanse_dataset(str(main_file))

            # クレンジング済みデータを保存
            cleansed_file = self.cleansed_dir / "nobel_fields_cot_cleansed.jsonl"
            self.save_dataset(cleansed_problems, str(cleansed_file))

            # 統計レポート保存
            stats_file = self.cleansed_dir / "cleansing_report.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            logger.info(f"Main dataset cleansing completed. Results saved to {self.cleansed_dir}")

        # カテゴリ別データセット
        for category in ['mathematics', 'physics', 'chemistry', 'biology']:
            category_file = self.data_dir / f"nobel_fields_cot_{category}.jsonl"
            if category_file.exists():
                logger.info(f"Processing {category} dataset...")
                category_problems, category_stats = self.cleanse_dataset(str(category_file))

                # カテゴリ別保存
                category_cleansed_file = self.cleansed_dir / f"nobel_fields_cot_{category}_cleansed.jsonl"
                self.save_dataset(category_problems, str(category_cleansed_file))

                # カテゴリ別統計保存
                category_stats_file = self.cleansed_dir / f"cleansing_report_{category}.json"
                with open(category_stats_file, 'w', encoding='utf-8') as f:
                    json.dump(category_stats, f, ensure_ascii=False, indent=2)

                logger.info(f"{category} dataset cleansing completed")

        logger.info("All datasets processing completed")

def main():
    """メイン実行関数"""
    cleanser = NobelFieldsDataCleanser()

    print("SO8T Nobel Fields Data Cleansing and Quad Classification")
    print("=" * 60)

    # データクレンジング実行
    cleanser.process_all_datasets()

    print("\nData cleansing completed successfully!")

    # クレンジング結果の確認
    cleansed_dir = Path("data/nobel_fields_cot/cleansed")
    if cleansed_dir.exists():
        cleansed_files = list(cleansed_dir.glob("*.jsonl"))
        print(f"\nCleansed datasets saved:")
        for file_path in cleansed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"  - {file_path.name}: {line_count} problems")
            except Exception as e:
                print(f"  - {file_path.name}: Error reading file ({e})")

    # 音声通知
    try:
        import winsound
        winsound.Beep(1200, 300)  # 成功音（高音）
        print("[AUDIO] Data cleansing completed successfully")
    except ImportError:
        print("[AUDIO] Data cleansing completed (winsound not available)")

if __name__ == "__main__":
    main()
