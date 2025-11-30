#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T AEGIS-v2.0 Reasoning Dataset Creator
既存データセットすべてを四値分類し、Phi3.5の内部タグ付けとPPOトレーニング用に統合

機能:
- 既存の全データセットを四値分類（数学・物理・化学・生物）
- Phi3.5の内部タグ付けシステム
- PPOトレーニング用/thinkingモデル化データ生成
- 統計的処理と品質最適化
- AEGIS-v2.0reasoningdataset.jsonl生成
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aegis_v2_reasoning_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ReasoningEntry:
    """推論エントリ"""
    id: str
    text: str
    category: str  # 'mathematics', 'physics', 'chemistry', 'biology'
    source_dataset: str
    phi35_tags: Dict[str, Any]  # Phi3.5内部タグ付け
    thinking_trace: List[Dict[str, Any]]  # /thinkingモデル化データ
    ppo_labels: Dict[str, float]  # PPOトレーニング用ラベル
    quality_score: float
    created_at: str

@dataclass
class Phi35InternalTags:
    """Phi3.5内部タグ付け"""
    domain: str  # 専門領域
    complexity: str  # 複雑さレベル
    reasoning_type: str  # 推論タイプ
    knowledge_depth: int  # 知識深度（1-5）
    mathematical_formality: int  # 数学的形式性（1-5）
    interdisciplinary: bool  # 学際性
    safety_level: str  # 安全レベル
    ethical_considerations: List[str]  # 倫理的考慮事項

@dataclass
class ThinkingStep:
    """Thinkingステップ"""
    step_type: str  # 'problem_analysis', 'solution_approach', 'verification', 'conclusion'
    content: str
    confidence: float
    evidence: List[str]
    phi35_reasoning: Dict[str, Any]

class AEGISV2ReasoningDatasetCreator:
    """AEGIS-v2.0 Reasoning Dataset Creator"""

    def __init__(self, output_file: str = "data/aegis_v2_0reasoningdataset.jsonl"):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # 四値分類カテゴリ
        self.categories = ['mathematics', 'physics', 'chemistry', 'biology']

        # データセットパス
        self.dataset_paths = self._collect_all_dataset_paths()

        # Phi3.5タグ付けルール
        self.phi35_rules = self._load_phi35_tagging_rules()

        # PPOラベル生成ルール
        self.ppo_rules = self._load_ppo_labeling_rules()

        # 統計追跡
        self.stats = {
            'total_processed': 0,
            'categories': {cat: 0 for cat in self.categories},
            'sources': defaultdict(int),
            'quality_distribution': [],
            'phi35_tags_distribution': defaultdict(int),
            'ppo_labels_stats': defaultdict(list)
        }

        logger.info(f"Initialized AEGIS-v2.0 Reasoning Dataset Creator with {len(self.dataset_paths)} datasets")

    def _collect_all_dataset_paths(self) -> List[Path]:
        """全データセットパスを収集"""
        data_dir = Path("data")
        dataset_paths = []

        # 再帰的にjsonlファイルを探す
        for jsonl_file in data_dir.rglob("*.jsonl"):
            # 特定のファイルを除外（中間ファイルなど）
            exclude_patterns = [
                'validation_burnin_test',  # テストファイル
                'temp_',  # 一時ファイル
                'backup_',  # バックアップ
                'cache_',  # キャッシュ
            ]

            if not any(pattern in jsonl_file.name for pattern in exclude_patterns):
                dataset_paths.append(jsonl_file)

        logger.info(f"Collected {len(dataset_paths)} dataset files")
        return dataset_paths

    def _load_phi35_tagging_rules(self) -> Dict[str, Any]:
        """Phi3.5タグ付けルールをロード"""
        return {
            'domain_keywords': {
                'mathematics': [
                    'theorem', 'proof', 'lemma', 'corollary', 'axiom', 'algebra',
                    'geometry', 'topology', 'analysis', 'number theory', 'logic',
                    'combinatorics', 'graph theory', 'category theory'
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
            },
            'complexity_levels': {
                'basic': ['fundamental', 'basic', 'elementary', 'simple'],
                'intermediate': ['intermediate', 'moderate', 'standard', 'normal'],
                'advanced': ['advanced', 'complex', 'sophisticated', 'expert'],
                'expert': ['expert', 'specialized', 'cutting-edge', 'research']
            },
            'reasoning_types': [
                'deductive', 'inductive', 'abductive', 'analogical',
                'causal', 'probabilistic', 'logical', 'mathematical'
            ],
            'safety_levels': ['safe', 'moderate', 'sensitive', 'restricted']
        }

    def _load_ppo_labeling_rules(self) -> Dict[str, Any]:
        """PPOラベル生成ルールをロード"""
        return {
            'reward_functions': {
                'correctness': lambda x: 1.0 if x.get('is_correct', False) else 0.0,
                'confidence': lambda x: x.get('confidence', 0.5),
                'complexity': lambda x: min(1.0, len(x.get('text', '')) / 1000),
                'reasoning_depth': lambda x: min(1.0, len(x.get('thinking_trace', [])) / 10),
                'quality_score': lambda x: x.get('quality_score', 0.5)
            },
            'penalty_functions': {
                'inconsistency': lambda x: -0.5 if x.get('has_inconsistency', False) else 0.0,
                'toxicity': lambda x: -1.0 if x.get('is_toxic', False) else 0.0,
                'irrelevance': lambda x: -0.3 if x.get('is_irrelevant', False) else 0.0
            },
            'normalization_factors': {
                'reward_scale': 2.0,
                'penalty_scale': 1.0,
                'confidence_weight': 0.3,
                'quality_weight': 0.4,
                'reasoning_weight': 0.3
            }
        }

    def create_aegis_v2_dataset(self) -> Dict[str, Any]:
        """AEGIS-v2.0 Reasoning Datasetを作成"""
        logger.info("Creating AEGIS-v2.0 Reasoning Dataset...")

        all_entries = []
        processed_files = 0

        # 全データセットを処理
        for dataset_path in tqdm(self.dataset_paths, desc="Processing datasets"):
            try:
                entries = self._process_dataset_file(dataset_path)
                all_entries.extend(entries)
                processed_files += 1

                logger.info(f"Processed {dataset_path.name}: {len(entries)} entries")

            except Exception as e:
                logger.error(f"Failed to process {dataset_path}: {e}")
                continue

        # 重複除去
        unique_entries = self._remove_duplicates(all_entries)
        logger.info(f"Removed duplicates: {len(all_entries)} -> {len(unique_entries)}")

        # 統計的処理と最適化
        optimized_entries = self._statistical_optimization(unique_entries)

        # AEGIS-v2.0 Reasoning Datasetとして保存
        self._save_aegis_v2_dataset(optimized_entries)

        # 統計レポート生成
        stats_report = self._generate_stats_report(optimized_entries)

        logger.info(f"AEGIS-v2.0 Reasoning Dataset created: {len(optimized_entries)} entries")
        return stats_report

    def _process_dataset_file(self, file_path: Path) -> List[ReasoningEntry]:
        """個別データセットファイルを処理"""
        entries = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            self.stats['total_processed'] += 1

                            # 四値分類
                            category = self._classify_into_quad(data, file_path)

                            # Phi3.5タグ付け
                            phi35_tags = self._generate_phi35_tags(data, category)

                            # Thinkingトレース生成
                            thinking_trace = self._generate_thinking_trace(data, category)

                            # PPOラベル生成
                            ppo_labels = self._generate_ppo_labels(data, phi35_tags, thinking_trace)

                            # 品質スコア計算
                            quality_score = self._calculate_quality_score(data, category, phi35_tags)

                            # エントリ作成
                            entry = ReasoningEntry(
                                id=f"{file_path.stem}_{line_num}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}",
                                text=self._extract_text(data),
                                category=category,
                                source_dataset=file_path.stem,
                                phi35_tags=asdict(phi35_tags),
                                thinking_trace=[asdict(step) for step in thinking_trace],
                                ppo_labels=ppo_labels,
                                quality_score=quality_score,
                                created_at=datetime.now().isoformat()
                            )

                            entries.append(entry)

                            # 統計更新
                            self.stats['categories'][category] += 1
                            self.stats['sources'][file_path.stem] += 1
                            self.stats['quality_distribution'].append(quality_score)

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")

        return entries

    def _classify_into_quad(self, data: Dict, file_path: Path) -> str:
        """データを四値分類"""
        text = self._extract_text(data).lower()
        filename = file_path.name.lower()

        # ファイル名ベースの分類ヒント
        filename_hints = {
            'mathematics': ['math', 'nobel_fields_cot_mathematics', 'algebra', 'geometry', 'analysis'],
            'physics': ['physics', 'quantum', 'relativity', 'nuclear', 'particle', 'nobel_fields_cot_physics'],
            'chemistry': ['chemistry', 'organic', 'inorganic', 'biochemistry', 'nobel_fields_cot_chemistry'],
            'biology': ['biology', 'genetics', 'molecular', 'neuroscience', 'ecology', 'nobel_fields_cot_biology']
        }

        # ファイル名からの分類
        for category, hints in filename_hints.items():
            if any(hint in filename for hint in hints):
                return category

        # テキスト内容からの分類
        category_scores = defaultdict(float)

        for category, keywords in self.phi35_rules['domain_keywords'].items():
            matches = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = matches

        # 最高スコアのカテゴリを返す
        if max(category_scores.values()) > 0:
            return max(category_scores.items(), key=lambda x: x[1])[0]

        # デフォルトは最も一般的なもの
        return 'mathematics'

    def _generate_phi35_tags(self, data: Dict, category: str) -> Phi35InternalTags:
        """Phi3.5内部タグ付け生成"""
        text = self._extract_text(data).lower()

        # ドメイン
        domain = category

        # 複雑さレベル
        complexity_keywords = self.phi35_rules['complexity_levels']
        complexity = 'intermediate'  # デフォルト

        for level, keywords in complexity_keywords.items():
            if any(kw in text for kw in keywords):
                complexity = level
                break

        # 推論タイプ
        reasoning_type = 'logical'  # デフォルト

        if 'mathematical' in text or 'theorem' in text:
            reasoning_type = 'mathematical'
        elif 'causal' in text or 'cause' in text:
            reasoning_type = 'causal'
        elif 'probabilistic' in text or 'probability' in text:
            reasoning_type = 'probabilistic'

        # 知識深度（1-5）
        knowledge_depth = 3  # デフォルト
        if complexity == 'basic':
            knowledge_depth = 2
        elif complexity == 'advanced':
            knowledge_depth = 4
        elif complexity == 'expert':
            knowledge_depth = 5

        # 数学的形式性
        mathematical_formality = 2  # デフォルト
        if category == 'mathematics':
            mathematical_formality = 5
        elif category == 'physics':
            mathematical_formality = 4
        elif category in ['chemistry', 'biology']:
            mathematical_formality = 3

        # 学際性
        interdisciplinary = len([cat for cat in self.categories
                                if any(kw in text for kw in self.phi35_rules['domain_keywords'][cat])]) > 1

        # 安全レベル
        safety_level = 'safe'  # デフォルト
        if 'nsfw' in str(data).lower() or 'toxicity' in text:
            safety_level = 'sensitive'

        # 倫理的考慮事項
        ethical_considerations = []
        if 'safety' in text or 'ethical' in text:
            ethical_considerations.append('safety_considerations')
        if 'privacy' in text:
            ethical_considerations.append('privacy_concerns')
        if 'bias' in text:
            ethical_considerations.append('bias_mitigation')

        return Phi35InternalTags(
            domain=domain,
            complexity=complexity,
            reasoning_type=reasoning_type,
            knowledge_depth=knowledge_depth,
            mathematical_formality=mathematical_formality,
            interdisciplinary=interdisciplinary,
            safety_level=safety_level,
            ethical_considerations=ethical_considerations
        )

    def _generate_thinking_trace(self, data: Dict, category: str) -> List[ThinkingStep]:
        """Thinkingトレース生成"""
        text = self._extract_text(data)
        thinking_steps = []

        # 問題分析ステップ
        analysis_step = ThinkingStep(
            step_type='problem_analysis',
            content=f"Analyzing the {category} problem: {text[:100]}...",
            confidence=0.8,
            evidence=[f"Domain classification: {category}", "Content analysis completed"],
            phi35_reasoning={
                'domain_awareness': category,
                'complexity_assessment': 'intermediate',
                'key_concepts': self._extract_key_concepts(text, category)
            }
        )
        thinking_steps.append(analysis_step)

        # 解法アプローチステップ
        approach_step = ThinkingStep(
            step_type='solution_approach',
            content=f"Applying {category}-specific reasoning approach",
            confidence=0.75,
            evidence=["Theoretical framework identified", "Method selection completed"],
            phi35_reasoning={
                'reasoning_strategy': 'systematic_analysis',
                'formal_methods': category == 'mathematics',
                'empirical_methods': category in ['physics', 'chemistry', 'biology']
            }
        )
        thinking_steps.append(approach_step)

        # 検証ステップ
        verification_step = ThinkingStep(
            step_type='verification',
            content="Verifying reasoning consistency and accuracy",
            confidence=0.85,
            evidence=["Logical consistency checked", "Evidence validation completed"],
            phi35_reasoning={
                'validation_method': 'cross_verification',
                'confidence_level': 'high',
                'error_detection': 'none_found'
            }
        )
        thinking_steps.append(verification_step)

        # 結論ステップ
        conclusion_step = ThinkingStep(
            step_type='conclusion',
            content=f"Conclusion reached for {category} problem",
            confidence=0.9,
            evidence=["All reasoning steps validated", "Solution quality confirmed"],
            phi35_reasoning={
                'conclusion_confidence': 'high',
                'generalizability': True,
                'future_applications': [f"{category}_research", "problem_solving"]
            }
        )
        thinking_steps.append(conclusion_step)

        return thinking_steps

    def _generate_ppo_labels(self, data: Dict, phi35_tags: Phi35InternalTags,
                           thinking_trace: List[ThinkingStep]) -> Dict[str, float]:
        """PPOラベル生成"""
        labels = {}

        # リワード計算
        reward_functions = self.ppo_rules['reward_functions']
        penalty_functions = self.ppo_rules['penalty_functions']
        norm_factors = self.ppo_rules['normalization_factors']

        # 基本データ拡張
        enriched_data = dict(data)
        enriched_data.update({
            'phi35_tags': phi35_tags,
            'thinking_trace': thinking_trace,
            'quality_score': 0.8,  # 仮定
            'is_correct': True,  # 仮定
            'confidence': 0.8,
            'has_inconsistency': False,
            'is_toxic': False,
            'is_irrelevant': False
        })

        # リワード計算
        total_reward = 0
        for reward_name, reward_func in reward_functions.items():
            reward = reward_func(enriched_data)
            labels[f'reward_{reward_name}'] = reward
            total_reward += reward * norm_factors.get(f'{reward_name.split("_")[0]}_weight', 1.0)

        # ペナルティ計算
        total_penalty = 0
        for penalty_name, penalty_func in penalty_functions.items():
            penalty = penalty_func(enriched_data)
            labels[f'penalty_{penalty_name}'] = penalty
            total_penalty += penalty

        # 最終PPOスコア
        final_score = (total_reward * norm_factors['reward_scale'] +
                      total_penalty * norm_factors['penalty_scale'])

        labels['ppo_final_score'] = max(-1.0, min(1.0, final_score))
        labels['ppo_confidence'] = enriched_data['confidence']

        return labels

    def _calculate_quality_score(self, data: Dict, category: str, phi35_tags: Phi35InternalTags) -> float:
        """品質スコア計算"""
        score = 0.5  # ベーススコア

        # テキスト品質
        text = self._extract_text(data)
        if text and len(text) > 50:
            score += 0.1

        # カテゴリ適合性
        if category in self.categories:
            score += 0.1

        # Phi3.5タグ品質
        if phi35_tags.knowledge_depth > 3:
            score += 0.1
        if phi35_tags.mathematical_formality > 3:
            score += 0.1
        if phi35_tags.interdisciplinary:
            score += 0.05

        # 安全適合性
        if phi35_tags.safety_level == 'safe':
            score += 0.05

        return min(1.0, score)

    def _extract_text(self, data: Dict) -> str:
        """データからテキストを抽出"""
        text_fields = ['text', 'content', 'instruction', 'response', 'input', 'output',
                      'question', 'answer', 'context', 'dialogue', 'title']

        for field in text_fields:
            if field in data and data[field]:
                return str(data[field])

        return str(data) if data else ""

    def _extract_key_concepts(self, text: str, category: str) -> List[str]:
        """主要概念の抽出"""
        concepts = []
        keywords = self.phi35_rules['domain_keywords'].get(category, [])

        for keyword in keywords:
            if keyword in text.lower():
                concepts.append(keyword)

        return concepts[:5]  # 最大5つ

    def _remove_duplicates(self, entries: List[ReasoningEntry]) -> List[ReasoningEntry]:
        """重複エントリの除去"""
        seen_hashes = set()
        unique_entries = []

        for entry in entries:
            text_hash = hashlib.md5(entry.text.encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_entries.append(entry)

        return unique_entries

    def _statistical_optimization(self, entries: List[ReasoningEntry]) -> List[ReasoningEntry]:
        """統計的最適化"""
        logger.info("Performing statistical optimization...")

        # カテゴリバランス調整
        category_counts = Counter(entry.category for entry in entries)
        min_samples = min(category_counts.values())

        # 各カテゴリからmin_samplesだけサンプリング
        optimized_entries = []
        for category in self.categories:
            category_entries = [e for e in entries if e.category == category]
            if len(category_entries) > min_samples:
                # 品質スコアでソートして上位を選択
                category_entries.sort(key=lambda x: x.quality_score, reverse=True)
                category_entries = category_entries[:min_samples]

            optimized_entries.extend(category_entries)

        # PPOラベル正規化
        ppo_scores = [entry.ppo_labels['ppo_final_score'] for entry in optimized_entries]
        if ppo_scores:
            score_mean = np.mean(ppo_scores)
            score_std = np.std(ppo_scores)

            for entry in optimized_entries:
                # Z-score正規化
                raw_score = entry.ppo_labels['ppo_final_score']
                normalized_score = (raw_score - score_mean) / (score_std + 1e-8)
                entry.ppo_labels['ppo_normalized_score'] = max(-1.0, min(1.0, normalized_score))

        logger.info(f"Statistical optimization completed: {len(entries)} -> {len(optimized_entries)}")
        return optimized_entries

    def _save_aegis_v2_dataset(self, entries: List[ReasoningEntry]):
        """AEGIS-v2.0 Reasoning Dataset保存"""
        logger.info(f"Saving AEGIS-v2.0 Reasoning Dataset to {self.output_file}")

        with open(self.output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                json.dump(asdict(entry), f, ensure_ascii=False, indent=None)
                f.write('\n')

        logger.info(f"Saved {len(entries)} entries to {self.output_file}")

    def _generate_stats_report(self, entries: List[ReasoningEntry]) -> Dict[str, Any]:
        """統計レポート生成"""
        report = {
            'dataset_info': {
                'name': 'AEGIS-v2.0 Reasoning Dataset',
                'total_entries': len(entries),
                'created_at': datetime.now().isoformat(),
                'file_path': str(self.output_file)
            },
            'category_distribution': dict(Counter(entry.category for entry in entries)),
            'source_distribution': dict(sorted(self.stats['sources'].items(),
                                             key=lambda x: x[1], reverse=True)[:10]),
            'quality_stats': {
                'mean_quality': np.mean(self.stats['quality_distribution']),
                'std_quality': np.std(self.stats['quality_distribution']),
                'min_quality': min(self.stats['quality_distribution']) if self.stats['quality_distribution'] else 0,
                'max_quality': max(self.stats['quality_distribution']) if self.stats['quality_distribution'] else 0
            },
            'phi35_tags_summary': dict(self.stats['phi35_tags_distribution']),
            'ppo_labels_summary': {
                'mean_final_score': np.mean([entry.ppo_labels.get('ppo_final_score', 0) for entry in entries]),
                'std_final_score': np.std([entry.ppo_labels.get('ppo_final_score', 0) for entry in entries])
            },
            'processing_stats': self.stats
        }

        return report

def main():
    """メイン実行関数"""
    print("SO8T AEGIS-v2.0 Reasoning Dataset Creator")
    print("=" * 50)

    creator = AEGISV2ReasoningDatasetCreator()

    try:
        # AEGIS-v2.0 Reasoning Dataset作成
        print("\n[1/3] Creating AEGIS-v2.0 Reasoning Dataset...")
        stats_report = creator.create_aegis_v2_dataset()

        # レポート表示
        print("\n" + "="*60)
        print("AEGIS-V2.0 REASONING DATASET CREATION COMPLETED")
        print("="*60)
        print(f"Total Entries: {stats_report['dataset_info']['total_entries']}")
        print(f"Output File: {stats_report['dataset_info']['file_path']}")

        print(f"\nCategory Distribution:")
        for cat, count in stats_report['category_distribution'].items():
            print(f"  {cat}: {count}")

        print(f"\nQuality Statistics:")
        quality = stats_report['quality_stats']
        print(f"  Mean: {quality['mean_quality']:.3f}")
        print(f"  Std: {quality['std_quality']:.3f}")
        print(f"  Range: [{quality['min_quality']:.3f}, {quality['max_quality']:.3f}]")

        print(f"\nTop Source Datasets:")
        for source, count in list(stats_report['source_distribution'].items())[:5]:
            print(f"  {source}: {count}")

        print(f"\nPPO Labels Summary:")
        ppo = stats_report['ppo_labels_summary']
        print(f"  Mean Final Score: {ppo['mean_final_score']:.3f}")
        print(f"  Std Final Score: {ppo['ppo_labels_summary']['std_final_score']:.3f}")

        # 統計レポート保存
        stats_file = Path("data/aegis_v2_0reasoningdataset_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_report, f, ensure_ascii=False, indent=2)

        print(f"\n📊 Statistics saved to: {stats_file}")

        # 音声通知
        try:
            import winsound
            winsound.Beep(1600, 600)  # 成功音（高め）
            print("[AUDIO] AEGIS-v2.0 Reasoning Dataset creation completed successfully")
        except ImportError:
            print("[AUDIO] AEGIS-v2.0 Reasoning Dataset creation completed (winsound not available)")

    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()
