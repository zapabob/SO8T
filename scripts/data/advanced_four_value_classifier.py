#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高度な四値分類器 (Observation, Deduction, Abduction, Integration)
ノーベル賞・フィールズ賞級の思考プロセス分析のための分類システム
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import argparse

# パス設定
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "utils"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedFourValueClassifier:
    """高度な四値分類器"""

    def __init__(self):
        # 分類パターン
        self.classification_patterns = {
            'observation': {
                'keywords': [
                    'observe', 'see', 'notice', 'data', 'input', '情報', '事実',
                    'given', 'provided', 'stated', 'shows', 'indicates',
                    '測定', '観測', 'データ', '入力', '条件'
                ],
                'patterns': [
                    r'given.*that', r'we.*see', r'we.*observe',
                    r'データ.*から', r'情報.*から', r'観測.*結果'
                ],
                'weight': 1.0
            },
            'deduction': {
                'keywords': [
                    'prove', 'deduce', 'logic', 'reason', 'therefore', 'thus',
                    '証明', '論理', '推論', 'したがって', 'ゆえに',
                    'follows', 'conclude', 'derive', 'mathematical', 'theorem'
                ],
                'patterns': [
                    r'therefore', r'thus', r'we.*conclude',
                    r'証明.*する', r'論理的.*に', r'数学的.*に'
                ],
                'weight': 1.2
            },
            'abduction': {
                'keywords': [
                    'hypothesize', 'assume', 'suppose', 'imagine', 'perhaps',
                    '仮説', '仮定', '推測', '可能性', 'もしかすると',
                    'assume', 'suppose', 'likely', 'possible', 'creative'
                ],
                'patterns': [
                    r'perhaps', r'maybe', r'we.*assume',
                    r'仮定.*して', r'可能性.*として', r'創造的.*に'
                ],
                'weight': 1.1
            },
            'integration': {
                'keywords': [
                    'integrate', 'combine', 'synthesize', 'unify', 'holistic',
                    '統合', '統合', '総合', '統一', '全体',
                    'combine', 'synthesize', 'unify', 'connect', 'relate'
                ],
                'patterns': [
                    r'combine.*with', r'integrate.*into', r'synthesize',
                    r'統合.*する', r'総合的.*に', r'全体.*として'
                ],
                'weight': 1.3
            }
        }

        # ドメイン固有の分類ルール
        self.domain_rules = {
            'mathematics': {
                'observation': ['given', 'let', 'consider', 'データ'],
                'deduction': ['prove', 'theorem', 'lemma', '証明'],
                'abduction': ['assume', 'suppose', 'hypothesis', '仮定'],
                'integration': ['combine', 'unify', 'generalize', '統合']
            },
            'physics': {
                'observation': ['measured', 'observed', 'data', '実験'],
                'deduction': ['derive', 'calculate', '理論', '導出'],
                'abduction': ['hypothesize', 'model', 'assume', '仮説'],
                'integration': ['unify', 'combine', 'synthesize', '統一']
            },
            'chemistry': {
                'observation': ['react', 'compound', 'measured', '反応'],
                'deduction': ['mechanism', 'explain', 'theory', '機構'],
                'abduction': ['propose', 'suggest', 'hypothesis', '提案'],
                'integration': ['synthesize', 'combine', 'network', '合成']
            },
            'biology': {
                'observation': ['observed', 'data', 'sequence', '観察'],
                'deduction': ['explain', 'mechanism', 'pathway', '機構'],
                'abduction': ['hypothesize', 'model', 'evolution', '仮説'],
                'integration': ['system', 'network', 'ecology', '生態']
            }
        }

        # 統計情報
        self.stats = {
            'processed_samples': 0,
            'classification_distribution': defaultdict(int),
            'confidence_scores': [],
            'domain_distribution': defaultdict(int)
        }

    def classify_inference_type(self, content: str, domain: str = 'general',
                              metadata: Dict[str, Any] = None) -> Tuple[str, float]:
        """
        高度な四値分類を実行

        Args:
            content: 分類対象のコンテンツ
            domain: ドメイン（mathematics, physics, etc.）
            metadata: 追加のメタデータ

        Returns:
            (inference_type, confidence_score)
        """
        if not content:
            return 'deduction', 0.5

        content_lower = content.lower()

        # 各タイプのスコアを計算
        scores = {}
        for inference_type, patterns in self.classification_patterns.items():
            score = 0.0

            # キーワードマッチング
            for keyword in patterns['keywords']:
                count = content_lower.count(keyword.lower())
                score += count * 0.1

            # 正規表現パターンマッチング
            for pattern in patterns['patterns']:
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                score += matches * 0.2

            # ドメイン固有のルール適用
            if domain in self.domain_rules:
                domain_keywords = self.domain_rules[domain].get(inference_type, [])
                for keyword in domain_keywords:
                    count = content_lower.count(keyword.lower())
                    score += count * 0.15

            # コンテンツの構造的特徴
            score += self._analyze_structural_features(content, inference_type)

            # 重み適用
            scores[inference_type] = score * patterns['weight']

        # 最大スコアのタイプを選択
        best_type = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_type] / total_score if total_score > 0 else 0.5

        return best_type, min(confidence, 1.0)

    def _analyze_structural_features(self, content: str, inference_type: str) -> float:
        """構造的特徴を分析"""
        score = 0.0

        # LaTeX数式の密度
        latex_density = len(re.findall(r'\\[a-zA-Z]+\{', content)) / max(len(content.split()), 1) * 1000
        if inference_type in ['deduction', 'integration']:
            score += latex_density * 0.1

        # 論理的接続詞の使用
        logical_connectors = ['therefore', 'thus', 'hence', 'consequently', 'accordingly']
        if inference_type == 'deduction':
            for connector in logical_connectors:
                if connector in content.lower():
                    score += 0.2

        # 仮説的表現
        hypothetical_phrases = ['if', 'suppose', 'assume', 'imagine', 'perhaps']
        if inference_type == 'abduction':
            for phrase in hypothetical_phrases:
                if phrase in content.lower():
                    score += 0.15

        # 統合的表現
        integration_phrases = ['combine', 'integrate', 'synthesize', 'unify', 'holistic']
        if inference_type == 'integration':
            for phrase in integration_phrases:
                if phrase in content.lower():
                    score += 0.2

        return score

    def classify_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """サンプルを分類"""
        # コンテンツ抽出
        content_parts = []

        if 'instruction' in sample:
            content_parts.append(str(sample['instruction']))
        if 'problem_statement' in sample:
            content_parts.append(str(sample['problem_statement']))
        if 'solution' in sample:
            content_parts.append(str(sample['solution']))
        if 'output' in sample:
            content_parts.append(str(sample['output']))

        content = ' '.join(content_parts)

        # ドメイン判定
        domain = sample.get('domain') or sample.get('category', 'general')

        # 分類実行
        inference_type, confidence = self.classify_inference_type(content, domain, sample)

        # 結果更新
        sample['inference_type'] = inference_type
        sample['classification_confidence'] = confidence

        # 統計更新
        self.stats['processed_samples'] += 1
        self.stats['classification_distribution'][inference_type] += 1
        self.stats['confidence_scores'].append(confidence)
        self.stats['domain_distribution'][domain] += 1

        return sample

    def batch_classify(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """バッチ分類を実行"""
        logger.info(f"Starting batch classification of {len(samples)} samples...")

        classified_samples = []
        for sample in tqdm(samples, desc="Classifying samples"):
            try:
                classified_sample = self.classify_sample(sample)
                classified_samples.append(classified_sample)
            except Exception as e:
                logger.error(f"Error classifying sample: {e}")
                # エラーの場合は元のサンプルをそのまま追加
                classified_samples.append(sample)

        logger.info("Batch classification completed")
        self._log_statistics()

        return classified_samples

    def _log_statistics(self):
        """統計情報をログ出力"""
        logger.info("=== Classification Statistics ===")
        logger.info(f"Processed samples: {self.stats['processed_samples']}")

        logger.info("Inference type distribution:")
        for inf_type, count in self.stats['classification_distribution'].items():
            percentage = (count / self.stats['processed_samples']) * 100
            logger.info(f"  {inf_type}: {count} ({percentage:.1f}%)")

        if self.stats['confidence_scores']:
            avg_confidence = np.mean(self.stats['confidence_scores'])
            logger.info(f"Average confidence: {avg_confidence:.3f}")

        logger.info("Domain distribution:")
        for domain, count in self.stats['domain_distribution'].items():
            percentage = (count / self.stats['processed_samples']) * 100
            logger.info(f"  {domain}: {count} ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="高度な四値分類器")
    parser.add_argument("--input_file", type=str, required=True,
                       help="入力データセットファイル")
    parser.add_argument("--output_file", type=str, required=True,
                       help="出力データセットファイル")

    args = parser.parse_args()

    # 分類器の初期化
    classifier = AdvancedFourValueClassifier()

    # データセット読み込み
    logger.info(f"Loading dataset from {args.input_file}")
    samples = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    # 分類実行
    classified_samples = classifier.batch_classify(samples)

    # 結果保存
    logger.info(f"Saving classified dataset to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in classified_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    logger.info("Classification completed!")

if __name__ == "__main__":
    main()
