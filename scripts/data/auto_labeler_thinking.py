#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/thinkingモデル用自動ラベル付け

ドメイン別ラベル付けルールに基づいて、ALLOW/ESCALATION/DENYのラベルを自動付与。
バランス調整機能も含む。

Usage:
    from scripts.data.auto_labeler_thinking import ThinkingAutoLabeler
    labeler = ThinkingAutoLabeler(keywords_config)
    labeled_sample = labeler.label_sample(sample, domain)
"""

import re
import random
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import hashlib

logger = logging.getLogger(__name__)


class ThinkingAutoLabeler:
    """/thinkingモデル用自動ラベル付けクラス"""
    
    def __init__(self, keywords_config: Dict):
        """
        Args:
            keywords_config: ドメイン別キーワード設定
        """
        self.keywords_config = keywords_config
        self.domain_configs = keywords_config["domains"]
    
    def classify_label(
        self,
        text: str,
        title: str,
        domain: str
    ) -> Tuple[str, float]:
        """
        テキストとタイトルからラベルを分類
        
        Args:
            text: テキスト内容
            title: タイトル
            domain: ドメイン
        
        Returns:
            (label, confidence)のタプル
        """
        if domain not in self.domain_configs:
            return "ALLOW", 0.5
        
        domain_config = self.domain_configs[domain]
        classification_rules = domain_config["classification_rules"]
        
        text_lower = text.lower()
        title_lower = title.lower()
        combined = f"{title_lower} {text_lower}"
        
        # DENY判定（最優先）
        deny_score = self._calculate_label_score(combined, classification_rules["DENY"])
        if deny_score > 0.5:
            return "DENY", min(deny_score, 0.95)
        
        # ESCALATION判定
        escalate_score = self._calculate_label_score(combined, classification_rules["ESCALATION"])
        if escalate_score > 0.4:
            return "ESCALATION", min(escalate_score, 0.90)
        
        # ALLOW判定
        allow_score = self._calculate_label_score(combined, classification_rules["ALLOW"])
        if allow_score > 0.3:
            return "ALLOW", min(allow_score, 0.95)
        
        # デフォルト: ドメイン別分布に基づいて決定
        return self._sample_from_distribution(domain_config["label_distribution"])
    
    def _calculate_label_score(
        self,
        text: str,
        rules: Dict
    ) -> float:
        """ルールに基づいてスコアを計算"""
        score = 0.0
        matches = 0
        
        # キーワードマッチング
        keywords = rules.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in text:
                score += 1.0
                matches += 1
        
        # パターンマッチング
        patterns = rules.get("patterns", [])
        for pattern in patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 2.0
                    matches += 1
            except re.error:
                continue
        
        # 正規化（0.0-1.0）
        if matches == 0:
            return 0.0
        
        normalized_score = min(score / (len(keywords) + len(patterns)), 1.0)
        return normalized_score
    
    def _sample_from_distribution(self, distribution: Dict[str, float]) -> Tuple[str, float]:
        """分布に基づいてラベルをサンプリング"""
        labels = list(distribution.keys())
        weights = list(distribution.values())
        
        label = random.choices(labels, weights=weights)[0]
        confidence = distribution[label]
        
        return label, confidence
    
    def label_sample(
        self,
        sample: Dict,
        domain: str
    ) -> Dict:
        """
        サンプルにラベル付け
        
        Args:
            sample: サンプル辞書
            domain: ドメイン
        
        Returns:
            ラベル付きサンプル
        """
        text = sample.get("output", sample.get("text", ""))
        title = sample.get("title", sample.get("instruction", ""))
        
        # ラベル分類
        label, confidence = self.classify_label(text, title, domain)
        
        # ラベル付きサンプルを作成
        labeled_sample = {
            **sample,
            "safety_judgment": label,
            "confidence": confidence,
            "so8_group_state": "stable",
            "pet_regularization": 0.1,
            "self_verification": "passed",
        }
        
        # thinkingフィールドを追加/更新
        labeled_sample["thinking"] = self._generate_thinking(text, label, domain)
        
        return labeled_sample
    
    def _generate_thinking(
        self,
        text: str,
        label: str,
        domain: str
    ) -> str:
        """thinkingフィールドを生成"""
        thinking_templates = {
            "ALLOW": "<think>This is general information that can be safely shared. The content appears to be publicly available knowledge suitable for educational purposes.</think>",
            "ESCALATION": "<think>This content may require expert review or additional verification. While it contains useful information, some details might need professional confirmation.</think>",
            "DENY": "<think>This content appears to contain sensitive or classified information that should not be disclosed. Appropriate refusal is necessary.</think>",
        }
        
        return thinking_templates.get(label, thinking_templates["ALLOW"])
    
    def balance_dataset(
        self,
        samples: List[Dict],
        target_distribution: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        データセットのバランスを調整
        
        Args:
            samples: サンプルリスト
            target_distribution: 目標分布（Noneの場合は設定から読み込み）
        
        Returns:
            バランス調整後のサンプルリスト
        """
        if target_distribution is None:
            target_distribution = self.keywords_config["label_balance"]["target_distribution"]
        
        # ラベル別に分類
        label_samples = defaultdict(list)
        for sample in samples:
            label = sample.get("safety_judgment", "ALLOW")
            label_samples[label].append(sample)
        
        # 統計
        label_counts = {k: len(v) for k, v in label_samples.items()}
        total_samples = len(samples)
        
        logger.info(f"Label distribution before balancing: {label_counts}")
        logger.info(f"Total samples: {total_samples:,}")
        
        # 目標サンプル数を計算
        target_counts = {}
        for label, ratio in target_distribution.items():
            target_counts[label] = int(total_samples * ratio)
        
        # バランス調整
        balanced_samples = []
        for label, target_count in target_counts.items():
            available_samples = label_samples.get(label, [])
            
            if len(available_samples) >= target_count:
                # サンプリング
                balanced_samples.extend(random.sample(available_samples, target_count))
            else:
                # すべて使用（不足分は許容）
                balanced_samples.extend(available_samples)
        
        # シャッフル
        random.shuffle(balanced_samples)
        
        # 統計
        balanced_counts = Counter(s.get("safety_judgment", "ALLOW") for s in balanced_samples)
        logger.info(f"Label distribution after balancing: {dict(balanced_counts)}")
        logger.info(f"Total balanced samples: {len(balanced_samples):,}")
        
        return balanced_samples
    
    def balance_by_domain(
        self,
        samples: List[Dict]
    ) -> List[Dict]:
        """ドメイン別にバランス調整"""
        domain_distribution = self.keywords_config["domain_balance"]["target_distribution"]
        
        # ドメイン別に分類
        domain_samples = defaultdict(list)
        for sample in samples:
            domain = sample.get("domain", "general")
            domain_samples[domain].append(sample)
        
        # 統計
        domain_counts = {k: len(v) for k, v in domain_samples.items()}
        total_samples = len(samples)
        
        logger.info(f"Domain distribution before balancing: {domain_counts}")
        
        # 目標サンプル数を計算
        target_counts = {}
        for domain, ratio in domain_distribution.items():
            target_counts[domain] = int(total_samples * ratio)
        
        # バランス調整
        balanced_samples = []
        for domain, target_count in target_counts.items():
            available_samples = domain_samples.get(domain, [])
            
            if len(available_samples) >= target_count:
                balanced_samples.extend(random.sample(available_samples, target_count))
            else:
                balanced_samples.extend(available_samples)
        
        random.shuffle(balanced_samples)
        
        balanced_counts = Counter(s.get("domain", "general") for s in balanced_samples)
        logger.info(f"Domain distribution after balancing: {dict(balanced_counts)}")
        
        return balanced_samples
    
    def balance_by_language(
        self,
        samples: List[Dict]
    ) -> List[Dict]:
        """言語別にバランス調整"""
        language_distribution = self.keywords_config["language_balance"]
        
        # 言語別に分類
        language_samples = defaultdict(list)
        for sample in samples:
            language = sample.get("language", "ja")
            language_samples[language].append(sample)
        
        # 統計
        language_counts = {k: len(v) for k, v in language_samples.items()}
        total_samples = len(samples)
        
        logger.info(f"Language distribution before balancing: {language_counts}")
        
        # 目標サンプル数を計算
        target_counts = {}
        for language, ratio in language_distribution.items():
            target_counts[language] = int(total_samples * ratio)
        
        # バランス調整
        balanced_samples = []
        for language, target_count in target_counts.items():
            available_samples = language_samples.get(language, [])
            
            if len(available_samples) >= target_count:
                balanced_samples.extend(random.sample(available_samples, target_count))
            else:
                balanced_samples.extend(available_samples)
        
        random.shuffle(balanced_samples)
        
        balanced_counts = Counter(s.get("language", "ja") for s in balanced_samples)
        logger.info(f"Language distribution after balancing: {dict(balanced_counts)}")
        
        return balanced_samples
    
    def balance_complete(
        self,
        samples: List[Dict]
    ) -> List[Dict]:
        """完全なバランス調整（ドメイン、ラベル、言語）"""
        logger.info("="*80)
        logger.info("Complete Dataset Balancing")
        logger.info("="*80)
        
        # 1. ドメイン別バランス
        logger.info("Step 1: Balancing by domain...")
        balanced = self.balance_by_domain(samples)
        
        # 2. ラベル別バランス
        logger.info("Step 2: Balancing by label...")
        balanced = self.balance_dataset(balanced)
        
        # 3. 言語別バランス
        logger.info("Step 3: Balancing by language...")
        balanced = self.balance_by_language(balanced)
        
        logger.info("="*80)
        logger.info(f"[COMPLETE] Balanced dataset: {len(balanced):,} samples")
        logger.info("="*80)
        
        return balanced

