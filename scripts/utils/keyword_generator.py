#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
キーワード自動生成機能（ローカル処理）

既存キーワードから関連キーワードを自動生成する機能を提供します。
APIを使用せず、ローカル処理のみで実装されています。

Usage:
    from scripts.utils.keyword_generator import KeywordGenerator
    
    generator = KeywordGenerator()
    related_keywords = generator.generate_related_keywords("Python", max_count=10)
"""

import sys
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from difflib import SequenceMatcher

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class KeywordGenerator:
    """キーワード自動生成クラス（ローカル処理）"""
    
    def __init__(
        self,
        keyword_coordinator: Optional[Any] = None,
        similarity_threshold: float = 0.6
    ):
        """
        初期化
        
        Args:
            keyword_coordinator: KeywordCoordinatorインスタンス（既存キーワード取得用）
            similarity_threshold: 類似度の閾値（0.0-1.0）
        """
        self.keyword_coordinator = keyword_coordinator
        self.similarity_threshold = similarity_threshold
        
        # 一般的な関連語パターン（技術用語）
        self.related_patterns = {
            'python': ['django', 'flask', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn'],
            'rust': ['cargo', 'tokio', 'serde', 'actix', 'rocket', 'wasm'],
            'typescript': ['angular', 'react', 'vue', 'node', 'nestjs', 'express'],
            'javascript': ['node', 'react', 'vue', 'angular', 'express', 'webpack'],
            'java': ['spring', 'hibernate', 'maven', 'gradle', 'junit', 'kotlin'],
            'c++': ['qt', 'boost', 'cmake', 'stl', 'template'],
            'c': ['gcc', 'clang', 'make', 'cmake', 'linux'],
            'swift': ['ios', 'xcode', 'cocoa', 'swiftui', 'combine'],
            'kotlin': ['android', 'gradle', 'coroutines', 'ktor'],
            'c#': ['dotnet', 'aspnet', 'unity', 'xamarin', 'linq'],
            'php': ['laravel', 'symfony', 'composer', 'wordpress'],
            'go': ['golang', 'goroutine', 'gin', 'echo', 'gorm'],
            'ruby': ['rails', 'rubygems', 'rspec', 'sinatra'],
            'scala': ['spark', 'akka', 'play', 'sbt'],
            'r': ['tidyverse', 'ggplot2', 'shiny', 'dplyr'],
            'matlab': ['simulink', 'matlab', 'octave'],
            'sql': ['mysql', 'postgresql', 'sqlite', 'mongodb'],
            'html': ['css', 'javascript', 'dom', 'bootstrap'],
            'css': ['sass', 'less', 'bootstrap', 'tailwind'],
        }
        
        logger.info("="*80)
        logger.info("Keyword Generator Initialized")
        logger.info("="*80)
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
    
    def generate_related_keywords(
        self,
        keyword: str,
        max_count: int = 10,
        use_existing_keywords: bool = True,
        use_patterns: bool = True,
        use_similarity: bool = True
    ) -> List[str]:
        """
        関連キーワードを生成
        
        Args:
            keyword: 元のキーワード
            max_count: 最大生成数
            use_existing_keywords: 既存キーワードから検索するか
            use_patterns: パターンマッチングを使用するか
            use_similarity: 類似度ベースの検索を使用するか
        
        Returns:
            related_keywords: 関連キーワードのリスト
        """
        keyword_lower = keyword.lower().strip()
        related_keywords: Set[str] = set()
        
        # パターンマッチングによる生成
        if use_patterns:
            pattern_keywords = self._generate_from_patterns(keyword_lower)
            related_keywords.update(pattern_keywords)
        
        # 既存キーワードから類似キーワードを検索
        if use_existing_keywords and self.keyword_coordinator:
            existing_keywords = self._generate_from_existing_keywords(keyword_lower)
            related_keywords.update(existing_keywords)
        
        # 類似度ベースの生成
        if use_similarity and self.keyword_coordinator:
            similarity_keywords = self._generate_from_similarity(keyword_lower)
            related_keywords.update(similarity_keywords)
        
        # 元のキーワードを除外
        related_keywords.discard(keyword_lower)
        related_keywords.discard(keyword)
        
        # リストに変換してソート（類似度順）
        result = list(related_keywords)
        
        # 類似度でソート
        if use_similarity:
            result.sort(key=lambda k: self._calculate_similarity(keyword_lower, k.lower()), reverse=True)
        
        # 最大数に制限
        result = result[:max_count]
        
        logger.info(f"[GENERATOR] Generated {len(result)} related keywords for '{keyword}'")
        return result
    
    def _generate_from_patterns(self, keyword: str) -> List[str]:
        """
        パターンマッチングから関連キーワードを生成
        
        Args:
            keyword: キーワード
        
        Returns:
            related_keywords: 関連キーワードのリスト
        """
        related_keywords = []
        keyword_lower = keyword.lower()
        
        # パターンマッチング
        for pattern_key, pattern_values in self.related_patterns.items():
            if pattern_key in keyword_lower or keyword_lower in pattern_key:
                related_keywords.extend(pattern_values)
        
        # 部分一致チェック
        for pattern_key, pattern_values in self.related_patterns.items():
            if self._calculate_similarity(keyword_lower, pattern_key) >= self.similarity_threshold:
                related_keywords.extend(pattern_values)
        
        return related_keywords
    
    def _generate_from_existing_keywords(self, keyword: str) -> List[str]:
        """
        既存キーワードから類似キーワードを検索
        
        Args:
            keyword: キーワード
        
        Returns:
            related_keywords: 関連キーワードのリスト
        """
        if not self.keyword_coordinator:
            return []
        
        related_keywords = []
        
        try:
            # すべてのキーワードを取得
            all_keywords = self.keyword_coordinator.get_all_keywords()
            
            for kw_data in all_keywords:
                existing_keyword = kw_data.get('keyword', '')
                if not existing_keyword:
                    continue
                
                existing_keyword_lower = existing_keyword.lower()
                
                # 類似度チェック
                similarity = self._calculate_similarity(keyword, existing_keyword_lower)
                if similarity >= self.similarity_threshold and existing_keyword_lower != keyword:
                    related_keywords.append(existing_keyword)
                
                # 部分文字列チェック
                if keyword in existing_keyword_lower or existing_keyword_lower in keyword:
                    if existing_keyword_lower != keyword:
                        related_keywords.append(existing_keyword)
        
        except Exception as e:
            logger.warning(f"[GENERATOR] Failed to generate from existing keywords: {e}")
        
        return related_keywords
    
    def _generate_from_similarity(self, keyword: str) -> List[str]:
        """
        類似度ベースで関連キーワードを生成
        
        Args:
            keyword: キーワード
        
        Returns:
            related_keywords: 関連キーワードのリスト
        """
        if not self.keyword_coordinator:
            return []
        
        related_keywords = []
        
        try:
            # すべてのキーワードを取得
            all_keywords = self.keyword_coordinator.get_all_keywords()
            
            # 類似度でソート
            keyword_similarities = []
            for kw_data in all_keywords:
                existing_keyword = kw_data.get('keyword', '')
                if not existing_keyword:
                    continue
                
                existing_keyword_lower = existing_keyword.lower()
                if existing_keyword_lower == keyword:
                    continue
                
                similarity = self._calculate_similarity(keyword, existing_keyword_lower)
                if similarity >= self.similarity_threshold:
                    keyword_similarities.append((similarity, existing_keyword))
            
            # 類似度順にソート
            keyword_similarities.sort(key=lambda x: x[0], reverse=True)
            
            # 上位10件を取得
            related_keywords = [kw for _, kw in keyword_similarities[:10]]
        
        except Exception as e:
            logger.warning(f"[GENERATOR] Failed to generate from similarity: {e}")
        
        return related_keywords
    
    def _calculate_similarity(self, keyword1: str, keyword2: str) -> float:
        """
        2つのキーワードの類似度を計算
        
        Args:
            keyword1: キーワード1
            keyword2: キーワード2
        
        Returns:
            similarity: 類似度（0.0-1.0）
        """
        if not keyword1 or not keyword2:
            return 0.0
        
        # SequenceMatcherを使用した類似度計算
        similarity = SequenceMatcher(None, keyword1.lower(), keyword2.lower()).ratio()
        
        # 部分文字列の一致も考慮
        if keyword1.lower() in keyword2.lower() or keyword2.lower() in keyword1.lower():
            similarity = max(similarity, 0.7)
        
        return similarity
    
    def generate_variations(self, keyword: str) -> List[str]:
        """
        キーワードのバリエーションを生成
        
        Args:
            keyword: キーワード
        
        Returns:
            variations: バリエーションのリスト
        """
        variations = []
        keyword_lower = keyword.lower()
        
        # 一般的な接頭辞・接尾辞
        prefixes = ['', 'learn ', 'master ', 'advanced ', 'beginner ', 'introduction to ']
        suffixes = ['', ' tutorial', ' guide', ' examples', ' best practices', ' patterns']
        
        for prefix in prefixes:
            for suffix in suffixes:
                variation = f"{prefix}{keyword}{suffix}".strip()
                if variation.lower() != keyword_lower:
                    variations.append(variation)
        
        # 複数形・動詞形など
        if keyword_lower.endswith('s'):
            variations.append(keyword[:-1])  # 単数形
        else:
            variations.append(keyword + 's')  # 複数形
        
        return variations[:10]  # 最大10件


def main():
    """メイン関数（テスト用）"""
    from scripts.utils.keyword_coordinator import KeywordCoordinator
    
    coordinator = KeywordCoordinator()
    generator = KeywordGenerator(keyword_coordinator=coordinator)
    
    # テストキーワード
    test_keyword = "Python"
    related = generator.generate_related_keywords(test_keyword, max_count=10)
    print(f"Related keywords for '{test_keyword}': {related}")
    
    # バリエーション生成
    variations = generator.generate_variations(test_keyword)
    print(f"Variations for '{test_keyword}': {variations}")


if __name__ == "__main__":
    main()































































