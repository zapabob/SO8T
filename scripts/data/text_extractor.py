#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高度なテキスト抽出モジュール

Wikipedia固有の構造を考慮した抽出、参考文献・脚注の除去、表・図表の適切な処理、
セクション構造の保持、品質スコア計算を行う。

Usage:
    from scripts.data.text_extractor import WikipediaTextExtractor
    extractor = WikipediaTextExtractor()
    text, quality_score = extractor.extract_text(html, url)
"""

import re
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from bs4 import BeautifulSoup, Tag, NavigableString
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class WikipediaTextExtractor:
    """Wikipediaテキスト抽出器"""
    
    def __init__(
        self,
        min_text_length: int = 200,
        max_text_length: int = 5000,
        remove_references: bool = True,
        remove_tables: bool = True,
        remove_templates: bool = True,
        preserve_sections: bool = True
    ):
        """
        Args:
            min_text_length: 最小テキスト長
            max_text_length: 最大テキスト長
            remove_references: 参考文献を除去するか
            remove_tables: 表を除去するか
            remove_templates: テンプレートを除去するか
            preserve_sections: セクション構造を保持するか
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.remove_references = remove_references
        self.remove_tables = remove_tables
        self.remove_templates = remove_templates
        self.preserve_sections = preserve_sections
    
    def extract_text(
        self,
        html: str,
        url: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        HTMLからテキストを抽出
        
        Args:
            html: HTML文字列
            url: URL（オプション、デバッグ用）
        
        Returns:
            (抽出されたテキスト, 品質スコア)のタプル
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # メインコンテンツを取得
        content = self._get_main_content(soup)
        if content is None:
            return "", 0.0
        
        # 不要な要素を除去
        self._remove_unwanted_elements(content)
        
        # テキストを抽出
        text = self._extract_text_from_content(content)
        
        # 正規化
        text = self._normalize_text(text)
        
        # 品質スコアを計算
        quality_score = self._calculate_quality_score(text, content)
        
        # 長さチェック
        if len(text) < self.min_text_length:
            logger.debug(f"[TEXT_EXTRACT] Text too short: {len(text)} < {self.min_text_length}")
            return "", 0.0
        
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
            quality_score *= 0.9  # 切り詰めによる品質低下
        
        return text, quality_score
    
    def _get_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """メインコンテンツを取得"""
        # Wikipediaのメインコンテンツエリア
        content = soup.find('div', {'id': 'mw-content-text'})
        
        if content is None:
            # フォールバック: body全体
            content = soup.find('body')
        
        return content
    
    def _remove_unwanted_elements(self, content: Tag):
        """不要な要素を除去"""
        # スクリプトとスタイル
        for tag in content.find_all(['script', 'style']):
            tag.decompose()
        
        # ナビゲーション要素
        for tag in content.find_all(['nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # 参考文献セクション
        if self.remove_references:
            # 参考文献セクション
            ref_section = content.find('div', {'id': 'references'}) or content.find('div', {'class': 'reflist'})
            if ref_section:
                ref_section.decompose()
            
            # 脚注
            for tag in content.find_all(['sup', 'span'], class_=re.compile(r'reference|cite')):
                tag.decompose()
        
        # 表
        if self.remove_tables:
            for tag in content.find_all('table'):
                tag.decompose()
        
        # テンプレート
        if self.remove_templates:
            # Wikipediaテンプレート
            for tag in content.find_all(['div', 'span'], class_=re.compile(r'template|infobox|navbox')):
                tag.decompose()
            
            # スタブ、要出典などのテンプレート
            for tag in content.find_all(['span', 'div'], class_=re.compile(r'stub|citation|dablink')):
                tag.decompose()
        
        # 外部リンクセクション
        ext_links = content.find('div', {'id': 'external_links'}) or content.find('div', {'class': 'external-links'})
        if ext_links:
            ext_links.decompose()
        
        # カテゴリセクション
        cat_links = content.find('div', {'id': 'catlinks'}) or content.find('div', {'class': 'catlinks'})
        if cat_links:
            cat_links.decompose()
    
    def _extract_text_from_content(self, content: Tag) -> str:
        """コンテンツからテキストを抽出"""
        if self.preserve_sections:
            # セクション構造を保持
            sections = []
            
            # 見出しと段落を順序付きで抽出
            for element in content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                text = element.get_text(separator=' ', strip=True)
                if text:
                    # 見出しの場合は強調
                    if element.name.startswith('h'):
                        sections.append(f"\n\n## {text}\n\n")
                    else:
                        sections.append(text)
            
            return ' '.join(sections)
        else:
            # シンプルなテキスト抽出
            return content.get_text(separator='\n', strip=True)
    
    def _normalize_text(self, text: str) -> str:
        """テキストを正規化"""
        # 連続する改行を削減
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # 連続する空白を削減
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 行頭・行末の空白を削除
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        return text.strip()
    
    def _calculate_quality_score(
        self,
        text: str,
        content: Tag
    ) -> float:
        """
        品質スコアを計算（0.0-1.0）
        
        評価項目:
        1. 長さスコア（30%）
        2. 日本語含有率（40%）
        3. 句読点適切さ（20%）
        4. 語彙多様性（10%）
        """
        if not text:
            return 0.0
        
        score = 0.0
        length = len(text)
        
        # 1. 長さスコア（30%）
        if self.min_text_length <= length <= self.max_text_length:
            length_score = 0.3
        elif length < self.min_text_length:
            length_score = (length / self.min_text_length) * 0.3
        else:
            length_score = 0.3 * (1.0 - (length - self.max_text_length) / self.max_text_length)
        score += max(0.0, length_score)
        
        # 2. 日本語含有率（40%）
        japanese_chars = sum(1 for c in text if ord(c) > 0x3040 and ord(c) < 0x309F or 
                            ord(c) > 0x30A0 and ord(c) < 0x30FF or
                            ord(c) > 0x4E00 and ord(c) < 0x9FFF)
        japanese_ratio = japanese_chars / max(length, 1)
        score += japanese_ratio * 0.4
        
        # 3. 句読点適切さ（20%）
        punctuation = text.count('。') + text.count('、') + text.count('.') + text.count(',')
        expected_punct = length / 50
        if 0.5 * expected_punct <= punctuation <= 1.5 * expected_punct:
            score += 0.2
        elif punctuation > 0:
            score += 0.1
        
        # 4. 語彙多様性（10%）
        words = text.split()
        if len(words) > 0:
            unique_words = len(set(words))
            diversity = unique_words / len(words)
            if diversity > 0.3:
                score += 0.1
            elif diversity > 0.2:
                score += 0.05
        
        return min(score, 1.0)
    
    def extract_metadata(self, html: str) -> Dict[str, Any]:
        """
        HTMLからメタデータを抽出
        
        Args:
            html: HTML文字列
        
        Returns:
            メタデータ辞書
        """
        soup = BeautifulSoup(html, 'lxml')
        metadata = {}
        
        # タイトル
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # 説明
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        # 言語
        html_tag = soup.find('html')
        if html_tag:
            metadata['language'] = html_tag.get('lang', '')
        
        # セクション数
        content = self._get_main_content(soup)
        if content:
            sections = content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            metadata['section_count'] = len(sections)
        
        return metadata

