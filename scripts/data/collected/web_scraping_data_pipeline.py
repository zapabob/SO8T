#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webスクレイピングデータパイプライン処理

データクレンジング、ラベル付け、四値分類を実行するパイプライン

Usage:
    python scripts/pipelines/web_scraping_data_pipeline.py --input D:/webdataset/processed --output D:/webdataset/cleaned
"""

import sys
import json
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import time
import statistics

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# SO8Tモデルインポート
try:
    from so8t_mmllm.src.models.so8t_thinking_model import SO8TThinkingModel
    from so8t_mmllm.src.models.thinking_tokens import (
        add_thinking_tokens_to_tokenizer,
        build_quadruple_thinking_prompt,
        extract_quadruple_thinking
    )
    from transformers import AutoTokenizer
    SO8T_AVAILABLE = True
except ImportError:
    SO8T_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SO8T model not available for quadruple classification")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/web_scraping_data_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """データクレンジングクラス"""
    
    def __init__(self, exclude_empty_text: bool = True, fill_missing_from_content: bool = True):
        """
        初期化
        
        Args:
            exclude_empty_text: テキスト長が0のサンプルを除外するか
            fill_missing_from_content: テキストフィールドが欠損している場合、contentフィールドから補完を試みるか
        """
        self.exclude_empty_text = exclude_empty_text
        self.fill_missing_from_content = fill_missing_from_content
        
        # 必須フィールド
        self.required_fields = ['text', 'nsfw_label', 'category', 'domain']
        
        # 不要なパターン
        self.noise_patterns = [
            r'<script[^>]*>.*?</script>',  # スクリプトタグ
            r'<style[^>]*>.*?</style>',  # スタイルタグ
            r'<[^>]+>',  # HTMLタグ
            r'&[a-z]+;',  # HTMLエンティティ
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URL
            r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF]',  # 特殊文字（日本語文字は保持）
        ]
        
        # 空白文字の正規化
        self.whitespace_patterns = [
            (r'\s+', ' '),  # 連続する空白を1つに
            (r'\n+', '\n'),  # 連続する改行を1つに
            (r'\t+', ' '),  # タブを空白に
        ]
    
    def clean_text(self, text: str) -> str:
        """
        テキストをクレンジング
        
        Args:
            text: クレンジングするテキスト
        
        Returns:
            クレンジングされたテキスト
        """
        if not text:
            return ""
        
        # 不要なパターンを除去
        cleaned = text
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # 空白文字の正規化
        for pattern, replacement in self.whitespace_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # 前後の空白を削除
        cleaned = cleaned.strip()
        
        return cleaned
    
    def validate_sample(self, sample: Dict) -> Tuple[bool, List[str]]:
        """
        サンプルのバリデーション
        
        Args:
            sample: バリデーションするサンプル
        
        Returns:
            (is_valid, missing_fields) タプル
        """
        missing_fields = []
        
        for field in self.required_fields:
            if field not in sample or not sample[field]:
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    
    def fill_missing_text(self, sample: Dict) -> Dict:
        """
        欠損しているテキストフィールドを補完
        
        Args:
            sample: 補完するサンプル
        
        Returns:
            補完されたサンプル
        """
        filled_sample = sample.copy()
        
        # textフィールドが欠損または空の場合、contentフィールドから補完を試みる
        if 'text' not in filled_sample or not filled_sample.get('text'):
            if 'content' in filled_sample and filled_sample['content']:
                filled_sample['text'] = filled_sample['content']
                logger.debug(f"[CLEANER] Filled missing text from content field")
            elif 'title' in filled_sample and filled_sample['title']:
                filled_sample['text'] = filled_sample['title']
                logger.debug(f"[CLEANER] Filled missing text from title field")
        
        # nsfw_labelが欠損している場合、デフォルト値を設定
        if 'nsfw_label' not in filled_sample or not filled_sample.get('nsfw_label'):
            filled_sample['nsfw_label'] = 'safe'
            logger.debug(f"[CLEANER] Filled missing nsfw_label with default 'safe'")
        
        # categoryが欠損している場合、デフォルト値を設定
        if 'category' not in filled_sample or not filled_sample.get('category'):
            filled_sample['category'] = 'general'
            logger.debug(f"[CLEANER] Filled missing category with default 'general'")
        
        # domainが欠損している場合、デフォルト値を設定
        if 'domain' not in filled_sample or not filled_sample.get('domain'):
            # URLからドメインを抽出を試みる
            url = filled_sample.get('url', '')
            if url:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    filled_sample['domain'] = parsed.netloc or 'unknown'
                except Exception:
                    filled_sample['domain'] = 'unknown'
            else:
                filled_sample['domain'] = 'unknown'
            logger.debug(f"[CLEANER] Filled missing domain with default 'unknown'")
        
        return filled_sample
    
    def clean_sample(self, sample: Dict) -> Optional[Dict]:
        """
        サンプルをクレンジング
        
        Args:
            sample: クレンジングするサンプル
        
        Returns:
            クレンジングされたサンプル（無効な場合はNone）
        """
        cleaned_sample = sample.copy()
        
        # 欠損データの補完
        if self.fill_missing_from_content:
            cleaned_sample = self.fill_missing_text(cleaned_sample)
        
        # テキストをクレンジング
        if 'text' in cleaned_sample:
            cleaned_sample['text'] = self.clean_text(cleaned_sample['text'])
            cleaned_sample['text_length'] = len(cleaned_sample['text'])
            
            # テキスト長が0のサンプルを除外
            if self.exclude_empty_text and cleaned_sample['text_length'] == 0:
                logger.debug(f"[CLEANER] Excluding sample with empty text")
                return None
        else:
            # テキストフィールドが存在しない場合は除外
            if self.exclude_empty_text:
                logger.debug(f"[CLEANER] Excluding sample without text field")
                return None
        
        # タイトルをクレンジング
        if 'title' in cleaned_sample:
            cleaned_sample['title'] = self.clean_text(cleaned_sample['title'])
        
        # バリデーション
        is_valid, missing_fields = self.validate_sample(cleaned_sample)
        if not is_valid:
            logger.warning(f"[CLEANER] Sample validation failed, missing fields: {missing_fields}")
            # 必須フィールドが欠損している場合は除外
            return None
        
        # メタデータを追加
        cleaned_sample['cleaned_at'] = datetime.now().isoformat()
        cleaned_sample['cleaning_version'] = '1.1'  # 欠損データ処理版
        
        return cleaned_sample


class DataLabeler:
    """データラベル付けクラス"""
    
    def __init__(self):
        """初期化"""
        # カテゴリキーワードマッピング
        self.category_keywords = {
            'technology': ['技術', 'テクノロジー', 'IT', 'プログラミング', 'ソフトウェア', 'ハードウェア', 'AI', '機械学習', 'technology', 'tech', 'programming', 'software', 'hardware', 'artificial intelligence', 'machine learning'],
            'science': ['科学', 'サイエンス', '研究', '実験', '論文', 'science', 'research', 'experiment', 'paper', 'study'],
            'business': ['ビジネス', '経営', '企業', '経済', 'マーケティング', 'business', 'management', 'company', 'economy', 'marketing'],
            'entertainment': ['エンターテインメント', '娯楽', '映画', '音楽', 'ゲーム', 'entertainment', 'movie', 'music', 'game'],
            'news': ['ニュース', '報道', '記事', 'news', 'article', 'report'],
            'education': ['教育', '学習', '学校', '大学', 'education', 'learning', 'school', 'university'],
            'health': ['健康', '医療', '病気', '治療', 'health', 'medical', 'disease', 'treatment'],
            'nsfw_detection': ['nsfw', 'adult', 'explicit', 'mature', '18+', 'xxx'],
        }
        
        # 言語検出キーワード
        self.language_keywords = {
            'ja': ['の', 'は', 'を', 'に', 'が', 'で', 'と', 'から', 'まで', 'です', 'ます', 'である', 'だ'],
            'en': ['the', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'],
        }
    
    def detect_category(self, text: str, existing_category: Optional[str] = None) -> str:
        """
        カテゴリを検出
        
        Args:
            text: テキスト
            existing_category: 既存のカテゴリ
        
        Returns:
            検出されたカテゴリ
        """
        if existing_category:
            return existing_category
        
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def detect_language(self, text: str, existing_language: Optional[str] = None) -> str:
        """
        言語を検出
        
        Args:
            text: テキスト
            existing_language: 既存の言語
        
        Returns:
            検出された言語
        """
        if existing_language:
            return existing_language
        
        text_lower = text.lower()
        
        # 日本語文字の検出
        japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF]', text)
        japanese_ratio = len(japanese_chars) / max(len(text), 1)
        
        # 英語キーワードの検出
        english_keywords = sum(1 for keyword in self.language_keywords['en'] if keyword in text_lower)
        
        if japanese_ratio > 0.1:
            return 'ja'
        elif english_keywords > 5:
            return 'en'
        else:
            return 'unknown'
    
    def label_sample(self, sample: Dict) -> Dict:
        """
        サンプルにラベルを付与
        
        Args:
            sample: ラベル付けするサンプル
        
        Returns:
            ラベル付けされたサンプル
        """
        labeled_sample = sample.copy()
        
        text = labeled_sample.get('text', '')
        
        # カテゴリ検出
        if 'category' not in labeled_sample or not labeled_sample['category']:
            labeled_sample['category'] = self.detect_category(text, labeled_sample.get('category'))
        
        # 言語検出
        if 'language' not in labeled_sample or not labeled_sample['language']:
            labeled_sample['language'] = self.detect_language(text, labeled_sample.get('language'))
        
        # ラベル付け時刻を記録
        labeled_sample['labeled_at'] = datetime.now().isoformat()
        labeled_sample['labeling_version'] = '1.0'
        
        return labeled_sample


class QuadrupleClassifier:
    """四値分類クラス（SO8T四重推論）"""
    
    def __init__(self, so8t_model_path: Optional[str] = None):
        """
        初期化
        
        Args:
            so8t_model_path: SO8Tモデルのパス
        """
        self.so8t_model = None
        self.so8t_tokenizer = None
        self.so8t_model_path = so8t_model_path
        self.last_availability_check = 0
        self.availability_check_interval = 300  # 5分ごとにチェック
        
        if SO8T_AVAILABLE:
            try:
                self._initialize_so8t_model(so8t_model_path)
                logger.info("[QUADRUPLE] SO8T model initialized for quadruple classification")
            except Exception as e:
                logger.warning(f"[QUADRUPLE] Failed to initialize SO8T model: {e}")
                logger.warning("[QUADRUPLE] Continuing without SO8T classification")
    
    def _initialize_so8t_model(self, model_path: Optional[str] = None):
        """SO8Tモデルを初期化"""
        try:
            # SO8Tモデルローダーを使用
            try:
                from scripts.utils.so8t_model_loader import load_so8t_model, find_so8t_model_paths
                SO8T_LOADER_AVAILABLE = True
            except ImportError:
                SO8T_LOADER_AVAILABLE = False
            
            if SO8T_LOADER_AVAILABLE:
                # SO8Tモデルローダーを使用してモデルをロード
                model, tokenizer, success = load_so8t_model(
                    model_path=model_path,
                    device="auto",
                    use_quadruple_thinking=True,
                    use_redacted_tokens=False,
                    fallback_to_default=True
                )
                
                if success and model is not None and tokenizer is not None:
                    self.so8t_model = model
                    self.so8t_tokenizer = tokenizer
                    logger.info("[QUADRUPLE] Model loaded successfully using model loader")
                    return
                else:
                    logger.warning("[QUADRUPLE] SO8T model loader failed, falling back to manual initialization")
            
            # フォールバック: 手動初期化
            if model_path is None:
                # デフォルトモデルパスを探す
                if SO8T_LOADER_AVAILABLE:
                    found_paths = find_so8t_model_paths()
                    if found_paths:
                        model_path = str(found_paths[0])
                        logger.info(f"[QUADRUPLE] Auto-detected SO8T model path: {model_path}")
                    else:
                        raise FileNotFoundError("SO8T model not found")
                else:
                    default_paths = [
                        "D:/webdataset/models/so8t-phi4-so8t-ja-finetuned",
                        "models/so8t-phi4-so8t-ja-finetuned",
                        "so8t-mmllm/models/so8t-phi4-so8t-ja-finetuned"
                    ]
                    for path in default_paths:
                        if Path(path).exists():
                            model_path = path
                            break
                    
                    if model_path is None:
                        raise FileNotFoundError("SO8T model not found")
            
            logger.info(f"[QUADRUPLE] Loading model from: {model_path}")
            
            # トークナイザーを読み込み
            self.so8t_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # SO8T設定（簡易版）
            from so8t_mmllm.src.models.safety_aware_so8t import SafetyAwareSO8TConfig
            so8t_config = SafetyAwareSO8TConfig(
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=8,
                intermediate_size=11008,
                max_position_embeddings=4096,
                use_so8_rotation=True,
                use_safety_head=True,
                use_verifier_head=True
            )
            
            # SO8TThinkingModelを読み込み
            self.so8t_model = SO8TThinkingModel(
                base_model_name_or_path=model_path,
                so8t_config=so8t_config,
                use_redacted_tokens=False,
                use_quadruple_thinking=True
            )
            
            # トークナイザーを設定
            self.so8t_model.set_tokenizer(self.so8t_tokenizer)
            
            # 評価モードに設定
            self.so8t_model.eval()
            
            logger.info("[QUADRUPLE] Model loaded successfully")
            
        except Exception as e:
            logger.error(f"[QUADRUPLE] Failed to initialize model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def classify_quadruple(self, sample: Dict) -> Dict:
        """
        サンプルを四値分類（SO8T四重推論、精度向上版）
        
        Args:
            sample: 分類するサンプル
        
        Returns:
            四値分類結果を含むサンプル
        """
        classified_sample = sample.copy()
        
        # SO8Tモデルの利用可能性をチェック
        so8t_available = self._check_so8t_availability()
        
        # ルールベース分類を常に実行（ハイブリッド分類のため）
        rule_based_label = self._classify_with_rules(classified_sample)
        
        if not so8t_available:
            # SO8Tモデルが利用できない場合はルールベース分類を使用
            classified_sample['quadruple_classification'] = {
                'task': 'rule_based',
                'safety': 'rule_based',
                'policy': 'rule_based',
                'final': rule_based_label,
                'four_class_label': rule_based_label,
                'four_class_label_id': {'ALLOW': 0, 'ESCALATION': 1, 'DENY': 2, 'REFUSE': 3}.get(rule_based_label, 0),
                'reasoning': 'SO8T model not available, using rule-based classification',
                'classification_method': 'rule_based'
            }
            classified_sample['classified_at'] = datetime.now().isoformat()
            classified_sample['classification_version'] = '1.2'  # ルールベース分類版
            return classified_sample
        
        try:
            text = classified_sample.get('text', '')
            category = classified_sample.get('category', 'unknown')
            language = classified_sample.get('language', 'unknown')
            nsfw_label = classified_sample.get('nsfw_label', 'safe')
            
            # プロンプト構築（精度向上版）
            prompt = self._build_enhanced_classification_prompt(text, category, language, nsfw_label)
            
            # SO8T四重推論を実行（パラメータ調整）
            result = self.so8t_model.generate_thinking(
                self.so8t_tokenizer,
                prompt,
                max_new_tokens=512,  # 推論内容を増やす
                temperature=0.3,  # より決定論的に
                top_p=0.95,  # より多様な推論
                top_k=50,  # top-kサンプリング
                do_sample=True,
                device="cuda" if __import__('torch').cuda.is_available() else "cpu"
            )
            
            # 四重推論を抽出
            if self.so8t_model.use_quadruple_thinking:
                task_text, safety_text, policy_text, final_text = extract_quadruple_thinking(
                    result.get('full_text', '')
                )
                
                # 分類結果を抽出（精度向上版）
                task_class = self._extract_enhanced_classification(task_text, 'task')
                safety_class = self._extract_enhanced_classification(safety_text, 'safety')
                policy_class = self._extract_enhanced_classification(policy_text, 'policy')
                final_class = self._extract_enhanced_classification(final_text, 'final')
                
                # 信頼度スコアを計算
                task_confidence = self._calculate_confidence(task_text, task_class)
                safety_confidence = self._calculate_confidence(safety_text, safety_class)
                policy_confidence = self._calculate_confidence(policy_text, policy_class)
                final_confidence = self._calculate_confidence(final_text, final_class)
                
                # 四値分類ラベルにマッピング
                four_class_label = self._map_to_four_class(final_class, classified_sample)
                
                so8t_result = {
                    'task': task_class,
                    'safety': safety_class,
                    'policy': policy_class,
                    'final': final_class,
                    'four_class_label': four_class_label,
                    'four_class_label_id': {'ALLOW': 0, 'ESCALATION': 1, 'DENY': 2, 'REFUSE': 3}.get(four_class_label, 0),
                    'task_confidence': task_confidence,
                    'safety_confidence': safety_confidence,
                    'policy_confidence': policy_confidence,
                    'final_confidence': final_confidence,
                    'task_reasoning': task_text,
                    'safety_reasoning': safety_text,
                    'policy_reasoning': policy_text,
                    'final_reasoning': final_text,
                    'classification_method': 'so8t'
                }
                
                # ハイブリッド分類を実行
                classified_sample['quadruple_classification'] = self._hybrid_classification(
                    so8t_result, rule_based_label, classified_sample
                )
            else:
                # 基本形式の場合
                thinking_text = result.get('thinking', '')
                final_text = result.get('final', '')
                
                final_class = self._extract_enhanced_classification(final_text, 'final')
                final_confidence = self._calculate_confidence(final_text, final_class)
                
                # 四値分類ラベルにマッピング
                four_class_label = self._map_to_four_class(final_class, classified_sample)
                
                so8t_result = {
                    'task': 'unknown',
                    'safety': 'unknown',
                    'policy': 'unknown',
                    'final': final_class,
                    'four_class_label': four_class_label,
                    'four_class_label_id': {'ALLOW': 0, 'ESCALATION': 1, 'DENY': 2, 'REFUSE': 3}.get(four_class_label, 0),
                    'final_confidence': final_confidence,
                    'thinking': thinking_text,
                    'final_reasoning': final_text,
                    'classification_method': 'so8t_basic'
                }
                
                # ハイブリッド分類を実行
                classified_sample['quadruple_classification'] = self._hybrid_classification(
                    so8t_result, rule_based_label, classified_sample
                )
            
            classified_sample['classified_at'] = datetime.now().isoformat()
            classified_sample['classification_version'] = '1.3'  # ハイブリッド分類版
            
        except Exception as e:
            logger.error(f"[QUADRUPLE] Classification failed: {e}")
            # エラー時はルールベース分類を使用
            classified_sample['quadruple_classification'] = {
                'task': 'error',
                'safety': 'error',
                'policy': 'error',
                'final': 'error',
                'four_class_label': rule_based_label,
                'four_class_label_id': {'ALLOW': 0, 'ESCALATION': 1, 'DENY': 2, 'REFUSE': 3}.get(rule_based_label, 0),
                'error': str(e),
                'classification_method': 'rule_based_fallback',
                'rule_based_result': rule_based_label
            }
        
        return classified_sample
    
    def _build_enhanced_classification_prompt(self, text: str, category: str, language: str, nsfw_label: str) -> str:
        """精度向上版分類用プロンプトを構築"""
        # テキストの要約（長い場合は要約）
        text_preview = text[:1000] if len(text) > 1000 else text
        if len(text) > 1000:
            text_preview += "..."
        
        prompt = f"""以下のテキストを四重推論で分類してください。各推論で詳細な分析を行い、明確な判断を下してください。

カテゴリ: {category}
言語: {language}
NSFWラベル: {nsfw_label}
テキスト: {text_preview}

四重推論を行い、以下を分類してください：

1. <think-task>: このテキストのタスク適切性を評価
   - 適切: タスクに適切で有用なコンテンツ
   - 不適切: タスクに不適切で無関係なコンテンツ
   - 要検討: タスクとの関連性が不明確で検討が必要
   - 理由を明確に説明してください

2. <think-safety>: このテキストの安全性を評価
   - 安全: 安全で問題のないコンテンツ
   - 要注意: 注意が必要なコンテンツ（NSFW、暴力的表現など）
   - 危険: 危険なコンテンツ（違法、有害など）
   - 理由を明確に説明してください

3. <think-policy>: このテキストのポリシー準拠性を評価
   - 準拠: ポリシーに完全に準拠している
   - 要確認: ポリシー準拠性が不明確で確認が必要
   - 違反: ポリシーに違反している可能性がある
   - 理由を明確に説明してください

4. <final>: 最終判断を下してください
   - 承認: このテキストを承認し、使用可能
   - 要確認: 確認が必要で、人間の判断を要する
   - 拒否: このテキストを拒否し、使用不可
   - 理由を明確に説明してください

回答: <think-task>"""
        
        return prompt
    
    def _extract_enhanced_classification(self, text: str, classification_type: str) -> str:
        """精度向上版分類結果を抽出"""
        text_lower = text.lower()
        
        # 分類タイプごとのキーワード
        if classification_type == 'task':
            if any(kw in text for kw in ['適切', 'appropriate', '有用', 'useful', '関連', 'relevant']):
                if any(kw in text for kw in ['不適切', 'inappropriate', '無関係', 'irrelevant']):
                    return 'inappropriate'
                return 'appropriate'
            elif any(kw in text for kw in ['不適切', 'inappropriate', '無関係', 'irrelevant']):
                return 'inappropriate'
            elif any(kw in text for kw in ['要検討', 'review', '検討', 'consider']):
                return 'review_needed'
        
        elif classification_type == 'safety':
            if any(kw in text for kw in ['安全', 'safe', '問題なし', 'no issue']):
                return 'safe'
            elif any(kw in text for kw in ['要注意', 'caution', '注意', 'warning']):
                return 'caution'
            elif any(kw in text for kw in ['危険', 'dangerous', '有害', 'harmful', '違法', 'illegal']):
                return 'dangerous'
        
        elif classification_type == 'policy':
            if any(kw in text for kw in ['準拠', 'compliant', '遵守', '遵守']):
                return 'compliant'
            elif any(kw in text for kw in ['要確認', 'review', '確認', 'verify']):
                return 'review_needed'
            elif any(kw in text for kw in ['違反', 'violation', '違反', 'breach']):
                return 'violation'
        
        elif classification_type == 'final':
            if any(kw in text for kw in ['承認', 'approved', '許可', 'allow']):
                return 'approved'
            elif any(kw in text for kw in ['要確認', 'review', '確認', 'verify']):
                return 'review_needed'
            elif any(kw in text for kw in ['拒否', 'rejected', '拒否', 'deny']):
                return 'rejected'
        
        return 'unknown'
    
    def _calculate_confidence(self, reasoning_text: str, classification: str) -> float:
        """分類の信頼度スコアを計算（0.0-1.0）"""
        if classification == 'unknown' or classification == 'error':
            return 0.0
        
        # 推論テキストの長さと明確さから信頼度を計算
        text_length = len(reasoning_text)
        has_reasoning = len(reasoning_text) > 50
        has_keywords = any(kw in reasoning_text.lower() for kw in ['理由', 'reason', 'because', 'ため', 'ので'])
        
        confidence = 0.5  # ベース信頼度
        
        if has_reasoning:
            confidence += 0.2
        if has_keywords:
            confidence += 0.2
        if text_length > 200:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _refine_nsfw_label(self, text: str, current_label: str) -> str:
        """
        NSFWラベルを詳細化（nsfw_detectedをより具体的なラベルに分類）
        
        Args:
            text: テキスト内容
            current_label: 現在のNSFWラベル
        
        Returns:
            詳細化されたNSFWラベル
        """
        # nsfw_detectedでない場合はそのまま返す
        if current_label != 'nsfw_detected':
            return current_label
        
        import re
        text_lower = text.lower()
        
        # キーワードパターン（優先度順：より厳格なものから）
        keyword_patterns = {
            'illegal_content': [
                r'爆弾', r'テロ', r'テロリスト', r'爆発物', r'製造方法',
                r'bomb', r'terror', r'terrorist', r'explosive', r'manufacture',
                r'違法', r'犯罪', r'殺人', r'殺害', r'暗殺'
            ],
            'self_harm': [
                r'自殺', r'自傷', r'自殺方法', r'自傷行為', r'リストカット',
                r'suicide', r'self.harm', r'self.injury', r'cutting',
                r'首つり', r'飛び降り', r'過剰摂取'
            ],
            'violence': [
                r'暴力', r'暴行', r'傷害', r'殺人', r'虐待',
                r'violence', r'assault', r'murder', r'abuse',
                r'拷問', r'リンチ', r'虐殺'
            ],
            'harassment': [
                r'差別', r'ヘイト', r'誹謗', r'中傷', r'いじめ',
                r'discrimination', r'hate', r'slander', r'bullying',
                r'人種差別', r'性差別', r'宗教差別'
            ],
            'nsfw_block': [
                r'性的', r'ポルノ', r'アダルト', r'エロ', r'わいせつ',
                r'sexual', r'porn', r'pornography', r'adult', r'obscene',
                r'児童', r'未成年', r'child'
            ],
            'weapons_detail': [
                r'武器', r'銃', r'ナイフ', r'爆発物', r'製造',
                r'weapon', r'gun', r'knife', r'explosive', r'manufacture',
                r'手順', r'作り方', r'how.to'
            ],
            'medical_advice_high_risk': [
                r'医療', r'診断', r'治療', r'薬', r'処方',
                r'medical', r'diagnosis', r'treatment', r'prescription',
                r'手術', r'投薬', r'副作用'
            ],
            'nsfw_soft': [
                r'恋愛', r'デート', r'関係', r'romance', r'relationship'
            ]
        }
        
        # 優先度順にチェック（より厳格なラベルを優先）
        priority_order = [
            'illegal_content', 'self_harm', 'violence', 'harassment',
            'nsfw_block', 'weapons_detail', 'medical_advice_high_risk', 'nsfw_soft'
        ]
        
        for label_type in priority_order:
            patterns = keyword_patterns[label_type]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    logger.debug(f"[NSFW_REFINE] Refined '{current_label}' to '{label_type}' (matched: {pattern})")
                    return label_type
        
        # マッチしない場合はnsfw_detectedのまま
        return current_label
    
    def _classify_with_rules(self, sample: Dict) -> str:
        """
        ルールベースのフォールバック分類
        
        Args:
            sample: 分類するサンプル
        
        Returns:
            四値分類ラベル（ALLOW/ESCALATION/DENY/REFUSE）
        """
        nsfw_label = sample.get('nsfw_label', 'safe')
        safety_label = sample.get('safety_label', 'ALLOW')
        domain = sample.get('domain', sample.get('domain_label', 'general'))
        text = sample.get('text', sample.get('content', ''))
        category = sample.get('category', 'general')
        
        # NSFWラベルの詳細化
        if nsfw_label == 'nsfw_detected':
            nsfw_label = self._refine_nsfw_label(text, nsfw_label)
        
        # デフォルトはALLOW
        four_class_label = 'ALLOW'
        
        # NSFW検知結果に基づく分類
        if nsfw_label in ['nsfw_block', 'violence', 'harassment', 'self_harm', 'illegal_content']:
            four_class_label = 'REFUSE'
        elif nsfw_label in ['nsfw_soft', 'weapons_detail', 'medical_advice_high_risk']:
            four_class_label = 'DENY'
        
        # Safety labelに基づく分類（NSFWラベルより優先度が高い）
        if safety_label == 'DENY':
            four_class_label = 'DENY'
        elif safety_label == 'REFUSE':
            four_class_label = 'REFUSE'
        elif safety_label == 'ESCALATION':
            four_class_label = 'ESCALATION'
        
        # ドメインに基づく分類（ESCALATION分類の追加）
        sensitive_domains = ['defense', 'medical', 'financial']
        domain_lower = domain.lower()
        
        # 機密ドメインのチェック（部分一致も含む）
        is_sensitive_domain = any(sd in domain_lower for sd in sensitive_domains)
        
        if is_sensitive_domain:
            # 機密ドメインのサンプルはESCALATION
            if four_class_label == 'ALLOW':
                four_class_label = 'ESCALATION'
            elif four_class_label == 'DENY' and len(text) > 1000:
                # 長文の機密ドメインでDENYの場合はESCALATIONに変更
                four_class_label = 'ESCALATION'
        elif len(text) > 1000:
            # 長文サンプル（>1000文字）はESCALATION
            if four_class_label == 'ALLOW':
                four_class_label = 'ESCALATION'
        
        # カテゴリに基づく分類
        sensitive_categories = ['violence', 'illegal', 'harmful']
        if category in sensitive_categories:
            if four_class_label == 'ALLOW':
                four_class_label = 'DENY'
        
        # キーワード分析結果を統合
        keyword_analysis = self._analyze_text_keywords(text)
        if keyword_analysis['has_dangerous_keywords']:
            if four_class_label == 'ALLOW':
                four_class_label = 'DENY'
            elif four_class_label in ['DENY', 'ESCALATION']:
                # より厳格な分類に更新
                if keyword_analysis['danger_level'] == 'high':
                    four_class_label = 'REFUSE'
        
        # ドメイン信頼度スコアを考慮
        domain_trust_score = self._calculate_domain_trust_score(domain, text)
        if domain_trust_score < 0.5:
            # 信頼度が低いドメインはエスカレーション
            if four_class_label == 'ALLOW':
                four_class_label = 'ESCALATION'
        
        # カテゴリの詳細化を適用
        refined_category, category_confidence = self._refine_category(category, text)
        if refined_category != category:
            # 詳細化されたカテゴリに基づく分類を再評価
            if refined_category in sensitive_categories:
                if four_class_label == 'ALLOW':
                    four_class_label = 'DENY'
            # 機密カテゴリ（medical、financial、defense）の場合はESCALATION
            if refined_category in ['medical', 'financial', 'defense']:
                if four_class_label == 'ALLOW':
                    four_class_label = 'ESCALATION'
        
        return four_class_label
    
    def _analyze_text_keywords(self, text: str) -> Dict:
        """
        テキスト内容のキーワード分析
        
        Args:
            text: 分析するテキスト
        
        Returns:
            キーワード分析結果の辞書
        """
        import re
        text_lower = text.lower()
        
        # 危険キーワードパターン
        dangerous_keywords = {
            'high': [
                r'爆弾', r'テロ', r'殺人', r'自殺', r'違法',
                r'bomb', r'terror', r'murder', r'suicide', r'illegal',
                r'武器', r'銃', r'weapon', r'gun'
            ],
            'medium': [
                r'暴力', r'暴行', r'傷害', r'差別', r'ヘイト',
                r'violence', r'assault', r'discrimination', r'hate',
                r'医療', r'診断', r'治療', r'medical', r'diagnosis'
            ],
            'low': [
                r'注意', r'警告', r'危険', r'caution', r'warning', r'danger'
            ]
        }
        
        # 機密キーワードパターン
        confidential_keywords = [
            r'機密', r'秘密', r'内部', r'confidential', r'secret', r'internal',
            r'個人情報', r'プライバシー', r'personal', r'privacy'
        ]
        
        # 違法キーワードパターン
        illegal_keywords = [
            r'違法', r'犯罪', r'不正', r'illegal', r'crime', r'fraud',
            r'詐欺', r'窃盗', r'強盗', r'theft', r'robbery'
        ]
        
        # キーワード検出
        has_dangerous_keywords = False
        danger_level = 'none'
        dangerous_count = 0
        confidential_count = 0
        illegal_count = 0
        
        for level, patterns in dangerous_keywords.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    has_dangerous_keywords = True
                    dangerous_count += matches
                    if level == 'high' or (level == 'medium' and danger_level != 'high'):
                        danger_level = level
        
        for pattern in confidential_keywords:
            confidential_count += len(re.findall(pattern, text_lower))
        
        for pattern in illegal_keywords:
            illegal_count += len(re.findall(pattern, text_lower))
        
        return {
            'has_dangerous_keywords': has_dangerous_keywords,
            'danger_level': danger_level,
            'dangerous_count': dangerous_count,
            'confidential_count': confidential_count,
            'illegal_count': illegal_count,
            'total_keyword_matches': dangerous_count + confidential_count + illegal_count
        }
    
    def _calculate_domain_trust_score(self, domain: str, text: str) -> float:
        """
        ドメインの信頼度スコアを計算（0.0-1.0）
        
        Args:
            domain: ドメイン名
            text: テキスト内容
        
        Returns:
            信頼度スコア（0.0-1.0、高いほど信頼できる）
        """
        # 信頼できるドメインリスト
        trusted_domains = [
            'wikipedia.org', 'edu', 'gov', 'ac.jp', 'go.jp',
            'nature.com', 'science.org', 'ieee.org', 'acm.org'
        ]
        
        # 信頼度が低いドメインリスト
        untrusted_domains = [
            'blog', 'forum', 'social', 'anonymous', 'tor',
            'bit.ly', 'tinyurl', 'short.link'
        ]
        
        domain_lower = domain.lower()
        
        # 信頼できるドメイン
        for trusted in trusted_domains:
            if trusted in domain_lower:
                return 0.9
        
        # 信頼度が低いドメイン
        for untrusted in untrusted_domains:
            if untrusted in domain_lower:
                return 0.3
        
        # ドメインの長さと構造から信頼度を推定
        # 短いドメインや特殊文字が多いドメインは信頼度が低い
        if len(domain) < 5:
            return 0.4
        elif len(domain) > 50:
            return 0.5
        
        # テキストの品質から信頼度を推定
        text_length = len(text)
        if text_length < 50:
            return 0.5
        elif text_length > 5000:
            return 0.7
        
        # デフォルト信頼度
        return 0.6
    
    def _refine_category(self, category: str, text: str) -> Tuple[str, float]:
        """
        カテゴリを詳細化
        
        Args:
            category: 現在のカテゴリ
            text: テキスト内容
        
        Returns:
            (詳細化されたカテゴリ, 信頼度スコア) タプル
        """
        import re
        text_lower = text.lower()
        
        # テキスト内容からカテゴリを推測（より詳細なキーワードマッピング）
        category_keywords = {
            'science': [
                r'科学', r'物理', r'化学', r'生物', r'数学',
                r'science', r'physics', r'chemistry', r'biology', r'mathematics',
                r'実験', r'研究', r'experiment', r'research', r'study',
                r'理論', r'仮説', r'theory', r'hypothesis'
            ],
            'technology': [
                r'技術', r'テクノロジー', r'コンピュータ', r'プログラミング',
                r'technology', r'computer', r'programming', r'software', r'hardware',
                r'AI', r'人工知能', r'機械学習', r'machine.learning', r'artificial.intelligence',
                r'アルゴリズム', r'algorithm', r'データ', r'data'
            ],
            'medicine': [
                r'医療', r'医学', r'診断', r'治療', r'薬', r'処方',
                r'medical', r'medicine', r'diagnosis', r'treatment', r'prescription',
                r'病気', r'疾患', r'disease', r'illness', r'health',
                r'患者', r'症状', r'patient', r'symptom'
            ],
            'history': [
                r'歴史', r'過去', r'時代', r'戦争', r'文明',
                r'history', r'historical', r'past', r'war', r'civilization',
                r'古代', r'中世', r'ancient', r'medieval', r'era'
            ],
            'culture': [
                r'文化', r'芸術', r'音楽', r'文学', r'映画',
                r'culture', r'art', r'music', r'literature', r'movie',
                r'伝統', r'習慣', r'tradition', r'custom', r'festival'
            ],
            'business': [
                r'ビジネス', r'企業', r'経営', r'市場', r'経済',
                r'business', r'company', r'management', r'market', r'economy',
                r'投資', r'株式', r'investment', r'stock', r'trade'
            ],
            'violence': [
                r'暴力', r'暴行', r'傷害', r'殺人', r'虐待',
                r'violence', r'assault', r'murder', r'abuse', r'attack'
            ],
            'illegal': [
                r'違法', r'犯罪', r'不正', r'illegal', r'crime',
                r'詐欺', r'窃盗', r'強盗', r'fraud', r'theft'
            ],
            'harmful': [
                r'有害', r'危険', r'有害物質', r'harmful', r'dangerous',
                r'毒', r'poison', r'toxin', r'hazard'
            ],
            'financial': [
                r'金融', r'投資', r'株式', r'finance', r'investment',
                r'stock', r'currency', r'bitcoin', r'banking', r'loan'
            ],
            'defense': [
                r'防衛', r'軍事', r'武器', r'defense', r'military',
                r'weapon', r'security', r'army', r'navy', r'air.force'
            ],
            'education': [
                r'教育', r'学習', r'学校', r'大学', r'教育',
                r'education', r'learning', r'school', r'university', r'student',
                r'授業', r'講義', r'lesson', r'lecture', r'course'
            ]
        }
        
        # カテゴリが汎用的な場合、テキストから推測
        if category in ['general', 'unknown', 'other']:
            category_scores = {}
            
            for refined_cat, patterns in category_keywords.items():
                score = 0.0
                matches = 0
                
                for pattern in patterns:
                    found_matches = len(re.findall(pattern, text_lower))
                    if found_matches > 0:
                        matches += found_matches
                        # マッチ数に応じてスコアを加算
                        score += found_matches * 0.1
                
                if matches > 0:
                    # マッチ数が多いほど信頼度が高い
                    category_scores[refined_cat] = min(score, 1.0)
            
            if category_scores:
                # 最もスコアが高いカテゴリを選択
                best_category = max(category_scores.items(), key=lambda x: x[1])
                confidence = best_category[1]
                
                # 信頼度が0.3以上の場合のみカテゴリを変更
                if confidence >= 0.3:
                    logger.debug(f"[CATEGORY_REFINE] Refined '{category}' to '{best_category[0]}' (confidence: {confidence:.2f})")
                    return best_category[0], confidence
        
        return category, 1.0
    
    def _check_so8t_availability(self) -> bool:
        """
        SO8Tモデルの利用可能性をチェック
        
        Returns:
            SO8Tモデルが利用可能な場合True
        """
        import time
        
        # 前回のチェックから一定時間経過していない場合はスキップ
        current_time = time.time()
        if current_time - self.last_availability_check < self.availability_check_interval:
            return self.so8t_model is not None
        
        self.last_availability_check = current_time
        
        # 既に初期化されている場合はTrue
        if self.so8t_model is not None:
            return True
        
        # SO8Tが利用可能でない場合はFalse
        if not SO8T_AVAILABLE:
            return False
        
        # SO8Tモデルの再初期化を試みる
        try:
            self._initialize_so8t_model(self.so8t_model_path)
            logger.info("[QUADRUPLE] SO8T model became available, switching to SO8T classification")
            return True
        except Exception as e:
            logger.debug(f"[QUADRUPLE] SO8T model still not available: {e}")
            return False
    
    def _hybrid_classification(
        self,
        so8t_result: Dict,
        rule_based_result: str,
        sample: Dict
    ) -> Dict:
        """
        ハイブリッド分類（SO8T分類とルールベース分類を統合）
        
        Args:
            so8t_result: SO8T分類結果
            rule_based_result: ルールベース分類結果
            sample: サンプルデータ
        
        Returns:
            統合された分類結果
        """
        so8t_four_class = so8t_result.get('four_class_label', 'unknown')
        so8t_confidence = so8t_result.get('final_confidence', 0.0)
        
        # SO8T分類の信頼度が高い場合（0.7以上）
        if so8t_confidence >= 0.7:
            # SO8T分類を優先
            final_label = so8t_four_class
            classification_method = 'so8t_primary'
            reasoning = f"SO8T classification (confidence: {so8t_confidence:.2f})"
        # SO8T分類の信頼度が中程度の場合（0.4-0.7）
        elif so8t_confidence >= 0.4:
            # SO8T分類とルールベース分類の両方を考慮
            # より厳格な分類を優先
            label_priority = {'REFUSE': 4, 'DENY': 3, 'ESCALATION': 2, 'ALLOW': 1}
            so8t_priority = label_priority.get(so8t_four_class, 0)
            rule_priority = label_priority.get(rule_based_result, 0)
            
            if so8t_priority >= rule_priority:
                final_label = so8t_four_class
            else:
                final_label = rule_based_result
            
            classification_method = 'hybrid'
            reasoning = f"Hybrid classification (SO8T: {so8t_four_class}, Rule: {rule_based_result}, SO8T confidence: {so8t_confidence:.2f})"
        # SO8T分類の信頼度が低い場合（0.4未満）
        else:
            # ルールベース分類を優先
            final_label = rule_based_result
            classification_method = 'rule_based_primary'
            reasoning = f"Rule-based classification (SO8T confidence too low: {so8t_confidence:.2f})"
        
        # 統合結果を構築
        hybrid_result = so8t_result.copy()
        hybrid_result['four_class_label'] = final_label
        hybrid_result['four_class_label_id'] = {'ALLOW': 0, 'ESCALATION': 1, 'DENY': 2, 'REFUSE': 3}.get(final_label, 0)
        hybrid_result['classification_method'] = classification_method
        hybrid_result['rule_based_result'] = rule_based_result
        hybrid_result['so8t_result'] = so8t_four_class
        hybrid_result['so8t_confidence'] = so8t_confidence
        hybrid_result['hybrid_reasoning'] = reasoning
        
        return hybrid_result
    
    def _map_to_four_class(self, final_class: str, sample: Dict) -> str:
        """
        SO8T分類結果を四値分類（ALLOW/ESCALATION/DENY/REFUSE）にマッピング
        
        Args:
            final_class: SO8T分類結果（approved/review_needed/rejected/unknown）
            sample: サンプルデータ
        
        Returns:
            四値分類ラベル（ALLOW/ESCALATION/DENY/REFUSE）
        """
        # SO8T分類結果に基づくマッピング
        if final_class == 'approved':
            # 承認された場合でも、ルールベースで再確認
            return self._classify_with_rules(sample)
        elif final_class == 'review_needed':
            return 'ESCALATION'
        elif final_class == 'rejected':
            # 拒否された場合、NSFWラベルでREFUSE/DENYを判定
            nsfw_label = sample.get('nsfw_label', 'safe')
            if nsfw_label in ['nsfw_block', 'violence', 'harassment', 'self_harm', 'illegal_content']:
                return 'REFUSE'
            else:
                return 'DENY'
        else:
            # unknown/errorの場合はルールベース分類を使用
            return self._classify_with_rules(sample)
    


class WebScrapingDataPipeline:
    """Webスクレイピングデータパイプライン"""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        use_so8t_classification: bool = True,
        so8t_model_path: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            use_so8t_classification: SO8T分類を使用するか
            so8t_model_path: SO8Tモデルのパス
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cleaner = DataCleaner()
        self.labeler = DataLabeler()
        
        if use_so8t_classification:
            self.classifier = QuadrupleClassifier(so8t_model_path)
        else:
            self.classifier = None
        
        logger.info("="*80)
        logger.info("Web Scraping Data Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"SO8T classification: {use_so8t_classification}")
    
    def load_samples(self) -> List[Dict]:
        """サンプルを読み込み"""
        samples = []
        
        # JSONLファイルを探す
        jsonl_files = list(self.input_dir.glob("*.jsonl"))
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to load {jsonl_file}: {e}")
        
        logger.info(f"[LOAD] Loaded {len(samples)} samples from {len(jsonl_files)} files")
        return samples
    
    def process_sample(self, sample: Dict) -> Dict:
        """
        サンプルを処理（クレンジング→ラベル付け→四値分類）
        
        Args:
            sample: 処理するサンプル
        
        Returns:
            処理されたサンプル
        """
        # 1. データクレンジング
        cleaned_sample = self.cleaner.clean_sample(sample)
        
        # 2. ラベル付け
        labeled_sample = self.labeler.label_sample(cleaned_sample)
        
        # 3. 四値分類（SO8T使用時）
        if self.classifier:
            classified_sample = self.classifier.classify_quadruple(labeled_sample)
        else:
            classified_sample = labeled_sample
            classified_sample['quadruple_classification'] = {
                'task': 'not_classified',
                'safety': 'not_classified',
                'policy': 'not_classified',
                'final': 'not_classified',
                'reasoning': 'SO8T classification disabled'
            }
        
        return classified_sample
    
    def process_sample_batch(self, samples_batch: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        サンプルのバッチを処理（並列処理用）
        
        Args:
            samples_batch: 処理するサンプルのバッチ
        
        Returns:
            (処理済みサンプル, 失敗したサンプル)のタプル
        """
        processed_samples = []
        failed_samples = []
        
        for sample in samples_batch:
            try:
                processed_sample = self.process_sample(sample)
                processed_samples.append(processed_sample)
            except Exception as e:
                failed_samples.append({'sample': sample, 'error': str(e)})
        
        return processed_samples, failed_samples
    
    def process_pipeline(self, num_workers: int = 4, batch_size: int = 100) -> Dict:
        """
        パイプライン処理を実行（並列処理対応）
        
        Args:
            num_workers: 並列処理ワーカー数
            batch_size: バッチサイズ
        
        Returns:
            処理結果統計
        """
        start_time = time.time()
        logger.info("[PIPELINE] Starting pipeline processing...")
        logger.info(f"[PIPELINE] Workers: {num_workers}, Batch size: {batch_size}")
        
        # サンプルを読み込み
        samples = self.load_samples()
        
        if not samples:
            logger.warning("[PIPELINE] No samples found")
            return {'total': 0, 'processed': 0, 'failed': 0}
        
        logger.info(f"[PIPELINE] Loaded {len(samples)} samples")
        
        # 処理結果
        processed_samples = []
        failed_samples = []
        
        # 処理時間計測
        processing_times = []
        
        # バッチに分割
        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        
        logger.info(f"[PIPELINE] Processing {len(batches)} batches...")
        
        # 並列処理（ThreadPoolExecutor使用）
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # プログレスバー表示
            with tqdm(total=len(samples), desc="Processing samples") as pbar:
                futures = []
                for batch in batches:
                    future = executor.submit(self.process_sample_batch, batch)
                    futures.append(future)
                
                # 結果を取得
                for future in futures:
                    batch_start_time = time.time()
                    batch_processed, batch_failed = future.result()
                    batch_processing_time = time.time() - batch_start_time
                    processing_times.append(batch_processing_time)
                    
                    processed_samples.extend(batch_processed)
                    failed_samples.extend(batch_failed)
                    
                    pbar.update(len(batch_processed) + len(batch_failed))
        
        # 結果を保存
        output_file = self.output_dir / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in processed_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        total_time = time.time() - start_time
        
        logger.info(f"[PIPELINE] Saved {len(processed_samples)} processed samples to {output_file}")
        
        # 詳細な統計情報を計算
        stats = self.calculate_detailed_statistics(
            samples,
            processed_samples,
            failed_samples,
            processing_times,
            total_time
        )
        
        stats['output_file'] = str(output_file)
        
        # 統計情報をログに出力
        self.log_statistics(stats)
        
        # 統計情報をJSONファイルに保存
        stats_file = self.output_dir / f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"[PIPELINE] Statistics saved to {stats_file}")
        
        logger.info(f"[PIPELINE] Pipeline completed: {stats['processed']}/{stats['total']} samples processed in {total_time:.2f}s")
        
        return stats
    
    def calculate_detailed_statistics(
        self,
        original_samples: List[Dict],
        processed_samples: List[Dict],
        failed_samples: List[Dict],
        processing_times: List[float],
        total_time: float
    ) -> Dict:
        """
        詳細な統計情報を計算
        
        Args:
            original_samples: 元のサンプル
            processed_samples: 処理済みサンプル
            failed_samples: 失敗したサンプル
            processing_times: 処理時間のリスト
            total_time: 総処理時間
        
        Returns:
            統計情報辞書
        """
        stats = {
            'total': len(original_samples),
            'processed': len(processed_samples),
            'failed': len(failed_samples),
            'success_rate': len(processed_samples) / max(len(original_samples), 1) * 100,
            'total_time_seconds': total_time,
            'avg_processing_time_per_sample': statistics.mean(processing_times) / max(len(processing_times), 1) if processing_times else 0,
            'min_processing_time': min(processing_times) if processing_times else 0,
            'max_processing_time': max(processing_times) if processing_times else 0,
            'median_processing_time': statistics.median(processing_times) if processing_times else 0,
            'samples_per_second': len(processed_samples) / max(total_time, 1),
        }
        
        # カテゴリ別統計
        category_counts = Counter()
        language_counts = Counter()
        classification_counts = {
            'task': Counter(),
            'safety': Counter(),
            'policy': Counter(),
            'final': Counter()
        }
        
        text_lengths = []
        
        for sample in processed_samples:
            # カテゴリ統計
            category = sample.get('category', 'unknown')
            category_counts[category] += 1
            
            # 言語統計
            language = sample.get('language', 'unknown')
            language_counts[language] += 1
            
            # 四値分類統計
            quadruple = sample.get('quadruple_classification', {})
            if isinstance(quadruple, dict):
                classification_counts['task'][quadruple.get('task', 'unknown')] += 1
                classification_counts['safety'][quadruple.get('safety', 'unknown')] += 1
                classification_counts['policy'][quadruple.get('policy', 'unknown')] += 1
                classification_counts['final'][quadruple.get('final', 'unknown')] += 1
            
            # テキスト長統計
            text_length = sample.get('text_length', 0)
            if text_length > 0:
                text_lengths.append(text_length)
        
        stats['category_distribution'] = dict(category_counts)
        stats['language_distribution'] = dict(language_counts)
        stats['classification_distribution'] = {
            'task': dict(classification_counts['task']),
            'safety': dict(classification_counts['safety']),
            'policy': dict(classification_counts['policy']),
            'final': dict(classification_counts['final'])
        }
        
        if text_lengths:
            stats['text_length_stats'] = {
                'min': min(text_lengths),
                'max': max(text_lengths),
                'mean': statistics.mean(text_lengths),
                'median': statistics.median(text_lengths),
                'stdev': statistics.stdev(text_lengths) if len(text_lengths) > 1 else 0
            }
        else:
            stats['text_length_stats'] = {}
        
        # エラー統計
        error_types = Counter()
        for failed in failed_samples:
            error_msg = failed.get('error', 'unknown')
            error_type = error_msg.split(':')[0] if ':' in error_msg else 'unknown'
            error_types[error_type] += 1
        
        stats['error_distribution'] = dict(error_types)
        
        return stats
    
    def log_statistics(self, stats: Dict):
        """統計情報をログに出力"""
        logger.info("="*80)
        logger.info("PIPELINE PROCESSING STATISTICS")
        logger.info("="*80)
        logger.info(f"Total samples: {stats['total']}")
        logger.info(f"Processed samples: {stats['processed']}")
        logger.info(f"Failed samples: {stats['failed']}")
        logger.info(f"Success rate: {stats['success_rate']:.2f}%")
        logger.info(f"Total time: {stats['total_time_seconds']:.2f}s")
        logger.info(f"Average processing time per sample: {stats['avg_processing_time_per_sample']:.4f}s")
        logger.info(f"Samples per second: {stats['samples_per_second']:.2f}")
        logger.info("")
        logger.info("Category Distribution:")
        for category, count in stats.get('category_distribution', {}).items():
            logger.info(f"  {category}: {count}")
        logger.info("")
        logger.info("Language Distribution:")
        for language, count in stats.get('language_distribution', {}).items():
            logger.info(f"  {language}: {count}")
        logger.info("")
        logger.info("Classification Distribution:")
        for classification_type, distribution in stats.get('classification_distribution', {}).items():
            logger.info(f"  {classification_type}:")
            for label, count in distribution.items():
                logger.info(f"    {label}: {count}")
        logger.info("")
        if stats.get('text_length_stats'):
            text_stats = stats['text_length_stats']
            logger.info("Text Length Statistics:")
            logger.info(f"  Min: {text_stats.get('min', 0)}")
            logger.info(f"  Max: {text_stats.get('max', 0)}")
            logger.info(f"  Mean: {text_stats.get('mean', 0):.2f}")
            logger.info(f"  Median: {text_stats.get('median', 0):.2f}")
            logger.info(f"  Std Dev: {text_stats.get('stdev', 0):.2f}")
        logger.info("")
        if stats.get('error_distribution'):
            logger.info("Error Distribution:")
            for error_type, count in stats['error_distribution'].items():
                logger.info(f"  {error_type}: {count}")
        logger.info("="*80)


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Web Scraping Data Pipeline")
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('D:/webdataset/processed'),
        help='Input directory'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('D:/webdataset/cleaned'),
        help='Output directory'
    )
    parser.add_argument(
        '--use-so8t',
        action='store_true',
        default=True,
        help='Use SO8T for quadruple classification'
    )
    parser.add_argument(
        '--so8t-model-path',
        type=str,
        default=None,
        help='Path to SO8T model'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing'
    )
    
    args = parser.parse_args()
    
    # パイプライン作成
    pipeline = WebScrapingDataPipeline(
        input_dir=args.input,
        output_dir=args.output,
        use_so8t_classification=args.use_so8t,
        so8t_model_path=args.so8t_model_path
    )
    
    # パイプライン処理実行
    stats = pipeline.process_pipeline(
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    
    logger.info(f"[SUCCESS] Pipeline completed: {stats}")
    return stats


if __name__ == "__main__":
    asyncio.run(main())

