#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
検知用NSFWデータセット収集スクリプト

安全判定と拒否挙動の学習を目的としたNSFW検知用データセットを収集します。
生成目的ではなく、検知・防止・教育を目的としています。

Usage:
    python scripts/data/collect_nsfw_detection_dataset.py --output D:/webdataset/nsfw_detection_dataset
"""

import sys
import json
import logging
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_nsfw_detection_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# NSFW分類器のインポート
try:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))
    from train_nsfw_classifier import NSFWClassifier
    from multimodal_nsfw_detector import MultimodalNSFWDetector
    NSFW_AVAILABLE = True
except ImportError as e:
    NSFW_AVAILABLE = False
    logger.warning(f"NSFW classifier not available: {e}")

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available")


class NSFWDetectionDatasetCollector:
    """検知用NSFWデータセット収集クラス"""
    
    def __init__(
        self,
        output_dir: Path,
        nsfw_classifier_path: Optional[Path] = None,
        use_multimodal: bool = True
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            nsfw_classifier_path: NSFW分類器のパス
            use_multimodal: マルチモーダル検知を使用するか
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NSFW分類器の初期化
        self.nsfw_classifier = None
        self.multimodal_detector = None
        
        if NSFW_AVAILABLE:
            if use_multimodal:
                try:
                    self.multimodal_detector = MultimodalNSFWDetector(
                        text_classifier_path=nsfw_classifier_path
                    )
                    logger.info("[NSFW] Multimodal NSFW detector initialized")
                except Exception as e:
                    logger.warning(f"[NSFW] Failed to initialize multimodal detector: {e}")
            
            if not self.multimodal_detector:
                try:
                    if nsfw_classifier_path and nsfw_classifier_path.exists():
                        self.nsfw_classifier = NSFWClassifier(model_path=nsfw_classifier_path)
                        logger.info("[NSFW] Text NSFW classifier initialized")
                    else:
                        default_path = Path("models/nsfw_classifier.joblib")
                        if default_path.exists():
                            self.nsfw_classifier = NSFWClassifier(model_path=default_path)
                            logger.info("[NSFW] Text NSFW classifier loaded from default path")
                except Exception as e:
                    logger.warning(f"[NSFW] Failed to initialize NSFW classifier: {e}")
        
        # データソース（合法・倫理的）
        self.data_sources = {
            'legal_cases': [],  # 公開裁判例
            'administrative_guidance': [],  # 行政指導事例
            'sns_violations': [],  # SNS利用規約違反事例（匿名化）
            'literature_film': [],  # 文学作品・映画（検閲歴史）
            'academic_papers': []  # 学術論文（法学・社会学）
        }
        
        # NSFWカテゴリ定義（検知目的、詳細版20+カテゴリ）
        self.nsfw_categories = {
            # 性的コンテンツ（3カテゴリ）
            'sexual_content_explicit': {
                'keywords': ['性的描写', 'ポルノ', 'アダルト', 'エロ', 'わいせつ', 'sex', 'porn', 'explicit', 'xxx'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': False
            },
            'sexual_content_implicit': {
                'keywords': ['性的', 'セクシー', 'エッチ', 'sexual', 'sexy', 'erotic'],
                'severity': 'medium',
                'legal_status': 'context_dependent',
                'context_required': True
            },
            'sexual_content_educational': {
                'keywords': ['性教育', '性器', '生殖', 'sex education', 'reproductive', 'anatomy'],
                'severity': 'low',
                'legal_status': 'safe',
                'context_required': True
            },
            # 暴力（3カテゴリ）
            'violence_graphic': {
                'keywords': ['殺人', 'テロ', '爆弾', '暴力', 'murder', 'terror', 'bomb', 'graphic violence'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': False
            },
            'violence_implied': {
                'keywords': ['戦争', '争い', 'war', 'conflict', 'fight'],
                'severity': 'medium',
                'legal_status': 'context_dependent',
                'context_required': True
            },
            'violence_educational': {
                'keywords': ['暴力防止', '平和', 'violence prevention', 'peace'],
                'severity': 'low',
                'legal_status': 'safe',
                'context_required': True
            },
            # 差別（4カテゴリ）
            'hate_speech_racial': {
                'keywords': ['人種差別', 'racial', 'discrimination', 'racism'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': False
            },
            'hate_speech_gender': {
                'keywords': ['性差別', 'gender discrimination', 'sexism'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': False
            },
            'hate_speech_religious': {
                'keywords': ['宗教差別', 'religious discrimination'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': False
            },
            'hate_speech_other': {
                'keywords': ['ヘイト', '誹謗', '中傷', 'hate', 'slander', 'defamation'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': False
            },
            # 自傷行為（2カテゴリ）
            'self_harm_suicide': {
                'keywords': ['自殺', 'suicide', 'kill myself', 'end my life'],
                'severity': 'high',
                'legal_status': 'safe_but_harmful',
                'context_required': False
            },
            'self_harm_self_injury': {
                'keywords': ['自傷', 'self-harm', 'self-injury', 'cutting'],
                'severity': 'high',
                'legal_status': 'safe_but_harmful',
                'context_required': False
            },
            # 武器（3カテゴリ）
            'weapons_manufacturing': {
                'keywords': ['武器製造', '爆弾製造', 'weapon manufacturing', 'bomb making'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': False
            },
            'weapons_usage': {
                'keywords': ['武器使用', '銃の使い方', 'weapon usage', 'gun handling'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': False
            },
            'weapons_educational': {
                'keywords': ['武器の歴史', '軍事史', 'weapon history', 'military history'],
                'severity': 'low',
                'legal_status': 'safe',
                'context_required': True
            },
            # 医療助言（3カテゴリ）
            'medical_advice_diagnosis': {
                'keywords': ['診断', '病気の診断', 'diagnosis', 'disease diagnosis'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': True
            },
            'medical_advice_treatment': {
                'keywords': ['治療', '薬の処方', 'treatment', 'prescription'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': True
            },
            'medical_advice_educational': {
                'keywords': ['医学教育', '医療情報', 'medical education', 'health information'],
                'severity': 'low',
                'legal_status': 'safe',
                'context_required': True
            },
            # 違法コンテンツ（5カテゴリ）
            'illegal_drugs': {
                'keywords': ['違法薬物', '麻薬', 'illegal drugs', 'narcotics'],
                'severity': 'high',
                'legal_status': 'illegal',
                'context_required': False
            },
            'illegal_financial': {
                'keywords': ['金融犯罪', '詐欺', 'financial crime', 'fraud'],
                'severity': 'high',
                'legal_status': 'illegal',
                'context_required': False
            },
            'illegal_cybercrime': {
                'keywords': ['サイバー犯罪', 'ハッキング', 'cybercrime', 'hacking'],
                'severity': 'high',
                'legal_status': 'illegal',
                'context_required': False
            },
            'child_exploitation': {
                'keywords': ['児童搾取', '児童ポルノ', 'child exploitation', 'child pornography'],
                'severity': 'critical',
                'legal_status': 'illegal',
                'context_required': False
            },
            'privacy_violation': {
                'keywords': ['プライバシー侵害', '個人情報漏洩', 'privacy violation', 'data breach'],
                'severity': 'high',
                'legal_status': 'potentially_illegal',
                'context_required': False
            }
        }
        
        logger.info("="*80)
        logger.info("NSFW Detection Dataset Collector Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"NSFW classifier available: {self.nsfw_classifier is not None or self.multimodal_detector is not None}")
    
    def detect_nsfw(
        self,
        text: str,
        url: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        audience: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        NSFW検知（拡張版：詳細メタデータ対応）
        
        Args:
            text: テキスト
            url: URL（オプション）
            context: 文脈情報（オプション）
            intent: 意図情報（オプション）
            audience: 対象読者（オプション）
        
        Returns:
            NSFW検知結果の辞書（詳細メタデータ含む）
        """
        result = {
            'label': 'safe',
            'confidence': 1.0,
            'category': 'none',
            'detailed_category': 'none',
            'severity': 'none',
            'legal_status': 'safe',
            'detection_method': 'rule_based',
            'context': context or 'unknown',
            'intent': intent or 'unknown',
            'audience': audience or 'general',
            'reasoning': '',
            'requires_human_review': False
        }
        
        # マルチモーダル検知を使用
        if self.multimodal_detector:
            try:
                label, confidence = self.multimodal_detector.detect_text_nsfw(text)
                result['label'] = label
                result['confidence'] = float(confidence)
                result['detection_method'] = 'multimodal'
            except Exception as e:
                logger.debug(f"[NSFW] Multimodal detection failed: {e}")
        
        # テキスト分類器を使用
        if result['label'] == 'safe' and self.nsfw_classifier:
            try:
                label, confidence = self.nsfw_classifier.predict(text)
                result['label'] = label
                result['confidence'] = float(confidence)
                result['detection_method'] = 'classifier'
            except Exception as e:
                logger.debug(f"[NSFW] Classifier detection failed: {e}")
        
        # ルールベース検知（フォールバック）
        if result['label'] == 'safe':
            result = self._rule_based_detection(text, context, intent, audience)
        
        # 詳細カテゴリ分類
        detailed_category = self._classify_detailed_category(text, context, intent, audience)
        result['detailed_category'] = detailed_category
        
        if detailed_category != 'none':
            category_info = self.nsfw_categories.get(detailed_category, {})
            result['severity'] = category_info.get('severity', 'none')
            result['legal_status'] = category_info.get('legal_status', 'safe')
            result['requires_human_review'] = category_info.get('context_required', False) and not context
            
            # 判定理由を生成
            result['reasoning'] = self._generate_reasoning(
                text, detailed_category, category_info, context, intent, audience
            )
        
        return result
    
    def _rule_based_detection(
        self,
        text: str,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        audience: Optional[str] = None
    ) -> Dict[str, Any]:
        """ルールベースNSFW検知（フォールバック、拡張版）"""
        import re
        
        text_lower = text.lower()
        context_lower = (context or '').lower()
        intent_lower = (intent or '').lower()
        
        # 各カテゴリをチェック（優先度順）
        for category, info in self.nsfw_categories.items():
            keywords = info['keywords']
            for keyword in keywords:
                if re.search(keyword, text_lower):
                    # 文脈依存カテゴリの場合は文脈を確認
                    if info.get('context_required', False):
                        # 教育・学術的文脈の場合は安全と判定
                        if any(ctx in context_lower for ctx in ['教育', '学術', '研究', 'education', 'academic', 'research']):
                            continue
                        # 意図が教育・学術の場合は安全と判定
                        if any(int in intent_lower for int in ['教育', '学術', '研究', 'education', 'academic', 'research']):
                            continue
                    
                    return {
                        'label': 'nsfw_detected',
                        'confidence': 0.7,
                        'category': category.split('_')[0] if '_' in category else category,
                        'detailed_category': category,
                        'severity': info['severity'],
                        'legal_status': info['legal_status'],
                        'detection_method': 'rule_based',
                        'context': context or 'unknown',
                        'intent': intent or 'unknown',
                        'audience': audience or 'general',
                        'requires_human_review': info.get('context_required', False)
                    }
        
        return {
            'label': 'safe',
            'confidence': 1.0,
            'category': 'none',
            'detailed_category': 'none',
            'severity': 'none',
            'legal_status': 'safe',
            'detection_method': 'rule_based',
            'context': context or 'unknown',
            'intent': intent or 'unknown',
            'audience': audience or 'general',
            'requires_human_review': False
        }
    
    def _classify_category(self, text: str) -> str:
        """カテゴリ分類（基本版、後方互換性のため保持）"""
        import re
        
        text_lower = text.lower()
        
        # 各カテゴリのキーワードをチェック
        for category, info in self.nsfw_categories.items():
            keywords = info['keywords']
            for keyword in keywords:
                if re.search(keyword, text_lower):
                    return category.split('_')[0] if '_' in category else category
        
        return 'none'
    
    def _classify_detailed_category(
        self,
        text: str,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        audience: Optional[str] = None
    ) -> str:
        """詳細カテゴリ分類（20+カテゴリ対応）"""
        import re
        
        text_lower = text.lower()
        context_lower = (context or '').lower()
        intent_lower = (intent or '').lower()
        audience_lower = (audience or '').lower()
        
        # 優先度順にカテゴリをチェック（より具体的なカテゴリを優先）
        category_scores = {}
        
        for category, info in self.nsfw_categories.items():
            keywords = info['keywords']
            score = 0
            
            for keyword in keywords:
                if re.search(keyword, text_lower):
                    score += 1
            
            # 文脈依存カテゴリの場合は文脈を確認
            if info.get('context_required', False):
                # 教育・学術的文脈の場合はスコアを下げる
                if any(ctx in context_lower for ctx in ['教育', '学術', '研究', 'education', 'academic', 'research']):
                    score *= 0.3
                # 意図が教育・学術の場合はスコアを下げる
                if any(int in intent_lower for int in ['教育', '学術', '研究', 'education', 'academic', 'research']):
                    score *= 0.3
                # 対象読者が専門家の場合はスコアを下げる
                if any(aud in audience_lower for aud in ['専門家', '研究者', '医師', 'expert', 'researcher', 'doctor']):
                    score *= 0.3
            
            if score > 0:
                category_scores[category] = score
        
        # 最もスコアの高いカテゴリを返す
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'none'
    
    def _generate_reasoning(
        self,
        text: str,
        category: str,
        category_info: Dict,
        context: Optional[str],
        intent: Optional[str],
        audience: Optional[str]
    ) -> str:
        """判定理由を生成"""
        reasoning_parts = []
        
        if category != 'none':
            reasoning_parts.append(f"カテゴリ: {category}")
            reasoning_parts.append(f"深刻度: {category_info.get('severity', 'none')}")
            reasoning_parts.append(f"法的状態: {category_info.get('legal_status', 'safe')}")
        
        if context and context != 'unknown':
            reasoning_parts.append(f"文脈: {context}")
        
        if intent and intent != 'unknown':
            reasoning_parts.append(f"意図: {intent}")
        
        if audience and audience != 'general':
            reasoning_parts.append(f"対象読者: {audience}")
        
        if category_info.get('context_required', False) and not context:
            reasoning_parts.append("文脈情報が必要なため、人間による確認を推奨")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "安全なコンテンツ"
    
    def generate_synthetic_samples(self, num_samples: int = 1000) -> List[Dict]:
        """
        合成データを生成（倫理的、検知目的のみ）
        
        Args:
            num_samples: 生成するサンプル数
        
        Returns:
            合成サンプルのリスト
        """
        logger.info(f"[SYNTHETIC] Generating {num_samples} synthetic NSFW detection samples...")
        
        synthetic_samples = []
        
        # 合成データテンプレート（倫理的、実際のNSFWコンテンツは含まない）
        templates = [
            {
                'text': '性器の解剖学的構造について説明します。これは医学教育の一環です。',
                'context': '医学教育',
                'intent': '学術',
                'audience': '医学生',
                'expected_category': 'sexual_content_educational',
                'expected_judgment': 'SAFE'
            },
            {
                'text': '彼女の裸体は月光に照らされていた。これは文学作品の一節です。',
                'context': '文学作品',
                'intent': '芸術表現',
                'audience': '一般読者',
                'expected_category': 'sexual_content_implicit',
                'expected_judgment': 'SAFE'
            },
            {
                'text': '暴力を防止するための教育プログラムについて説明します。',
                'context': '教育',
                'intent': '教育',
                'audience': '学生',
                'expected_category': 'violence_educational',
                'expected_judgment': 'SAFE'
            },
            {
                'text': '部落の歴史について研究しています。これは学術研究です。',
                'context': '学術研究',
                'intent': '歴史理解',
                'audience': '研究者',
                'expected_category': 'hate_speech_other',
                'expected_judgment': 'SAFE'
            },
            {
                'text': '武器の歴史について説明します。これは軍事史の研究です。',
                'context': '学術研究',
                'intent': '歴史理解',
                'audience': '研究者',
                'expected_category': 'weapons_educational',
                'expected_judgment': 'SAFE'
            }
        ]
        
        # テンプレートからサンプルを生成
        for i in range(num_samples):
            template = templates[i % len(templates)]
            
            # NSFW検知を実行
            nsfw_result = self.detect_nsfw(
                text=template['text'],
                context=template.get('context'),
                intent=template.get('intent'),
                audience=template.get('audience')
            )
            
            sample = {
                'text': template['text'],
                'context': template.get('context', 'unknown'),
                'intent': template.get('intent', 'unknown'),
                'audience': template.get('audience', 'general'),
                'nsfw_detection': nsfw_result,
                'nsfw_label': nsfw_result['label'],
                'nsfw_confidence': nsfw_result['confidence'],
                'nsfw_category': nsfw_result.get('category', 'none'),
                'detailed_category': nsfw_result.get('detailed_category', 'none'),
                'nsfw_severity': nsfw_result.get('severity', 'none'),
                'nsfw_legal_status': nsfw_result.get('legal_status', 'safe'),
                'reasoning': nsfw_result.get('reasoning', ''),
                'expected_category': template.get('expected_category'),
                'expected_judgment': template.get('expected_judgment'),
                'detection_purpose': 'safety_training',
                'source': 'synthetic',
                'collected_at': datetime.now().isoformat()
            }
            
            synthetic_samples.append(sample)
        
        logger.info(f"[SYNTHETIC] Generated {len(synthetic_samples)} synthetic samples")
        return synthetic_samples
    
    def collect_from_existing_data(
        self,
        input_dir: Path,
        max_samples: int = 50000,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        既存データからNSFW検知用データセットを収集
        
        Args:
            input_dir: 入力ディレクトリ
            max_samples: 最大サンプル数
        
        Returns:
            NSFW検知済みサンプルのリスト
        """
        logger.info(f"[COLLECT] Collecting NSFW detection dataset from {input_dir}")
        
        nsfw_samples = []
        total_checked = 0
        
        # JSONLファイルを読み込み
        jsonl_files = list(input_dir.glob("*.jsonl"))
        logger.info(f"[COLLECT] Found {len(jsonl_files)} JSONL files")
        
        for jsonl_file in jsonl_files:
            if len(nsfw_samples) >= max_samples:
                break
            
            logger.info(f"[COLLECT] Processing {jsonl_file.name}...")
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(nsfw_samples) >= max_samples:
                        break
                    
                    try:
                        sample = json.loads(line.strip())
                        text = sample.get('text', '') or sample.get('cleaned_text', '')
                        
                        if not text or len(text.strip()) < 10:
                            continue
                        
                        total_checked += 1
                        
                        # メタデータを抽出
                        context = sample.get('context') or sample.get('category') or 'unknown'
                        intent = sample.get('intent') or 'unknown'
                        audience = sample.get('audience') or sample.get('domain') or 'general'
                        
                        # NSFW検知（詳細メタデータ対応）
                        nsfw_result = self.detect_nsfw(
                            text=text,
                            url=sample.get('url'),
                            context=context if include_metadata else None,
                            intent=intent if include_metadata else None,
                            audience=audience if include_metadata else None
                        )
                        
                        # NSFWが検知された場合、または安全サンプルとして保存（バランス調整）
                        should_save = False
                        if nsfw_result['label'] != 'safe':
                            should_save = True
                        elif total_checked % 50 == 0:  # 安全サンプルも2%保存（バランス調整）
                            should_save = True
                        
                        if should_save:
                            sample['nsfw_detection'] = nsfw_result
                            sample['nsfw_label'] = nsfw_result['label']
                            sample['nsfw_confidence'] = nsfw_result['confidence']
                            sample['nsfw_category'] = nsfw_result.get('category', 'none')
                            sample['detailed_category'] = nsfw_result.get('detailed_category', 'none')
                            sample['nsfw_severity'] = nsfw_result.get('severity', 'none')
                            sample['nsfw_legal_status'] = nsfw_result.get('legal_status', 'safe')
                            sample['reasoning'] = nsfw_result.get('reasoning', '')
                            sample['requires_human_review'] = nsfw_result.get('requires_human_review', False)
                            
                            # メタデータを追加
                            if include_metadata:
                                sample['context'] = context
                                sample['intent'] = intent
                                sample['audience'] = audience
                            
                            sample['detection_purpose'] = 'safety_training'  # 生成目的ではない
                            sample['source'] = 'existing_data'
                            sample['collected_at'] = datetime.now().isoformat()
                            
                            nsfw_samples.append(sample)
                    
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"[COLLECT] Collected {len(nsfw_samples)} NSFW detection samples from {total_checked} checked samples")
        
        # カテゴリ分布を確認
        category_dist = Counter(s.get('detailed_category', 'none') for s in nsfw_samples)
        logger.info(f"[COLLECT] Category distribution: {dict(category_dist)}")
        
        return nsfw_samples
    
    def save_dataset(self, samples: List[Dict], split_ratio: float = 0.8):
        """
        データセットを保存
        
        Args:
            samples: サンプルのリスト
            split_ratio: 訓練/検証の分割比率
        """
        logger.info(f"[SAVE] Saving {len(samples)} samples...")
        
        # ラベル分布を確認
        label_dist = Counter(s.get('nsfw_label', 'unknown') for s in samples)
        logger.info(f"[SAVE] Label distribution: {dict(label_dist)}")
        
        # 訓練/検証に分割
        import random
        random.shuffle(samples)
        split_idx = int(len(samples) * split_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # 訓練データを保存
        train_file = self.output_dir / "nsfw_detection_train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"[SAVE] Training data saved: {train_file} ({len(train_samples)} samples)")
        
        # 検証データを保存
        val_file = self.output_dir / "nsfw_detection_val.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"[SAVE] Validation data saved: {val_file} ({len(val_samples)} samples)")
        
        # 詳細統計を計算
        category_dist = Counter(s.get('detailed_category', 'none') for s in samples)
        severity_dist = Counter(s.get('nsfw_severity', 'none') for s in samples)
        legal_status_dist = Counter(s.get('nsfw_legal_status', 'safe') for s in samples)
        source_dist = Counter(s.get('source', 'unknown') for s in samples)
        
        # メタデータを保存
        metadata = {
            'total_samples': len(samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'label_distribution': dict(label_dist),
            'category_distribution': dict(category_dist),
            'severity_distribution': dict(severity_dist),
            'legal_status_distribution': dict(legal_status_dist),
            'source_distribution': dict(source_dist),
            'created_at': datetime.now().isoformat(),
            'purpose': 'safety_training',  # 生成目的ではない
            'categories': list(self.nsfw_categories.keys()),
            'total_categories': len(self.nsfw_categories),
            'metadata_included': True
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"[SAVE] Metadata saved: {metadata_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='NSFW Detection Dataset Collector (Enhanced)')
    parser.add_argument('--input', type=Path, help='Input directory (processed data)')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--nsfw-classifier', type=Path, help='NSFW classifier path')
    parser.add_argument('--max-samples', type=int, default=50000, help='Maximum samples to collect')
    parser.add_argument('--use-multimodal', action='store_true', help='Use multimodal detection')
    parser.add_argument('--include-synthetic', action='store_true', help='Include synthetic samples')
    parser.add_argument('--synthetic-samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--include-metadata', action='store_true', default=True, help='Include detailed metadata')
    
    args = parser.parse_args()
    
    collector = NSFWDetectionDatasetCollector(
        output_dir=args.output,
        nsfw_classifier_path=args.nsfw_classifier,
        use_multimodal=args.use_multimodal
    )
    
    all_samples = []
    
    # 既存データから収集
    if args.input:
        existing_samples = collector.collect_from_existing_data(
            input_dir=args.input,
            max_samples=args.max_samples,
            include_metadata=args.include_metadata
        )
        all_samples.extend(existing_samples)
        logger.info(f"[MAIN] Collected {len(existing_samples)} samples from existing data")
    
    # 合成データを生成
    if args.include_synthetic:
        synthetic_samples = collector.generate_synthetic_samples(
            num_samples=args.synthetic_samples
        )
        all_samples.extend(synthetic_samples)
        logger.info(f"[MAIN] Generated {len(synthetic_samples)} synthetic samples")
    
    if all_samples:
        collector.save_dataset(all_samples)
        logger.info(f"[OK] NSFW detection dataset collection completed: {len(all_samples)} total samples")
    else:
        logger.warning("[WARNING] No NSFW samples collected")


if __name__ == '__main__':
    main()

