#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習ベースドメイン分類モジュール

TF-IDFベースの特徴抽出、シンプルな分類器（scikit-learn）、分類確信度の計算、
分類結果の検証を行う。

Usage:
    from scripts.data.domain_classifier import MLDomainClassifier
    classifier = MLDomainClassifier(keywords_config)
    classifier.train(training_data)  # オプション: 事前学習
    domain, confidence = classifier.classify(text, title)
"""

import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available. ML-based classification will be disabled.")

logger = logging.getLogger(__name__)


class MLDomainClassifier:
    """機械学習ベースドメイン分類器"""
    
    def __init__(
        self,
        keywords_config: Dict,
        use_ml: bool = True,
        classifier_type: str = "naive_bayes"
    ):
        """
        Args:
            keywords_config: ドメイン別キーワード設定
            use_ml: 機械学習を使用するか（Falseの場合はキーワードベースにフォールバック）
            classifier_type: 分類器タイプ（"naive_bayes" または "logistic_regression"）
        """
        self.keywords_config = keywords_config
        self.use_ml = use_ml and SKLEARN_AVAILABLE
        self.classifier_type = classifier_type
        
        self.domains = list(keywords_config["domains"].keys())
        self.label_encoder = LabelEncoder() if self.use_ml else None
        self.classifier = None
        self.vectorizer = None
        self.is_trained = False
    
    def train(
        self,
        training_data: List[Dict[str, Any]],
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        分類器を訓練
        
        Args:
            training_data: 訓練データ（各要素は{"text": str, "title": str, "domain": str}）
            test_size: テストデータの割合
        
        Returns:
            訓練結果（精度など）
        """
        if not self.use_ml:
            logger.warning("[ML_CLASSIFIER] ML is disabled, skipping training")
            return {}
        
        if not training_data:
            logger.warning("[ML_CLASSIFIER] No training data provided")
            return {}
        
        try:
            # データ準備
            texts = []
            titles = []
            labels = []
            
            for item in training_data:
                text = item.get("text", "") or item.get("output", "")
                title = item.get("title", "")
                domain = item.get("domain")
                
                if text and domain and domain in self.domains:
                    # テキストとタイトルを結合
                    combined = f"{title} {text}"
                    texts.append(combined)
                    labels.append(domain)
            
            if not texts:
                logger.warning("[ML_CLASSIFIER] No valid training samples")
                return {}
            
            # ラベルエンコーディング
            self.label_encoder.fit(self.domains)
            encoded_labels = self.label_encoder.transform(labels)
            
            # ベクトル化器と分類器の作成
            if self.classifier_type == "naive_bayes":
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
                self.classifier = MultinomialNB(alpha=1.0)
            else:  # logistic_regression
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
                self.classifier = LogisticRegression(max_iter=1000, random_state=42)
            
            # 特徴抽出
            X = self.vectorizer.fit_transform(texts)
            
            # 訓練
            self.classifier.fit(X, encoded_labels)
            self.is_trained = True
            
            # 精度評価（簡易）
            predictions = self.classifier.predict(X)
            accuracy = (predictions == encoded_labels).mean()
            
            logger.info(f"[ML_CLASSIFIER] Training completed. Accuracy: {accuracy:.4f}")
            
            return {
                "accuracy": float(accuracy),
                "training_samples": len(texts),
                "classifier_type": self.classifier_type
            }
        
        except Exception as e:
            logger.error(f"[ML_CLASSIFIER] Training failed: {e}")
            self.use_ml = False  # フォールバック
            return {}
    
    def classify(
        self,
        text: str,
        title: str
    ) -> Tuple[Optional[str], float]:
        """
        テキストとタイトルからドメインを分類
        
        Args:
            text: テキスト内容
            title: タイトル
        
        Returns:
            (ドメイン, 確信度)のタプル
        """
        # 機械学習ベース分類
        if self.use_ml and self.is_trained:
            try:
                combined = f"{title} {text}"
                X = self.vectorizer.transform([combined])
                prediction = self.classifier.predict(X)[0]
                probabilities = self.classifier.predict_proba(X)[0]
                
                domain = self.label_encoder.inverse_transform([prediction])[0]
                confidence = float(probabilities[prediction])
                
                logger.debug(f"[ML_CLASSIFY] {title} -> {domain} (confidence: {confidence:.4f})")
                return domain, confidence
            
            except Exception as e:
                logger.warning(f"[ML_CLASSIFY] ML classification failed: {e}, falling back to keyword-based")
        
        # キーワードベース分類（フォールバック）
        return self._keyword_based_classify(text, title)
    
    def _keyword_based_classify(
        self,
        text: str,
        title: str
    ) -> Tuple[Optional[str], float]:
        """キーワードベース分類（フォールバック）"""
        text_lower = text.lower()
        title_lower = title.lower()
        combined = f"{title_lower} {text_lower}"
        
        domain_scores = {}
        
        for domain_key, domain_config in self.keywords_config["domains"].items():
            score = 0
            
            # 日本語キーワード
            for keyword in domain_config["keywords_ja"]:
                if keyword in combined:
                    score += 2
                if keyword in title_lower:
                    score += 3
            
            # 英語キーワード
            for keyword in domain_config["keywords_en"]:
                keyword_lower = keyword.lower()
                if keyword_lower in combined:
                    score += 2
                if keyword_lower in title_lower:
                    score += 3
            
            if score > 0:
                domain_scores[domain_key] = score
        
        if not domain_scores:
            return None, 0.0
        
        # 最高スコアのドメインを返す
        best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        max_score = domain_scores[best_domain]
        total_score = sum(domain_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0
        
        return best_domain, confidence
    
    def save_model(self, model_path: Path):
        """モデルを保存"""
        if not self.is_trained:
            logger.warning("[ML_CLASSIFIER] Model is not trained, cannot save")
            return
        
        model_data = {
            "classifier": self.classifier,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
            "classifier_type": self.classifier_type,
            "domains": self.domains,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"[ML_CLASSIFIER] Model saved to {model_path}")
    
    def load_model(self, model_path: Path):
        """モデルを読み込み"""
        if not model_path.exists():
            logger.warning(f"[ML_CLASSIFIER] Model file not found: {model_path}")
            return
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data["classifier"]
        self.vectorizer = model_data["vectorizer"]
        self.label_encoder = model_data["label_encoder"]
        self.classifier_type = model_data["classifier_type"]
        self.domains = model_data["domains"]
        self.is_trained = True
        
        logger.info(f"[ML_CLASSIFIER] Model loaded from {model_path}")





















