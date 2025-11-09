#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチモーダルNSFW検知モジュール

画像+テキストの組み合わせでのNSFW検知機能を提供します。
検知目的であり、生成目的ではありません。

Usage:
    from scripts.data.multimodal_nsfw_detector import MultimodalNSFWDetector
    detector = MultimodalNSFWDetector()
    label, confidence = detector.detect_multimodal(text, image_path)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# NSFW分類器のインポート
try:
    from scripts.data.train_nsfw_classifier import NSFWClassifier
    NSFW_CLASSIFIER_AVAILABLE = True
except ImportError:
    NSFW_CLASSIFIER_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultimodalNSFWDetector:
    """マルチモーダルNSFW検知クラス"""
    
    def __init__(
        self,
        text_classifier_path: Optional[Path] = None,
        image_classifier_path: Optional[Path] = None
    ):
        """
        Args:
            text_classifier_path: テキストNSFW分類器のパス
            image_classifier_path: 画像NSFW分類器のパス（将来の拡張用）
        """
        # テキストNSFW分類器
        self.text_classifier = None
        if NSFW_CLASSIFIER_AVAILABLE:
            if text_classifier_path and text_classifier_path.exists():
                try:
                    self.text_classifier = NSFWClassifier(model_path=text_classifier_path)
                    logger.info(f"[NSFW] Text classifier loaded from {text_classifier_path}")
                except Exception as e:
                    logger.warning(f"[NSFW] Failed to load text classifier: {e}")
            else:
                # デフォルトパスを試行
                default_path = Path("models/nsfw_classifier.joblib")
                if default_path.exists():
                    try:
                        self.text_classifier = NSFWClassifier(model_path=default_path)
                        logger.info(f"[NSFW] Text classifier loaded from default path")
                    except Exception as e:
                        logger.warning(f"[NSFW] Failed to load default text classifier: {e}")
        
        # 画像NSFW分類器（将来の拡張用）
        # 現在はテキストベースの検知のみ実装
        self.image_classifier = None
        if image_classifier_path and image_classifier_path.exists():
            # 画像分類器の実装は将来の拡張
            logger.info(f"[NSFW] Image classifier path specified but not implemented yet")
        
        if not self.text_classifier:
            logger.warning("[NSFW] No NSFW classifier available. Detection will be limited.")
    
    def detect_text_nsfw(self, text: str) -> Tuple[str, float]:
        """
        テキストのNSFW検知
        
        Args:
            text: 検知対象テキスト
        
        Returns:
            (label, confidence): NSFWラベルと信頼度
        """
        if not self.text_classifier:
            # 分類器がない場合は安全側に倒す
            return "safe", 1.0
        
        try:
            label, confidence = self.text_classifier.predict(text)
            return label, confidence
        except Exception as e:
            logger.warning(f"[NSFW] Text detection failed: {e}")
            return "unknown", 0.0
    
    def detect_image_nsfw(self, image_path: Path) -> Tuple[str, float]:
        """
        画像のNSFW検知（将来の拡張用）
        
        Args:
            image_path: 画像ファイルパス
        
        Returns:
            (label, confidence): NSFWラベルと信頼度
        """
        # 現在は実装なし（将来の拡張）
        # 画像分類モデルまたはAPIを使用して実装可能
        return "unknown", 0.0
    
    def detect_multimodal(
        self,
        text: str,
        image_paths: Optional[List[Path]] = None,
        context_weight: float = 0.6
    ) -> Dict[str, any]:
        """
        マルチモーダルNSFW検知（テキスト+画像の組み合わせ）
        
        Args:
            text: テキストコンテンツ
            image_paths: 画像ファイルパスのリスト
            context_weight: コンテキスト（テキスト）の重み（0.0-1.0）
        
        Returns:
            detection_result: 検知結果
        """
        # テキストNSFW検知
        text_label, text_confidence = self.detect_text_nsfw(text)
        
        # 画像NSFW検知（将来の拡張）
        image_labels = []
        image_confidences = []
        
        if image_paths:
            for image_path in image_paths:
                if image_path.exists():
                    img_label, img_confidence = self.detect_image_nsfw(image_path)
                    image_labels.append(img_label)
                    image_confidences.append(img_confidence)
        
        # マルチモーダル判定（テキスト+画像の組み合わせ）
        # 現在はテキストベースの判定を優先
        if text_label != "safe":
            # テキストがNSFWの場合は、画像も考慮
            if image_labels:
                # 画像もNSFWの場合は確信度を上げる
                if any(label != "safe" for label in image_labels):
                    final_label = text_label
                    final_confidence = min(text_confidence + 0.1, 1.0)
                else:
                    # 画像が安全な場合はテキストの判定を維持
                    final_label = text_label
                    final_confidence = text_confidence
            else:
                # 画像がない場合はテキストの判定を使用
                final_label = text_label
                final_confidence = text_confidence
        else:
            # テキストが安全な場合
            if image_labels and any(label != "safe" for label in image_labels):
                # 画像がNSFWの場合は画像の判定を優先
                final_label = image_labels[0]
                final_confidence = image_confidences[0] if image_confidences else 0.5
            else:
                # すべて安全
                final_label = "safe"
                final_confidence = text_confidence
        
        # 検知結果
        result = {
            "label": final_label,
            "confidence": final_confidence,
            "text_label": text_label,
            "text_confidence": text_confidence,
            "image_labels": image_labels,
            "image_confidences": image_confidences,
            "multimodal": len(image_paths) > 0 if image_paths else False,
            "detection_purpose": "safety_training"  # 検知目的、生成目的ではない
        }
        
        return result
    
    def detect_batch(
        self,
        samples: List[Dict]
    ) -> List[Dict]:
        """
        バッチNSFW検知
        
        Args:
            samples: サンプルリスト（text, imagesフィールドを含む）
        
        Returns:
            labeled_samples: NSFWラベル付きサンプル
        """
        labeled_samples = []
        
        for sample in tqdm(samples, desc="NSFW detection"):
            text = sample.get("text", sample.get("content", ""))
            
            # 画像パス取得
            image_paths = []
            if "images" in sample:
                for img_info in sample["images"]:
                    if "path" in img_info:
                        image_path = Path(img_info["path"])
                        if image_path.exists():
                            image_paths.append(image_path)
            
            # マルチモーダルNSFW検知
            detection_result = self.detect_multimodal(text, image_paths)
            
            # サンプルにNSFWラベルを追加
            sample["nsfw_label"] = detection_result["label"]
            sample["nsfw_confidence"] = detection_result["confidence"]
            sample["nsfw_text_label"] = detection_result["text_label"]
            sample["nsfw_text_confidence"] = detection_result["text_confidence"]
            sample["nsfw_image_labels"] = detection_result["image_labels"]
            sample["nsfw_detection_purpose"] = detection_result["detection_purpose"]
            sample["nsfw_multimodal"] = detection_result["multimodal"]
            
            labeled_samples.append(sample)
        
        return labeled_samples


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal NSFW detection")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input data file (JSONL)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output data file (JSONL)")
    parser.add_argument("--text-classifier", type=Path, default=None,
                        help="Text NSFW classifier path")
    args = parser.parse_args()
    
    detector = MultimodalNSFWDetector(text_classifier_path=args.text_classifier)
    
    # データ読み込み
    samples = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # NSFW検知
    labeled_samples = detector.detect_batch(samples)
    
    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in labeled_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 統計
    nsfw_count = sum(1 for s in labeled_samples if s.get("nsfw_label") != "safe")
    print(f"\n[SUCCESS] Labeled {len(labeled_samples):,} samples")
    print(f"[STATS] NSFW detected: {nsfw_count:,} ({nsfw_count/len(labeled_samples)*100:.1f}%)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()







