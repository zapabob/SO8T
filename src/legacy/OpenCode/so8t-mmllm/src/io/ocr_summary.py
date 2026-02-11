"""
OCR要約プロセッサ
OpenCV + Tesseract でローカル画像処理
"""

import cv2
import pytesseract
import json
import base64
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union
import io
import hashlib


class OCRSummaryProcessor:
    """
    ローカルOCR要約プロセッサ
    画像を外部に送信せずにローカルで処理
    """
    
    def __init__(
        self,
        tesseract_config: str = "--oem 3 --psm 6",
        languages: str = "jpn+eng",
        min_confidence: float = 30.0
    ):
        """
        Args:
            tesseract_config: Tesseract設定
            languages: 認識言語
            min_confidence: 最小信頼度
        """
        self.tesseract_config = tesseract_config
        self.languages = languages
        self.min_confidence = min_confidence
        
        # 画像前処理パラメータ
        self.preprocess_params = {
            "blur_kernel": (3, 3),
            "threshold_block_size": 11,
            "threshold_c": 2,
            "morph_kernel": np.ones((2, 2), np.uint8)
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        画像の前処理
        
        Args:
            image: 入力画像 [H, W, C]
            
        Returns:
            前処理済み画像
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # ノイズ除去
        blurred = cv2.GaussianBlur(gray, self.preprocess_params["blur_kernel"], 0)
        
        # 適応的閾値処理
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.preprocess_params["threshold_block_size"],
            self.preprocess_params["threshold_c"]
        )
        
        # モルフォロジー処理
        kernel = self.preprocess_params["morph_kernel"]
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def extract_text_blocks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        テキストブロックを抽出
        
        Args:
            image: 前処理済み画像
            
        Returns:
            テキストブロックのリスト
        """
        # テキスト領域の検出
        data = pytesseract.image_to_data(
            image,
            config=self.tesseract_config,
            lang=self.languages,
            output_type=pytesseract.Output.DICT
        )
        
        blocks = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if conf > self.min_confidence and text:
                block = {
                    "text": text,
                    "confidence": conf,
                    "bbox": {
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i]
                    }
                }
                blocks.append(block)
        
        return blocks
    
    def generate_summary(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        テキストブロックから要約を生成
        
        Args:
            blocks: テキストブロックのリスト
            
        Returns:
            要約辞書
        """
        if not blocks:
            return {
                "text": "",
                "blocks": [],
                "lang": "unknown",
                "confidence": 0.0,
                "summary": "No text detected"
            }
        
        # 全テキストを結合
        full_text = " ".join([block["text"] for block in blocks])
        
        # 言語推定（簡易版）
        lang = "mixed"
        if any(ord(char) > 127 for char in full_text):
            lang = "japanese"
        else:
            lang = "english"
        
        # 平均信頼度
        avg_confidence = np.mean([block["confidence"] for block in blocks])
        
        # 簡易要約（最初の100文字）
        summary = full_text[:100] + "..." if len(full_text) > 100 else full_text
        
        return {
            "text": full_text,
            "blocks": blocks,
            "lang": lang,
            "confidence": float(avg_confidence),
            "summary": summary
        }
    
    def process_image(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        画像を処理してOCR要約を生成
        
        Args:
            image: 画像（パス、numpy配列、PIL画像）
            
        Returns:
            OCR要約辞書
        """
        # 画像の読み込み
        if isinstance(image, str):
            # ファイルパス
            if image.startswith("file://"):
                image_path = image[7:]  # "file://" を除去
                pil_image = Image.open(image_path)
            elif image.startswith("data:image"):
                # Base64エンコード画像
                header, data = image.split(",", 1)
                image_data = base64.b64decode(data)
                pil_image = Image.open(io.BytesIO(image_data))
            else:
                # 通常のファイルパス
                pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # RGBに変換
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # numpy配列に変換
        image_array = np.array(pil_image)
        
        # 前処理
        processed_image = self.preprocess_image(image_array)
        
        # テキストブロック抽出
        blocks = self.extract_text_blocks(processed_image)
        
        # 要約生成
        summary = self.generate_summary(blocks)
        
        # 画像ハッシュ（プライバシー保護）
        image_hash = hashlib.sha256(image_array.tobytes()).hexdigest()[:16]
        summary["image_hash"] = image_hash
        
        return summary
    
    def process_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """
        複数画像をバッチ処理
        
        Args:
            images: 画像のリスト
            
        Returns:
            OCR要約のリスト
        """
        summaries = []
        for image in images:
            try:
                summary = self.process_image(image)
                summaries.append(summary)
            except Exception as e:
                # エラー時は空の要約を追加
                summaries.append({
                    "text": "",
                    "blocks": [],
                    "lang": "unknown",
                    "confidence": 0.0,
                    "summary": f"Error processing image: {str(e)}",
                    "image_hash": "error"
                })
        
        return summaries
    
    def create_prompt_with_ocr(
        self, 
        text_prompt: str, 
        image_summaries: List[Dict[str, Any]]
    ) -> str:
        """
        OCR要約を含むプロンプトを作成
        
        Args:
            text_prompt: テキストプロンプト
            image_summaries: 画像要約のリスト
            
        Returns:
            結合されたプロンプト
        """
        if not image_summaries:
            return text_prompt
        
        ocr_parts = []
        for i, summary in enumerate(image_summaries):
            ocr_part = f"Image {i+1} OCR Summary:\n"
            ocr_part += f"Language: {summary['lang']}\n"
            ocr_part += f"Confidence: {summary['confidence']:.1f}%\n"
            ocr_part += f"Text: {summary['text']}\n"
            ocr_part += f"Summary: {summary['summary']}\n"
            ocr_parts.append(ocr_part)
        
        ocr_text = "\n".join(ocr_parts)
        return f"{ocr_text}\n\n{text_prompt}"
    
    def log_privacy_policy(self, audit_logger: Optional[Any] = None) -> None:
        """
        プライバシーポリシーをログに記録
        
        Args:
            audit_logger: 監査ロガー
        """
        policy = {
            "image_processing": "LOCAL_ONLY",
            "external_sharing": "FORBIDDEN",
            "data_retention": "NONE",
            "ocr_only": True
        }
        
        if audit_logger:
            audit_logger.log_audit(
                change_type="privacy_policy",
                change_description="OCR processing policy applied",
                change_data=policy
            )
