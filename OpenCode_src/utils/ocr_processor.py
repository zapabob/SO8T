"""
SO8T OCR Processor

This module provides OCR functionality using Tesseract and OpenCV for the SO8T safety pipeline.
It handles image preprocessing, text extraction, and confidence scoring for multimodal input processing.
"""

import cv2
import pytesseract
import numpy as np
from PIL import Image
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class SO8TOCRProcessor:
    """
    SO8T OCR Processor for multimodal input processing
    
    Features:
    - Image preprocessing (denoising, contrast enhancement)
    - Multi-language OCR (Japanese + English)
    - Confidence scoring and bounding box detection
    - Text quality assessment
    - Batch processing support
    """
    
    def __init__(self, tesseract_path: Optional[str] = None, language: str = 'jpn+eng'):
        """
        Initialize OCR processor
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
            language: OCR language configuration (default: 'jpn+eng')
        """
        self.language = language
        self.tesseract_path = tesseract_path
        
        # Set tesseract path if provided
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Tesseract path set to: {tesseract_path}")
        
        # OCR configuration
        self.ocr_config = '--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
        
        # Image preprocessing parameters
        self.preprocessing_params = {
            'denoise_strength': 10,
            'contrast_alpha': 1.2,
            'brightness_beta': 10,
            'gaussian_kernel': (5, 5),
            'morphology_kernel': np.ones((3, 3), np.uint8)
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_confidence': 30.0,
            'min_text_length': 3,
            'max_skew_angle': 15.0
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(
                gray, 
                h=self.preprocessing_params['denoise_strength']
            )
            
            # Contrast enhancement
            enhanced = cv2.convertScaleAbs(
                denoised,
                alpha=self.preprocessing_params['contrast_alpha'],
                beta=self.preprocessing_params['brightness_beta']
            )
            
            # Gaussian blur for noise reduction
            blurred = cv2.GaussianBlur(
                enhanced,
                self.preprocessing_params['gaussian_kernel'],
                0
            )
            
            # Morphological operations to clean up text
            kernel = self.preprocessing_params['morphology_kernel']
            cleaned = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in image using contour detection
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of text region dictionaries
        """
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (remove very small regions)
                if w > 20 and h > 10:
                    text_regions.append({
                        'bbox': (x, y, w, h),
                        'area': w * h,
                        'aspect_ratio': w / h if h > 0 else 0
                    })
            
            # Sort by area (largest first)
            text_regions.sort(key=lambda x: x['area'], reverse=True)
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []
    
    def extract_text_from_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Extract text from image with confidence scoring
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Detect text regions
            text_regions = self.detect_text_regions(processed_image)
            
            # Extract text using Tesseract
            text_data = pytesseract.image_to_data(
                processed_image,
                lang=self.language,
                config=self.ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results
            extracted_text = pytesseract.image_to_string(
                processed_image,
                lang=self.language,
                config=self.ocr_config
            ).strip()
            
            # Calculate confidence scores
            confidences = [int(conf) for conf in text_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract bounding boxes
            boxes = []
            for i, conf in enumerate(text_data['conf']):
                if int(conf) > self.quality_thresholds['min_confidence']:
                    boxes.append({
                        'text': text_data['text'][i],
                        'confidence': int(conf),
                        'bbox': (
                            text_data['left'][i],
                            text_data['top'][i],
                            text_data['width'][i],
                            text_data['height'][i]
                        )
                    })
            
            # Assess text quality
            quality_score = self.assess_text_quality(extracted_text, avg_confidence, boxes)
            
            # Determine if native VL processing is needed
            complexity_score = self.calculate_image_complexity(image)
            use_native_vl = complexity_score > 0.3
            
            result = {
                'text': extracted_text,
                'confidence': avg_confidence,
                'boxes': boxes,
                'text_regions': text_regions,
                'quality_score': quality_score,
                'complexity_score': complexity_score,
                'use_native_vl': use_native_vl,
                'language': self.language,
                'image_size': image.shape[:2],
                'processing_successful': True
            }
            
            logger.info(f"OCR completed for {image_path}: {len(extracted_text)} chars, confidence: {avg_confidence:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'boxes': [],
                'text_regions': [],
                'quality_score': 0.0,
                'complexity_score': 0.0,
                'use_native_vl': False,
                'language': self.language,
                'image_size': (0, 0),
                'processing_successful': False,
                'error': str(e)
            }
    
    def assess_text_quality(self, text: str, confidence: float, boxes: List[Dict]) -> float:
        """
        Assess quality of extracted text
        
        Args:
            text: Extracted text
            confidence: Average confidence score
            boxes: Bounding box information
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            # Base quality from confidence
            quality = confidence / 100.0
            
            # Length penalty (too short or too long)
            text_length = len(text.strip())
            if text_length < self.quality_thresholds['min_text_length']:
                quality *= 0.5
            elif text_length > 1000:  # Very long text might be noise
                quality *= 0.8
            
            # Character diversity (more diverse = better)
            unique_chars = len(set(text))
            if unique_chars > 10:
                quality *= 1.1
            
            # Box consistency (similar sized boxes = better)
            if len(boxes) > 1:
                box_heights = [box['bbox'][3] for box in boxes]
                height_std = np.std(box_heights)
                if height_std < 10:  # Consistent heights
                    quality *= 1.05
            
            # Japanese character presence bonus
            japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309F' or 
                               '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF')
            if japanese_chars > 0:
                quality *= 1.1
            
            return min(quality, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing text quality: {e}")
            return 0.0
    
    def calculate_image_complexity(self, image: np.ndarray) -> float:
        """
        Calculate image complexity score for VL processing decision
        
        Args:
            image: Input image
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Texture complexity (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            texture_complexity = min(laplacian_var / 1000.0, 1.0)
            
            # Color diversity (if color image)
            color_diversity = 0.0
            if len(image.shape) == 3:
                hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
                hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
                
                # Calculate entropy for each channel
                entropy = 0
                for hist in [hist_b, hist_g, hist_r]:
                    hist_norm = hist / np.sum(hist)
                    hist_norm = hist_norm[hist_norm > 0]
                    entropy += -np.sum(hist_norm * np.log2(hist_norm))
                
                color_diversity = entropy / (3 * 8)  # Normalize to 0-1
            
            # Combined complexity score
            complexity = (edge_density * 0.4 + texture_complexity * 0.3 + color_diversity * 0.3)
            
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating image complexity: {e}")
            return 0.0
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image for OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing OCR results
        """
        try:
            # Load image
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # Preprocess image
            processed_image = self.preprocess_image(image_np)
            
            # Extract text and confidence
            ocr_result = self.extract_text_from_image(image_path)
            text = ocr_result.get('text', '')
            confidence = ocr_result.get('confidence', 0.0)
            details = ocr_result.get('details', [])
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(text, confidence)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(image_path)
            
            return {
                'text': text,
                'confidence': confidence,
                'quality_score': quality_score,
                'complexity_score': complexity_score,
                'details': details,
                'image_path': image_path,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'quality_score': 0.0,
                'complexity_score': 0.0,
                'details': [],
                'image_path': image_path,
                'success': False,
                'error': str(e)
            }
    
    def batch_process_images(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of OCR results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.extract_text_from_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'text': '',
                    'confidence': 0.0,
                    'boxes': [],
                    'text_regions': [],
                    'quality_score': 0.0,
                    'complexity_score': 0.0,
                    'use_native_vl': False,
                    'language': self.language,
                    'image_size': (0, 0),
                    'processing_successful': False,
                    'error': str(e)
                })
        
        return results
    
    def save_ocr_result(self, result: Dict, output_path: Union[str, Path]) -> bool:
        """
        Save OCR result to JSON file
        
        Args:
            result: OCR result dictionary
            output_path: Output file path
            
        Returns:
            Success status
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"OCR result saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving OCR result: {e}")
            return False

def main():
    """Test OCR processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SO8T OCR Processor Test')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--language', default='jpn+eng', help='OCR language')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize OCR processor
    ocr = SO8TOCRProcessor(tesseract_path=args.tesseract_path, language=args.language)
    
    # Process image
    print(f"Processing image: {args.image_path}")
    result = ocr.extract_text_from_image(args.image_path)
    
    # Print results
    print(f"\nExtracted Text:")
    print(f"'{result['text']}'")
    print(f"\nConfidence: {result['confidence']:.1f}%")
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"Complexity Score: {result['complexity_score']:.2f}")
    print(f"Use Native VL: {result['use_native_vl']}")
    print(f"Text Regions: {len(result['text_regions'])}")
    print(f"Bounding Boxes: {len(result['boxes'])}")
    
    # Save result if output path specified
    if args.output:
        ocr.save_ocr_result(result, args.output)
        print(f"\nResult saved to: {args.output}")

if __name__ == "__main__":
    main()
