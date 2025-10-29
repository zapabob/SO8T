"""
SO8T Multimodal Processor

This module provides hybrid multimodal processing for the SO8T safety pipeline.
It combines OCR processing with native VL capabilities for optimal image understanding.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import json
from PIL import Image
import torch

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ocr_processor import SO8TOCRProcessor
from models.so8t_safety_judge import SO8TSafetyJudge

logger = logging.getLogger(__name__)

class SO8TMultimodalProcessor:
    """
    SO8T Multimodal Processor for hybrid image processing
    
    Features:
    - OCR + Native VL hybrid processing
    - Image complexity assessment
    - Automatic processing method selection
    - SO(8) group structure integration
    - Safety-aware multimodal analysis
    """
    
    def __init__(self, 
                 ocr_processor: Optional[SO8TOCRProcessor] = None,
                 safety_judge: Optional[SO8TSafetyJudge] = None,
                 vl_model=None,
                 complexity_threshold: float = 0.3):
        """
        Initialize multimodal processor
        
        Args:
            ocr_processor: OCR processor instance
            safety_judge: Safety judge instance
            vl_model: Optional native VL model (Qwen2-VL)
            complexity_threshold: Threshold for VL processing decision
        """
        self.ocr_processor = ocr_processor or SO8TOCRProcessor()
        self.safety_judge = safety_judge or SO8TSafetyJudge()
        self.vl_model = vl_model
        self.complexity_threshold = complexity_threshold
        
        # Processing statistics
        self.stats = {
            'ocr_processed': 0,
            'vl_processed': 0,
            'hybrid_processed': 0,
            'total_processed': 0
        }
    
    def process_input(self, 
                     text: str = "", 
                     image_path: str = "", 
                     audio_path: str = "") -> Dict[str, Any]:
        """
        Process multimodal input (text, image, audio)
        
        Args:
            text: Text input
            image_path: Path to image file
            audio_path: Path to audio file
            
        Returns:
            Processing result dictionary
        """
        result = {
            "text_input": text,
            "image_path": image_path,
            "audio_path": audio_path,
            "processing_method": "unknown",
            "extracted_text": "",
            "safety_judgment": "UNKNOWN",
            "confidence": 0.0,
            "success": False
        }
        
        try:
            # Process text
            if text:
                result["processing_method"] = "text"
                result["extracted_text"] = text
                result["safety_judgment"] = "ALLOW"
                result["confidence"] = 1.0
                result["success"] = True
            
            # Process image
            if image_path and os.path.exists(image_path):
                image_result = self.process_image(image_path)
                result["processing_method"] = image_result.get("processing_method", "ocr")
                result["extracted_text"] = image_result.get("text", "")
                result["safety_judgment"] = image_result.get("safety_judgment", "ALLOW")
                result["confidence"] = image_result.get("confidence", 0.0)
                result["success"] = image_result.get("success", False)
            
            # Process audio (not implemented yet)
            if audio_path and os.path.exists(audio_path):
                result["audio_processing"] = "not_implemented"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in process_input: {e}")
            result["error"] = str(e)
            return result
    
    def process_image(self, 
                     image_path: Union[str, Path],
                     force_method: Optional[str] = None,
                     safety_check: bool = True) -> Dict[str, Any]:
        """
        Process image using hybrid approach
        
        Args:
            image_path: Path to image file
            force_method: Force specific method ('ocr', 'vl', 'hybrid')
            safety_check: Enable safety checking
            
        Returns:
            Processing result dictionary
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image for analysis
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Determine processing method
            if force_method:
                method = force_method
            else:
                method = self._determine_processing_method(image, image_path)
            
            # Process image based on selected method
            if method == 'ocr':
                result = self._process_with_ocr(image, image_path)
                self.stats['ocr_processed'] += 1
            elif method == 'vl' and self.vl_model:
                result = self._process_with_vl(image, image_path)
                self.stats['vl_processed'] += 1
            elif method == 'hybrid':
                result = self._process_hybrid(image, image_path)
                self.stats['hybrid_processed'] += 1
            else:
                # Fallback to OCR
                result = self._process_with_ocr(image, image_path)
                self.stats['ocr_processed'] += 1
            
            # Safety check if enabled
            if safety_check and result.get('text'):
                safety_result = self.safety_judge.judge_text(result['text'])
                result['safety_judgment'] = safety_result['action']
                result['safety_confidence'] = safety_result['confidence']
                result['safety_reasoning'] = safety_result['reasoning']
            else:
                result['safety_judgment'] = 'ALLOW'
                result['safety_confidence'] = 1.0
                result['safety_reasoning'] = 'No safety check performed'
            
            # Add metadata
            result['processing_method'] = method
            result['image_path'] = str(image_path)
            result['image_size'] = image.shape[:2]
            result['processing_successful'] = True
            
            self.stats['total_processed'] += 1
            
            logger.info(f"Processed {image_path.name} using {method} method")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'processing_method': 'error',
                'image_path': str(image_path),
                'processing_successful': False,
                'error': str(e),
                'safety_judgment': 'DENY',
                'safety_confidence': 0.0,
                'safety_reasoning': f'Processing error: {e}'
            }
    
    def _determine_processing_method(self, image: np.ndarray, image_path: Path) -> str:
        """
        Determine optimal processing method based on image characteristics
        
        Args:
            image: Input image
            image_path: Image file path
            
        Returns:
            Processing method ('ocr', 'vl', 'hybrid')
        """
        try:
            # Calculate image complexity
            complexity = self.ocr_processor.calculate_image_complexity(image)
            
            # Check file extension for hints
            ext = image_path.suffix.lower()
            is_diagram = ext in ['.png', '.svg'] and 'diagram' in image_path.name.lower()
            is_chart = 'chart' in image_path.name.lower() or 'graph' in image_path.name.lower()
            
            # Decision logic
            if complexity > self.complexity_threshold:
                if self.vl_model and (is_diagram or is_chart):
                    return 'vl'  # Use native VL for complex diagrams/charts
                else:
                    return 'hybrid'  # Use hybrid approach for complex images
            else:
                return 'ocr'  # Use OCR for simple images
                
        except Exception as e:
            logger.error(f"Error determining processing method: {e}")
            return 'ocr'  # Fallback to OCR
    
    def _process_with_ocr(self, image: np.ndarray, image_path: Path) -> Dict[str, Any]:
        """
        Process image using OCR only
        
        Args:
            image: Input image
            image_path: Image file path
            
        Returns:
            OCR processing result
        """
        try:
            # Use OCR processor
            ocr_result = self.ocr_processor.extract_text_from_image(image_path)
            
            return {
                'text': ocr_result['text'],
                'confidence': ocr_result['confidence'],
                'quality_score': ocr_result['quality_score'],
                'complexity_score': ocr_result['complexity_score'],
                'use_native_vl': ocr_result['use_native_vl'],
                'boxes': ocr_result['boxes'],
                'text_regions': ocr_result['text_regions'],
                'processing_details': {
                    'method': 'ocr_only',
                    'denoising_applied': True,
                    'contrast_enhanced': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'quality_score': 0.0,
                'complexity_score': 0.0,
                'use_native_vl': False,
                'boxes': [],
                'text_regions': [],
                'processing_details': {'method': 'ocr_only', 'error': str(e)}
            }
    
    def _process_with_vl(self, image: np.ndarray, image_path: Path) -> Dict[str, Any]:
        """
        Process image using native VL model
        
        Args:
            image: Input image
            image_path: Image file path
            
        Returns:
            VL processing result
        """
        try:
            if not self.vl_model:
                raise ValueError("VL model not available")
            
            # Convert image to PIL format for VL model
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Process with VL model (placeholder - would need actual VL model)
            # This is a simplified interface
            vl_result = self._call_vl_model(pil_image)
            
            return {
                'text': vl_result.get('text', ''),
                'confidence': vl_result.get('confidence', 0.9),
                'quality_score': vl_result.get('quality_score', 0.9),
                'complexity_score': vl_result.get('complexity_score', 0.8),
                'use_native_vl': True,
                'vl_analysis': vl_result.get('analysis', {}),
                'processing_details': {
                    'method': 'vl_native',
                    'model_used': 'qwen2-vl',
                    'image_understanding': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in VL processing: {e}")
            # Fallback to OCR
            return self._process_with_ocr(image, image_path)
    
    def _process_hybrid(self, image: np.ndarray, image_path: Path) -> Dict[str, Any]:
        """
        Process image using hybrid OCR + VL approach
        
        Args:
            image: Input image
            image_path: Image file path
            
        Returns:
            Hybrid processing result
        """
        try:
            # First, try OCR
            ocr_result = self.ocr_processor.extract_text_from_image(image_path)
            
            # If OCR confidence is low, try VL as well
            if ocr_result['confidence'] < 0.7 and self.vl_model:
                vl_result = self._process_with_vl(image, image_path)
                
                # Combine results
                combined_text = self._combine_text_results(
                    ocr_result['text'], 
                    vl_result['text']
                )
                
                return {
                    'text': combined_text,
                    'confidence': max(ocr_result['confidence'], vl_result['confidence']),
                    'quality_score': max(ocr_result['quality_score'], vl_result['quality_score']),
                    'complexity_score': ocr_result['complexity_score'],
                    'use_native_vl': True,
                    'ocr_text': ocr_result['text'],
                    'vl_text': vl_result['text'],
                    'ocr_confidence': ocr_result['confidence'],
                    'vl_confidence': vl_result['confidence'],
                    'processing_details': {
                        'method': 'hybrid',
                        'ocr_used': True,
                        'vl_used': True,
                        'text_combined': True
                    }
                }
            else:
                # OCR is sufficient
                return {
                    'text': ocr_result['text'],
                    'confidence': ocr_result['confidence'],
                    'quality_score': ocr_result['quality_score'],
                    'complexity_score': ocr_result['complexity_score'],
                    'use_native_vl': False,
                    'ocr_text': ocr_result['text'],
                    'processing_details': {
                        'method': 'hybrid_ocr_only',
                        'ocr_used': True,
                        'vl_used': False,
                        'reason': 'OCR confidence sufficient'
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in hybrid processing: {e}")
            # Fallback to OCR only
            return self._process_with_ocr(image, image_path)
    
    def _call_vl_model(self, image: Image.Image) -> Dict[str, Any]:
        """
        Call native VL model (placeholder implementation)
        
        Args:
            image: PIL Image
            
        Returns:
            VL model result
        """
        # This is a placeholder - in practice, you would call the actual VL model
        # For now, return a mock result
        return {
            'text': f"VL analysis of image ({image.size[0]}x{image.size[1]})",
            'confidence': 0.85,
            'quality_score': 0.9,
            'complexity_score': 0.7,
            'analysis': {
                'objects_detected': ['text', 'image'],
                'text_regions': 1,
                'confidence': 0.85
            }
        }
    
    def _combine_text_results(self, ocr_text: str, vl_text: str) -> str:
        """
        Combine OCR and VL text results
        
        Args:
            ocr_text: OCR extracted text
            vl_text: VL extracted text
            
        Returns:
            Combined text
        """
        # Simple combination strategy
        if not ocr_text and not vl_text:
            return ""
        elif not ocr_text:
            return vl_text
        elif not vl_text:
            return ocr_text
        else:
            # Both available - combine intelligently
            if len(ocr_text) > len(vl_text):
                # OCR seems more complete
                return ocr_text
            else:
                # VL might have better understanding
                return vl_text
    
    def batch_process_images(self, 
                           image_paths: List[Union[str, Path]],
                           force_method: Optional[str] = None,
                           safety_check: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            force_method: Force specific method for all images
            safety_check: Enable safety checking
            
        Returns:
            List of processing results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.process_image(
                    image_path, 
                    force_method=force_method,
                    safety_check=safety_check
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'text': '',
                    'confidence': 0.0,
                    'processing_method': 'error',
                    'image_path': str(image_path),
                    'processing_successful': False,
                    'error': str(e),
                    'safety_judgment': 'DENY',
                    'safety_confidence': 0.0,
                    'safety_reasoning': f'Processing error: {e}'
                })
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics
        
        Returns:
            Statistics dictionary
        """
        total = self.stats['total_processed']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'ocr_percentage': (self.stats['ocr_processed'] / total) * 100,
            'vl_percentage': (self.stats['vl_processed'] / total) * 100,
            'hybrid_percentage': (self.stats['hybrid_processed'] / total) * 100
        }
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            'ocr_processed': 0,
            'vl_processed': 0,
            'hybrid_processed': 0,
            'total_processed': 0
        }
    
    def save_processing_result(self, result: Dict[str, Any], output_path: Union[str, Path]) -> bool:
        """
        Save processing result to file
        
        Args:
            result: Processing result dictionary
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
            
            logger.info(f"Processing result saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processing result: {e}")
            return False

def create_multimodal_processor(ocr_processor: Optional[SO8TOCRProcessor] = None,
                               safety_judge: Optional[SO8TSafetyJudge] = None,
                               vl_model=None) -> SO8TMultimodalProcessor:
    """Create and initialize SO8T Multimodal Processor"""
    return SO8TMultimodalProcessor(
        ocr_processor=ocr_processor,
        safety_judge=safety_judge,
        vl_model=vl_model
    )

def main():
    """Test multimodal processor"""
    print("SO8T Multimodal Processor Test")
    print("=" * 50)
    
    # Create test images
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create simple test image
    simple_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(simple_img, "Simple Text", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    simple_path = test_dir / "multimodal_simple.png"
    cv2.imwrite(str(simple_path), simple_img)
    
    # Create complex test image
    complex_img = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
    cv2.putText(complex_img, "Complex Diagram", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    complex_path = test_dir / "multimodal_complex.png"
    cv2.imwrite(str(complex_path), complex_img)
    
    # Create multimodal processor
    processor = create_multimodal_processor()
    
    # Test images
    test_images = [simple_path, complex_path]
    
    print("Testing multimodal processing:")
    print("-" * 30)
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n--- Testing Image {i}: {image_path.name} ---")
        
        try:
            result = processor.process_image(image_path)
            
            print(f"Text: '{result['text']}'")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Quality Score: {result['quality_score']:.2f}")
            print(f"Complexity Score: {result['complexity_score']:.2f}")
            print(f"Processing Method: {result['processing_method']}")
            print(f"Use Native VL: {result['use_native_vl']}")
            print(f"Safety Judgment: {result['safety_judgment']}")
            print(f"Safety Confidence: {result['safety_confidence']:.2f}")
            print(f"Processing Successful: {result['processing_successful']}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Test batch processing
    print(f"\n--- Testing Batch Processing ---")
    try:
        batch_results = processor.batch_process_images(test_images)
        print(f"Batch processed {len(batch_results)} images")
        
        successful_count = sum(1 for r in batch_results if r['processing_successful'])
        print(f"Successful: {successful_count}/{len(batch_results)}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
    
    # Show statistics
    stats = processor.get_processing_statistics()
    print(f"\n--- Processing Statistics ---")
    print(f"Total Processed: {stats['total_processed']}")
    print(f"OCR Processed: {stats['ocr_processed']} ({stats.get('ocr_percentage', 0):.1f}%)")
    print(f"VL Processed: {stats['vl_processed']} ({stats.get('vl_percentage', 0):.1f}%)")
    print(f"Hybrid Processed: {stats['hybrid_processed']} ({stats.get('hybrid_percentage', 0):.1f}%)")
    
    print(f"\n[OK] Multimodal processor test completed!")

if __name__ == "__main__":
    main()
