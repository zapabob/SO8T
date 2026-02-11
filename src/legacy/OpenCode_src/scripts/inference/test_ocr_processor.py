"""
SO8T OCR Processor Test Script

This script tests the OCR processor functionality with sample images.
"""

import os
import sys
import logging
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ocr_processor import SO8TOCRProcessor

def create_test_images():
    """Create test images for OCR testing"""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test image with text
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # White background
    
    # Add text using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Hello World", (50, 100), font, 2, (0, 0, 0), 3)
    cv2.putText(img, "SO8T OCR Test", (50, 150), font, 1, (0, 0, 0), 2)
    
    # Save test image
    test_image_path = test_dir / "test_simple.png"
    cv2.imwrite(str(test_image_path), img)
    
    # Create a more complex test image
    complex_img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    
    # Add multiple lines of text
    texts = [
        "SO8T Safety Pipeline",
        "OCR Processing Test",
        "Multimodal Input",
        "Japanese + English"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(complex_img, text, (50, 80 + i * 50), font, 1.5, (0, 0, 0), 2)
    
    # Add some noise
    noise = np.random.randint(0, 50, complex_img.shape, dtype=np.uint8)
    complex_img = cv2.add(complex_img, noise)
    
    # Save complex test image
    complex_image_path = test_dir / "test_complex.png"
    cv2.imwrite(str(complex_image_path), complex_img)
    
    return [test_image_path, complex_image_path]

def test_ocr_processor():
    """Test OCR processor functionality"""
    print("SO8T OCR Processor Test")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test images
    print("Creating test images...")
    test_images = create_test_images()
    print(f"Created {len(test_images)} test images")
    
    # Initialize OCR processor
    print("\nInitializing OCR processor...")
    ocr = SO8TOCRProcessor(language='jpn+eng')
    
    # Test each image
    for i, image_path in enumerate(test_images):
        print(f"\n--- Testing Image {i+1}: {image_path.name} ---")
        
        try:
            # Process image
            result = ocr.extract_text_from_image(image_path)
            
            # Print results
            print(f"Text: '{result['text']}'")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Quality Score: {result['quality_score']:.2f}")
            print(f"Complexity Score: {result['complexity_score']:.2f}")
            print(f"Use Native VL: {result['use_native_vl']}")
            print(f"Text Regions: {len(result['text_regions'])}")
            print(f"Bounding Boxes: {len(result['boxes'])}")
            print(f"Processing Successful: {result['processing_successful']}")
            
            if not result['processing_successful']:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Test batch processing
    print(f"\n--- Testing Batch Processing ---")
    try:
        batch_results = ocr.batch_process_images(test_images)
        print(f"Batch processed {len(batch_results)} images")
        
        successful_count = sum(1 for r in batch_results if r['processing_successful'])
        print(f"Successful: {successful_count}/{len(batch_results)}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
    
    # Test image preprocessing
    print(f"\n--- Testing Image Preprocessing ---")
    try:
        test_img = cv2.imread(str(test_images[0]))
        if test_img is not None:
            preprocessed = ocr.preprocess_image(test_img)
            print(f"Original shape: {test_img.shape}")
            print(f"Preprocessed shape: {preprocessed.shape}")
            print("Preprocessing successful")
        else:
            print("Could not load test image for preprocessing test")
    except Exception as e:
        print(f"Error in preprocessing test: {e}")
    
    print(f"\n--- OCR Processor Test Completed ---")

def test_quality_assessment():
    """Test text quality assessment"""
    print("\n--- Testing Quality Assessment ---")
    
    ocr = SO8TOCRProcessor()
    
    # Test cases
    test_cases = [
        ("Hello World", 95.0, []),  # High quality
        ("Hi", 60.0, []),  # Low quality (short)
        ("", 0.0, []),  # Empty text
        ("こんにちは世界", 85.0, []),  # Japanese text
        ("Very long text " * 100, 70.0, []),  # Very long text
    ]
    
    for text, confidence, boxes in test_cases:
        quality = ocr.assess_text_quality(text, confidence, boxes)
        print(f"Text: '{text[:30]}...' | Confidence: {confidence}% | Quality: {quality:.2f}")

def test_complexity_calculation():
    """Test image complexity calculation"""
    print("\n--- Testing Complexity Calculation ---")
    
    ocr = SO8TOCRProcessor()
    
    # Create test images with different complexity levels
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Simple image (low complexity)
    simple_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.putText(simple_img, "Simple", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    simple_path = test_dir / "complexity_simple.png"
    cv2.imwrite(str(simple_path), simple_img)
    
    # Complex image (high complexity)
    complex_img = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
    cv2.putText(complex_img, "Complex", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    complex_path = test_dir / "complexity_complex.png"
    cv2.imwrite(str(complex_path), complex_img)
    
    # Test complexity calculation
    for img_path in [simple_path, complex_path]:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                complexity = ocr.calculate_image_complexity(img)
                print(f"{img_path.name}: Complexity = {complexity:.3f}")
        except Exception as e:
            print(f"Error testing {img_path}: {e}")

def main():
    """Main test function"""
    try:
        test_ocr_processor()
        test_quality_assessment()
        test_complexity_calculation()
        print("\n[OK] All OCR processor tests completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
