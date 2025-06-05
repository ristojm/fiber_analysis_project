#!/usr/bin/env python3
"""
Intel GPU Optimized OCR Comparison
Tests different OCR libraries with Intel GPU acceleration
"""

import time
import cv2
import numpy as np
from pathlib import Path

def test_paddleocr_intel(image_path):
    """Test PaddleOCR with Intel GPU via OpenVINO"""
    try:
        from paddleocr import PaddleOCR
        
        print("\n=== Testing PaddleOCR with Intel GPU ===")
        
        # Initialize with OpenVINO backend for Intel GPU
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=True,  # This will use Intel GPU with OpenVINO
            use_openvino=True,
            precision='fp16',  # Use FP16 for better Intel GPU performance
            show_log=False
        )
        
        # Warm up
        _ = ocr.ocr(image_path, cls=True)
        
        # Measure performance
        start_time = time.perf_counter()
        result = ocr.ocr(image_path, cls=True)
        end_time = time.perf_counter()
        
        print(f"Time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Detected {len(result[0])} text regions")
        
        # Print some results
        for idx, line in enumerate(result[0][:3]):  # First 3 lines
            text = line[1][0]
            confidence = line[1][1]
            print(f"  Text: '{text}' (confidence: {confidence:.3f})")
            
        return result
        
    except ImportError:
        print("PaddleOCR not installed. Install with:")
        print("pip install paddlepaddle paddleocr")
        return None

def test_easyocr_cpu(image_path):
    """Test EasyOCR (CPU only as it doesn't support Intel GPU)"""
    try:
        import easyocr
        
        print("\n=== Testing EasyOCR (CPU) ===")
        
        # Initialize reader
        reader = easyocr.Reader(['en'], gpu=False)  # No Intel GPU support
        
        # Measure performance
        start_time = time.perf_counter()
        result = reader.readtext(image_path)
        end_time = time.perf_counter()
        
        print(f"Time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Detected {len(result)} text regions")
        
        # Print some results
        for idx, (bbox, text, confidence) in enumerate(result[:3]):
            print(f"  Text: '{text}' (confidence: {confidence:.3f})")
            
        return result
        
    except ImportError:
        print("EasyOCR not installed. Install with:")
        print("pip install easyocr")
        return None

def test_tesseract(image_path):
    """Test Tesseract OCR"""
    try:
        import pytesseract
        from PIL import Image
        
        print("\n=== Testing Tesseract ===")
        
        # Load image
        image = Image.open(image_path)
        
        # Measure performance
        start_time = time.perf_counter()
        text = pytesseract.image_to_string(image)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        end_time = time.perf_counter()
        
        print(f"Time: {(end_time - start_time)*1000:.2f} ms")
        
        # Count detected words
        n_boxes = len(data['text'])
        words = [data['text'][i] for i in range(n_boxes) if data['text'][i].strip()]
        print(f"Detected {len(words)} words")
        
        # Print first few words
        print(f"  Text preview: {' '.join(words[:10])}...")
        
        return text
        
    except ImportError:
        print("Pytesseract not installed. Install with:")
        print("pip install pytesseract pillow")
        print("Also need Tesseract binary: https://github.com/tesseract-ocr/tesseract")
        return None

def test_rapidocr_openvino(image_path):
    """Test RapidOCR with OpenVINO backend (good Intel GPU support)"""
    try:
        from rapidocr_openvino import RapidOCR
        
        print("\n=== Testing RapidOCR with OpenVINO ===")
        
        # Initialize with OpenVINO
        ocr = RapidOCR()
        
        # Load image
        img = cv2.imread(image_path)
        
        # Warm up
        _ = ocr(img)
        
        # Measure performance
        start_time = time.perf_counter()
        result, elapse = ocr(img)
        end_time = time.perf_counter()
        
        print(f"Time: {(end_time - start_time)*1000:.2f} ms")
        if result:
            print(f"Detected {len(result)} text regions")
            for idx, (bbox, text, score) in enumerate(result[:3]):
                print(f"  Text: '{text}' (confidence: {score:.3f})")
        
        return result
        
    except ImportError:
        print("RapidOCR not installed. Install with:")
        print("pip install rapidocr-openvino")
        return None

def create_test_image():
    """Create a test image with various text types"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Different text styles to test
    texts = [
        ("HEADER TEXT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5),
        ("Normal paragraph text here", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8),
        ("Small text 123456", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5),
        ("Email: test@example.com", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7),
        ("Phone: +1-234-567-8900", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7),
        ("MIXED CaSe TeXt", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9),
        ("Special chars: @#$%&*()", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7),
        ("Rotated text example", (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8),
    ]
    
    for text, pos, font, scale in texts:
        cv2.putText(img, text, pos, font, scale, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Add a slightly rotated text
    center = (600, 400)
    angle = -15
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_region = cv2.warpAffine(img[350:450, 350:750], M, (400, 100))
    
    cv2.imwrite("test_ocr_comparison.png", img)
    return "test_ocr_comparison.png"

def compare_ocr_libraries():
    """Compare different OCR libraries"""
    print("=== OCR Library Comparison for Intel GPU ===")
    print(f"Creating test image...")
    
    # Create or use existing test image
    image_path = create_test_image()
    
    # Test each library
    results = {}
    
    # PaddleOCR - Best for Intel GPU
    results['paddle'] = test_paddleocr_intel(image_path)
    
    # RapidOCR - Also good for Intel GPU
    results['rapid'] = test_rapidocr_openvino(image_path)
    
    # EasyOCR - CPU only but good accuracy
    results['easy'] = test_easyocr_cpu(image_path)
    
    # Tesseract - Baseline comparison
    results['tesseract'] = test_tesseract(image_path)
    
    print("\n=== Recommendations ===")
    print("For Intel GPU acceleration:")
    print("1. PaddleOCR with OpenVINO - Best overall (accuracy + speed)")
    print("2. RapidOCR-OpenVINO - Lightweight and fast")
    print("3. Custom OpenVINO models - For specific use cases")
    print("\nFor accuracy without GPU:")
    print("1. EasyOCR - Best accuracy but slower")
    print("2. TrOCR - Best for difficult text")
    print("3. Tesseract - Good for clean documents")

if __name__ == "__main__":
    # Check Intel GPU availability
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        print(f"Available devices: {devices}")
        if "GPU" in devices:
            gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            print(f"Intel GPU: {gpu_name}")
    except:
        print("OpenVINO not available")
    
    # Run comparison
    compare_ocr_libraries()