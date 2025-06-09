#!/usr/bin/env python3
"""
Simple test to isolate the CrumblyDetector hanging issue
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time
import threading

# Add paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))
sys.path.insert(0, str(current_dir))

try:
    from crumbly_detection import CrumblyDetector
    print("✅ CrumblyDetector imported successfully")
except ImportError as e:
    print(f"❌ CrumblyDetector import failed: {e}")
    sys.exit(1)

def test_crumbly_detector_simple():
    """Test CrumblyDetector with minimal input"""
    print("\n🧪 Testing CrumblyDetector with simple inputs...")
    
    try:
        # Create simple test data
        test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        test_mask = np.ones((200, 200), dtype=np.uint8) * 255
        test_mask[50:150, 50:150] = 0  # Create some "fiber" area
        
        detector = CrumblyDetector(porosity_aware=True)
        print("✅ CrumblyDetector initialized")
        
        # Test with timeout
        result = None
        error_occurred = None
        
        def analysis_worker():
            nonlocal result, error_occurred
            try:
                print("🔄 Running analyze_crumbly_texture...")
                result = detector.analyze_crumbly_texture(
                    test_image, test_mask, None, 1.0, debug=False
                )
                print("✅ analyze_crumbly_texture completed")
            except Exception as e:
                error_occurred = e
                print(f"❌ Error in analyze_crumbly_texture: {e}")
        
        # Run with 10 second timeout
        thread = threading.Thread(target=analysis_worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout=10)
        
        if thread.is_alive():
            print("❌ CrumblyDetector.analyze_crumbly_texture() HANGS after 10 seconds")
            return False
        elif error_occurred:
            print(f"❌ CrumblyDetector error: {error_occurred}")
            return False
        elif result:
            print(f"✅ CrumblyDetector test PASSED!")
            print(f"   Classification: {result.get('classification', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Crumbly Score: {result.get('crumbly_score', 0):.3f}")
            return True
        else:
            print("❌ CrumblyDetector returned no result")
            return False
            
    except Exception as e:
        print(f"❌ CrumblyDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_crumbly_detector_real_image():
    """Test with a real image"""
    print("\n🖼️ Testing CrumblyDetector with real image...")
    
    # Look for a test image
    image_paths = [
        "C:/Users/risto/Nextcloud/KalvoTech/Operations/Python/fiber_analysis_project/sample_images/Crumble_test_score/crumbly/14a_001.jpg"
    ]
    
    test_image_path = None
    for path in image_paths:
        if Path(path).exists():
            test_image_path = path
            break
    
    if not test_image_path:
        print("⚠️ No test image found, skipping real image test")
        return True
    
    try:
        # Load real image
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"❌ Could not load image: {test_image_path}")
            return False
        
        print(f"✅ Loaded real image: {image.shape}")
        
        # Create basic mask
        mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        detector = CrumblyDetector(porosity_aware=True)
        
        # Test with timeout
        result = None
        error_occurred = None
        
        def analysis_worker():
            nonlocal result, error_occurred
            try:
                print("🔄 Running analyze_crumbly_texture on real image...")
                result = detector.analyze_crumbly_texture(
                    image, mask, None, 0.4437, debug=False
                )
                print("✅ Real image analysis completed")
            except Exception as e:
                error_occurred = e
                print(f"❌ Error in real image analysis: {e}")
        
        # Run with 30 second timeout
        thread = threading.Thread(target=analysis_worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)
        
        if thread.is_alive():
            print("❌ CrumblyDetector HANGS on real image after 30 seconds")
            return False
        elif error_occurred:
            print(f"❌ Real image analysis error: {error_occurred}")
            import traceback
            traceback.print_exc()
            return False
        elif result:
            print(f"✅ Real image analysis PASSED!")
            print(f"   Classification: {result.get('classification', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            return True
        else:
            print("❌ Real image analysis returned no result")
            return False
            
    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 CrumblyDetector Isolation Test")
    print("=" * 50)
    
    # Test 1: Simple synthetic data
    test1_passed = test_crumbly_detector_simple()
    
    # Test 2: Real image (if simple test passes)
    test2_passed = True
    if test1_passed:
        test2_passed = test_crumbly_detector_real_image()
    
    print("\n📊 Test Results:")
    print(f"   Simple test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Real image test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 CrumblyDetector is working correctly!")
        print("   The hanging issue might be in the threading or evaluation system.")
    else:
        print("\n⚠️ CrumblyDetector has issues that need to be fixed.")
        print("   Consider using a simplified version or debugging the detector.")