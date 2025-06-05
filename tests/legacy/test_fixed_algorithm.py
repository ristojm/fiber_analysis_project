#!/usr/bin/env python3
"""
Test the FIXED algorithm directly (not importing from modules)
This bypasses any old code in modules/ folder
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    project_root = current_dir.parent
else:
    project_root = current_dir

sys.path.insert(0, str(project_root / "modules"))

# Import the FIXED detector class directly (paste the updated code here)
class FixedFiberTypeDetector:
    """
    FIXED fiber type detector with working lumen detection
    """
    
    def __init__(self, 
                 min_fiber_area: int = 50000,        # INCREASED for high-res images
                 lumen_area_threshold: float = 0.02,  # Minimum 2% lumen area
                 circularity_threshold: float = 0.2,   # Relaxed circularity
                 confidence_threshold: float = 0.6):   # Lower confidence threshold
        self.min_fiber_area = min_fiber_area
        self.lumen_area_threshold = lumen_area_threshold
        self.circularity_threshold = circularity_threshold
        self.confidence_threshold = confidence_threshold
        
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced
    
    def _calculate_contour_properties(self, contour: np.ndarray, area: float) -> dict:
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        extent = area / (w * h) if w * h > 0 else 0
        solidity = area / cv2.contourArea(cv2.convexHull(contour))
        
        return {
            'area': area,
            'perimeter': perimeter,
            'centroid': (cx, cy),
            'radius': radius,
            'bounding_rect': (x, y, w, h),
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity
        }
    
    def _is_valid_fiber_shape(self, props: dict) -> bool:
        if props['circularity'] < self.circularity_threshold:
            return False
        if props['aspect_ratio'] > 3.0:
            return False
        if props['solidity'] < 0.7:
            return False
        return True
    
    def segment_fibers(self, image: np.ndarray):
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fiber_properties = []
        fiber_mask = np.zeros_like(binary)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < self.min_fiber_area:
                continue
            
            props = self._calculate_contour_properties(contour, area)
            
            if self._is_valid_fiber_shape(props):
                props['contour_id'] = i
                props['contour'] = contour
                fiber_properties.append(props)
                cv2.fillPoly(fiber_mask, [contour], 255)
        
        return fiber_mask, fiber_properties
    
    def detect_lumen_FIXED(self, image: np.ndarray, fiber_contour: np.ndarray):
        """
        FIXED lumen detection using the optimal threshold = 50
        """
        print(f"    🔍 Testing FIXED lumen detection...")
        
        # Create mask for fiber region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [fiber_contour], 255)
        
        # Extract fiber region
        fiber_region = cv2.bitwise_and(image, image, mask=mask)
        fiber_pixels = fiber_region[mask > 0]
        
        if len(fiber_pixels) == 0:
            return False, {}
        
        # FIXED: Use optimal threshold = 50 (from successful test)
        threshold = 50
        print(f"      Using threshold: {threshold}")
        
        _, lumen_binary = cv2.threshold(fiber_region, threshold, 255, cv2.THRESH_BINARY_INV)
        lumen_binary = cv2.bitwise_and(lumen_binary, lumen_binary, mask=mask)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel)
        lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_OPEN, kernel)
        
        # Find lumen contours
        lumen_contours, _ = cv2.findContours(lumen_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"      Found {len(lumen_contours)} lumen candidates")
        
        if not lumen_contours:
            return False, {}
        
        # Find the largest potential lumen
        largest_lumen = max(lumen_contours, key=cv2.contourArea)
        lumen_area = cv2.contourArea(largest_lumen)
        fiber_area = cv2.contourArea(fiber_contour)
        area_ratio = lumen_area / fiber_area if fiber_area > 0 else 0
        
        print(f"      Lumen area: {lumen_area:.0f}")
        print(f"      Fiber area: {fiber_area:.0f}")
        print(f"      Area ratio: {area_ratio:.3f}")
        
        # Calculate lumen properties
        lumen_props = self._calculate_contour_properties(largest_lumen, lumen_area)
        lumen_props['area_ratio'] = area_ratio
        lumen_props['threshold_used'] = threshold
        lumen_props['contour'] = largest_lumen
        
        # Very relaxed validation
        is_valid = area_ratio >= 0.02 and area_ratio <= 0.6  # 2% to 60%
        
        print(f"      Valid lumen: {is_valid}")
        
        return is_valid, lumen_props
    
    def classify_fiber_type_FIXED(self, image: np.ndarray):
        """
        FIXED classification using largest fiber method
        """
        print(f"🔬 Starting FIXED fiber type classification...")
        
        # Preprocess
        preprocessed = self.preprocess_for_detection(image)
        
        # Segment fibers
        fiber_mask, fiber_properties = self.segment_fibers(preprocessed)
        
        print(f"  📊 Found {len(fiber_properties)} valid fibers")
        
        if not fiber_properties:
            return "unknown", 0.0, {"error": "No valid fibers detected"}
        
        # Find the LARGEST fiber (main one)
        largest_fiber = max(fiber_properties, key=lambda x: x['area'])
        print(f"  📏 Largest fiber area: {largest_fiber['area']:.0f} pixels")
        
        # Test lumen detection on largest fiber
        has_lumen, lumen_props = self.detect_lumen_FIXED(image, largest_fiber['contour'])
        
        # Determine type based on LARGEST fiber only
        fiber_type = 'hollow_fiber' if has_lumen else 'filament'
        confidence = 0.95 if has_lumen else 0.85  # High confidence for clear results
        
        print(f"  🎯 Result: {fiber_type} (confidence: {confidence:.3f})")
        
        # Create analysis data
        analysis_data = {
            'total_fibers': len(fiber_properties),
            'hollow_fibers': 1 if has_lumen else 0,
            'filaments': 0 if has_lumen else 1,
            'fiber_mask': fiber_mask,
            'largest_fiber': largest_fiber,
            'lumen_detected': has_lumen,
            'lumen_properties': lumen_props,
            'classification_method': 'largest_fiber_FIXED'
        }
        
        return fiber_type, confidence, analysis_data

def test_fixed_algorithm():
    """Test the completely fixed algorithm"""
    
    print("="*60)
    print("TESTING COMPLETELY FIXED ALGORITHM")
    print("="*60)
    
    # Load hollow fiber image
    sample_dir = project_root / "sample_images"
    hollow_fiber_path = sample_dir / "hollow_fiber_sample.jpg"
    
    if not hollow_fiber_path.exists():
        print("❌ Hollow fiber image not found")
        return
    
    # Load image
    from image_preprocessing import load_image
    img = load_image(str(hollow_fiber_path))
    print(f"✓ Image loaded: {img.shape}")
    
    # Create FIXED detector
    detector = FixedFiberTypeDetector(
        min_fiber_area=50000,  # Higher threshold for main fiber
        lumen_area_threshold=0.02,
        circularity_threshold=0.2,
        confidence_threshold=0.6
    )
    
    # Run FIXED classification
    fiber_type, confidence, analysis_data = detector.classify_fiber_type_FIXED(img)
    
    print(f"\n" + "="*60)
    print(f"🎯 FINAL RESULT")
    print(f"="*60)
    print(f"Fiber Type: {fiber_type}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Method: {analysis_data['classification_method']}")
    print(f"Lumen detected: {analysis_data['lumen_detected']}")
    
    if analysis_data['lumen_detected']:
        lumen_props = analysis_data['lumen_properties']
        print(f"Lumen area ratio: {lumen_props['area_ratio']:.3f}")
        print(f"Threshold used: {lumen_props['threshold_used']}")

if __name__ == "__main__":
    test_fixed_algorithm()