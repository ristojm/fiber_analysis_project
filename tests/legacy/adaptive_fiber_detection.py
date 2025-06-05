"""
Adaptive Fiber Detection with Dynamic Thresholds
Automatically adjusts parameters based on image resolution and content
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class AdaptiveFiberTypeDetector:
    """
    Adaptive fiber type detector that adjusts thresholds based on image characteristics
    """
    
    def __init__(self, 
                 min_fiber_ratio: float = 0.001,      # Min fiber area as fraction of image
                 max_fiber_ratio: float = 0.8,        # Max fiber area as fraction of image
                 lumen_area_threshold: float = 0.02,  # Minimum 2% lumen area
                 circularity_threshold: float = 0.2,   # Relaxed circularity
                 confidence_threshold: float = 0.6):   # Lower confidence threshold
        """
        Initialize adaptive detector with ratio-based parameters
        """
        self.min_fiber_ratio = min_fiber_ratio
        self.max_fiber_ratio = max_fiber_ratio
        self.lumen_area_threshold = lumen_area_threshold
        self.circularity_threshold = circularity_threshold
        self.confidence_threshold = confidence_threshold
        
    def calculate_adaptive_thresholds(self, image: np.ndarray) -> Dict:
        """
        Calculate adaptive thresholds based on image characteristics
        """
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Adaptive thresholds based on image size
        min_fiber_area = int(total_pixels * self.min_fiber_ratio)
        max_fiber_area = int(total_pixels * self.max_fiber_ratio)
        
        # Ensure reasonable minimums
        min_fiber_area = max(min_fiber_area, 500)   # At least 500 pixels
        
        # Adaptive morphological kernel size based on image resolution
        kernel_size = max(3, min(15, int(np.sqrt(total_pixels) / 200)))
        
        # Adaptive minimum lumen area
        min_lumen_area = max(50, int(min_fiber_area * 0.02))  # At least 2% of min fiber
        
        thresholds = {
            'min_fiber_area': min_fiber_area,
            'max_fiber_area': max_fiber_area,
            'kernel_size': kernel_size,
            'min_lumen_area': min_lumen_area,
            'image_total_pixels': total_pixels,
            'image_diagonal': np.sqrt(height**2 + width**2)
        }
        
        return thresholds
    
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Adaptive preprocessing based on image characteristics"""
        
        # Adaptive blur based on image size
        blur_size = max(3, min(9, int(np.sqrt(image.shape[0] * image.shape[1]) / 300)))
        if blur_size % 2 == 0:
            blur_size += 1
            
        blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        
        # Adaptive CLAHE
        clip_limit = 3.0
        tile_size = max(4, min(16, int(min(image.shape) / 100)))
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def segment_fibers_adaptive(self, image: np.ndarray, thresholds: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """
        Adaptive fiber segmentation using calculated thresholds
        """
        # Multi-scale Otsu thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive morphological operations
        kernel_size = thresholds['kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by adaptive thresholds
        fiber_properties = []
        fiber_mask = np.zeros_like(binary)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Adaptive area filtering
            if area < thresholds['min_fiber_area'] or area > thresholds['max_fiber_area']:
                continue
            
            # Calculate geometric properties
            props = self._calculate_contour_properties(contour, area)
            
            # Adaptive shape validation
            if self._is_valid_fiber_shape_adaptive(props, thresholds):
                props['contour_id'] = i
                props['contour'] = contour
                fiber_properties.append(props)
                cv2.fillPoly(fiber_mask, [contour], 255)
        
        return fiber_mask, fiber_properties
    
    def _calculate_contour_properties(self, contour: np.ndarray, area: float) -> Dict:
        """Calculate geometric properties of a contour"""
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Calculate shape descriptors
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
    
    def _is_valid_fiber_shape_adaptive(self, props: Dict, thresholds: Dict) -> bool:
        """
        Adaptive shape validation based on image characteristics
        """
        # Adaptive circularity threshold based on fiber size
        min_circularity = self.circularity_threshold
        
        # Smaller fibers can be less circular (artifacts)
        area_ratio = props['area'] / thresholds['image_total_pixels']
        if area_ratio < 0.001:  # Very small fibers
            min_circularity = max(0.3, self.circularity_threshold * 1.5)
        
        if props['circularity'] < min_circularity:
            return False
        
        # Adaptive aspect ratio based on size
        max_aspect_ratio = 3.0 if area_ratio > 0.01 else 2.5  # Stricter for larger fibers
        if props['aspect_ratio'] > max_aspect_ratio:
            return False
        
        # Solidity check
        if props['solidity'] < 0.7:
            return False
        
        return True
    
    def detect_lumen_adaptive(self, image: np.ndarray, fiber_contour: np.ndarray, thresholds: Dict) -> Tuple[bool, Dict]:
        """
        Adaptive lumen detection with resolution-aware parameters
        """
        # Create mask for fiber region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [fiber_contour], 255)
        
        # Extract fiber region
        fiber_region = cv2.bitwise_and(image, image, mask=mask)
        fiber_pixels = fiber_region[mask > 0]
        
        if len(fiber_pixels) == 0:
            return False, {}
        
        # Adaptive threshold based on fiber characteristics
        threshold = 50  # Start with our proven threshold
        
        # Adjust threshold based on image characteristics
        mean_intensity = fiber_pixels.mean()
        if mean_intensity > 120:  # Bright image
            threshold = 60
        elif mean_intensity < 60:  # Dark image
            threshold = 40
        
        _, lumen_binary = cv2.threshold(fiber_region, threshold, 255, cv2.THRESH_BINARY_INV)
        lumen_binary = cv2.bitwise_and(lumen_binary, lumen_binary, mask=mask)
        
        # Adaptive morphological operations
        kernel_size = max(3, min(9, thresholds['kernel_size']))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel)
        lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_OPEN, kernel)
        
        # Find lumen contours
        lumen_contours, _ = cv2.findContours(lumen_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not lumen_contours:
            return False, {}
        
        # Find the largest potential lumen
        largest_lumen = max(lumen_contours, key=cv2.contourArea)
        lumen_area = cv2.contourArea(largest_lumen)
        fiber_area = cv2.contourArea(fiber_contour)
        
        # Calculate lumen properties
        lumen_props = self._calculate_contour_properties(largest_lumen, lumen_area)
        lumen_props['area_ratio'] = lumen_area / fiber_area if fiber_area > 0 else 0
        lumen_props['threshold_used'] = threshold
        lumen_props['contour'] = largest_lumen
        
        # Adaptive validation
        is_lumen = self._validate_lumen_adaptive(lumen_props, fiber_contour, thresholds)
        
        return is_lumen, lumen_props
    
    def _validate_lumen_adaptive(self, lumen_props: Dict, fiber_contour: np.ndarray, thresholds: Dict) -> bool:
        """
        Adaptive lumen validation based on fiber and image characteristics
        """
        # Area ratio check (adaptive based on fiber size)
        min_area_ratio = self.lumen_area_threshold
        max_area_ratio = 0.6
        
        # For very small fibers, allow smaller lumen ratios
        fiber_area = cv2.contourArea(fiber_contour)
        if fiber_area < thresholds['min_fiber_area'] * 2:
            min_area_ratio = 0.01  # More lenient for small fibers
        
        if not (min_area_ratio <= lumen_props['area_ratio'] <= max_area_ratio):
            return False
        
        # Adaptive circularity check
        min_circularity = 0.05  # Very lenient for irregular lumens
        if lumen_props['circularity'] < min_circularity:
            return False
        
        # Adaptive centrality check
        fiber_moments = cv2.moments(fiber_contour)
        if fiber_moments['m00'] > 0:
            fiber_cx = fiber_moments['m10'] / fiber_moments['m00']
            fiber_cy = fiber_moments['m01'] / fiber_moments['m00']
            
            lumen_cx, lumen_cy = lumen_props['centroid']
            distance = np.sqrt((fiber_cx - lumen_cx)**2 + (fiber_cy - lumen_cy)**2)
            
            # Adaptive centrality tolerance based on fiber size
            fiber_radius = np.sqrt(fiber_area / np.pi)
            max_distance_ratio = 0.5  # Allow 50% offset
            
            # More lenient for smaller fibers
            if fiber_area < thresholds['min_fiber_area'] * 3:
                max_distance_ratio = 0.7
            
            if distance > max_distance_ratio * fiber_radius:
                return False
        
        return True
    
    def classify_fiber_type_adaptive(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Main adaptive classification function
        """
        # Calculate adaptive thresholds
        thresholds = self.calculate_adaptive_thresholds(image)
        
        # Preprocess with adaptive parameters
        preprocessed = self.preprocess_for_detection(image)
        
        # Segment fibers with adaptive thresholds
        fiber_mask, fiber_properties = self.segment_fibers_adaptive(preprocessed, thresholds)
        
        if not fiber_properties:
            return "unknown", 0.0, {
                "error": "No valid fibers detected", 
                "thresholds": thresholds,
                "total_fibers": 0
            }
        
        # Analyze each fiber
        analysis_results = []
        
        for fiber_props in fiber_properties:
            contour = fiber_props['contour']
            has_lumen, lumen_props = self.detect_lumen_adaptive(image, contour, thresholds)
            
            # Calculate confidence
            confidence = self._calculate_confidence_adaptive(fiber_props, has_lumen, lumen_props, thresholds)
            
            analysis_results.append({
                'fiber_properties': fiber_props,
                'has_lumen': has_lumen,
                'lumen_properties': lumen_props,
                'confidence': confidence,
                'type': 'hollow_fiber' if has_lumen else 'filament'
            })
        
        # Classification based on largest fiber (most reliable)
        largest_fiber_result = max(analysis_results, key=lambda x: x['fiber_properties']['area'])
        
        final_type = largest_fiber_result['type']
        final_confidence = largest_fiber_result['confidence']
        
        # Count totals
        hollow_count = sum(1 for result in analysis_results if result['has_lumen'])
        total_count = len(analysis_results)
        
        analysis_data = {
            'total_fibers': total_count,
            'hollow_fibers': hollow_count,
            'filaments': total_count - hollow_count,
            'fiber_mask': fiber_mask,
            'individual_results': analysis_results,
            'preprocessed_image': preprocessed,
            'thresholds': thresholds,
            'classification_method': 'adaptive_largest_fiber'
        }
        
        return final_type, final_confidence, analysis_data
    
    def _calculate_confidence_adaptive(self, fiber_props: Dict, has_lumen: bool, lumen_props: Dict, thresholds: Dict) -> float:
        """
        Adaptive confidence calculation based on multiple factors
        """
        base_confidence = 0.5
        
        # Boost for good fiber shape
        shape_quality = min(1.0, fiber_props['circularity'] / 0.8)
        base_confidence += 0.2 * shape_quality
        
        # Boost for appropriate size
        area_ratio = fiber_props['area'] / thresholds['image_total_pixels']
        size_score = min(1.0, area_ratio / 0.1) if area_ratio < 0.1 else 1.0
        base_confidence += 0.1 * size_score
        
        if has_lumen and lumen_props:
            # Boost for clear lumen
            lumen_quality = min(1.0, lumen_props['area_ratio'] / 0.2)
            base_confidence += 0.2 * lumen_quality
        else:
            # Boost for solid fiber characteristics
            solidity_factor = min(1.0, fiber_props['solidity'] / 0.9)
            base_confidence += 0.2 * solidity_factor
        
        return min(1.0, base_confidence)

def test_adaptive_detection():
    """Test the adaptive detection on both images"""
    
    print("="*60)
    print("TESTING ADAPTIVE FIBER DETECTION")
    print("="*60)
    
    # Setup paths
    from pathlib import Path
    import sys
    
    current_dir = Path(__file__).parent
    if current_dir.name == 'tests':
        project_root = current_dir.parent
    else:
        project_root = current_dir
        
    modules_dir = project_root / "modules"
    sys.path.insert(0, str(modules_dir))
    
    from image_preprocessing import load_image
    
    # Test both images
    test_images = [
        ("hollow_fiber_sample.jpg", "hollow_fiber"),
        ("solid_filament_sample.jpg", "filament")
    ]
    
    detector = AdaptiveFiberTypeDetector()
    
    for img_name, expected in test_images:
        print(f"\n📷 Testing: {img_name}")
        print("-" * 40)
        
        img = load_image(str(project_root / "sample_images" / img_name))
        
        # Run adaptive classification
        fiber_type, confidence, analysis_data = detector.classify_fiber_type_adaptive(img)
        
        print(f"Image size: {img.shape}")
        print(f"Adaptive thresholds:")
        thresholds = analysis_data['thresholds']
        print(f"  min_fiber_area: {thresholds['min_fiber_area']:,}")
        print(f"  max_fiber_area: {thresholds['max_fiber_area']:,}")
        print(f"  kernel_size: {thresholds['kernel_size']}")
        
        print(f"\nResults:")
        print(f"  Type: {fiber_type}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Total fibers: {analysis_data['total_fibers']}")
        print(f"  Hollow: {analysis_data['hollow_fibers']}")
        print(f"  Filaments: {analysis_data['filaments']}")
        
        # Check correctness
        if fiber_type == expected:
            print(f"  ✅ CORRECT!")
        else:
            print(f"  ❌ Expected: {expected}")
        
        # Show fiber details
        if analysis_data['individual_results']:
            print(f"\nFiber details:")
            for i, result in enumerate(analysis_data['individual_results']):
                props = result['fiber_properties']
                area = props['area']
                has_lumen = result['has_lumen']
                conf = result['confidence']
                print(f"  Fiber {i+1}: {area:,} pixels, lumen={has_lumen}, conf={conf:.3f}")

if __name__ == "__main__":
    test_adaptive_detection()