"""
SEM Fiber Analysis System - Adaptive Fiber Type Detection Module
FIXED: Ensures proper fiber mask creation and storage
UPDATED: Dynamic thresholds that adapt to image resolution and content
"""

import cv2
import numpy as np
from skimage import filters, morphology, measure, feature
from scipy import ndimage, spatial
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

class FiberTypeDetector:
    """
    Adaptive fiber type detector that automatically adjusts to image characteristics.
    Uses resolution-aware thresholds for robust performance across different imaging conditions.
    FIXED: Now properly creates and stores comprehensive fiber masks.
    """
    
    def __init__(self, 
                 min_fiber_ratio: float = 0.001,      # Min fiber area as fraction of image
                 max_fiber_ratio: float = 0.8,        # Max fiber area as fraction of image
                 lumen_area_threshold: float = 0.02,  # Minimum 2% lumen area
                 circularity_threshold: float = 0.2,   # Relaxed circularity
                 confidence_threshold: float = 0.6):   # Lower confidence threshold
        """
        Initialize adaptive detector with ratio-based parameters.
        
        Args:
            min_fiber_ratio: Minimum fiber area as fraction of total image pixels
            max_fiber_ratio: Maximum fiber area as fraction of total image pixels
            lumen_area_threshold: Minimum lumen area ratio for hollow fiber classification
            circularity_threshold: Minimum circularity for fiber validation
            confidence_threshold: Minimum confidence for classification
        """
        self.min_fiber_ratio = min_fiber_ratio
        self.max_fiber_ratio = max_fiber_ratio
        self.lumen_area_threshold = lumen_area_threshold
        self.circularity_threshold = circularity_threshold
        self.confidence_threshold = confidence_threshold
        
    def calculate_adaptive_thresholds(self, image: np.ndarray) -> Dict:
        """
        Calculate adaptive thresholds based on image characteristics.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of adaptive thresholds
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
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd number
        
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
        """
        Adaptive preprocessing based on image characteristics.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image optimized for segmentation
        """
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
    
    def segment_fibers(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Adaptive fiber segmentation using calculated thresholds.
        FIXED: Ensures proper fiber mask creation
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Tuple of (binary_mask, fiber_properties_list)
        """
        # Calculate adaptive thresholds
        thresholds = self.calculate_adaptive_thresholds(image)
        
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
        fiber_mask = np.zeros_like(binary, dtype=np.uint8)  # FIXED: Proper initialization
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Adaptive area filtering
            if area < thresholds['min_fiber_area'] or area > thresholds['max_fiber_area']:
                continue
            
            # Calculate geometric properties
            props = self._calculate_contour_properties(contour, area)
            
            # Adaptive shape validation
            if self._is_valid_fiber_shape(props, thresholds):
                props['contour_id'] = i
                props['contour'] = contour
                props['thresholds'] = thresholds  # Store for later use
                fiber_properties.append(props)
                
                # FIXED: Add this fiber to the mask immediately
                cv2.fillPoly(fiber_mask, [contour], 255)
        
        return fiber_mask, fiber_properties
    
    def _calculate_contour_properties(self, contour: np.ndarray, area: float) -> Dict:
        """
        Calculate geometric properties of a contour.
        
        Args:
            contour: OpenCV contour
            area: Contour area
            
        Returns:
            Dictionary of geometric properties
        """
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Fit ellipse (if enough points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
        else:
            ellipse = None
            ellipse_area = 0
        
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
            'ellipse': ellipse,
            'ellipse_area': ellipse_area,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity
        }
    
    def _is_valid_fiber_shape(self, props: Dict, thresholds: Dict) -> bool:
        """
        Adaptive shape validation based on image characteristics.
        
        Args:
            props: Contour properties dictionary
            thresholds: Adaptive thresholds dictionary
            
        Returns:
            True if valid fiber shape
        """
        # Adaptive circularity threshold based on fiber size
        min_circularity = self.circularity_threshold
        
        # Smaller fibers can be less circular (more likely to be artifacts)
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
    
    def detect_lumen(self, image: np.ndarray, fiber_contour: np.ndarray) -> Tuple[bool, Dict]:
        """
        Adaptive lumen detection with resolution-aware parameters.
        
        Args:
            image: Original grayscale image
            fiber_contour: Contour of the fiber
            
        Returns:
            Tuple of (has_lumen, lumen_properties)
        """
        # Get adaptive thresholds
        thresholds = self.calculate_adaptive_thresholds(image)
        
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
        is_lumen = self._validate_lumen_enhanced(lumen_props, fiber_contour, thresholds)
        
        return is_lumen, lumen_props
    
    def _validate_lumen_enhanced(self, lumen_props: Dict, fiber_contour: np.ndarray, thresholds: Dict) -> bool:
        """
        Enhanced adaptive lumen validation.
        
        Args:
            lumen_props: Properties of detected lumen
            fiber_contour: Contour of parent fiber
            thresholds: Adaptive thresholds dictionary
            
        Returns:
            True if valid lumen
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
        
        # Adaptive circularity check (very lenient for irregular lumens)
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
    
    def classify_fiber_type(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Main adaptive classification function.
        FIXED: Now properly creates and stores comprehensive fiber mask
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (fiber_type, confidence, analysis_data)
        """
        # Calculate adaptive thresholds
        thresholds = self.calculate_adaptive_thresholds(image)
        
        # Preprocess with adaptive parameters
        preprocessed = self.preprocess_for_detection(image)
        
        # Segment fibers with adaptive thresholds
        fiber_mask, fiber_properties = self.segment_fibers(preprocessed)
        
        if not fiber_properties:
            return "unknown", 0.0, {
                "error": "No valid fibers detected", 
                "thresholds": thresholds,
                "total_fibers": 0,
                "hollow_fibers": 0,
                "filaments": 0,
                "fiber_mask": np.zeros_like(image, dtype=np.uint8),  # FIXED: Always include mask
                "individual_results": [],
                "preprocessed_image": preprocessed,
                "classification_method": "adaptive_largest_fiber"
            }
        
        # Analyze each fiber
        analysis_results = []
        
        for fiber_props in fiber_properties:
            contour = fiber_props['contour']
            has_lumen, lumen_props = self.detect_lumen(image, contour)
            
            # Calculate confidence
            confidence = self._calculate_type_confidence(fiber_props, has_lumen, lumen_props, thresholds)
            
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
        
        # FIXED: Create comprehensive fiber mask from ALL detected fibers
        comprehensive_fiber_mask = self._create_comprehensive_fiber_mask(image, analysis_results)
        
        analysis_data = {
            'total_fibers': total_count,
            'hollow_fibers': hollow_count,
            'filaments': total_count - hollow_count,
            'fiber_mask': comprehensive_fiber_mask,  # FIXED: Proper comprehensive mask
            'individual_results': analysis_results,
            'preprocessed_image': preprocessed,
            'thresholds': thresholds,
            'classification_method': 'adaptive_largest_fiber'
        }
        
        return final_type, final_confidence, analysis_data
    
    def _create_comprehensive_fiber_mask(self, image: np.ndarray, analysis_results: List[Dict]) -> np.ndarray:
        """
        FIXED: Create a comprehensive fiber mask from all detected fibers
        
        Args:
            image: Original image for shape reference
            analysis_results: List of individual fiber analysis results
            
        Returns:
            Comprehensive binary mask with all fibers (uint8, 0 or 255)
        """
        # Create empty mask with proper dtype
        comprehensive_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Add each fiber to the comprehensive mask
        for result in analysis_results:
            fiber_props = result.get('fiber_properties', {})
            contour = fiber_props.get('contour')
            
            if contour is not None:
                try:
                    # Fill this fiber's area in the comprehensive mask
                    cv2.fillPoly(comprehensive_mask, [contour], 255)
                except Exception as e:
                    # Skip this contour if there's an error
                    print(f"Warning: Could not add fiber contour to mask: {e}")
                    continue
        
        return comprehensive_mask
    
    def _calculate_type_confidence(self, fiber_props: Dict, has_lumen: bool, lumen_props: Dict, thresholds: Dict) -> float:
        """
        Adaptive confidence calculation based on multiple factors.
        
        Args:
            fiber_props: Fiber geometric properties
            has_lumen: Whether lumen was detected
            lumen_props: Lumen properties (if detected)
            thresholds: Adaptive thresholds dictionary
            
        Returns:
            Confidence score (0-1)
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

# Convenience function
def detect_fiber_type(image: np.ndarray, **kwargs) -> Tuple[str, float]:
    """
    Convenience function for fiber type detection with adaptive algorithm.
    
    Args:
        image: Input grayscale image
        **kwargs: Additional parameters for FiberTypeDetector
        
    Returns:
        Tuple of (fiber_type, confidence)
    """
    detector = FiberTypeDetector(**kwargs)
    fiber_type, confidence, _ = detector.classify_fiber_type(image)
    return fiber_type, confidence

def visualize_fiber_type_analysis(image: np.ndarray, analysis_data: Dict, figsize: Tuple[int, int] = (15, 10)):
    """
    Visualize the fiber type detection results.
    
    Args:
        image: Original image
        analysis_data: Analysis results from classify_fiber_type
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Preprocessed image
    axes[1].imshow(analysis_data['preprocessed_image'], cmap='gray')
    axes[1].set_title('Preprocessed')
    axes[1].axis('off')
    
    # Fiber mask
    axes[2].imshow(analysis_data['fiber_mask'], cmap='gray')
    axes[2].set_title('Detected Fibers')
    axes[2].axis('off')
    
    # Overlay with fiber boundaries and lumen detection
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for result in analysis_data['individual_results']:
        contour = result['fiber_properties']['contour']
        color = (0, 255, 0) if result['has_lumen'] else (255, 0, 0)  # Green for hollow, red for solid
        cv2.drawContours(overlay, [contour], -1, color, 3)
        
        # Draw lumen if detected
        if result['has_lumen'] and 'contour' in result['lumen_properties']:
            lumen_contour = result['lumen_properties']['contour']
            cv2.drawContours(overlay, [lumen_contour], -1, (0, 255, 255), 2)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Classification Results\n(Green=Hollow, Red=Solid, Cyan=Lumen)')
    axes[3].axis('off')
    
    # Classification summary with adaptive info
    thresholds = analysis_data.get('thresholds', {})
    method = analysis_data.get('classification_method', 'adaptive')
    
    summary_text = f"Method: {method}\n"
    summary_text += f"Total Fibers: {analysis_data['total_fibers']}\n"
    summary_text += f"Hollow: {analysis_data['hollow_fibers']}\n"
    summary_text += f"Filaments: {analysis_data['filaments']}\n\n"
    summary_text += f"Adaptive Thresholds:\n"
    summary_text += f"Min area: {thresholds.get('min_fiber_area', 0):,}\n"
    summary_text += f"Max area: {thresholds.get('max_fiber_area', 0):,}\n"
    summary_text += f"Kernel: {thresholds.get('kernel_size', 0)}\n\n"
    summary_text += f"Fiber Mask:\n"
    summary_text += f"Pixels: {np.sum(analysis_data['fiber_mask'] > 0):,}\n"
    summary_text += f"Coverage: {np.sum(analysis_data['fiber_mask'] > 0) / analysis_data['fiber_mask'].size * 100:.1f}%"
    
    axes[4].text(0.05, 0.95, summary_text, transform=axes[4].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[4].set_title('Summary')
    axes[4].axis('off')
    
    # Individual confidence scores
    confidences = [result['confidence'] for result in analysis_data['individual_results']]
    if confidences:
        axes[5].bar(range(len(confidences)), confidences)
        axes[5].set_title('Individual Confidence Scores')
        axes[5].set_xlabel('Fiber ID')
        axes[5].set_ylabel('Confidence')
        axes[5].set_ylim(0, 1)
    else:
        axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()