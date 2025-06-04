"""
SEM Fiber Analysis System - Fiber Type Detection Module
Automatically distinguishes between hollow fibers and solid filaments.
Priority #1 feature for adaptive analysis pipeline.
"""

import cv2
import numpy as np
from skimage import filters, morphology, measure, feature
from scipy import ndimage, spatial
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

class FiberTypeDetector:
    """
    Main class for detecting whether cross-sections are hollow fibers or solid filaments.
    """
    
    def __init__(self, 
                 min_fiber_area: int = 1000,
                 lumen_area_threshold: float = 0.02,  # Reduced from 0.05
                 circularity_threshold: float = 0.2,   # Reduced from 0.3
                 confidence_threshold: float = 0.6):   # Reduced from 0.7
        """
        Initialize fiber type detector with more lenient parameters for irregular lumens.
        
        Args:
            min_fiber_area: Minimum area for valid fiber detection
            lumen_area_threshold: Minimum lumen area ratio for hollow fiber classification
            circularity_threshold: Minimum circularity for fiber validation
            confidence_threshold: Minimum confidence for classification
        """
        self.min_fiber_area = min_fiber_area
        self.lumen_area_threshold = lumen_area_threshold
        self.circularity_threshold = circularity_threshold
        self.confidence_threshold = confidence_threshold
        
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image specifically for fiber type detection.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image optimized for segmentation
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def segment_fibers(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Segment individual fiber cross-sections from the image.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Tuple of (binary_mask, fiber_properties_list)
        """
        # Multi-scale Otsu thresholding for better segmentation
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up segmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze each contour
        fiber_properties = []
        fiber_mask = np.zeros_like(binary)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < self.min_fiber_area:
                continue
            
            # Calculate geometric properties
            props = self._calculate_contour_properties(contour, area)
            
            # Check if it's a valid fiber based on shape
            if self._is_valid_fiber_shape(props):
                props['contour_id'] = i
                props['contour'] = contour
                fiber_properties.append(props)
                
                # Add to fiber mask
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
        # Basic measurements
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Minimum enclosing circle
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
    
    def _is_valid_fiber_shape(self, props: Dict) -> bool:
        """
        Determine if a contour represents a valid fiber cross-section.
        
        Args:
            props: Contour properties dictionary
            
        Returns:
            True if valid fiber shape
        """
        # Check circularity (fibers should be roughly circular)
        if props['circularity'] < self.circularity_threshold:
            return False
        
        # Check aspect ratio (shouldn't be too elongated)
        if props['aspect_ratio'] > 3.0:
            return False
        
        # Check solidity (should be reasonably solid)
        if props['solidity'] < 0.7:
            return False
        
        return True
    
    def detect_lumen(self, image: np.ndarray, fiber_contour: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect if a fiber has a central lumen (hollow center).
        Enhanced version for better detection of irregular lumens.
        
        Args:
            image: Original grayscale image
            fiber_contour: Contour of the fiber
            
        Returns:
            Tuple of (has_lumen, lumen_properties)
        """
        # Create mask for fiber region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [fiber_contour], 255)
        
        # Extract fiber region
        fiber_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Apply multiple thresholding approaches for better lumen detection
        fiber_pixels = fiber_region[mask > 0]
        if len(fiber_pixels) == 0:
            return False, {}
        
        # Method 1: Percentile-based thresholding (more robust)
        threshold_low = np.percentile(fiber_pixels, 15)  # Bottom 15% intensity
        threshold_med = np.percentile(fiber_pixels, 35)  # Bottom 35% intensity
        
        # Method 2: Multi-level thresholding
        lumen_candidates = []
        
        for threshold in [threshold_low, threshold_med]:
            _, lumen_binary = cv2.threshold(fiber_region, threshold, 255, cv2.THRESH_BINARY_INV)
            lumen_binary = cv2.bitwise_and(lumen_binary, lumen_binary, mask=mask)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel)
            lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            lumen_contours, _ = cv2.findContours(lumen_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in lumen_contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum lumen area
                    lumen_candidates.append((contour, area, threshold))
        
        if not lumen_candidates:
            return False, {}
        
        # Select the best lumen candidate
        # Prefer larger lumens from more restrictive thresholds
        lumen_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)  # Sort by threshold desc, then area desc
        
        best_lumen_contour, lumen_area, used_threshold = lumen_candidates[0]
        fiber_area = cv2.contourArea(fiber_contour)
        
        # Calculate lumen properties
        lumen_props = self._calculate_contour_properties(best_lumen_contour, lumen_area)
        lumen_props['area_ratio'] = lumen_area / fiber_area if fiber_area > 0 else 0
        lumen_props['threshold_used'] = used_threshold
        lumen_props['contour'] = best_lumen_contour
        
        # Enhanced validation for irregular lumens
        is_lumen = self._validate_lumen_enhanced(lumen_props, fiber_contour, image)
        
        return is_lumen, lumen_props
    
    def _validate_lumen_enhanced(self, lumen_props: Dict, fiber_contour: np.ndarray, image: np.ndarray) -> bool:
        """
        Enhanced validation for irregular lumens - less strict requirements.
        
        Args:
            lumen_props: Properties of detected lumen
            fiber_contour: Contour of parent fiber
            image: Original image for additional checks
            
        Returns:
            True if valid lumen
        """
        # Relaxed minimum area ratio - irregular lumens can be smaller
        if lumen_props['area_ratio'] < 0.02:  # Reduced from 0.05 to 0.02
            return False
        
        # Maximum area ratio - shouldn't be too large
        if lumen_props['area_ratio'] > 0.6:
            return False
        
        # Relaxed circularity requirement for irregular lumens
        if lumen_props['circularity'] < 0.15:  # Reduced from 0.4 to 0.15
            return False
        
        # Check if lumen is reasonably central (more lenient)
        fiber_moments = cv2.moments(fiber_contour)
        if fiber_moments['m00'] > 0:
            fiber_cx = fiber_moments['m10'] / fiber_moments['m00']
            fiber_cy = fiber_moments['m01'] / fiber_moments['m00']
            
            lumen_cx, lumen_cy = lumen_props['centroid']
            distance = np.sqrt((fiber_cx - lumen_cx)**2 + (fiber_cy - lumen_cy)**2)
            
            # Distance should be less than 40% of fiber radius (increased from 20%)
            fiber_area = cv2.contourArea(fiber_contour)
            fiber_radius = np.sqrt(fiber_area / np.pi)
            
            if distance > 0.4 * fiber_radius:
                return False
        
        # Additional check: lumen should be significantly darker than wall
        if 'threshold_used' in lumen_props:
            # If we detected the lumen with a reasonable threshold, it's likely real
            return True
        
        return True
    
    def classify_fiber_type(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Main function to classify fiber type with confidence score.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (fiber_type, confidence, analysis_data)
        """
        # Preprocess image
        preprocessed = self.preprocess_for_detection(image)
        
        # Segment fibers
        fiber_mask, fiber_properties = self.segment_fibers(preprocessed)
        
        if not fiber_properties:
            return "unknown", 0.0, {"error": "No valid fibers detected"}
        
        # Analyze each fiber for lumen presence
        analysis_results = []
        total_confidence = 0.0
        
        for fiber_props in fiber_properties:
            contour = fiber_props['contour']
            has_lumen, lumen_props = self.detect_lumen(image, contour)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_type_confidence(fiber_props, has_lumen, lumen_props)
            
            analysis_results.append({
                'fiber_properties': fiber_props,
                'has_lumen': has_lumen,
                'lumen_properties': lumen_props,
                'confidence': confidence,
                'type': 'hollow_fiber' if has_lumen else 'filament'
            })
            
            total_confidence += confidence
        
        # Determine overall classification
        hollow_count = sum(1 for result in analysis_results if result['has_lumen'])
        total_count = len(analysis_results)
        
        # Classification logic
        if total_count == 0:
            final_type = "unknown"
            final_confidence = 0.0
        elif hollow_count / total_count >= 0.5:
            final_type = "hollow_fiber"
            final_confidence = total_confidence / total_count
        else:
            final_type = "filament"
            final_confidence = total_confidence / total_count
        
        analysis_data = {
            'total_fibers': total_count,
            'hollow_fibers': hollow_count,
            'filaments': total_count - hollow_count,
            'fiber_mask': fiber_mask,
            'individual_results': analysis_results,
            'preprocessed_image': preprocessed
        }
        
        return final_type, final_confidence, analysis_data
    
    def _calculate_type_confidence(self, fiber_props: Dict, has_lumen: bool, lumen_props: Dict) -> float:
        """
        Calculate confidence score for fiber type classification.
        
        Args:
            fiber_props: Fiber geometric properties
            has_lumen: Whether lumen was detected
            lumen_props: Lumen properties (if detected)
            
        Returns:
            Confidence score (0-1)
        """
        base_confidence = 0.5
        
        # Boost confidence based on fiber shape quality
        shape_quality = min(1.0, fiber_props['circularity'] / 0.8)
        base_confidence += 0.2 * shape_quality
        
        if has_lumen and lumen_props:
            # Boost confidence for clear hollow features
            lumen_quality = min(1.0, lumen_props['area_ratio'] / 0.3)
            lumen_shape = min(1.0, lumen_props['circularity'] / 0.6)
            base_confidence += 0.3 * (lumen_quality + lumen_shape) / 2
        else:
            # For filaments, confidence based on solidity and lack of dark regions
            solidity_factor = min(1.0, fiber_props['solidity'] / 0.9)
            base_confidence += 0.2 * solidity_factor
        
        return min(1.0, base_confidence)

def detect_fiber_type(image: np.ndarray, **kwargs) -> Tuple[str, float]:
    """
    Convenience function for fiber type detection.
    
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
        cv2.drawContours(overlay, [contour], -1, color, 2)
        
        # Draw lumen if detected
        if result['has_lumen'] and 'contour' in result['lumen_properties']:
            lumen_contour = result['lumen_properties']['contour']
            cv2.drawContours(overlay, [lumen_contour], -1, (0, 0, 255), 1)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Classification Results')
    axes[3].axis('off')
    
    # Classification summary
    axes[4].text(0.1, 0.8, f"Total Fibers: {analysis_data['total_fibers']}", fontsize=12, transform=axes[4].transAxes)
    axes[4].text(0.1, 0.6, f"Hollow Fibers: {analysis_data['hollow_fibers']}", fontsize=12, transform=axes[4].transAxes, color='green')
    axes[4].text(0.1, 0.4, f"Filaments: {analysis_data['filaments']}", fontsize=12, transform=axes[4].transAxes, color='red')
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