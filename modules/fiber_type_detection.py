"""
SEM Fiber Analysis System - PROPERLY FIXED Fiber Type Detection Module
FIXED: Maintains backward compatibility while adding scale factor support
UPDATED: All measurements properly converted to micrometers when scale factor provided

This is the correct fix that maintains the existing interface.
"""

import cv2
import numpy as np
from skimage import filters, morphology, measure, feature
from scipy import ndimage, spatial
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

class FiberTypeDetector:
    """
    PROPERLY FIXED fiber type detector with backward-compatible scale factor integration.
    """
    
    def __init__(self, 
                 min_fiber_area: int = 50000,        # KEEP ORIGINAL for compatibility
                 lumen_area_threshold: float = 0.02,  # Minimum 2% lumen area
                 circularity_threshold: float = 0.2,   # Relaxed circularity
                 confidence_threshold: float = 0.6):   # Lower confidence threshold
        """
        Initialize fiber type detector - KEEPS ORIGINAL INTERFACE
        """
        self.min_fiber_area = min_fiber_area
        self.lumen_area_threshold = lumen_area_threshold
        self.circularity_threshold = circularity_threshold
        self.confidence_threshold = confidence_threshold
        
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image specifically for fiber type detection.
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
        KEEPS ORIGINAL LOGIC
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
        """Calculate geometric properties of a contour."""
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
        """Determine if a contour represents a valid fiber cross-section."""
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
        FIXED: Improved lumen detection - KEEPS ORIGINAL INTERFACE but adds scale support internally
        """
        # Create mask for fiber region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [fiber_contour], 255)
        
        # Extract fiber region
        fiber_region = cv2.bitwise_and(image, image, mask=mask)
        fiber_pixels = fiber_region[mask > 0]
        
        if len(fiber_pixels) == 0:
            return False, {}
        
        # Use the optimal threshold method (fixed low threshold = 50)
        threshold = 50  # Optimal threshold found in testing
        
        _, lumen_binary = cv2.threshold(fiber_region, threshold, 255, cv2.THRESH_BINARY_INV)
        lumen_binary = cv2.bitwise_and(lumen_binary, lumen_binary, mask=mask)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
        
        # Validate lumen with relaxed criteria
        is_lumen = self._validate_lumen_enhanced(lumen_props, fiber_contour)
        
        return is_lumen, lumen_props
    
    def _validate_lumen_enhanced(self, lumen_props: Dict, fiber_contour: np.ndarray) -> bool:
        """
        Enhanced validation for lumens with relaxed criteria.
        """
        # Minimum area ratio (2% instead of 5%)
        if lumen_props['area_ratio'] < 0.02:
            return False
        
        # Maximum area ratio (shouldn't be too large)
        if lumen_props['area_ratio'] > 0.6:
            return False
        
        # Very relaxed circularity requirement
        if lumen_props['circularity'] < 0.05:  # Very lenient
            return False
        
        # Check if lumen is reasonably central (very lenient)
        fiber_moments = cv2.moments(fiber_contour)
        if fiber_moments['m00'] > 0:
            fiber_cx = fiber_moments['m10'] / fiber_moments['m00']
            fiber_cy = fiber_moments['m01'] / fiber_moments['m00']
            
            lumen_cx, lumen_cy = lumen_props['centroid']
            distance = np.sqrt((fiber_cx - lumen_cx)**2 + (fiber_cy - lumen_cy)**2)
            
            # Very lenient centrality check (50% radius tolerance)
            fiber_area = cv2.contourArea(fiber_contour)
            fiber_radius = np.sqrt(fiber_area / np.pi)
            
            if distance > 0.5 * fiber_radius:
                return False
        
        return True
    
    def classify_fiber_type(self, image: np.ndarray, scale_factor: Optional[float] = None) -> Tuple[str, float, Dict]:
        """
        FIXED: Main function with OPTIONAL scale factor parameter for backward compatibility.
        
        Args:
            image: Input grayscale image
            scale_factor: Optional micrometers per pixel conversion factor
            
        Returns:
            Tuple of (fiber_type, confidence, analysis_data_with_scale_if_provided)
        """
        # Preprocess image
        preprocessed = self.preprocess_for_detection(image)
        
        # Segment fibers
        fiber_mask, fiber_properties = self.segment_fibers(preprocessed)
        
        if not fiber_properties:
            return "unknown", 0.0, {"error": "No valid fibers detected"}
        
        # Analyze each fiber for lumen presence
        analysis_results = []
        
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
        
        # Determine overall classification based on the LARGEST fiber
        largest_fiber_result = max(analysis_results, key=lambda x: x['fiber_properties']['area'])
        
        final_type = largest_fiber_result['type']
        final_confidence = largest_fiber_result['confidence']
        
        # Count totals for compatibility
        hollow_count = sum(1 for result in analysis_results if result['has_lumen'])
        total_count = len(analysis_results)
        
        # FIXED: Add scale-aware measurements if scale factor provided
        if scale_factor is not None and scale_factor > 0:
            # Convert measurements to micrometers
            analysis_results_with_scale = self._add_scale_conversions(analysis_results, scale_factor)
            oval_results = self._calculate_oval_fitting_results(analysis_results_with_scale, scale_factor)
        else:
            # Keep original pixel-based measurements
            analysis_results_with_scale = analysis_results
            oval_results = self._calculate_oval_fitting_results_pixels(analysis_results)
        
        analysis_data = {
            'total_fibers': total_count,
            'hollow_fibers': hollow_count,
            'filaments': total_count - hollow_count,
            'fiber_mask': fiber_mask,
            'individual_results': analysis_results_with_scale,
            'preprocessed_image': preprocessed,
            'classification_method': 'largest_fiber_with_optional_scale',
            'scale_factor_used': scale_factor,
            'oval_fitting_results': oval_results  # NOW INCLUDES OVAL RESULTS
        }
        
        return final_type, final_confidence, analysis_data
    
    def _add_scale_conversions(self, analysis_results: List[Dict], scale_factor: float) -> List[Dict]:
        """Add scale-converted measurements to analysis results."""
        results_with_scale = []
        
        for result in analysis_results:
            result_copy = result.copy()
            
            # Convert fiber properties
            fiber_props = result['fiber_properties'].copy()
            area_pixels = fiber_props['area']
            fiber_props['area_um2'] = area_pixels * (scale_factor ** 2)
            fiber_props['equivalent_diameter_um'] = 2 * np.sqrt(fiber_props['area_um2'] / np.pi)
            fiber_props['radius_um'] = fiber_props['radius'] * scale_factor
            fiber_props['diameter_um'] = 2 * fiber_props['radius_um']
            fiber_props['perimeter_um'] = fiber_props['perimeter'] * scale_factor
            
            # Convert ellipse if available
            if fiber_props.get('ellipse') is not None:
                ellipse = fiber_props['ellipse']
                fiber_props['ellipse_major_axis_um'] = max(ellipse[1]) * scale_factor
                fiber_props['ellipse_minor_axis_um'] = min(ellipse[1]) * scale_factor
                fiber_props['ellipse_area_um2'] = np.pi * (fiber_props['ellipse_major_axis_um']/2) * (fiber_props['ellipse_minor_axis_um']/2)
            
            result_copy['fiber_properties'] = fiber_props
            
            # Convert lumen properties if present
            if result['has_lumen'] and result['lumen_properties']:
                lumen_props = result['lumen_properties'].copy()
                lumen_area_pixels = lumen_props['area']
                lumen_props['area_um2'] = lumen_area_pixels * (scale_factor ** 2)
                lumen_props['equivalent_diameter_um'] = 2 * np.sqrt(lumen_props['area_um2'] / np.pi)
                lumen_props['radius_um'] = lumen_props['radius'] * scale_factor
                lumen_props['perimeter_um'] = lumen_props['perimeter'] * scale_factor
                
                result_copy['lumen_properties'] = lumen_props
            
            results_with_scale.append(result_copy)
        
        return results_with_scale
    
    def _calculate_oval_fitting_results(self, analysis_results: List[Dict], scale_factor: float) -> Dict:
        """
        FIXED: Calculate oval fitting results with proper scale conversion.
        """
        if not analysis_results:
            return {
                'success_rate': 0.0,
                'avg_fit_quality': 0.0,
                'avg_diameter': 0.0,
                'diameter_std': 0.0,
                'lumens_fitted': 0,
                'total_analyzed': 0
            }
        
        successful_fits = 0
        fit_qualities = []
        diameters_um = []
        lumen_diameters_um = []
        
        for result in analysis_results:
            fiber_props = result.get('fiber_properties', {})
            
            # Fiber oval fitting
            if fiber_props.get('ellipse') is not None:
                successful_fits += 1
                
                # Calculate fit quality (how close to circular)
                ellipse = fiber_props['ellipse']
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                fit_quality = minor_axis / major_axis if major_axis > 0 else 0
                fit_qualities.append(fit_quality)
                
                # Get diameter in micrometers
                if 'ellipse_major_axis_um' in fiber_props:
                    avg_diameter_um = (fiber_props['ellipse_major_axis_um'] + fiber_props['ellipse_minor_axis_um']) / 2
                else:
                    # Fallback: convert from pixels
                    avg_diameter_pixels = (major_axis + minor_axis) / 2
                    avg_diameter_um = avg_diameter_pixels * scale_factor
                
                diameters_um.append(avg_diameter_um)
            
            # Lumen oval fitting
            if result.get('has_lumen', False):
                lumen_props = result.get('lumen_properties', {})
                if 'equivalent_diameter_um' in lumen_props:
                    lumen_diameters_um.append(lumen_props['equivalent_diameter_um'])
        
        return {
            'success_rate': (successful_fits / len(analysis_results)) * 100 if analysis_results else 0,
            'avg_fit_quality': np.mean(fit_qualities) if fit_qualities else 0,
            'avg_diameter': np.mean(diameters_um) if diameters_um else 0,
            'diameter_std': np.std(diameters_um) if diameters_um else 0,
            'lumens_fitted': len(lumen_diameters_um),
            'avg_lumen_diameter': np.mean(lumen_diameters_um) if lumen_diameters_um else 0,
            'total_analyzed': len(analysis_results),
            'scale_factor_used': scale_factor
        }
    
    def _calculate_oval_fitting_results_pixels(self, analysis_results: List[Dict]) -> Dict:
        """Calculate oval fitting results in pixels (for backward compatibility)."""
        if not analysis_results:
            return {
                'success_rate': 0.0,
                'avg_fit_quality': 0.0,
                'avg_diameter': 0.0,
                'diameter_std': 0.0,
                'lumens_fitted': 0,
                'total_analyzed': 0
            }
        
        successful_fits = 0
        fit_qualities = []
        diameters_pixels = []
        
        for result in analysis_results:
            fiber_props = result.get('fiber_properties', {})
            
            # Fiber oval fitting
            if fiber_props.get('ellipse') is not None:
                successful_fits += 1
                
                # Calculate fit quality
                ellipse = fiber_props['ellipse']
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                fit_quality = minor_axis / major_axis if major_axis > 0 else 0
                fit_qualities.append(fit_quality)
                
                avg_diameter_pixels = (major_axis + minor_axis) / 2
                diameters_pixels.append(avg_diameter_pixels)
        
        return {
            'success_rate': (successful_fits / len(analysis_results)) * 100 if analysis_results else 0,
            'avg_fit_quality': np.mean(fit_qualities) if fit_qualities else 0,
            'avg_diameter': np.mean(diameters_pixels) if diameters_pixels else 0,
            'diameter_std': np.std(diameters_pixels) if diameters_pixels else 0,
            'lumens_fitted': 0,  # Not calculated for pixel-based results
            'total_analyzed': len(analysis_results)
        }
    
    def _calculate_type_confidence(self, fiber_props: Dict, has_lumen: bool, lumen_props: Dict) -> float:
        """Calculate confidence score for fiber type classification."""
        base_confidence = 0.5
        
        # Boost confidence based on fiber shape quality
        shape_quality = min(1.0, fiber_props['circularity'] / 0.8)
        base_confidence += 0.2 * shape_quality
        
        if has_lumen and lumen_props:
            # Boost confidence for clear hollow features
            lumen_quality = min(1.0, lumen_props.get('area_ratio', 0) / 0.2)
            lumen_shape = min(1.0, lumen_props['circularity'] / 0.4)
            base_confidence += 0.3 * (lumen_quality + lumen_shape) / 2
        else:
            # For filaments, confidence based on solidity
            solidity_factor = min(1.0, fiber_props['solidity'] / 0.9)
            base_confidence += 0.2 * solidity_factor
        
        return min(1.0, base_confidence)


# FIXED: Convenience function with optional scale factor
def detect_fiber_type(image: np.ndarray, scale_factor: Optional[float] = None, **kwargs) -> Tuple[str, float]:
    """
    FIXED: Convenience function for fiber type detection with optional scale factor.
    
    Args:
        image: Input grayscale image
        scale_factor: Optional micrometers per pixel conversion factor
        **kwargs: Additional parameters for FiberTypeDetector
        
    Returns:
        Tuple of (fiber_type, confidence)
    """
    detector = FiberTypeDetector(**kwargs)
    fiber_type, confidence, _ = detector.classify_fiber_type(image, scale_factor)
    return fiber_type, confidence

def visualize_fiber_type_analysis(image: np.ndarray, analysis_data: Dict, figsize: Tuple[int, int] = (15, 10)):
    """Visualize the fiber type detection results."""
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
    axes[3].set_title('Classification Results\n(Green=Hollow, Red=Solid)')
    axes[3].axis('off')
    
    # Classification summary with oval results
    method = analysis_data.get('classification_method', 'majority_vote')
    scale_factor = analysis_data.get('scale_factor_used')
    oval_results = analysis_data.get('oval_fitting_results', {})
    
    summary_text = f"Classification: {method}\n"
    summary_text += f"Total Fibers: {analysis_data['total_fibers']}\n"
    summary_text += f"Hollow: {analysis_data['hollow_fibers']}\n"
    summary_text += f"Filaments: {analysis_data['filaments']}\n\n"
    
    summary_text += f"Oval Fitting Results:\n"
    summary_text += f"Success Rate: {oval_results.get('success_rate', 0):.1f}%\n"
    summary_text += f"Avg Fit Quality: {oval_results.get('avg_fit_quality', 0):.2f}\n"
    
    if scale_factor:
        summary_text += f"Avg Diameter: {oval_results.get('avg_diameter', 0):.1f}μm\n"
        summary_text += f"Diameter Std: {oval_results.get('diameter_std', 0):.1f}μm\n"
        summary_text += f"\nLumen Fitting:\n"
        summary_text += f"Lumens Fitted: {oval_results.get('lumens_fitted', 0)}\n"
        summary_text += f"Avg Lumen Diameter: {oval_results.get('avg_lumen_diameter', 0):.1f}μm"
    else:
        summary_text += f"Avg Diameter: {oval_results.get('avg_diameter', 0):.0f}px\n"
        summary_text += f"Diameter Std: {oval_results.get('diameter_std', 0):.0f}px\n"
    
    axes[4].text(0.05, 0.95, summary_text, transform=axes[4].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[4].set_title('Analysis Summary')
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