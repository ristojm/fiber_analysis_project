"""
SEM Fiber Analysis System - Enhanced Fiber Type Detection Module
UPDATED: Added comprehensive oval fitting for fiber diameter measurements
"""

import cv2
import numpy as np
from skimage import filters, morphology, measure, feature
from scipy import ndimage, spatial
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

class FiberTypeDetector:
    """
    Enhanced fiber type detector with comprehensive oval fitting for diameter measurements.
    """
    
    def __init__(self, 
                 min_fiber_ratio: float = 0.001,
                 max_fiber_ratio: float = 0.8,
                 lumen_area_threshold: float = 0.02,
                 circularity_threshold: float = 0.1,
                 confidence_threshold: float = 0.6):
        """
        Initialize fiber type detector with oval fitting capabilities.
        """
        self.min_fiber_ratio = min_fiber_ratio
        self.max_fiber_ratio = max_fiber_ratio
        self.lumen_area_threshold = lumen_area_threshold
        self.circularity_threshold = circularity_threshold
        self.confidence_threshold = confidence_threshold
        
    def calculate_adaptive_thresholds(self, image: np.ndarray) -> Dict:
        """Calculate adaptive thresholds based on image characteristics."""
        height, width = image.shape[:2]
        total_pixels = height * width
        
        min_fiber_area = max(500, int(total_pixels * self.min_fiber_ratio))
        max_fiber_area = int(total_pixels * self.max_fiber_ratio)
        kernel_size = max(3, min(15, int(np.sqrt(total_pixels) / 200)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        min_lumen_area = max(50, int(min_fiber_area * 0.02))
        
        return {
            'min_fiber_area': min_fiber_area,
            'max_fiber_area': max_fiber_area,
            'kernel_size': kernel_size,
            'min_lumen_area': min_lumen_area,
            'image_total_pixels': total_pixels,
            'image_diagonal': np.sqrt(height**2 + width**2)
        }
        
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Adaptive preprocessing based on image characteristics."""
        blur_size = max(3, min(9, int(np.sqrt(image.shape[0] * image.shape[1]) / 300)))
        if blur_size % 2 == 0:
            blur_size += 1
            
        blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        
        clip_limit = 3.0
        tile_size = max(4, min(16, int(min(image.shape) / 100)))
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def segment_fibers(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Enhanced fiber segmentation with multiple fallback methods."""
        thresholds = self.calculate_adaptive_thresholds(image)
        
        # Try multiple segmentation methods
        methods = [
            self._try_inverted_otsu,
            self._try_bright_region_segmentation,
            self._try_adaptive_bright_segmentation,
            self._try_edge_with_filling
        ]
        
        for method in methods:
            fiber_mask, fiber_properties = method(image, thresholds)
            if len(fiber_properties) > 0 and np.sum(fiber_mask > 0) > thresholds['min_fiber_area']:
                return fiber_mask, fiber_properties
        
        # Emergency fallback
        return self._create_emergency_mask(image, thresholds)
    
    def _try_inverted_otsu(self, image: np.ndarray, thresholds: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Inverted Otsu for bright fibers on dark background."""
        try:
            threshold_val, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, binary = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)
            return self._process_binary_mask_enhanced(binary, thresholds)
        except:
            return np.zeros_like(image, dtype=np.uint8), []
    
    def _try_bright_region_segmentation(self, image: np.ndarray, thresholds: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Segment bright regions using percentile thresholds."""
        try:
            best_mask = None
            best_properties = []
            best_area = 0
            
            for percentile in [60, 70, 50, 80, 40]:
                threshold_val = np.percentile(image, percentile)
                _, binary = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)
                
                fiber_mask, fiber_properties = self._process_binary_mask_enhanced(binary, thresholds)
                total_area = np.sum(fiber_mask > 0)
                
                if len(fiber_properties) > 0 and total_area > best_area:
                    best_area = total_area
                    best_mask = fiber_mask
                    best_properties = fiber_properties
            
            return best_mask if best_mask is not None else np.zeros_like(image, dtype=np.uint8), best_properties
        except:
            return np.zeros_like(image, dtype=np.uint8), []
    
    def _try_adaptive_bright_segmentation(self, image: np.ndarray, thresholds: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Adaptive thresholding for bright regions."""
        try:
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 15, -2)
            return self._process_binary_mask_enhanced(binary, thresholds)
        except:
            return np.zeros_like(image, dtype=np.uint8), []
    
    def _try_edge_with_filling(self, image: np.ndarray, thresholds: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Edge detection with hole filling for fiber structures."""
        try:
            edges = cv2.Canny(image, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            binary = np.zeros_like(image, dtype=np.uint8)
            
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    cv2.fillPoly(binary, [contour], 255)
            
            return self._process_binary_mask_enhanced(binary, thresholds)
        except:
            return np.zeros_like(image, dtype=np.uint8), []
    
    def _process_binary_mask_enhanced(self, binary: np.ndarray, thresholds: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Enhanced binary mask processing with oval fitting."""
        kernel_size = max(5, thresholds['kernel_size'])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small_kernel, iterations=1)
        
        # Fill holes completely
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_binary = np.zeros_like(binary)
        
        for contour in contours:
            cv2.fillPoly(filled_binary, [contour], 255)
        
        binary = filled_binary
        
        # Find final contours and analyze
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fiber_properties = []
        fiber_mask = np.zeros_like(binary, dtype=np.uint8)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            min_area = max(1000, thresholds['min_fiber_area'] * 0.1)
            if area < min_area or area > thresholds['max_fiber_area']:
                continue
            
            # Calculate comprehensive geometric properties including oval fitting
            props = self._calculate_comprehensive_properties(contour, area, thresholds)
            
            if self._is_valid_sem_fiber_shape(props, thresholds):
                props['contour_id'] = i
                props['contour'] = contour
                props['thresholds'] = thresholds
                fiber_properties.append(props)
                cv2.fillPoly(fiber_mask, [contour], 255)
        
        return fiber_mask, fiber_properties
    
    def _calculate_comprehensive_properties(self, contour: np.ndarray, area: float, thresholds: Dict) -> Dict:
        """
        Calculate comprehensive geometric properties including oval fitting and diameter measurements.
        """
        try:
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            # Basic shape descriptors
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
            extent = area / (w * h) if w * h > 0 else 0
            
            try:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
            except:
                solidity = 0.8
            
            # ENHANCED: Oval/Ellipse fitting for diameter measurements
            oval_properties = self._fit_oval_to_contour(contour, area)
            
            # Combine all properties
            properties = {
                'area': area,
                'perimeter': perimeter,
                'centroid': (cx, cy),
                'radius': radius,
                'bounding_rect': (x, y, w, h),
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity,
                # NEW: Oval fitting properties
                **oval_properties
            }
            
            return properties
            
        except Exception as e:
            # Return safe defaults on error
            return {
                'area': area,
                'perimeter': 0,
                'centroid': (0, 0),
                'radius': 0,
                'bounding_rect': (0, 0, 0, 0),
                'circularity': 0.5,
                'aspect_ratio': 1.0,
                'extent': 0.5,
                'solidity': 0.8,
                # Default oval properties
                'oval_fitted': False,
                'oval_center': (0, 0),
                'oval_axes': (0, 0),
                'oval_angle': 0,
                'oval_major_diameter': 0,
                'oval_minor_diameter': 0,
                'oval_mean_diameter': 0,
                'oval_area': 0,
                'oval_eccentricity': 0,
                'oval_fit_quality': 0.0
            }
    
    def _fit_oval_to_contour(self, contour: np.ndarray, contour_area: float) -> Dict:
        """
        Fit an oval (ellipse) to the fiber contour and extract diameter measurements.
        
        Returns:
            Dictionary with oval fitting results and diameter measurements
        """
        try:
            # Require minimum number of points for ellipse fitting
            if len(contour) < 5:
                return self._get_default_oval_properties()
            
            # Fit ellipse to contour
            ellipse = cv2.fitEllipse(contour)
            
            # Extract ellipse parameters
            (center_x, center_y), (width, height), angle = ellipse
            
            # Calculate major and minor axes (convert from width/height to radii, then to diameters)
            major_axis = max(width, height)
            minor_axis = min(width, height)
            
            major_diameter = major_axis
            minor_diameter = minor_axis
            mean_diameter = (major_diameter + minor_diameter) / 2
            
            # Calculate ellipse area
            ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)
            
            # Calculate eccentricity
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0
            
            # Calculate fit quality (how well the ellipse matches the contour)
            fit_quality = self._calculate_ellipse_fit_quality(contour, ellipse, contour_area, ellipse_area)
            
            return {
                'oval_fitted': True,
                'oval_center': (center_x, center_y),
                'oval_axes': (major_axis, minor_axis),
                'oval_angle': angle,
                'oval_major_diameter': major_diameter,
                'oval_minor_diameter': minor_diameter,
                'oval_mean_diameter': mean_diameter,
                'oval_area': ellipse_area,
                'oval_eccentricity': eccentricity,
                'oval_fit_quality': fit_quality,
                'ellipse_params': ellipse  # Store full ellipse parameters for visualization
            }
            
        except Exception as e:
            return self._get_default_oval_properties()
    
    def _get_default_oval_properties(self) -> Dict:
        """Return default oval properties when fitting fails."""
        return {
            'oval_fitted': False,
            'oval_center': (0, 0),
            'oval_axes': (0, 0),
            'oval_angle': 0,
            'oval_major_diameter': 0,
            'oval_minor_diameter': 0,
            'oval_mean_diameter': 0,
            'oval_area': 0,
            'oval_eccentricity': 0,
            'oval_fit_quality': 0.0,
            'ellipse_params': None
        }
    
    def _calculate_ellipse_fit_quality(self, contour: np.ndarray, ellipse: Tuple, 
                                     contour_area: float, ellipse_area: float) -> float:
        """
        Calculate how well the fitted ellipse matches the original contour.
        
        Returns:
            Quality score from 0.0 (poor fit) to 1.0 (perfect fit)
        """
        try:
            # Method 1: Area similarity
            if ellipse_area > 0:
                area_similarity = min(contour_area, ellipse_area) / max(contour_area, ellipse_area)
            else:
                area_similarity = 0.0
            
            # Method 2: Point distance analysis
            (center_x, center_y), (width, height), angle = ellipse
            
            # Convert contour points to array
            contour_points = contour.reshape(-1, 2).astype(np.float32)
            
            # Calculate distances from contour points to ellipse boundary
            distances = []
            for point in contour_points[::5]:  # Sample every 5th point for efficiency
                dist = self._point_to_ellipse_distance(point, ellipse)
                distances.append(dist)
            
            if distances:
                mean_distance = np.mean(distances)
                # Normalize by ellipse size
                max_axis = max(width, height)
                normalized_distance = mean_distance / (max_axis / 2) if max_axis > 0 else 1.0
                distance_quality = max(0.0, 1.0 - normalized_distance)
            else:
                distance_quality = 0.0
            
            # Combine quality measures
            overall_quality = (area_similarity * 0.6 + distance_quality * 0.4)
            
            return float(np.clip(overall_quality, 0.0, 1.0))
            
        except Exception as e:
            return 0.0
    
    def _point_to_ellipse_distance(self, point: np.ndarray, ellipse: Tuple) -> float:
        """
        Calculate approximate distance from point to ellipse boundary.
        """
        try:
            (center_x, center_y), (width, height), angle = ellipse
            
            # Translate point to ellipse center
            px, py = point[0] - center_x, point[1] - center_y
            
            # Rotate point to align with ellipse axes
            angle_rad = np.radians(-angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            
            # Calculate distance to ellipse boundary
            a, b = width / 2, height / 2
            
            if a > 0 and b > 0:
                # Normalized coordinates
                nx, ny = rx / a, ry / b
                
                # Distance from center in normalized space
                dist_from_center = np.sqrt(nx**2 + ny**2)
                
                # Distance to boundary (1.0 in normalized space)
                if dist_from_center > 0:
                    boundary_distance = abs(dist_from_center - 1.0) * min(a, b)
                else:
                    boundary_distance = min(a, b)
            else:
                boundary_distance = float('inf')
            
            return boundary_distance
            
        except Exception as e:
            return float('inf')
    
    def _is_valid_sem_fiber_shape(self, props: Dict, thresholds: Dict) -> bool:
        """Enhanced shape validation including oval fit quality."""
        # Basic validations
        area_ratio = props['area'] / thresholds['image_total_pixels']
        if area_ratio < 0.00001:
            return False
        
        if props['circularity'] < 0.01:
            return False
        
        if props['aspect_ratio'] > 10.0:
            return False
        
        if props['solidity'] < 0.2:
            return False
        
        # NEW: Additional validation using oval fit quality
        if props.get('oval_fitted', False):
            oval_quality = props.get('oval_fit_quality', 0.0)
            # If oval fitting succeeded but quality is very poor, be more cautious
            if oval_quality < 0.1:
                # Require better basic shape metrics for poor oval fits
                return props['circularity'] > 0.05 and props['solidity'] > 0.4
        
        return True
    
    def _create_emergency_mask(self, image: np.ndarray, thresholds: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Create emergency fallback mask when all segmentation methods fail."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        radius = min(width, height) // 3
        
        emergency_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(emergency_mask, center, radius, 255, -1)
        
        emergency_contour = np.array([
            [center[0] - radius, center[1] - radius],
            [center[0] + radius, center[1] - radius],
            [center[0] + radius, center[1] + radius],
            [center[0] - radius, center[1] + radius]
        ]).reshape(-1, 1, 2)
        
        # Create comprehensive properties for emergency mask (in pixels)
        emergency_props = {
            'area': np.pi * radius * radius,
            'perimeter': 2 * np.pi * radius,
            'centroid': center,
            'radius': radius,
            'bounding_rect': (center[0] - radius, center[1] - radius, 2*radius, 2*radius),
            'circularity': 1.0,
            'aspect_ratio': 1.0,
            'extent': 0.785,
            'solidity': 1.0,
            'contour_id': 0,
            'contour': emergency_contour,
            'thresholds': thresholds,
            'emergency_mask': True,
            # Emergency oval properties (perfect circle in pixels)
            'oval_fitted': True,
            'oval_center': center,
            'oval_axes': (2*radius, 2*radius),
            'oval_angle': 0,
            'oval_major_diameter': 2*radius,
            'oval_minor_diameter': 2*radius,
            'oval_mean_diameter': 2*radius,
            'oval_area': np.pi * radius * radius,
            'oval_eccentricity': 0.0,
            'oval_fit_quality': 1.0,
            'ellipse_params': (center, (2*radius, 2*radius), 0)
        }
        
        return emergency_mask, [emergency_props]
    
    def detect_lumen(self, image: np.ndarray, fiber_contour: np.ndarray) -> Tuple[bool, Dict]:
        """Improved lumen detection with oval fitting for lumen measurements."""
        try:
            thresholds = self.calculate_adaptive_thresholds(image)
            
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [fiber_contour], 255)
            
            fiber_region = cv2.bitwise_and(image, image, mask=mask)
            fiber_pixels = fiber_region[mask > 0]
            
            if len(fiber_pixels) == 0:
                return False, {}
            
            # Adaptive threshold
            threshold = 50
            mean_intensity = fiber_pixels.mean()
            if mean_intensity > 120:
                threshold = 60
            elif mean_intensity < 60:
                threshold = 40
            
            _, lumen_binary = cv2.threshold(fiber_region, threshold, 255, cv2.THRESH_BINARY_INV)
            lumen_binary = cv2.bitwise_and(lumen_binary, lumen_binary, mask=mask)
            
            kernel_size = max(3, min(9, thresholds['kernel_size']))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel)
            lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_OPEN, kernel)
            
            lumen_contours, _ = cv2.findContours(lumen_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not lumen_contours:
                return False, {}
            
            largest_lumen = max(lumen_contours, key=cv2.contourArea)
            lumen_area = cv2.contourArea(largest_lumen)
            fiber_area = cv2.contourArea(fiber_contour)
            
            # ENHANCED: Calculate comprehensive lumen properties including oval fitting
            lumen_props = self._calculate_comprehensive_properties(largest_lumen, lumen_area, thresholds)
            lumen_props['area_ratio'] = lumen_area / fiber_area if fiber_area > 0 else 0
            lumen_props['threshold_used'] = threshold
            lumen_props['contour'] = largest_lumen
            
            # Validate lumen
            is_lumen = self._validate_lumen_enhanced(lumen_props, fiber_contour, thresholds)
            
            return is_lumen, lumen_props
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _validate_lumen_enhanced(self, lumen_props: Dict, fiber_contour: np.ndarray, thresholds: Dict) -> bool:
        """Enhanced lumen validation including oval fit quality."""
        try:
            # Basic area ratio checks
            min_area_ratio = self.lumen_area_threshold
            max_area_ratio = 0.6
            
            fiber_area = cv2.contourArea(fiber_contour)
            if fiber_area < thresholds['min_fiber_area'] * 2:
                min_area_ratio = 0.01
            
            if not (min_area_ratio <= lumen_props['area_ratio'] <= max_area_ratio):
                return False
            
            if lumen_props['circularity'] < 0.05:
                return False
            
            # Enhanced validation using oval fitting
            if lumen_props.get('oval_fitted', False):
                oval_quality = lumen_props.get('oval_fit_quality', 0.0)
                oval_eccentricity = lumen_props.get('oval_eccentricity', 1.0)
                
                # Good oval fit with reasonable eccentricity suggests a real lumen
                if oval_quality > 0.5 and oval_eccentricity < 0.8:
                    # Relax other requirements for well-fitted ovals
                    return True
            
            # Centrality check
            fiber_moments = cv2.moments(fiber_contour)
            if fiber_moments['m00'] > 0:
                fiber_cx = fiber_moments['m10'] / fiber_moments['m00']
                fiber_cy = fiber_moments['m01'] / fiber_moments['m00']
                
                lumen_cx, lumen_cy = lumen_props['centroid']
                distance = np.sqrt((fiber_cx - lumen_cx)**2 + (fiber_cy - lumen_cy)**2)
                
                fiber_radius = np.sqrt(fiber_area / np.pi)
                max_distance_ratio = 0.7
                
                if distance > max_distance_ratio * fiber_radius:
                    return False
            
            return True
            
        except Exception as e:
            return False
    
    def classify_fiber_type(self, image: np.ndarray, scale_factor: float = 1.0) -> Tuple[str, float, Dict]:
        """
        Enhanced fiber type classification with comprehensive oval fitting analysis.
        
        Args:
            image: Input grayscale image
            scale_factor: Micrometers per pixel conversion factor for real-world measurements
        """
        try:
            thresholds = self.calculate_adaptive_thresholds(image)
            preprocessed = self.preprocess_for_detection(image)
            fiber_mask, fiber_properties = self.segment_fibers(preprocessed)
            
            if fiber_mask is None or np.sum(fiber_mask > 0) < 1000:
                fiber_mask, fiber_properties = self._create_emergency_mask(image, thresholds)
            
            analysis_results = []
            
            for fiber_props in fiber_properties:
                contour = fiber_props['contour']
                has_lumen, lumen_props = self.detect_lumen(image, contour)
                
                # FIXED: Convert oval measurements to micrometers for fiber properties
                if fiber_props.get('oval_fitted', False):
                    fiber_props = self._convert_oval_measurements_to_um(fiber_props, scale_factor)
                
                # FIXED: Convert oval measurements to micrometers for lumen properties
                if has_lumen and lumen_props.get('oval_fitted', False):
                    lumen_props = self._convert_oval_measurements_to_um(lumen_props, scale_factor)
                
                confidence = self._calculate_type_confidence(fiber_props, has_lumen, lumen_props, thresholds)
                
                analysis_results.append({
                    'fiber_properties': fiber_props,
                    'has_lumen': has_lumen,
                    'lumen_properties': lumen_props,
                    'confidence': confidence,
                    'type': 'hollow_fiber' if has_lumen else 'filament'
                })
            
            # Classification based on largest fiber
            if analysis_results:
                largest_fiber_result = max(analysis_results, key=lambda x: x['fiber_properties']['area'])
                final_type = largest_fiber_result['type']
                final_confidence = largest_fiber_result['confidence']
            else:
                final_type = 'unknown'
                final_confidence = 0.3
            
            hollow_count = sum(1 for result in analysis_results if result['has_lumen'])
            total_count = len(analysis_results)
            
            # ENHANCED: Analysis data with comprehensive oval fitting information
            analysis_data = {
                'total_fibers': total_count,
                'hollow_fibers': hollow_count,
                'filaments': total_count - hollow_count,
                'fiber_mask': fiber_mask,
                'individual_results': analysis_results,
                'preprocessed_image': preprocessed,
                'thresholds': thresholds,
                'classification_method': 'adaptive_largest_fiber_with_oval_fitting',
                'mask_area_pixels': int(np.sum(fiber_mask > 0)),
                'mask_coverage_percent': float(np.sum(fiber_mask > 0) / fiber_mask.size * 100),
                'scale_factor_used': scale_factor,  # NEW: Record scale factor used
                # NEW: Aggregate oval fitting statistics (now in micrometers)
                'oval_fitting_summary': self._calculate_oval_fitting_summary(analysis_results, scale_factor)
            }
            
            return final_type, final_confidence, analysis_data
            
        except Exception as e:
            emergency_mask, emergency_props = self._create_emergency_mask(image, {'min_fiber_area': 1000, 'image_total_pixels': image.size})
            
            return "unknown", 0.1, {
                'total_fibers': 1,
                'hollow_fibers': 0,
                'filaments': 1,
                'fiber_mask': emergency_mask,
                'individual_results': [{
                    'fiber_properties': emergency_props[0],
                    'has_lumen': False,
                    'lumen_properties': {},
                    'confidence': 0.1,
                    'type': 'filament'
                }],
                'preprocessed_image': image,
                'thresholds': {'min_fiber_area': 1000},
                'classification_method': 'emergency_fallback',
                'error': str(e),
                'mask_area_pixels': int(np.sum(emergency_mask > 0)),
                'mask_coverage_percent': float(np.sum(emergency_mask > 0) / emergency_mask.size * 100),
                'oval_fitting_summary': {'total_fitted': 1, 'avg_fit_quality': 1.0, 'avg_mean_diameter': emergency_props[0]['oval_mean_diameter']}
            }
    
    def _convert_oval_measurements_to_um(self, props: Dict, scale_factor: float) -> Dict:
        """
        Convert oval measurements from pixels to micrometers using the scale factor.
        
        Args:
            props: Properties dictionary containing oval measurements in pixels
            scale_factor: Micrometers per pixel conversion factor
            
        Returns:
            Updated properties dictionary with measurements in micrometers
        """
        if not props.get('oval_fitted', False) or scale_factor <= 0:
            return props
        
        # Create a copy to avoid modifying the original
        converted_props = props.copy()
        
        # Convert diameter measurements
        converted_props['oval_major_diameter_um'] = props.get('oval_major_diameter', 0) * scale_factor
        converted_props['oval_minor_diameter_um'] = props.get('oval_minor_diameter', 0) * scale_factor
        converted_props['oval_mean_diameter_um'] = props.get('oval_mean_diameter', 0) * scale_factor
        
        # Convert area measurement
        converted_props['oval_area_um2'] = props.get('oval_area', 0) * (scale_factor ** 2)
        
        # Keep original pixel measurements for compatibility
        converted_props['oval_major_diameter_px'] = props.get('oval_major_diameter', 0)
        converted_props['oval_minor_diameter_px'] = props.get('oval_minor_diameter', 0)
        converted_props['oval_mean_diameter_px'] = props.get('oval_mean_diameter', 0)
        converted_props['oval_area_px'] = props.get('oval_area', 0)
        
        # Other measurements remain unchanged (unitless or already in correct units)
        # oval_eccentricity, oval_fit_quality, oval_angle, oval_center, oval_axes, ellipse_params
        
        return converted_props
    
    def _calculate_oval_fitting_summary(self, analysis_results: List[Dict], scale_factor: float = 1.0) -> Dict:
        """
        Calculate summary statistics for oval fitting across all detected fibers.
        
        Args:
            analysis_results: List of analysis results for each fiber
            scale_factor: Micrometers per pixel conversion factor
        """
        fitted_fibers = []
        fitted_lumens = []
        
        for result in analysis_results:
            fiber_props = result['fiber_properties']
            if fiber_props.get('oval_fitted', False):
                fitted_fibers.append(fiber_props)
            
            if result['has_lumen']:
                lumen_props = result['lumen_properties']
                if lumen_props.get('oval_fitted', False):
                    fitted_lumens.append(lumen_props)
        
        summary = {
            'total_fibers_analyzed': len(analysis_results),
            'fibers_successfully_fitted': len(fitted_fibers),
            'lumens_successfully_fitted': len(fitted_lumens),
            'fiber_fit_success_rate': len(fitted_fibers) / len(analysis_results) if analysis_results else 0,
            'scale_factor_used': scale_factor,
        }
        
        # Fiber oval statistics (now in micrometers)
        if fitted_fibers:
            fiber_qualities = [f['oval_fit_quality'] for f in fitted_fibers]
            # Use micrometers measurements if available, fall back to pixels * scale_factor
            fiber_mean_diameters_um = []
            fiber_major_diameters_um = []
            fiber_minor_diameters_um = []
            
            for f in fitted_fibers:
                if 'oval_mean_diameter_um' in f:
                    fiber_mean_diameters_um.append(f['oval_mean_diameter_um'])
                    fiber_major_diameters_um.append(f['oval_major_diameter_um'])
                    fiber_minor_diameters_um.append(f['oval_minor_diameter_um'])
                else:
                    # Fallback: convert from pixels
                    fiber_mean_diameters_um.append(f.get('oval_mean_diameter', 0) * scale_factor)
                    fiber_major_diameters_um.append(f.get('oval_major_diameter', 0) * scale_factor)
                    fiber_minor_diameters_um.append(f.get('oval_minor_diameter', 0) * scale_factor)
            
            fiber_eccentricities = [f['oval_eccentricity'] for f in fitted_fibers]
            
            summary.update({
                'fiber_avg_fit_quality': np.mean(fiber_qualities),
                'fiber_avg_mean_diameter_um': np.mean(fiber_mean_diameters_um),  # NOW IN MICROMETERS
                'fiber_avg_major_diameter_um': np.mean(fiber_major_diameters_um),  # NOW IN MICROMETERS
                'fiber_avg_minor_diameter_um': np.mean(fiber_minor_diameters_um),  # NOW IN MICROMETERS
                'fiber_avg_eccentricity': np.mean(fiber_eccentricities),
                'fiber_diameter_std_um': np.std(fiber_mean_diameters_um),  # NOW IN MICROMETERS
                'fiber_min_diameter_um': np.min(fiber_mean_diameters_um) if fiber_mean_diameters_um else 0,
                'fiber_max_diameter_um': np.max(fiber_mean_diameters_um) if fiber_mean_diameters_um else 0,
            })
        
        # Lumen oval statistics (now in micrometers)
        if fitted_lumens:
            lumen_qualities = [l['oval_fit_quality'] for l in fitted_lumens]
            lumen_mean_diameters_um = []
            
            for l in fitted_lumens:
                if 'oval_mean_diameter_um' in l:
                    lumen_mean_diameters_um.append(l['oval_mean_diameter_um'])
                else:
                    # Fallback: convert from pixels
                    lumen_mean_diameters_um.append(l.get('oval_mean_diameter', 0) * scale_factor)
            
            lumen_eccentricities = [l['oval_eccentricity'] for l in fitted_lumens]
            
            summary.update({
                'lumen_avg_fit_quality': np.mean(lumen_qualities),
                'lumen_avg_mean_diameter_um': np.mean(lumen_mean_diameters_um),  # NOW IN MICROMETERS
                'lumen_avg_eccentricity': np.mean(lumen_eccentricities),
                'lumen_diameter_std_um': np.std(lumen_mean_diameters_um),  # NOW IN MICROMETERS
            })
        
        return summary
    
    def _calculate_type_confidence(self, fiber_props: Dict, has_lumen: bool, lumen_props: Dict, thresholds: Dict) -> float:
        """Enhanced confidence calculation including oval fit quality."""
        try:
            base_confidence = 0.5
            
            # Shape quality boost
            shape_quality = min(1.0, fiber_props['circularity'] / 0.8)
            base_confidence += 0.15 * shape_quality
            
            # Size appropriateness boost
            area_ratio = fiber_props['area'] / thresholds['image_total_pixels']
            size_score = min(1.0, area_ratio / 0.1) if area_ratio < 0.1 else 1.0
            base_confidence += 0.05 * size_score
            
            # NEW: Oval fitting quality boost
            if fiber_props.get('oval_fitted', False):
                oval_quality = fiber_props.get('oval_fit_quality', 0.0)
                base_confidence += 0.1 * oval_quality
            
            if has_lumen and lumen_props:
                # Lumen quality boost
                lumen_quality = min(1.0, lumen_props.get('area_ratio', 0) / 0.2)
                base_confidence += 0.15 * lumen_quality
                
                # NEW: Lumen oval fitting boost
                if lumen_props.get('oval_fitted', False):
                    lumen_oval_quality = lumen_props.get('oval_fit_quality', 0.0)
                    base_confidence += 0.05 * lumen_oval_quality
            else:
                # Solid fiber characteristics boost
                solidity_factor = min(1.0, fiber_props['solidity'] / 0.9)
                base_confidence += 0.15 * solidity_factor
            
            return min(1.0, max(0.1, base_confidence))
            
        except:
            return 0.5


# Convenience function
def detect_fiber_type(image: np.ndarray, scale_factor: float = 1.0, **kwargs) -> Tuple[str, float]:
    """
    Convenience function for fiber type detection with oval fitting and scale conversion.
    
    Args:
        image: Input grayscale image
        scale_factor: Micrometers per pixel conversion factor
        **kwargs: Additional parameters for FiberTypeDetector
    """
    try:
        detector = FiberTypeDetector(**kwargs)
        fiber_type, confidence, _ = detector.classify_fiber_type(image, scale_factor)
        return fiber_type, confidence
    except Exception as e:
        return "unknown", 0.1

def visualize_fiber_type_analysis(image: np.ndarray, analysis_data: Dict, figsize: Tuple[int, int] = (20, 12)):
    """
    Enhanced visualization including oval fitting results.
    """
    try:
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # Preprocessed image
        preprocessed = analysis_data.get('preprocessed_image', image)
        axes[1].imshow(preprocessed, cmap='gray')
        axes[1].set_title('Preprocessed', fontweight='bold')
        axes[1].axis('off')
        
        # Fiber mask
        fiber_mask = analysis_data.get('fiber_mask', np.zeros_like(image))
        axes[2].imshow(fiber_mask, cmap='gray')
        mask_area = analysis_data.get('mask_area_pixels', np.sum(fiber_mask > 0))
        coverage = analysis_data.get('mask_coverage_percent', mask_area / fiber_mask.size * 100)
        axes[2].set_title(f'Detected Fibers\n({mask_area:,} pixels, {coverage:.1f}%)', fontweight='bold')
        axes[2].axis('off')
        
        # NEW: Oval fitting visualization
        overlay_ovals = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
        
        individual_results = analysis_data.get('individual_results', [])
        for result in individual_results:
            fiber_props = result.get('fiber_properties', {})
            
            # Draw fiber oval
            if fiber_props.get('oval_fitted', False) and 'ellipse_params' in fiber_props:
                ellipse_params = fiber_props['ellipse_params']
                if ellipse_params is not None:
                    cv2.ellipse(overlay_ovals, ellipse_params, (0, 255, 0), 3)  # Green for fiber oval
            
            # Draw lumen oval
            if result.get('has_lumen', False):
                lumen_props = result.get('lumen_properties', {})
                if lumen_props.get('oval_fitted', False) and 'ellipse_params' in lumen_props:
                    ellipse_params = lumen_props['ellipse_params']
                    if ellipse_params is not None:
                        cv2.ellipse(overlay_ovals, ellipse_params, (255, 255, 0), 2)  # Yellow for lumen oval
        
        axes[3].imshow(overlay_ovals)
        axes[3].set_title('Fitted Ovals\n(Green=Fiber, Yellow=Lumen)', fontweight='bold')
        axes[3].axis('off')
        
        # Classification results with oval info
        overlay_classification = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
        
        for result in individual_results:
            fiber_props = result.get('fiber_properties', {})
            contour = fiber_props.get('contour')
            
            if contour is not None:
                color = (0, 255, 0) if result.get('has_lumen', False) else (255, 0, 0)
                cv2.drawContours(overlay_classification, [contour], -1, color, 2)
                
                # Add diameter text if oval fitted
                if fiber_props.get('oval_fitted', False):
                    center = fiber_props.get('oval_center', (0, 0))
                    mean_diameter = fiber_props.get('oval_mean_diameter', 0)
                    cv2.putText(overlay_classification, f'{mean_diameter:.0f}px', 
                               (int(center[0]), int(center[1])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        axes[4].imshow(overlay_classification)
        axes[4].set_title('Classification + Diameters\n(Green=Hollow, Red=Solid)', fontweight='bold')
        axes[4].axis('off')
        
        # Oval fitting summary
        oval_summary = analysis_data.get('oval_fitting_summary', {})
        thresholds = analysis_data.get('thresholds', {})
        method = analysis_data.get('classification_method', 'adaptive')
        
        summary_text = f"Method: {method}\n"
        summary_text += f"Total Fibers: {analysis_data.get('total_fibers', 0)}\n"
        summary_text += f"Hollow: {analysis_data.get('hollow_fibers', 0)}\n"
        summary_text += f"Filaments: {analysis_data.get('filaments', 0)}\n\n"
        
        summary_text += f"Oval Fitting Results:\n"
        summary_text += f"Success Rate: {oval_summary.get('fiber_fit_success_rate', 0):.1%}\n"
        summary_text += f"Avg Fit Quality: {oval_summary.get('fiber_avg_fit_quality', 0):.2f}\n"
        summary_text += f"Avg Diameter: {oval_summary.get('fiber_avg_mean_diameter', 0):.1f}px\n"
        summary_text += f"Diameter Std: {oval_summary.get('fiber_diameter_std', 0):.1f}px\n"
        
        if oval_summary.get('lumens_successfully_fitted', 0) > 0:
            summary_text += f"\nLumen Fitting:\n"
            summary_text += f"Lumens Fitted: {oval_summary.get('lumens_successfully_fitted', 0)}\n"
            summary_text += f"Avg Lumen Diameter: {oval_summary.get('lumen_avg_mean_diameter', 0):.1f}px\n"
        
        if 'error' in analysis_data:
            summary_text += f"\nError: {analysis_data['error'][:30]}..."
        
        axes[5].text(0.05, 0.95, summary_text, transform=axes[5].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
        axes[5].set_title('Analysis Summary', fontweight='bold')
        axes[5].axis('off')
        
        # Diameter distribution plot
        all_diameters = []
        for result in individual_results:
            fiber_props = result.get('fiber_properties', {})
            if fiber_props.get('oval_fitted', False):
                all_diameters.append(fiber_props.get('oval_mean_diameter', 0))
        
        if all_diameters:
            axes[6].hist(all_diameters, bins=min(10, len(all_diameters)), alpha=0.7, color='blue', edgecolor='black')
            axes[6].set_title('Fiber Diameter Distribution', fontweight='bold')
            axes[6].set_xlabel('Diameter (pixels)')
            axes[6].set_ylabel('Count')
        else:
            axes[6].text(0.5, 0.5, 'No diameter\ndata available', 
                        ha='center', va='center', transform=axes[6].transAxes)
            axes[6].set_title('Diameter Distribution', fontweight='bold')
            axes[6].axis('off')
        
        # Individual confidence scores
        confidences = [result.get('confidence', 0) for result in individual_results]
        if confidences:
            bars = axes[7].bar(range(len(confidences)), confidences, alpha=0.7, color='green')
            axes[7].set_title('Individual Confidence Scores', fontweight='bold')
            axes[7].set_xlabel('Fiber ID')
            axes[7].set_ylabel('Confidence')
            axes[7].set_ylim(0, 1)
            
            # Add value labels on bars
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                axes[7].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{conf:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            axes[7].text(0.5, 0.5, 'No confidence\ndata available', 
                        ha='center', va='center', transform=axes[7].transAxes)
            axes[7].set_title('Confidence Scores', fontweight='bold')
            axes[7].axis('off')
        
        plt.suptitle(f"Enhanced Fiber Analysis with Oval Fitting", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Visualization error: {e}")
        # Show minimal visualization on error
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        fiber_mask = analysis_data.get('fiber_mask', np.zeros_like(image))
        plt.imshow(fiber_mask, cmap='gray')
        plt.title('Fiber Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        oval_summary = analysis_data.get('oval_fitting_summary', {})
        summary_text = f"Oval Fitting Summary:\nSuccess Rate: {oval_summary.get('fiber_fit_success_rate', 0):.1%}\nAvg Diameter: {oval_summary.get('fiber_avg_mean_diameter', 0):.1f}px"
        plt.text(0.1, 0.5, summary_text, fontsize=12, transform=plt.gca().transAxes)
        plt.title('Oval Fitting Results')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()