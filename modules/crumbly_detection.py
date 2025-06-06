"""
SEM Fiber Analysis System - Crumbly Texture Detection Module
Analyzes surface texture to distinguish between crumbly and smooth hollow fibers.
"""

import cv2
import numpy as np
from skimage import filters, feature, measure, morphology, segmentation
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import ndimage, spatial
from scipy.stats import entropy
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CrumblyDetector:
    """
    Detects and quantifies crumbly texture in SEM fiber images.
    """
    
    def __init__(self, 
                 lbp_radius: int = 3,
                 lbp_n_points: int = 24,
                 glcm_distances: List[int] = [1, 2, 3],
                 glcm_angles: List[float] = [0, 45, 90, 135],
                 edge_sigma: float = 1.0):
        """
        Initialize crumbly detector with configurable parameters.
        
        Args:
            lbp_radius: Radius for Local Binary Pattern analysis
            lbp_n_points: Number of points for LBP
            glcm_distances: Distances for Gray Level Co-occurrence Matrix
            glcm_angles: Angles for GLCM (in degrees)
            edge_sigma: Sigma for edge detection
        """
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.glcm_distances = glcm_distances
        self.glcm_angles = [np.radians(angle) for angle in glcm_angles]
        self.edge_sigma = edge_sigma
        
        # Thresholds for classification (will be tuned based on samples)
        self.crumbly_thresholds = {
            'edge_irregularity': 0.15,      # Higher = more irregular
            'lbp_uniformity': 0.3,          # Lower = more varied texture
            'surface_roughness': 0.25,      # Higher = rougher surface
            'glcm_contrast': 0.2,           # Higher = more contrast variation
            'boundary_fractal': 1.3         # Higher = more complex boundary
        }
    
    def analyze_crumbly_texture(self, image: np.ndarray, 
                               fiber_mask: np.ndarray,
                               lumen_mask: Optional[np.ndarray] = None,
                               scale_factor: float = 1.0,
                               debug: bool = False) -> Dict:
        """
        Comprehensive crumbly texture analysis.
        
        Args:
            image: Input SEM image
            fiber_mask: Binary mask of fiber region
            lumen_mask: Optional binary mask of lumen region
            scale_factor: Micrometers per pixel for real measurements
            debug: Enable debug visualizations
            
        Returns:
            Dictionary containing all texture metrics and classification
        """
        
        # Create fiber wall mask (exclude lumen if provided)
        if lumen_mask is not None:
            wall_mask = fiber_mask & (~lumen_mask)
        else:
            wall_mask = fiber_mask
        
        # Extract fiber region
        fiber_region = cv2.bitwise_and(image, image, mask=wall_mask.astype(np.uint8))
        
        # Initialize results
        results = {
            'classification': 'unknown',
            'confidence': 0.0,
            'crumbly_score': 0.0,
            'texture_metrics': {},
            'boundary_metrics': {},
            'surface_metrics': {},
            'scale_factor': scale_factor
        }
        
        try:
            # 1. Boundary Analysis
            boundary_metrics = self._analyze_boundary_irregularity(
                fiber_mask, lumen_mask, scale_factor
            )
            results['boundary_metrics'] = boundary_metrics
            
            # 2. Surface Texture Analysis
            surface_metrics = self._analyze_surface_texture(
                fiber_region, wall_mask, scale_factor
            )
            results['surface_metrics'] = surface_metrics
            
            # 3. Local Binary Pattern Analysis
            lbp_metrics = self._analyze_lbp_texture(
                fiber_region, wall_mask
            )
            results['texture_metrics']['lbp'] = lbp_metrics
            
            # 4. Gray Level Co-occurrence Matrix Analysis
            glcm_metrics = self._analyze_glcm_texture(
                fiber_region, wall_mask
            )
            results['texture_metrics']['glcm'] = glcm_metrics
            
            # 5. Edge-based Texture Analysis
            edge_metrics = self._analyze_edge_texture(
                fiber_region, wall_mask, scale_factor
            )
            results['texture_metrics']['edges'] = edge_metrics
            
            # 6. Combine metrics for final classification
            classification_result = self._classify_crumbly_texture(
                boundary_metrics, surface_metrics, lbp_metrics, 
                glcm_metrics, edge_metrics
            )
            
            results.update(classification_result)
            
            # 7. Generate debug visualization if requested
            if debug:
                self._create_debug_visualization(
                    image, fiber_mask, lumen_mask, results
                )
            
        except Exception as e:
            results['error'] = str(e)
            results['classification'] = 'error'
        
        return results
    
    def _analyze_boundary_irregularity(self, fiber_mask: np.ndarray, 
                                     lumen_mask: Optional[np.ndarray],
                                     scale_factor: float) -> Dict:
        """Analyze irregularity of fiber boundaries."""
        
        metrics = {}
        
        # Find fiber boundary contour
        fiber_contours, _ = cv2.findContours(
            fiber_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if fiber_contours:
            # Use largest contour (main fiber boundary)
            fiber_contour = max(fiber_contours, key=cv2.contourArea)
            
            # Calculate boundary irregularity metrics
            metrics['outer_boundary'] = self._calculate_contour_irregularity(
                fiber_contour, scale_factor
            )
        
        # Analyze lumen boundary if available
        if lumen_mask is not None:
            lumen_contours, _ = cv2.findContours(
                lumen_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            
            if lumen_contours:
                lumen_contour = max(lumen_contours, key=cv2.contourArea)
                metrics['lumen_boundary'] = self._calculate_contour_irregularity(
                    lumen_contour, scale_factor
                )
        
        return metrics
    
    def _calculate_contour_irregularity(self, contour: np.ndarray, 
                                      scale_factor: float) -> Dict:
        """Calculate irregularity metrics for a single contour."""
        
        # Basic contour properties
        perimeter = cv2.arcLength(contour, True) * scale_factor
        area = cv2.contourArea(contour) * (scale_factor ** 2)
        
        # Circularity (4π*area/perimeter²) - closer to 1 = more circular
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Convex hull analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) * (scale_factor ** 2)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Fractal dimension approximation (box counting method)
        fractal_dim = self._estimate_fractal_dimension(contour)
        
        # Curvature analysis
        curvature_stats = self._analyze_curvature(contour, scale_factor)
        
        # Roughness index (deviation from smooth ellipse)
        roughness_index = self._calculate_roughness_index(contour)
        
        return {
            'perimeter_um': perimeter,
            'area_um2': area,
            'circularity': circularity,
            'solidity': solidity,
            'fractal_dimension': fractal_dim,
            'roughness_index': roughness_index,
            'curvature_stats': curvature_stats
        }
    
    def _estimate_fractal_dimension(self, contour: np.ndarray) -> float:
        """Estimate fractal dimension using box counting method."""
        
        try:
            # Convert contour to binary image
            x_coords = contour[:, 0, 0]
            y_coords = contour[:, 0, 1]
            
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            
            # Create binary image of contour
            contour_img = np.zeros((height, width), dtype=np.uint8)
            shifted_contour = contour.copy()
            shifted_contour[:, 0, 0] -= min_x
            shifted_contour[:, 0, 1] -= min_y
            
            cv2.drawContours(contour_img, [shifted_contour], -1, 1, 1)
            
            # Box counting for different scales
            scales = [2, 4, 8, 16, 32]
            counts = []
            
            for scale in scales:
                if scale < min(width, height):
                    count = 0
                    for i in range(0, height, scale):
                        for j in range(0, width, scale):
                            box = contour_img[i:i+scale, j:j+scale]
                            if np.any(box):
                                count += 1
                    counts.append(count)
                else:
                    counts.append(1)
            
            # Calculate fractal dimension
            if len(counts) > 2:
                scales_log = np.log([1/s for s in scales[:len(counts)]])
                counts_log = np.log(counts)
                
                # Linear regression
                coeffs = np.polyfit(scales_log, counts_log, 1)
                fractal_dim = coeffs[0]
                
                # Clamp to reasonable range
                return max(1.0, min(2.0, fractal_dim))
            else:
                return 1.0
                
        except:
            return 1.0
    
    def _analyze_curvature(self, contour: np.ndarray, scale_factor: float) -> Dict:
        """Analyze curvature along the contour."""
        
        try:
            # Smooth contour slightly to reduce noise
            epsilon = 0.001 * cv2.arcLength(contour, True)
            smoothed = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(smoothed) < 5:
                smoothed = contour
            
            # Calculate curvature at each point
            curvatures = []
            points = smoothed[:, 0, :]
            
            for i in range(len(points)):
                p1 = points[i-1]
                p2 = points[i]
                p3 = points[(i+1) % len(points)]
                
                # Calculate curvature using three consecutive points
                curvature = self._point_curvature(p1, p2, p3, scale_factor)
                if not np.isnan(curvature) and not np.isinf(curvature):
                    curvatures.append(abs(curvature))
            
            if curvatures:
                return {
                    'mean_curvature': np.mean(curvatures),
                    'std_curvature': np.std(curvatures),
                    'max_curvature': np.max(curvatures),
                    'curvature_variation': np.std(curvatures) / np.mean(curvatures) if np.mean(curvatures) > 0 else 0
                }
            else:
                return {'mean_curvature': 0, 'std_curvature': 0, 'max_curvature': 0, 'curvature_variation': 0}
                
        except:
            return {'mean_curvature': 0, 'std_curvature': 0, 'max_curvature': 0, 'curvature_variation': 0}
    
    def _point_curvature(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, scale_factor: float) -> float:
        """Calculate curvature at a point using three consecutive points."""
        
        # Convert to real units
        p1 = p1 * scale_factor
        p2 = p2 * scale_factor
        p3 = p3 * scale_factor
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate cross product magnitude
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Calculate distances
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        
        if d1 > 0 and d2 > 0:
            # Curvature formula
            curvature = 2 * cross_product / (d1 * d2 * (d1 + d2))
            return curvature
        else:
            return 0.0
    
    def _calculate_roughness_index(self, contour: np.ndarray) -> float:
        """Calculate roughness index by comparing to fitted ellipse."""
        
        try:
            if len(contour) >= 5:
                # Fit ellipse to contour
                ellipse = cv2.fitEllipse(contour)
                
                # Create smooth ellipse contour
                ellipse_contour = cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(ellipse[1][0]/2), int(ellipse[1][1]/2)),
                    int(ellipse[2]), 0, 360, 5
                )
                
                # Calculate deviation from ellipse
                distances = []
                contour_points = contour[:, 0, :]
                
                for point in contour_points[::5]:  # Sample every 5th point for efficiency
                    # Find closest point on ellipse
                    dists_to_ellipse = np.sum((ellipse_contour - point) ** 2, axis=1)
                    min_dist = np.sqrt(np.min(dists_to_ellipse))
                    distances.append(min_dist)
                
                if distances:
                    roughness = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
                    return roughness
                else:
                    return 0.0
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _analyze_surface_texture(self, fiber_region: np.ndarray, 
                                wall_mask: np.ndarray, scale_factor: float) -> Dict:
        """Analyze surface texture within the fiber wall."""
        
        # Extract pixel values within fiber wall
        wall_pixels = fiber_region[wall_mask > 0]
        
        if len(wall_pixels) == 0:
            return {'error': 'No wall pixels found'}
        
        # Basic intensity statistics
        intensity_stats = {
            'mean_intensity': np.mean(wall_pixels),
            'std_intensity': np.std(wall_pixels),
            'intensity_range': np.max(wall_pixels) - np.min(wall_pixels),
            'intensity_cv': np.std(wall_pixels) / np.mean(wall_pixels) if np.mean(wall_pixels) > 0 else 0
        }
        
        # Surface roughness using local standard deviation
        roughness_metrics = self._calculate_surface_roughness(fiber_region, wall_mask)
        
        # Texture variation using sliding window analysis
        variation_metrics = self._analyze_texture_variation(fiber_region, wall_mask)
        
        return {
            'intensity_stats': intensity_stats,
            'roughness_metrics': roughness_metrics,
            'variation_metrics': variation_metrics
        }
    
    def _calculate_surface_roughness(self, fiber_region: np.ndarray, 
                                   wall_mask: np.ndarray) -> Dict:
        """Calculate surface roughness using local variation analysis."""
        
        # Create masked region
        masked_region = fiber_region.copy().astype(float)
        masked_region[wall_mask == 0] = np.nan
        
        # Calculate local standard deviation (roughness measure)
        kernel_sizes = [3, 5, 7]
        roughness_measures = {}
        
        for kernel_size in kernel_sizes:
            # Local standard deviation filter
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            
            # Mean filter
            local_mean = cv2.filter2D(fiber_region.astype(float), -1, kernel)
            
            # Variance calculation
            local_var = cv2.filter2D((fiber_region.astype(float)) ** 2, -1, kernel) - local_mean ** 2
            local_std = np.sqrt(np.maximum(local_var, 0))
            
            # Extract values within wall mask
            wall_roughness = local_std[wall_mask > 0]
            wall_roughness = wall_roughness[~np.isnan(wall_roughness)]
            
            if len(wall_roughness) > 0:
                roughness_measures[f'kernel_{kernel_size}'] = {
                    'mean_roughness': np.mean(wall_roughness),
                    'std_roughness': np.std(wall_roughness),
                    'max_roughness': np.max(wall_roughness)
                }
        
        return roughness_measures
    
    def _analyze_texture_variation(self, fiber_region: np.ndarray, 
                                 wall_mask: np.ndarray) -> Dict:
        """Analyze texture variation using sliding window approach."""
        
        # Define window size based on image size
        height, width = fiber_region.shape
        window_size = min(32, max(8, min(height, width) // 8))
        
        variations = []
        window_means = []
        
        # Sliding window analysis
        for y in range(0, height - window_size, window_size // 2):
            for x in range(0, width - window_size, window_size // 2):
                # Extract window
                window = fiber_region[y:y+window_size, x:x+window_size]
                mask_window = wall_mask[y:y+window_size, x:x+window_size]
                
                # Only analyze windows with sufficient fiber content
                if np.sum(mask_window) > (window_size ** 2) * 0.3:
                    window_pixels = window[mask_window > 0]
                    
                    if len(window_pixels) > 10:
                        window_mean = np.mean(window_pixels)
                        window_std = np.std(window_pixels)
                        
                        variations.append(window_std)
                        window_means.append(window_mean)
        
        if variations:
            return {
                'mean_local_variation': np.mean(variations),
                'std_local_variation': np.std(variations),
                'global_variation': np.std(window_means),
                'variation_uniformity': 1.0 / (1.0 + np.std(variations) / np.mean(variations)) if np.mean(variations) > 0 else 0
            }
        else:
            return {'mean_local_variation': 0, 'std_local_variation': 0, 'global_variation': 0, 'variation_uniformity': 0}
    
    def _analyze_lbp_texture(self, fiber_region: np.ndarray, wall_mask: np.ndarray) -> Dict:
        """Analyze texture using Local Binary Patterns."""
        
        # Apply LBP to fiber region
        lbp = local_binary_pattern(
            fiber_region, self.lbp_n_points, self.lbp_radius, method='uniform'
        )
        
        # Extract LBP values within wall mask
        lbp_values = lbp[wall_mask > 0]
        
        if len(lbp_values) == 0:
            return {'error': 'No LBP values found'}
        
        # Calculate LBP histogram
        n_bins = self.lbp_n_points + 2  # +2 for uniform patterns
        hist, _ = np.histogram(lbp_values, bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        # LBP texture metrics
        uniformity = np.sum(hist ** 2)  # Energy/Uniformity
        entropy_val = entropy(hist + 1e-7)  # Entropy
        
        # Pattern analysis
        uniform_patterns = hist[:self.lbp_n_points]  # First n_points are uniform
        non_uniform_ratio = np.sum(hist[self.lbp_n_points:])
        
        return {
            'lbp_uniformity': uniformity,
            'lbp_entropy': entropy_val,
            'non_uniform_ratio': non_uniform_ratio,
            'pattern_diversity': 1.0 - uniformity,
            'histogram': hist.tolist()
        }
    
    def _analyze_glcm_texture(self, fiber_region: np.ndarray, wall_mask: np.ndarray) -> Dict:
        """Analyze texture using Gray Level Co-occurrence Matrix."""
        
        # Reduce image to 64 gray levels for GLCM efficiency
        fiber_region_reduced = (fiber_region / 4).astype(np.uint8)
        
        # Create masked region
        masked_region = fiber_region_reduced.copy()
        masked_region[wall_mask == 0] = 0
        
        glcm_props = {}
        
        try:
            # Calculate GLCM for different distances and angles
            glcm = graycomatrix(
                masked_region, 
                distances=self.glcm_distances, 
                angles=self.glcm_angles,
                levels=64,
                symmetric=True,
                normed=True
            )
            
            # Calculate Haralick texture features
            contrast = graycoprops(glcm, 'contrast')
            dissimilarity = graycoprops(glcm, 'dissimilarity')
            homogeneity = graycoprops(glcm, 'homogeneity')
            energy = graycoprops(glcm, 'energy')
            correlation = graycoprops(glcm, 'correlation')
            
            # Average over all distances and angles
            glcm_props = {
                'contrast': np.mean(contrast),
                'dissimilarity': np.mean(dissimilarity),
                'homogeneity': np.mean(homogeneity),
                'energy': np.mean(energy),
                'correlation': np.mean(correlation),
                'contrast_std': np.std(contrast),
                'homogeneity_std': np.std(homogeneity)
            }
            
        except Exception as e:
            glcm_props = {
                'contrast': 0, 'dissimilarity': 0, 'homogeneity': 0,
                'energy': 0, 'correlation': 0, 'contrast_std': 0, 'homogeneity_std': 0,
                'error': str(e)
            }
        
        return glcm_props
    
    def _analyze_edge_texture(self, fiber_region: np.ndarray, 
                            wall_mask: np.ndarray, scale_factor: float) -> Dict:
        """Analyze texture using edge-based features."""
        
        # Apply Gaussian filter to reduce noise
        smoothed = cv2.GaussianBlur(fiber_region, (3, 3), self.edge_sigma)
        
        # Calculate gradients
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Extract values within wall mask
        mag_values = gradient_magnitude[wall_mask > 0]
        dir_values = gradient_direction[wall_mask > 0]
        
        if len(mag_values) == 0:
            return {'error': 'No edge values found'}
        
        # Edge density and strength
        edge_threshold = np.percentile(mag_values, 75)  # Top 25% of gradients
        strong_edges = mag_values > edge_threshold
        edge_density = np.sum(strong_edges) / len(mag_values)
        
        # Direction analysis
        direction_hist, _ = np.histogram(dir_values, bins=8, range=(-np.pi, np.pi))
        direction_uniformity = np.sum((direction_hist / np.sum(direction_hist)) ** 2)
        
        return {
            'mean_gradient_magnitude': np.mean(mag_values) * scale_factor,
            'std_gradient_magnitude': np.std(mag_values) * scale_factor,
            'edge_density': edge_density,
            'gradient_range': (np.max(mag_values) - np.min(mag_values)) * scale_factor,
            'direction_uniformity': direction_uniformity,
            'strong_edge_ratio': edge_density
        }
    
    def _classify_crumbly_texture(self, boundary_metrics: Dict, surface_metrics: Dict,
                                lbp_metrics: Dict, glcm_metrics: Dict, edge_metrics: Dict) -> Dict:
        """Combine all metrics to classify texture as crumbly or smooth."""
        
        # Initialize score
        crumbly_score = 0.0
        confidence_factors = []
        
        # Weight factors for different metric categories
        weights = {
            'boundary': 0.3,
            'surface': 0.25,
            'lbp': 0.2,
            'glcm': 0.15,
            'edge': 0.1
        }
        
        # Boundary irregularity contribution
        if 'outer_boundary' in boundary_metrics:
            outer = boundary_metrics['outer_boundary']
            
            # Roughness index (higher = more crumbly)
            roughness_score = min(1.0, outer.get('roughness_index', 0) / 0.3)
            
            # Fractal dimension (higher = more complex/crumbly)
            fractal_score = min(1.0, (outer.get('fractal_dimension', 1.0) - 1.0) / 0.5)
            
            # Circularity (lower = more irregular/crumbly)
            circularity_score = 1.0 - min(1.0, outer.get('circularity', 1.0))
            
            # Curvature variation (higher = more irregular)
            curvature_var = outer.get('curvature_stats', {}).get('curvature_variation', 0)
            curvature_score = min(1.0, curvature_var / 2.0)
            
            boundary_score = np.mean([roughness_score, fractal_score, circularity_score, curvature_score])
            crumbly_score += boundary_score * weights['boundary']
            confidence_factors.append(f"Boundary: {boundary_score:.2f}")
        
        # Surface texture contribution
        if 'roughness_metrics' in surface_metrics:
            roughness = surface_metrics['roughness_metrics']
            
            # Average roughness across different kernel sizes
            roughness_scores = []
            for kernel_key in roughness:
                if isinstance(roughness[kernel_key], dict):
                    mean_rough = roughness[kernel_key].get('mean_roughness', 0)
                    roughness_scores.append(min(1.0, mean_rough / 15.0))  # Normalize to ~15 intensity units
            
            if roughness_scores:
                surface_score = np.mean(roughness_scores)
                crumbly_score += surface_score * weights['surface']
                confidence_factors.append(f"Surface: {surface_score:.2f}")
        
        # LBP texture contribution
        if 'lbp_uniformity' in lbp_metrics:
            # Lower uniformity = more varied texture = more crumbly
            lbp_score = 1.0 - min(1.0, lbp_metrics['lbp_uniformity'] / 0.5)
            
            # Higher entropy = more varied = more crumbly
            entropy_score = min(1.0, lbp_metrics.get('lbp_entropy', 0) / 3.0)
            
            # Non-uniform patterns
            non_uniform_score = min(1.0, lbp_metrics.get('non_uniform_ratio', 0) / 0.3)
            
            lbp_total_score = np.mean([lbp_score, entropy_score, non_uniform_score])
            crumbly_score += lbp_total_score * weights['lbp']
            confidence_factors.append(f"LBP: {lbp_total_score:.2f}")
        
        # GLCM texture contribution
        if 'contrast' in glcm_metrics:
            # Higher contrast = more texture variation = more crumbly
            contrast_score = min(1.0, glcm_metrics.get('contrast', 0) / 50.0)
            
            # Lower homogeneity = more varied = more crumbly
            homogeneity_score = 1.0 - min(1.0, glcm_metrics.get('homogeneity', 1.0))
            
            # Higher dissimilarity = more crumbly
            dissimilarity_score = min(1.0, glcm_metrics.get('dissimilarity', 0) / 10.0)
            
            glcm_total_score = np.mean([contrast_score, homogeneity_score, dissimilarity_score])
            crumbly_score += glcm_total_score * weights['glcm']
            confidence_factors.append(f"GLCM: {glcm_total_score:.2f}")
        
        # Edge texture contribution
        if 'edge_density' in edge_metrics:
            # Higher edge density = more texture = more crumbly
            edge_density_score = min(1.0, edge_metrics.get('edge_density', 0) / 0.4)
            
            # Higher gradient variation = more texture variation = more crumbly
            grad_std = edge_metrics.get('std_gradient_magnitude', 0)
            grad_mean = edge_metrics.get('mean_gradient_magnitude', 1)
            grad_cv_score = min(1.0, (grad_std / grad_mean) if grad_mean > 0 else 0)
            
            edge_total_score = np.mean([edge_density_score, grad_cv_score])
            crumbly_score += edge_total_score * weights['edge']
            confidence_factors.append(f"Edge: {edge_total_score:.2f}")
        
        # Determine classification
        if crumbly_score > 0.6:
            classification = 'crumbly'
            confidence = min(0.95, 0.5 + crumbly_score)
        elif crumbly_score < 0.3:
            classification = 'smooth'
            confidence = min(0.95, 0.5 + (1.0 - crumbly_score))
        else:
            classification = 'intermediate'
            confidence = 0.3 + abs(0.5 - crumbly_score)
        
        return {
            'classification': classification,
            'confidence': confidence,
            'crumbly_score': crumbly_score,
            'confidence_factors': confidence_factors
        }
    
    def _create_debug_visualization(self, image: np.ndarray, fiber_mask: np.ndarray,
                                  lumen_mask: Optional[np.ndarray], results: Dict):
        """Create debug visualization showing texture analysis results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Fiber segmentation
        overlay = np.zeros((*image.shape, 3), dtype=np.uint8)
        overlay[fiber_mask > 0] = [0, 255, 0]  # Green for fiber
        if lumen_mask is not None:
            overlay[lumen_mask > 0] = [255, 0, 0]  # Red for lumen
        
        axes[1].imshow(cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), 0.7, overlay, 0.3, 0
        ))
        axes[1].set_title('Fiber Segmentation')
        axes[1].axis('off')
        
        # Edge detection visualization
        if 'edges' in results.get('texture_metrics', {}):
            smoothed = cv2.GaussianBlur(image, (3, 3), self.edge_sigma)
            edges = cv2.Canny(smoothed, 50, 150)
            axes[2].imshow(edges, cmap='gray')
            axes[2].set_title('Edge Detection')
            axes[2].axis('off')
        
        # LBP visualization
        if fiber_mask.any():
            lbp_img = local_binary_pattern(image, self.lbp_n_points, self.lbp_radius, method='uniform')
            lbp_masked = lbp_img.copy()
            lbp_masked[fiber_mask == 0] = 0
            axes[3].imshow(lbp_masked, cmap='viridis')
            axes[3].set_title('Local Binary Pattern')
            axes[3].axis('off')
        
        # Surface roughness visualization
        if fiber_mask.any():
            fiber_region = cv2.bitwise_and(image, image, mask=fiber_mask.astype(np.uint8))
            kernel = np.ones((5, 5)) / 25
            local_mean = cv2.filter2D(fiber_region.astype(float), -1, kernel)
            local_var = cv2.filter2D((fiber_region.astype(float))**2, -1, kernel) - local_mean**2
            local_std = np.sqrt(np.maximum(local_var, 0))
            local_std[fiber_mask == 0] = 0
            
            axes[4].imshow(local_std, cmap='hot')
            axes[4].set_title('Surface Roughness')
            axes[4].axis('off')
        
        # Results summary
        classification = results.get('classification', 'unknown')
        confidence = results.get('confidence', 0)
        crumbly_score = results.get('crumbly_score', 0)
        
        summary_text = f"Classification: {classification}\n"
        summary_text += f"Confidence: {confidence:.3f}\n"
        summary_text += f"Crumbly Score: {crumbly_score:.3f}\n\n"
        
        confidence_factors = results.get('confidence_factors', [])
        if confidence_factors:
            summary_text += "Contributing Factors:\n"
            for factor in confidence_factors:
                summary_text += f"  {factor}\n"
        
        axes[5].text(0.05, 0.95, summary_text, transform=axes[5].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
        axes[5].set_title('Analysis Results')
        axes[5].axis('off')
        
        plt.tight_layout()
        plt.show()

# Convenience functions
def detect_crumbly_texture(image: np.ndarray, fiber_mask: np.ndarray,
                          lumen_mask: Optional[np.ndarray] = None,
                          scale_factor: float = 1.0, **kwargs) -> Tuple[str, float]:
    """
    Convenience function for crumbly texture detection.
    
    Returns:
        Tuple of (classification, confidence)
    """
    detector = CrumblyDetector(**kwargs)
    results = detector.analyze_crumbly_texture(
        image, fiber_mask, lumen_mask, scale_factor
    )
    return results['classification'], results['confidence']

def batch_analyze_crumbly_texture(image_paths: List[str], 
                                 fiber_masks: List[np.ndarray],
                                 scale_factors: Optional[List[float]] = None) -> List[Dict]:
    """
    Batch analysis of crumbly texture for multiple images.
    """
    detector = CrumblyDetector()
    results = []
    
    if scale_factors is None:
        scale_factors = [1.0] * len(image_paths)
    
    for i, (img_path, fiber_mask) in enumerate(zip(image_paths, fiber_masks)):
        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                result = detector.analyze_crumbly_texture(
                    image, fiber_mask, scale_factor=scale_factors[i]
                )
                result['image_path'] = img_path
                results.append(result)
        except Exception as e:
            results.append({
                'image_path': img_path,
                'error': str(e),
                'classification': 'error'
            })
    
    return results