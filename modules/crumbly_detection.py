"""
SEM Fiber Analysis System - Refined Crumbly Texture Detection Module
Analyzes surface texture to distinguish between organized porosity and crumbly fragmentation.
"""

import cv2
import numpy as np
from skimage import filters, feature, measure, morphology, segmentation
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import ndimage, spatial
from scipy.stats import entropy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CrumblyDetector:
    """
    Detects and quantifies crumbly texture in SEM fiber images.
    Refined to distinguish between organized porosity and crumbly fragmentation.
    """
    
    def __init__(self, 
                 lbp_radius: int = 3,
                 lbp_n_points: int = 24,
                 glcm_distances: List[int] = [1, 2, 3],
                 glcm_angles: List[float] = [0, 45, 90, 135],
                 edge_sigma: float = 1.0,
                 porosity_aware: bool = True):
        """
        Initialize crumbly detector with configurable parameters.
        
        Args:
            lbp_radius: Radius for Local Binary Pattern analysis
            lbp_n_points: Number of points for LBP
            glcm_distances: Distances for Gray Level Co-occurrence Matrix
            glcm_angles: Angles for GLCM (in degrees)
            edge_sigma: Sigma for edge detection
            porosity_aware: Enable porosity-aware crumbly detection
        """
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.glcm_distances = glcm_distances
        self.glcm_angles = [np.radians(angle) for angle in glcm_angles]
        self.edge_sigma = edge_sigma
        self.porosity_aware = porosity_aware
        
        # Refined thresholds for porous vs crumbly distinction
        self.crumbly_thresholds = {
            'boundary_irregularity': 0.20,     # Higher = more irregular boundary
            'wall_fragmentation': 0.15,        # Higher = more fragmented walls
            'pore_edge_roughness': 0.25,       # Higher = rougher pore edges
            'structural_integrity': 0.7,       # Lower = less intact structure
            'texture_chaos': 0.3,              # Higher = more chaotic texture
            'organized_porosity': 0.6          # Lower = less organized pores
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
            # 1. Porosity-Aware Pore Analysis (NEW - Critical for distinction)
            if self.porosity_aware:
                pore_metrics = self._analyze_pore_characteristics(
                    fiber_region, wall_mask, scale_factor
                )
                results['pore_metrics'] = pore_metrics
            
            # 2. Boundary Analysis (Enhanced for porosity awareness)
            boundary_metrics = self._analyze_boundary_irregularity(
                fiber_mask, lumen_mask, scale_factor
            )
            results['boundary_metrics'] = boundary_metrics
            
            # 3. Wall Integrity Analysis (NEW - Key for crumbly vs porous)
            wall_integrity_metrics = self._analyze_wall_integrity(
                fiber_region, wall_mask, scale_factor
            )
            results['wall_integrity_metrics'] = wall_integrity_metrics
            
            # 4. Surface Texture Analysis (Refined for porosity)
            surface_metrics = self._analyze_surface_texture(
                fiber_region, wall_mask, scale_factor
            )
            results['surface_metrics'] = surface_metrics
            
            # 5. Local Binary Pattern Analysis (Enhanced)
            lbp_metrics = self._analyze_lbp_texture(
                fiber_region, wall_mask
            )
            results['texture_metrics']['lbp'] = lbp_metrics
            
            # 6. Gray Level Co-occurrence Matrix Analysis
            glcm_metrics = self._analyze_glcm_texture(
                fiber_region, wall_mask
            )
            results['texture_metrics']['glcm'] = glcm_metrics
            
            # 7. Edge-based Texture Analysis (Porosity-aware)
            edge_metrics = self._analyze_edge_texture(
                fiber_region, wall_mask, scale_factor
            )
            results['texture_metrics']['edges'] = edge_metrics
            
            # 8. Combine metrics for porosity-aware classification
            if self.porosity_aware:
                classification_result = self._classify_porosity_aware_crumbly(
                    boundary_metrics, wall_integrity_metrics, pore_metrics,
                    surface_metrics, lbp_metrics, glcm_metrics, edge_metrics
                )
            else:
                classification_result = self._classify_crumbly_texture(
                    boundary_metrics, surface_metrics, lbp_metrics, 
                    glcm_metrics, edge_metrics
                )
            
            results.update(classification_result)
            
            # 9. Generate debug visualization if requested
            if debug:
                self._create_debug_visualization(
                    image, fiber_mask, lumen_mask, results
                )
            
        except Exception as e:
            results['error'] = str(e)
            results['classification'] = 'error'
            print(f"Error in crumbly analysis: {e}")
        
        return results
    
    def _analyze_pore_characteristics(self, fiber_region: np.ndarray, 
                                    wall_mask: np.ndarray, scale_factor: float) -> Dict:
        """
        Analyze pore characteristics to distinguish organized porosity from crumbly fragmentation.
        Key insight: Porous fibers have round, well-defined pores. Crumbly fibers have irregular fragments.
        """
        
        try:
            # Create inverted mask to find pores within the fiber
            fiber_area = np.zeros_like(wall_mask, dtype=np.uint8)
            fiber_area[wall_mask] = 255
            
            # Find contour of fiber to define search area
            fiber_contours, _ = cv2.findContours(fiber_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not fiber_contours:
                return {'error': 'No fiber contour found'}
            
            main_fiber_contour = max(fiber_contours, key=cv2.contourArea)
            
            # Create mask for the entire fiber interior (including pores)
            fiber_interior_mask = np.zeros_like(wall_mask, dtype=np.uint8)
            cv2.fillPoly(fiber_interior_mask, [main_fiber_contour], 255)
            
            # Find pores: dark regions within the fiber interior that aren't the wall
            pore_candidates = fiber_interior_mask.astype(bool) & (~wall_mask)
            
            # Apply morphological operations to clean up small noise
            pore_mask = morphology.binary_opening(pore_candidates, morphology.disk(2))
            pore_mask = morphology.binary_closing(pore_mask, morphology.disk(3))
            
            # Find individual pore contours
            pore_mask_uint8 = pore_mask.astype(np.uint8) * 255
            pore_contours, _ = cv2.findContours(pore_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not pore_contours:
                return {
                    'pore_count': 0,
                    'organized_porosity_score': 1.0,  # No pores = perfectly organized
                    'mean_pore_circularity': 1.0,
                    'mean_pore_edge_smoothness': 1.0,
                    'total_pore_area_fraction': 0.0
                }
            
            # Filter pores by minimum size (avoid noise)
            min_pore_area = 50  # pixels
            valid_pores = [contour for contour in pore_contours if cv2.contourArea(contour) > min_pore_area]
            
            if not valid_pores:
                return {
                    'pore_count': 0,
                    'organized_porosity_score': 1.0,
                    'mean_pore_circularity': 1.0,
                    'mean_pore_edge_smoothness': 1.0,
                    'total_pore_area_fraction': 0.0
                }
            
            # Analyze each pore
            pore_analyses = []
            for contour in valid_pores:
                pore_analysis = self._analyze_single_pore(contour, scale_factor)
                pore_analyses.append(pore_analysis)
            
            # Aggregate pore characteristics
            circularities = [p['circularity'] for p in pore_analyses]
            smoothnesses = [p['edge_smoothness'] for p in pore_analyses]
            sizes = [p['area_um2'] for p in pore_analyses]
            
            # Organized porosity metrics
            mean_circularity = np.mean(circularities)
            circularity_consistency = 1.0 - min(1.0, np.std(circularities))  # Higher = more consistent
            
            mean_smoothness = np.mean(smoothnesses)
            size_variation = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
            
            # Calculate pore spatial organization
            spatial_organization = self._calculate_pore_spatial_organization(valid_pores, scale_factor)
            
            # Overall organized porosity score
            # High score = well-organized pores (NOT crumbly)
            # Low score = disorganized/fragmented (potentially crumbly)
            organized_porosity_score = np.mean([
                mean_circularity,           # Round pores = organized
                circularity_consistency,    # Consistent shapes = organized
                mean_smoothness,           # Smooth edges = organized
                1.0 - min(1.0, size_variation),  # Consistent sizes = organized
                spatial_organization       # Even distribution = organized
            ])
            
            return {
                'pore_count': len(valid_pores),
                'mean_pore_circularity': mean_circularity,
                'pore_circularity_consistency': circularity_consistency,
                'mean_pore_edge_smoothness': mean_smoothness,
                'pore_size_variation': size_variation,
                'spatial_organization': spatial_organization,
                'organized_porosity_score': organized_porosity_score,
                'mean_pore_area_um2': np.mean(sizes),
                'total_pore_area_fraction': np.sum(pore_mask) / np.sum(fiber_interior_mask) if np.sum(fiber_interior_mask) > 0 else 0
            }
            
        except Exception as e:
            return {
                'error': f'Pore analysis failed: {str(e)}',
                'pore_count': 0,
                'organized_porosity_score': 0.5,
                'mean_pore_circularity': 0.5,
                'mean_pore_edge_smoothness': 0.5,
                'total_pore_area_fraction': 0.0
            }
    
    def _analyze_single_pore(self, contour: np.ndarray, scale_factor: float) -> Dict:
        """Analyze characteristics of a single pore."""
        
        area = cv2.contourArea(contour) * (scale_factor ** 2)
        perimeter = cv2.arcLength(contour, True) * scale_factor
        
        # Circularity (1.0 = perfect circle)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        circularity = min(1.0, circularity)  # Cap at 1.0
        
        # Edge smoothness using curvature analysis
        edge_smoothness = self._calculate_pore_edge_smoothness(contour)
        
        # Aspect ratio analysis
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                aspect_ratio = minor_axis / major_axis if major_axis > 0 else 0
            except:
                aspect_ratio = 1.0
        else:
            aspect_ratio = 1.0
        
        return {
            'area_um2': area,
            'perimeter_um': perimeter,
            'circularity': circularity,
            'edge_smoothness': edge_smoothness,
            'aspect_ratio': aspect_ratio
        }
    
    def _calculate_pore_edge_smoothness(self, contour: np.ndarray) -> float:
        """Calculate how smooth the pore edges are (vs jagged/fragmented)."""
        
        try:
            if len(contour) < 4:
                return 1.0  # Very small, assume smooth
            
            # Approximate contour to reduce noise
            epsilon = 0.02 * cv2.arcLength(contour, True)
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(smoothed_contour) < 4:
                return 1.0
            
            # Calculate curvature variation
            points = smoothed_contour[:, 0, :]
            curvatures = []
            
            for i in range(len(points)):
                p1 = points[i-1]
                p2 = points[i]
                p3 = points[(i+1) % len(points)]
                
                # Simple curvature estimate
                v1 = p2 - p1
                v2 = p3 - p2
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    curvatures.append(angle)
            
            if curvatures:
                # Smooth edges have consistent, low curvature
                curvature_variation = np.std(curvatures)
                smoothness = 1.0 / (1.0 + curvature_variation)
                return smoothness
            else:
                return 1.0
                
        except:
            return 0.5  # Default moderate smoothness
    
    def _calculate_pore_spatial_organization(self, pore_contours: List[np.ndarray], scale_factor: float) -> float:
        """Calculate how well-organized the pores are spatially."""
        
        if len(pore_contours) < 2:
            return 1.0  # Single pore or no pores = perfectly organized
        
        try:
            # Get pore centers
            centers = []
            for contour in pore_contours:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    centers.append([cx, cy])
            
            if len(centers) < 2:
                return 1.0
            
            centers = np.array(centers)
            
            # Calculate nearest neighbor distances
            distances = pdist(centers) * scale_factor
            
            if len(distances) == 0:
                return 1.0
            
            # Organized pores have consistent spacing
            mean_distance = np.mean(distances)
            distance_variation = np.std(distances) / mean_distance if mean_distance > 0 else 0
            
            # Lower variation = more organized
            organization_score = 1.0 / (1.0 + distance_variation)
            
            return organization_score
            
        except:
            return 0.5  # Default moderate organization
    
    def _analyze_wall_integrity(self, fiber_region: np.ndarray, 
                              wall_mask: np.ndarray, scale_factor: float) -> Dict:
        """
        Analyze wall integrity to distinguish intact porous walls from fragmented crumbly walls.
        """
        
        try:
            # Find wall regions
            wall_pixels = fiber_region[wall_mask > 0]
            
            if len(wall_pixels) == 0:
                return {'error': 'No wall pixels found', 'wall_integrity_score': 0}
            
            # 1. Wall thickness analysis
            thickness_metrics = self._analyze_wall_thickness(wall_mask, scale_factor)
            
            # 2. Wall continuity analysis  
            continuity_metrics = self._analyze_wall_continuity(wall_mask, scale_factor)
            
            # 3. Fragmentation analysis
            fragmentation_metrics = self._analyze_wall_fragmentation(wall_mask, scale_factor)
            
            # 4. Structural coherence
            coherence_score = self._calculate_structural_coherence(
                thickness_metrics, continuity_metrics, fragmentation_metrics
            )
            
            return {
                'thickness_metrics': thickness_metrics,
                'continuity_metrics': continuity_metrics,
                'fragmentation_metrics': fragmentation_metrics,
                'wall_integrity_score': coherence_score
            }
        
        except Exception as e:
            return {
                'error': f'Wall integrity analysis failed: {str(e)}',
                'wall_integrity_score': 0.5
            }
    
    def _analyze_wall_thickness(self, wall_mask: np.ndarray, scale_factor: float) -> Dict:
        """Analyze wall thickness consistency."""
        
        try:
            # Distance transform to find wall thickness
            dist_transform = cv2.distanceTransform(wall_mask.astype(np.uint8), cv2.DIST_L2, 5)
            
            # Get thickness values (distance from edge)
            wall_thicknesses = dist_transform[wall_mask] * scale_factor
            wall_thicknesses = wall_thicknesses[wall_thicknesses > 0]
            
            if len(wall_thicknesses) == 0:
                return {'mean_thickness_um': 0, 'thickness_variation': 1.0, 'thickness_consistency': 0}
            
            mean_thickness = np.mean(wall_thicknesses)
            thickness_std = np.std(wall_thicknesses)
            thickness_variation = thickness_std / mean_thickness if mean_thickness > 0 else 1.0
            
            return {
                'mean_thickness_um': mean_thickness,
                'thickness_std_um': thickness_std,
                'thickness_variation': thickness_variation,
                'thickness_consistency': 1.0 / (1.0 + thickness_variation)
            }
        
        except:
            return {'mean_thickness_um': 0, 'thickness_variation': 1.0, 'thickness_consistency': 0}
    
    def _analyze_wall_continuity(self, wall_mask: np.ndarray, scale_factor: float) -> Dict:
        """Analyze how continuous/connected the walls are."""
        
        try:
            # Find connected components in wall
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                wall_mask.astype(np.uint8), connectivity=8
            )
            
            # Remove background label
            component_areas = stats[1:, cv2.CC_STAT_AREA] * (scale_factor ** 2)
            
            if len(component_areas) == 0:
                return {'continuity_score': 0, 'num_wall_components': 0}
            
            # Calculate continuity metrics
            total_wall_area = np.sum(component_areas)
            largest_component_area = np.max(component_areas)
            
            # High continuity = most wall area in single component
            continuity_score = largest_component_area / total_wall_area if total_wall_area > 0 else 0
            
            # Number of fragments
            significant_fragments = np.sum(component_areas > total_wall_area * 0.05)  # >5% of total
            
            return {
                'num_wall_components': len(component_areas),
                'significant_fragments': significant_fragments,
                'largest_component_fraction': continuity_score,
                'continuity_score': continuity_score
            }
        
        except:
            return {'continuity_score': 0, 'num_wall_components': 0}
    
    def _analyze_wall_fragmentation(self, wall_mask: np.ndarray, scale_factor: float) -> Dict:
        """Analyze degree of wall fragmentation."""
        
        try:
            # Morphological analysis to detect fragmentation
            # Closing operation to connect nearby fragments
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed_wall = cv2.morphologyEx(wall_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Compare original vs closed - large difference indicates fragmentation
            fragmentation_pixels = np.sum(closed_wall) - np.sum(wall_mask)
            total_pixels = np.sum(closed_wall)
            
            fragmentation_ratio = fragmentation_pixels / total_pixels if total_pixels > 0 else 0
            
            # Edge roughness of wall boundaries
            wall_contours, _ = cv2.findContours(wall_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            edge_roughness = 0
            if wall_contours:
                main_contour = max(wall_contours, key=cv2.contourArea)
                edge_roughness = self._calculate_roughness_index(main_contour)
            
            return {
                'fragmentation_ratio': fragmentation_ratio,
                'edge_roughness': edge_roughness,
                'fragmentation_score': (fragmentation_ratio + edge_roughness) / 2
            }
        
        except:
            return {'fragmentation_ratio': 0, 'edge_roughness': 0, 'fragmentation_score': 0}
    
    def _calculate_structural_coherence(self, thickness_metrics: Dict, 
                                      continuity_metrics: Dict, fragmentation_metrics: Dict) -> float:
        """Calculate overall structural coherence score."""
        
        try:
            # High coherence = intact structure (porous but not crumbly)
            # Low coherence = fragmented structure (crumbly)
            
            thickness_score = thickness_metrics.get('thickness_consistency', 0)
            continuity_score = continuity_metrics.get('continuity_score', 0)
            fragmentation_score = 1.0 - fragmentation_metrics.get('fragmentation_score', 1.0)
            
            coherence = np.mean([thickness_score, continuity_score, fragmentation_score])
            
            return coherence
        
        except:
            return 0.5
    
    def _analyze_boundary_irregularity(self, fiber_mask: np.ndarray, 
                                     lumen_mask: Optional[np.ndarray],
                                     scale_factor: float) -> Dict:
        """Analyze irregularity of fiber boundaries."""
        
        metrics = {}
        
        try:
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
        
        except Exception as e:
            metrics = {'error': f'Boundary analysis failed: {str(e)}'}
        
        return metrics
    
    def _calculate_contour_irregularity(self, contour: np.ndarray, 
                                      scale_factor: float) -> Dict:
        """Calculate irregularity metrics for a single contour."""
        
        try:
            # Basic contour properties
            perimeter = cv2.arcLength(contour, True) * scale_factor
            area = cv2.contourArea(contour) * (scale_factor ** 2)
            
            # Circularity (4π*area/perimeter²) - closer to 1 = more circular
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            circularity = min(1.0, circularity)  # Cap at 1.0
            
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
        
        except Exception as e:
            return {
                'perimeter_um': 0,
                'area_um2': 0,
                'circularity': 0.5,
                'solidity': 0.5,
                'fractal_dimension': 1.0,
                'roughness_index': 0.5,
                'curvature_stats': {'mean_curvature': 0, 'std_curvature': 0},
                'error': str(e)
            }
    
    def _estimate_fractal_dimension(self, contour: np.ndarray) -> float:
        """Estimate fractal dimension using box counting method."""
        
        try:
            if len(contour) < 10:
                return 1.0
            
            # Convert contour to binary image
            x_coords = contour[:, 0, 0]
            y_coords = contour[:, 0, 1]
            
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            
            if width < 10 or height < 10:
                return 1.0
            
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
                if scale < min(width, height) / 2:
                    count = 0
                    for i in range(0, height, scale):
                        for j in range(0, width, scale):
                            box = contour_img[i:i+scale, j:j+scale]
                            if np.any(box):
                                count += 1
                    counts.append(count)
                else:
                    break
            
            # Calculate fractal dimension
            if len(counts) > 2:
                scales_used = scales[:len(counts)]
                scales_log = np.log([1/s for s in scales_used])
                counts_log = np.log([max(1, c) for c in counts])
                
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
            if len(contour) < 5:
                return {'mean_curvature': 0, 'std_curvature': 0, 'max_curvature': 0, 'curvature_variation': 0}
            
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
                if not np.isnan(curvature) and not np.isinf(curvature) and abs(curvature) < 10:
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
        
        try:
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
        except:
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
                    return min(1.0, roughness)  # Cap at 1.0
                else:
                    return 0.0
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _analyze_surface_texture(self, fiber_region: np.ndarray, 
                                wall_mask: np.ndarray, scale_factor: float) -> Dict:
        """Analyze surface texture within the fiber wall."""
        
        try:
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
        
        except Exception as e:
            return {'error': f'Surface texture analysis failed: {str(e)}'}
    
    def _calculate_surface_roughness(self, fiber_region: np.ndarray, 
                                   wall_mask: np.ndarray) -> Dict:
        """Calculate surface roughness using local variation analysis."""
        
        try:
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
        
        except:
            return {}
    
    def _analyze_texture_variation(self, fiber_region: np.ndarray, 
                                 wall_mask: np.ndarray) -> Dict:
        """Analyze texture variation using sliding window approach."""
        
        try:
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
        
        except:
            return {'mean_local_variation': 0, 'std_local_variation': 0, 'global_variation': 0, 'variation_uniformity': 0}
    
    def _analyze_lbp_texture(self, fiber_region: np.ndarray, wall_mask: np.ndarray) -> Dict:
        """Analyze texture using Local Binary Patterns."""
        
        try:
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
        
        except Exception as e:
            return {'error': f'LBP analysis failed: {str(e)}'}
    
    def _analyze_glcm_texture(self, fiber_region: np.ndarray, wall_mask: np.ndarray) -> Dict:
        """Analyze texture using Gray Level Co-occurrence Matrix."""
        
        try:
            # Reduce image to 64 gray levels for GLCM efficiency
            fiber_region_reduced = (fiber_region / 4).astype(np.uint8)
            
            # Create masked region
            masked_region = fiber_region_reduced.copy()
            masked_region[wall_mask == 0] = 0
            
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
            return {
                'contrast': np.mean(contrast),
                'dissimilarity': np.mean(dissimilarity),
                'homogeneity': np.mean(homogeneity),
                'energy': np.mean(energy),
                'correlation': np.mean(correlation),
                'contrast_std': np.std(contrast),
                'homogeneity_std': np.std(homogeneity)
            }
            
        except Exception as e:
            return {
                'contrast': 0, 'dissimilarity': 0, 'homogeneity': 0,
                'energy': 0, 'correlation': 0, 'contrast_std': 0, 'homogeneity_std': 0,
                'error': str(e)
            }
    
    def _analyze_edge_texture(self, fiber_region: np.ndarray, 
                            wall_mask: np.ndarray, scale_factor: float) -> Dict:
        """Analyze texture using edge-based features."""
        
        try:
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
        
        except Exception as e:
            return {'error': f'Edge analysis failed: {str(e)}'}
    
    def _classify_porosity_aware_crumbly(self, boundary_metrics: Dict, wall_integrity_metrics: Dict,
                                       pore_metrics: Dict, surface_metrics: Dict, lbp_metrics: Dict,
                                       glcm_metrics: Dict, edge_metrics: Dict) -> Dict:
        """
        Enhanced classification that distinguishes between organized porosity and crumbly fragmentation.
        """
        
        try:
            # Initialize score components
            crumbly_indicators = []
            porous_but_intact_indicators = []
            confidence_factors = []
            
            # 1. PORE ORGANIZATION ANALYSIS (Most Important)
            # Well-organized pores indicate healthy porosity, not crumbly texture
            pore_organization = pore_metrics.get('organized_porosity_score', 0.5)
            
            if pore_organization > 0.7:
                porous_but_intact_indicators.append(pore_organization)
                confidence_factors.append(f"Well-organized pores: {pore_organization:.2f}")
            elif pore_organization < 0.4:
                crumbly_indicators.append(1.0 - pore_organization)
                confidence_factors.append(f"Disorganized pores: {pore_organization:.2f}")
            
            # Pore circularity - round pores = organized, irregular = fragmented
            pore_circularity = pore_metrics.get('mean_pore_circularity', 0.5)
            if pore_circularity > 0.6:
                porous_but_intact_indicators.append(pore_circularity)
                confidence_factors.append(f"Round pores: {pore_circularity:.2f}")
            elif pore_circularity < 0.3:
                crumbly_indicators.append(1.0 - pore_circularity)
                confidence_factors.append(f"Irregular pores: {pore_circularity:.2f}")
            
            # Pore edge smoothness
            pore_smoothness = pore_metrics.get('mean_pore_edge_smoothness', 0.5)
            if pore_smoothness > 0.6:
                porous_but_intact_indicators.append(pore_smoothness)
                confidence_factors.append(f"Smooth pore edges: {pore_smoothness:.2f}")
            elif pore_smoothness < 0.4:
                crumbly_indicators.append(1.0 - pore_smoothness)
                confidence_factors.append(f"Rough pore edges: {pore_smoothness:.2f}")
            
            # 2. WALL INTEGRITY ANALYSIS (Second Most Important)
            wall_integrity = wall_integrity_metrics.get('wall_integrity_score', 0.5)
            
            if wall_integrity > 0.7:
                porous_but_intact_indicators.append(wall_integrity)
                confidence_factors.append(f"Intact walls: {wall_integrity:.2f}")
            elif wall_integrity < 0.4:
                crumbly_indicators.append(1.0 - wall_integrity)
                confidence_factors.append(f"Fragmented walls: {wall_integrity:.2f}")
            
            # Wall continuity
            continuity = wall_integrity_metrics.get('continuity_metrics', {}).get('continuity_score', 0.5)
            if continuity > 0.8:
                porous_but_intact_indicators.append(continuity)
                confidence_factors.append(f"Continuous walls: {continuity:.2f}")
            elif continuity < 0.5:
                crumbly_indicators.append(1.0 - continuity)
                confidence_factors.append(f"Broken walls: {continuity:.2f}")
            
            # 3. BOUNDARY ANALYSIS (Refined for porosity)
            if 'outer_boundary' in boundary_metrics:
                outer = boundary_metrics['outer_boundary']
                
                # Circularity of overall fiber shape
                overall_circularity = outer.get('circularity', 0.5)
                if overall_circularity > 0.7:
                    porous_but_intact_indicators.append(overall_circularity)
                    confidence_factors.append(f"Regular fiber shape: {overall_circularity:.2f}")
                elif overall_circularity < 0.4:
                    crumbly_indicators.append(1.0 - overall_circularity)
                    confidence_factors.append(f"Irregular fiber shape: {overall_circularity:.2f}")
                
                # Roughness of fiber boundary
                boundary_roughness = outer.get('roughness_index', 0.5)
                if boundary_roughness > 0.4:  # Higher threshold for porous materials
                    crumbly_indicators.append(boundary_roughness)
                    confidence_factors.append(f"Rough boundary: {boundary_roughness:.2f}")
                elif boundary_roughness < 0.2:
                    porous_but_intact_indicators.append(1.0 - boundary_roughness)
                    confidence_factors.append(f"Smooth boundary: {boundary_roughness:.2f}")
            
            # 4. TEXTURE ANALYSIS (Supporting evidence)
            # LBP uniformity - but adjusted for porous materials
            lbp_uniformity = lbp_metrics.get('lbp_uniformity', 0.3)
            if lbp_uniformity > 0.4:  # More lenient for porous materials
                porous_but_intact_indicators.append(lbp_uniformity)
            elif lbp_uniformity < 0.2:
                crumbly_indicators.append(1.0 - lbp_uniformity)
            
            # GLCM contrast - high contrast can be from pores, not necessarily crumbly
            glcm_contrast = glcm_metrics.get('contrast', 0)
            normalized_contrast = min(1.0, glcm_contrast / 100.0)  # Normalize
            
            # Only consider extreme contrast as crumbly indicator
            if normalized_contrast > 0.8:
                crumbly_indicators.append(normalized_contrast)
                confidence_factors.append(f"High texture contrast: {normalized_contrast:.2f}")
            
            # 5. EDGE ANALYSIS (Supporting)
            edge_density = edge_metrics.get('edge_density', 0.3)
            # Moderate edge density is normal for porous materials
            if edge_density > 0.6:  # Very high edge density
                crumbly_indicators.append(edge_density)
                confidence_factors.append(f"High edge density: {edge_density:.2f}")
            
            # 6. CALCULATE FINAL SCORES
            
            # Calculate evidence for each classification
            crumbly_evidence = np.mean(crumbly_indicators) if crumbly_indicators else 0
            porous_intact_evidence = np.mean(porous_but_intact_indicators) if porous_but_intact_indicators else 0
            
            # Number of indicators for confidence
            total_indicators = len(crumbly_indicators) + len(porous_but_intact_indicators)
            
            # Final crumbly score (0 = definitely not crumbly, 1 = definitely crumbly)
            if porous_intact_evidence > crumbly_evidence:
                # Strong evidence for organized porosity
                crumbly_score = max(0, crumbly_evidence - porous_intact_evidence * 0.5)
            else:
                # Evidence leans toward crumbly
                crumbly_score = crumbly_evidence
            
            # Classification thresholds (adjusted for porosity awareness)
            if crumbly_score > 0.7 and crumbly_evidence > 0.6:
                classification = 'crumbly'
                confidence = min(0.95, 0.6 + crumbly_evidence)
            elif crumbly_score < 0.3 and porous_intact_evidence > 0.4:
                classification = 'porous_but_intact'  # More specific classification
                confidence = min(0.95, 0.6 + porous_intact_evidence)
            elif porous_intact_evidence > 0.6:
                classification = 'porous_but_intact'
                confidence = min(0.9, 0.5 + porous_intact_evidence)
            else:
                classification = 'intermediate'
                confidence = 0.3 + abs(0.5 - crumbly_score)
            
            # Boost confidence if we have many indicators
            if total_indicators > 4:
                confidence = min(0.98, confidence + 0.1)
            
            return {
                'classification': classification,
                'confidence': confidence,
                'crumbly_score': crumbly_score,
                'crumbly_evidence': crumbly_evidence,
                'porous_intact_evidence': porous_intact_evidence,
                'num_crumbly_indicators': len(crumbly_indicators),
                'num_intact_indicators': len(porous_but_intact_indicators),
                'confidence_factors': confidence_factors
            }
        
        except Exception as e:
            return {
                'classification': 'error',
                'confidence': 0.0,
                'crumbly_score': 0.5,
                'error': str(e)
            }
    
    def _classify_crumbly_texture(self, boundary_metrics: Dict, surface_metrics: Dict,
                                lbp_metrics: Dict, glcm_metrics: Dict, edge_metrics: Dict) -> Dict:
        """Original classification method (fallback)."""
        
        try:
            # Simplified classification for fallback
            crumbly_score = 0.5  # Default intermediate
            
            # Basic boundary analysis
            if 'outer_boundary' in boundary_metrics:
                outer = boundary_metrics['outer_boundary']
                circularity = outer.get('circularity', 0.5)
                roughness = outer.get('roughness_index', 0.5)
                
                # Lower circularity and higher roughness = more crumbly
                boundary_score = (1.0 - circularity + roughness) / 2
                crumbly_score = boundary_score
            
            # Determine classification
            if crumbly_score > 0.6:
                classification = 'crumbly'
                confidence = 0.7
            elif crumbly_score < 0.4:
                classification = 'smooth'
                confidence = 0.7
            else:
                classification = 'intermediate'
                confidence = 0.5
            
            return {
                'classification': classification,
                'confidence': confidence,
                'crumbly_score': crumbly_score,
                'confidence_factors': [f"Boundary score: {crumbly_score:.2f}"]
            }
        
        except Exception as e:
            return {
                'classification': 'error',
                'confidence': 0.0,
                'crumbly_score': 0.5,
                'error': str(e)
            }
    
    def _create_debug_visualization(self, image: np.ndarray, fiber_mask: np.ndarray,
                                  lumen_mask: Optional[np.ndarray], results: Dict):
        """Create debug visualization showing texture analysis results."""
        
        try:
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
                for factor in confidence_factors[:5]:  # Limit to first 5
                    summary_text += f"  {factor}\n"
            
            axes[5].text(0.05, 0.95, summary_text, transform=axes[5].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
            axes[5].set_title('Analysis Results')
            axes[5].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Visualization error: {e}")

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

# Test function for quick validation
def test_crumbly_detector():
    """Quick test function to validate the detector works."""
    print("Testing CrumblyDetector...")
    
    try:
        # Create a simple test image and mask
        test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        test_mask = np.zeros((200, 200), dtype=bool)
        test_mask[50:150, 50:150] = True  # Simple square mask
        
        detector = CrumblyDetector(porosity_aware=True)
        results = detector.analyze_crumbly_texture(
            test_image, test_mask, scale_factor=1.0, debug=False
        )
        
        print(f"✅ Test passed!")
        print(f"   Classification: {results['classification']}")
        print(f"   Confidence: {results['confidence']:.3f}")
        print(f"   Crumbly Score: {results['crumbly_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick test
    test_crumbly_detector()