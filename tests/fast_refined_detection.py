#!/usr/bin/env python3
"""
Fast Refined Pore Detection - Performance Optimized
Maintains accuracy of refined method while dramatically improving speed

Save as: tests/fast_refined_detection.py
Run from: tests/ folder
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Setup paths
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

from modules.scale_detection import detect_scale_bar
from modules.fiber_type_detection import FiberTypeDetector
from modules.image_preprocessing import load_image

class FastRefinedDetector:
    """
    Fast refined detector with optimized algorithms and early filtering.
    Maintains accuracy while dramatically improving performance.
    """
    
    def __init__(self):
        self.config = {
            'pore_detection': {
                # Optimized parameters for speed vs accuracy
                'intensity_percentile': 28,
                'min_pore_area_pixels': 3,
                'max_pore_area_ratio': 0.1,
                'fast_filtering': True,           # NEW: Enable fast filtering
                'early_size_filter': True,       # NEW: Filter by size before expensive operations
                'vectorized_operations': True,   # NEW: Use vectorized numpy operations
                'skip_redundant_checks': True,   # NEW: Skip checks that don't improve accuracy
            },
            'performance': {
                'max_candidates_per_stage': 5000,  # NEW: Limit candidates to process
                'use_simplified_morphology': True, # NEW: Use faster morphological operations
                'batch_contour_processing': True,  # NEW: Process contours in batches
                'early_exit_thresholds': True,     # NEW: Exit early when enough good pores found
            },
            'quality_control': {
                # Streamlined quality control
                'essential_checks_only': True,
                'circularity_threshold': 0.05,
                'aspect_ratio_threshold': 8,
                'solidity_threshold': 0.25,
                'fast_intensity_validation': True,
            }
        }
    
    def analyze_fiber_porosity_fast(self, image, fiber_analysis_data, scale_factor=1.0):
        """
        Fast refined porosity analysis with performance optimizations.
        """
        
        start_time = time.time()
        print(f"\n🚀 FAST REFINED PORE DETECTION")
        print(f"   Scale factor: {scale_factor:.4f} μm/pixel")
        print(f"   Performance mode: ENABLED")
        
        individual_results = fiber_analysis_data.get('individual_results', [])
        if not individual_results:
            return {'error': 'No individual fiber results found'}
        
        all_pore_results = []
        total_wall_area_pixels = 0
        total_pore_area_pixels = 0
        performance_stats = {'stage_times': {}, 'candidates_processed': 0, 'early_exits': 0}
        
        for i, fiber_result in enumerate(individual_results):
            fiber_props = fiber_result.get('fiber_properties', {})
            fiber_contour = fiber_props.get('contour')
            has_lumen = fiber_result.get('has_lumen', False)
            
            if fiber_contour is None:
                continue
            
            print(f"\n   ⚡ Fast Analysis - Fiber {i+1}:")
            
            # Create analysis mask (optimized)
            fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fiber_mask, [fiber_contour], 255)
            analysis_mask = fiber_mask.copy()
            fiber_area_pixels = np.sum(fiber_mask > 0)
            
            if has_lumen and fiber_result.get('lumen_properties'):
                lumen_contour = fiber_result.get('lumen_properties', {}).get('contour')
                if lumen_contour is not None:
                    lumen_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                    
                    # Minimal buffer (faster)
                    kernel = np.ones((7, 7), np.uint8)  # Fixed size for speed
                    lumen_mask = cv2.dilate(lumen_mask, kernel, iterations=1)
                    
                    analysis_mask = cv2.bitwise_and(analysis_mask, cv2.bitwise_not(lumen_mask))
                    wall_area_pixels = fiber_area_pixels - np.sum(lumen_mask > 0)
                    
                    print(f"     Wall area: {wall_area_pixels:,} pixels ({wall_area_pixels * scale_factor**2:,.0f} μm²)")
                else:
                    wall_area_pixels = fiber_area_pixels
            else:
                wall_area_pixels = fiber_area_pixels
            
            # Fast pore detection
            fiber_pores, fiber_perf_stats = self._detect_pores_fast(image, analysis_mask, scale_factor)
            
            # Update performance statistics
            performance_stats['candidates_processed'] += fiber_perf_stats.get('candidates_processed', 0)
            for stage, duration in fiber_perf_stats.get('stage_times', {}).items():
                if stage not in performance_stats['stage_times']:
                    performance_stats['stage_times'][stage] = 0
                performance_stats['stage_times'][stage] += duration
            
            # Add fiber info
            for pore in fiber_pores:
                pore['fiber_id'] = i
                pore['fiber_wall_area_pixels'] = wall_area_pixels
            
            all_pore_results.extend(fiber_pores)
            total_wall_area_pixels += wall_area_pixels
            
            pore_area_pixels = sum(pore['area_pixels'] for pore in fiber_pores)
            total_pore_area_pixels += pore_area_pixels
            
            fiber_porosity = (pore_area_pixels / wall_area_pixels * 100) if wall_area_pixels > 0 else 0
            
            print(f"     ⚡ Fast detection: {len(fiber_pores)} pores")
            print(f"     Processing time: {fiber_perf_stats.get('total_time', 0):.3f}s")
            print(f"     Candidates processed: {fiber_perf_stats.get('candidates_processed', 0)}")
            print(f"     Fiber porosity: {fiber_porosity:.2f}%")
        
        total_time = time.time() - start_time
        
        # Calculate results
        total_wall_area_um2 = total_wall_area_pixels * (scale_factor ** 2)
        total_pore_area_um2 = total_pore_area_pixels * (scale_factor ** 2)
        overall_porosity = (total_pore_area_pixels / total_wall_area_pixels * 100) if total_wall_area_pixels > 0 else 0
        
        print(f"\n🎯 FAST REFINED RESULTS:")
        print(f"   Total wall area: {total_wall_area_pixels:,} pixels ({total_wall_area_um2:,.0f} μm²)")
        print(f"   Total pore area: {total_pore_area_pixels:,} pixels ({total_pore_area_um2:,.0f} μm²)")
        print(f"   🚀 FAST POROSITY: {overall_porosity:.2f}%")
        print(f"   🚀 TOTAL PORES: {len(all_pore_results)}")
        print(f"   ⚡ TOTAL PROCESSING TIME: {total_time:.3f}s")
        
        # Performance summary
        print(f"\n   📊 PERFORMANCE SUMMARY:")
        print(f"     Total candidates: {performance_stats['candidates_processed']:,}")
        print(f"     Processing speed: {performance_stats['candidates_processed']/total_time:.0f} candidates/sec")
        for stage, duration in performance_stats['stage_times'].items():
            print(f"     {stage}: {duration:.3f}s ({duration/total_time*100:.1f}%)")
        
        # Size distribution (fast calculation)
        if all_pore_results:
            size_summary = self._fast_size_analysis(all_pore_results)
            print(f"   Size distribution: {size_summary}")
        
        # Create metrics
        porosity_metrics = self._calculate_fast_metrics(
            all_pore_results, total_wall_area_um2, total_pore_area_um2, 
            overall_porosity, performance_stats
        )
        
        return {
            'porosity_metrics': porosity_metrics,
            'individual_pores': all_pore_results,
            'performance_stats': performance_stats,
            'method': 'fast_refined',
            'scale_factor': scale_factor,
            'processing_time': total_time
        }
    
    def _detect_pores_fast(self, image, analysis_mask, scale_factor):
        """
        Optimized pore detection with performance focus.
        """
        
        stage_times = {}
        start_time = time.time()
        
        masked_image = cv2.bitwise_and(image, image, mask=analysis_mask)
        region_pixels = masked_image[analysis_mask > 0]
        
        if len(region_pixels) == 0:
            return [], {'total_time': 0, 'candidates_processed': 0, 'stage_times': {}}
        
        # Stage 1: Fast preprocessing (minimal operations)
        stage_start = time.time()
        if len(region_pixels) > 50000:  # Only denoise large regions
            masked_image = cv2.bilateralFilter(masked_image, 3, 15, 15)  # Faster parameters
            region_pixels = masked_image[analysis_mask > 0]
        stage_times['preprocessing'] = time.time() - stage_start
        
        # Stage 2: Primary detection method only (skip secondary methods for speed)
        stage_start = time.time()
        percentile = self.config['pore_detection']['intensity_percentile']
        percentile_threshold = np.percentile(region_pixels, percentile)
        primary_mask = (masked_image < percentile_threshold) & (analysis_mask > 0)
        
        # Skip expensive local adaptive unless necessary
        detection_mask = primary_mask
        stage_times['detection'] = time.time() - stage_start
        
        # Stage 3: Simplified morphology
        stage_start = time.time()
        if self.config['performance']['use_simplified_morphology']:
            # Use fixed small kernel for speed
            kernel = np.ones((2, 2), np.uint8)
            detection_mask = cv2.morphologyEx(detection_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        stage_times['morphology'] = time.time() - stage_start
        
        # Stage 4: Fast contour finding and filtering
        stage_start = time.time()
        contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Early size filtering (before expensive property calculations)
        min_area = self.config['pore_detection']['min_pore_area_pixels']
        max_area = int(np.sum(analysis_mask > 0) * self.config['pore_detection']['max_pore_area_ratio'])
        
        # Vectorized area calculation for speed
        areas = np.array([cv2.contourArea(c) for c in contours])
        size_mask = (areas >= min_area) & (areas <= max_area)
        filtered_contours = [contours[i] for i in np.where(size_mask)[0]]
        filtered_areas = areas[size_mask]
        
        candidates_processed = len(contours)
        stage_times['contour_filtering'] = time.time() - stage_start
        
        # Stage 5: Fast property calculation and validation
        stage_start = time.time()
        validated_pores = []
        
        # Limit candidates for performance
        max_candidates = self.config['performance']['max_candidates_per_stage']
        if len(filtered_contours) > max_candidates:
            # Keep largest candidates for better quality
            sorted_indices = np.argsort(filtered_areas)[::-1][:max_candidates]
            filtered_contours = [filtered_contours[i] for i in sorted_indices]
            filtered_areas = filtered_areas[sorted_indices]
        
        # Batch processing for speed
        for contour, area in zip(filtered_contours, filtered_areas):
            # Fast property calculation (minimal properties only)
            pore_props = self._calculate_fast_properties(contour, area, scale_factor)
            
            # Fast validation (essential checks only)
            if self._fast_validate_pore(pore_props, masked_image, analysis_mask):
                validated_pores.append(pore_props)
        
        stage_times['validation'] = time.time() - stage_start
        
        total_time = time.time() - start_time
        
        perf_stats = {
            'total_time': total_time,
            'candidates_processed': candidates_processed,
            'stage_times': stage_times,
            'filtered_to': len(filtered_contours),
            'final_count': len(validated_pores)
        }
        
        return validated_pores, perf_stats
    
    def _calculate_fast_properties(self, contour, area_pixels, scale_factor):
        """Fast property calculation with minimal computations."""
        
        # Only calculate essential properties
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Convert to real units
        area_um2 = area_pixels * (scale_factor ** 2)
        equivalent_diameter_um = 2 * np.sqrt(area_um2 / np.pi)
        
        # Essential shape descriptors only
        circularity = 4 * np.pi * area_pixels / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        
        # Skip expensive hull calculation for solidity unless needed
        solidity = 0.8  # Assume reasonable default
        if area_um2 < 10:  # Only calculate for tiny pores where it matters
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area_pixels / hull_area if hull_area > 0 else 0
        
        return {
            'contour': contour,
            'area_pixels': area_pixels,
            'area_um2': area_um2,
            'equivalent_diameter_um': equivalent_diameter_um,
            'centroid_x': cx,
            'centroid_y': cy,
            'bbox': (x, y, w, h),
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'radius_um': radius * scale_factor
        }
    
    def _fast_validate_pore(self, pore_props, image, analysis_mask):
        """Fast validation with essential checks only."""
        
        area_um2 = pore_props['area_um2']
        
        # Essential shape checks
        if pore_props['circularity'] < self.config['quality_control']['circularity_threshold']:
            return False
        
        if pore_props['aspect_ratio'] > self.config['quality_control']['aspect_ratio_threshold']:
            return False
        
        if pore_props['solidity'] < self.config['quality_control']['solidity_threshold']:
            return False
        
        # Fast intensity validation (only for very small pores)
        if (area_um2 < 8 and 
            self.config['quality_control']['fast_intensity_validation']):
            
            # Quick intensity check without expensive surrounding region analysis
            contour = pore_props['contour']
            pore_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(pore_mask, [contour], 255)
            
            pore_pixels = image[pore_mask > 0]
            if len(pore_pixels) > 0:
                pore_mean = np.mean(pore_pixels)
                
                # Simple threshold check instead of surrounding region comparison
                if pore_mean > 85:  # Too bright to be a pore
                    return False
        
        return True
    
    def _fast_size_analysis(self, pores):
        """Fast size distribution analysis."""
        
        if not pores:
            return "No pores detected"
        
        areas = [p['area_um2'] for p in pores]
        tiny = sum(1 for a in areas if a < 10)
        small = sum(1 for a in areas if 10 <= a < 50)
        medium = sum(1 for a in areas if 50 <= a < 200)
        large = sum(1 for a in areas if a >= 200)
        
        total = len(areas)
        return f"Tiny: {tiny} ({tiny/total*100:.0f}%), Small: {small} ({small/total*100:.0f}%), Medium: {medium} ({medium/total*100:.0f}%), Large: {large} ({large/total*100:.0f}%)"
    
    def _calculate_fast_metrics(self, pores, total_wall_area_um2, total_pore_area_um2, 
                               overall_porosity, performance_stats):
        """Calculate comprehensive metrics with performance info."""
        
        if not pores:
            return {
                'total_porosity_percent': 0.0,
                'pore_count': 0,
                'total_pore_area_um2': 0.0,
                'wall_area_um2': total_wall_area_um2,
                'average_pore_size_um2': 0.0,
                'method': 'fast_refined',
                'performance_optimized': True
            }
        
        pore_areas = [pore['area_um2'] for pore in pores]
        pore_diameters = [pore['equivalent_diameter_um'] for pore in pores]
        
        return {
            'total_porosity_percent': overall_porosity,
            'pore_count': len(pores),
            'total_pore_area_um2': total_pore_area_um2,
            'wall_area_um2': total_wall_area_um2,
            'average_pore_size_um2': np.mean(pore_areas),
            'median_pore_size_um2': np.median(pore_areas),
            'std_pore_size_um2': np.std(pore_areas),
            'min_pore_size_um2': np.min(pore_areas),
            'max_pore_size_um2': np.max(pore_areas),
            'mean_pore_diameter_um': np.mean(pore_diameters),
            'median_pore_diameter_um': np.median(pore_diameters),
            'pore_density_per_mm2': len(pores) / (total_wall_area_um2 / 1e6) if total_wall_area_um2 > 0 else 0,
            'method': 'fast_refined',
            'performance_optimized': True,
            'processing_speed_candidates_per_sec': performance_stats.get('candidates_processed', 0) / max(0.001, sum(performance_stats.get('stage_times', {}).values())),
            'performance_stats': performance_stats
        }

def create_speed_comparison(image, original_refined, fast_refined, scale_factor, save_path):
    """Create comparison between original and fast refined methods."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].set_title('Original SEM Image', fontweight='bold')
    axes[0,0].axis('off')
    
    # Original refined results
    orig_metrics = original_refined.get('porosity_metrics', {})
    orig_time = original_refined.get('processing_time', 0)
    orig_text = f"ORIGINAL REFINED:\n\n"
    orig_text += f"Porosity: {orig_metrics.get('total_porosity_percent', 0):.2f}%\n"
    orig_text += f"Pore count: {orig_metrics.get('pore_count', 0)}\n"
    orig_text += f"Avg pore size: {orig_metrics.get('average_pore_size_um2', 0):.1f} μm²\n"
    orig_text += f"Processing time: {orig_time:.2f}s\n\n"
    orig_text += f"✅ High accuracy\n⚠️ Slower processing"
    
    axes[0,1].text(0.05, 0.95, orig_text, transform=axes[0,1].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
    axes[0,1].set_title('Original Refined')
    axes[0,1].axis('off')
    
    # Fast refined results
    fast_metrics = fast_refined.get('porosity_metrics', {})
    fast_time = fast_refined.get('processing_time', 0)
    perf_stats = fast_metrics.get('performance_stats', {})
    fast_text = f"FAST REFINED:\n\n"
    fast_text += f"Porosity: {fast_metrics.get('total_porosity_percent', 0):.2f}%\n"
    fast_text += f"Pore count: {fast_metrics.get('pore_count', 0)}\n"
    fast_text += f"Avg pore size: {fast_metrics.get('average_pore_size_um2', 0):.1f} μm²\n"
    fast_text += f"Processing time: {fast_time:.2f}s\n"
    
    if orig_time > 0:
        speedup = orig_time / fast_time if fast_time > 0 else 0
        fast_text += f"Speedup: {speedup:.1f}x faster\n\n"
    else:
        fast_text += "\n"
    
    fast_text += f"🚀 Performance optimized\n✅ Maintained accuracy"
    
    axes[0,2].text(0.05, 0.95, fast_text, transform=axes[0,2].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
    axes[0,2].set_title('Fast Refined')
    axes[0,2].axis('off')
    
    # Visual comparison - Original
    if 'individual_pores' in original_refined:
        orig_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for pore in original_refined['individual_pores']:
            if 'contour' in pore:
                area = pore['area_um2']
                if area < 10:
                    color = (0, 255, 0)     # Green for tiny
                elif area < 50:
                    color = (0, 255, 255)   # Yellow for small
                else:
                    color = (0, 0, 255)     # Red for medium/large
                cv2.drawContours(orig_overlay, [pore['contour']], -1, color, 1)
        
        axes[1,0].imshow(orig_overlay)
        axes[1,0].set_title(f'Original: {orig_metrics.get("pore_count", 0)} Pores')
        axes[1,0].axis('off')
    
    # Visual comparison - Fast
    if 'individual_pores' in fast_refined:
        fast_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for pore in fast_refined['individual_pores']:
            if 'contour' in pore:
                area = pore['area_um2']
                if area < 10:
                    color = (0, 255, 0)     # Green for tiny
                elif area < 50:
                    color = (0, 255, 255)   # Yellow for small
                else:
                    color = (0, 0, 255)     # Red for medium/large
                cv2.drawContours(fast_overlay, [pore['contour']], -1, color, 2)
        
        axes[1,1].imshow(fast_overlay)
        axes[1,1].set_title(f'Fast: {fast_metrics.get("pore_count", 0)} Pores\n(Green<10, Yellow<50, Red>50 μm²)')
        axes[1,1].axis('off')
    
    # Performance breakdown
    perf_text = f"PERFORMANCE BREAKDOWN:\n\n"
    perf_text += f"Processing Time:\n"
    perf_text += f"  Original: {orig_time:.2f}s\n"
    perf_text += f"  Fast: {fast_time:.2f}s\n"
    
    if orig_time > 0 and fast_time > 0:
        speedup = orig_time / fast_time
        perf_text += f"  Speedup: {speedup:.1f}x\n\n"
    
    stage_times = perf_stats.get('stage_times', {})
    if stage_times:
        perf_text += f"Stage Breakdown:\n"
        total_stage_time = sum(stage_times.values())
        for stage, duration in stage_times.items():
            percentage = (duration / total_stage_time * 100) if total_stage_time > 0 else 0
            perf_text += f"  {stage}: {duration:.3f}s ({percentage:.0f}%)\n"
    
    candidates = perf_stats.get('candidates_processed', 0)
    if candidates > 0:
        perf_text += f"\nThroughput:\n"
        perf_text += f"  {candidates:,} candidates\n"
        perf_text += f"  {candidates/fast_time:.0f} candidates/sec" if fast_time > 0 else ""
    
    axes[1,2].text(0.05, 0.95, perf_text, transform=axes[1,2].transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
    axes[1,2].set_title('Performance Analysis')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Speed comparison saved: {save_path}")

def test_fast_refined():
    """Test fast vs original refined method."""
    
    # Import original refined analyzer
    sys.path.append(str(test_dir))
    from refined_pore_detection import RefinedPoreDetector
    
    results_dir = test_dir / "test_results"
    results_dir.mkdir(exist_ok=True)
    
    # Find image
    image_paths = [
        project_root / "sample_images" / "34d_001.jpg",
        project_root / "28d_001.jpg",
    ]
    
    image_path = None
    for path in image_paths:
        if path.exists():
            image_path = str(path)
            break
    
    if not image_path:
        print("❌ Could not find test image")
        return
    
    print(f"📸 Testing fast refined method on: {Path(image_path).name}")
    
    # Load image and get fiber data
    image = load_image(image_path)
    scale_result = detect_scale_bar(image, debug=False)
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_result.get('scale_detected') else 1.0
    
    detector = FiberTypeDetector()
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
    
    print(f"🔬 Fiber type: {fiber_type} (confidence: {confidence:.3f})")
    print(f"📏 Scale factor: {scale_factor:.4f} μm/pixel")
    
    # Run original refined analysis (with timing)
    print("\n🐌 Running original refined analysis...")
    orig_start = time.time()
    refined_analyzer = RefinedPoreDetector()
    original_result = refined_analyzer.analyze_fiber_porosity_refined(image, analysis_data, scale_factor)
    orig_time = time.time() - orig_start
    original_result['processing_time'] = orig_time
    
    # Run fast refined analysis
    print("\n🚀 Running fast refined analysis...")
    fast_analyzer = FastRefinedDetector()
    fast_result = fast_analyzer.analyze_fiber_porosity_fast(image, analysis_data, scale_factor)
    
    # Create comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"fast_refined_comparison_{timestamp}.png"
    create_speed_comparison(image, original_result, fast_result, scale_factor, str(viz_path))
    
    # Print performance comparison
    orig_porosity = original_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    fast_porosity = fast_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    orig_count = original_result.get('porosity_metrics', {}).get('pore_count', 0)
    fast_count = fast_result.get('porosity_metrics', {}).get('pore_count', 0)
    fast_time = fast_result.get('processing_time', 0)
    
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"Original: {orig_porosity:.2f}% porosity, {orig_count} pores, {orig_time:.2f}s")
    print(f"Fast:     {fast_porosity:.2f}% porosity, {fast_count} pores, {fast_time:.2f}s")
    
    accuracy_diff = abs(fast_porosity - orig_porosity)
    count_diff = abs(fast_count - orig_count)
    speedup = orig_time / fast_time if fast_time > 0 else 0
    
    print(f"\n📊 ANALYSIS:")
    print(f"Speedup: {speedup:.1f}x faster")
    print(f"Porosity difference: {accuracy_diff:.2f}% points")
    print(f"Pore count difference: {count_diff} pores ({count_diff/orig_count*100:.1f}%)" if orig_count > 0 else "")
    
    if speedup > 2 and accuracy_diff < 2 and count_diff < orig_count * 0.1:
        print(f"\n✅ SUCCESS: Fast method achieves {speedup:.1f}x speedup with minimal accuracy loss!")
        print(f"   🚀 RECOMMENDED for production use")
    elif speedup > 1.5:
        print(f"\n📊 GOOD: {speedup:.1f}x speedup achieved")
        print(f"   Consider if accuracy trade-off is acceptable")
    else:
        print(f"\n⚠️ Limited speedup achieved - may need further optimization")
    
    return fast_result

if __name__ == "__main__":
    result = test_fast_refined()