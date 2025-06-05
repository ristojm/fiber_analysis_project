#!/usr/bin/env python3
"""
Porosity Analysis Testing Script - Fixed Version

This script tests the porosity analysis functionality with correct imports
for the fiber_analysis_project directory structure.

Usage from project root:
    python tests/test_porosity_analysis_fixed.py

Author: Fiber Analysis Project
Date: 2025
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Test imports first
def test_imports():
    """Test if the porosity analysis module can be imported."""
    try:
        # Try to import from modules directory
        from modules.porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity, quick_porosity_check
        print("✅ Successfully imported porosity analysis modules")
        return True, PorosityAnalyzer
    except ImportError as e:
        print(f"❌ Could not import porosity modules: {e}")
        print("Creating mock analyzer for testing...")
        return False, None

def create_synthetic_fiber_data(shape=(512, 512), fiber_type='hollow_fiber', 
                               porosity_percent=15.0, pore_count=40, noise_level=0.1):
    """
    Generate synthetic SEM-like fiber image with known ground truth porosity.
    """
    height, width = shape
    center_y, center_x = height // 2, width // 2
    
    # Create fiber mask
    y, x = np.ogrid[:height, :width]
    
    if fiber_type == 'hollow_fiber':
        # Hollow fiber with lumen
        outer_radius_y = height * 0.35
        outer_radius_x = width * 0.35
        fiber_outer = ((x - center_x) ** 2 / outer_radius_x ** 2 + 
                      (y - center_y) ** 2 / outer_radius_y ** 2) <= 1
        
        # Inner lumen
        lumen_radius_y = outer_radius_y * 0.3
        lumen_radius_x = outer_radius_x * 0.3
        lumen_mask = ((x - center_x) ** 2 / lumen_radius_x ** 2 + 
                     (y - center_y) ** 2 / lumen_radius_y ** 2) <= 1
        
        fiber_mask = fiber_outer & ~lumen_mask
        
        # Create base image
        image = np.ones(shape, dtype=np.float32) * 0.7
        image[fiber_mask] = 0.9
        image[lumen_mask] = 0.2
        
    else:  # solid filament
        radius_y = height * 0.4
        radius_x = width * 0.4
        fiber_mask = ((x - center_x) ** 2 / radius_x ** 2 + 
                     (y - center_y) ** 2 / radius_y ** 2) <= 1
        
        # Create base image
        image = np.ones(shape, dtype=np.float32) * 0.7
        image[fiber_mask] = 0.9
    
    # Add synthetic pores
    pore_mask = np.zeros_like(fiber_mask, dtype=bool)
    pores_added = 0
    target_pore_area = (porosity_percent / 100) * np.sum(fiber_mask)
    current_pore_area = 0
    
    for attempt in range(pore_count * 5):  # Try multiple times
        if pores_added >= pore_count or current_pore_area >= target_pore_area:
            break
            
        # Random position within fiber
        fiber_coords = np.where(fiber_mask)
        if len(fiber_coords[0]) == 0:
            break
            
        idx = np.random.randint(len(fiber_coords[0]))
        pore_y, pore_x = fiber_coords[0][idx], fiber_coords[1][idx]
        
        # Random pore size
        radius = np.random.randint(3, 15)
        
        # Create circular pore
        pore_candidate = ((x - pore_x) ** 2 + (y - pore_y) ** 2) <= radius ** 2
        pore_candidate = pore_candidate & fiber_mask
        
        # Check overlap
        if np.sum(pore_candidate) > 0:
            overlap_ratio = np.sum(pore_candidate & pore_mask) / np.sum(pore_candidate)
            
            if overlap_ratio < 0.3 and np.sum(pore_candidate) > 4:
                pore_mask |= pore_candidate
                image[pore_candidate] = np.random.uniform(0.1, 0.3)
                pores_added += 1
                current_pore_area += np.sum(pore_candidate)
    
    # Add noise
    noise = np.random.normal(0, noise_level, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Calculate ground truth
    actual_porosity = (np.sum(pore_mask) / np.sum(fiber_mask)) * 100 if np.sum(fiber_mask) > 0 else 0
    
    ground_truth = {
        'porosity_percent': actual_porosity,
        'pore_count': pores_added,
        'pore_mask': pore_mask,
        'fiber_mask': fiber_mask,
        'total_pore_area': np.sum(pore_mask),
        'total_fiber_area': np.sum(fiber_mask)
    }
    
    return image_uint8, fiber_mask, ground_truth

def simple_porosity_analysis(image, fiber_mask):
    """
    Simple porosity analysis implementation for testing when main module isn't available.
    """
    try:
        from skimage import filters, measure, morphology
        from skimage.morphology import remove_small_objects
    except ImportError:
        print("❌ scikit-image not available. Cannot run analysis.")
        return None
    
    # Basic analysis within fiber regions
    fiber_region = image.astype(np.float32) / 255.0
    fiber_pixels = fiber_region[fiber_mask > 0]
    
    if len(fiber_pixels) == 0:
        return {'porosity_percent': 0, 'pore_count': 0, 'pore_mask': np.zeros_like(fiber_mask)}
    
    # Otsu thresholding
    threshold = filters.threshold_otsu(fiber_pixels)
    pore_threshold = threshold * 0.7  # More aggressive for pore detection
    
    # Create pore mask
    pore_mask = (fiber_region < pore_threshold) & fiber_mask
    
    # Clean up small objects
    pore_mask = remove_small_objects(pore_mask, min_size=5)
    
    # Label and count pores
    pore_labels = measure.label(pore_mask)
    pore_count = np.max(pore_labels)
    
    # Calculate porosity
    total_pore_area = np.sum(pore_mask)
    total_fiber_area = np.sum(fiber_mask)
    porosity_percent = (total_pore_area / total_fiber_area) * 100 if total_fiber_area > 0 else 0
    
    # Get pore properties
    props = measure.regionprops(pore_labels)
    pore_areas = [prop.area for prop in props] if props else []
    
    return {
        'porosity_percent': porosity_percent,
        'pore_count': pore_count,
        'pore_mask': pore_mask,
        'pore_areas': pore_areas,
        'total_pore_area': total_pore_area,
        'total_fiber_area': total_fiber_area,
        'mean_pore_area': np.mean(pore_areas) if pore_areas else 0
    }

def visualize_test_results(image, fiber_mask, ground_truth, analysis_results, title="Porosity Analysis Test"):
    """Create visualization of test results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Synthetic SEM Image')
    axes[0, 0].axis('off')
    
    # Fiber mask
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].contour(fiber_mask, colors='red', linewidths=2)
    axes[0, 1].set_title('Fiber Region')
    axes[0, 1].axis('off')
    
    # Ground truth pores
    axes[0, 2].imshow(image, cmap='gray')
    axes[0, 2].contour(ground_truth['pore_mask'], colors='yellow', linewidths=1)
    axes[0, 2].contour(fiber_mask, colors='red', linewidths=2)
    axes[0, 2].set_title(f"Ground Truth\\n{ground_truth['porosity_percent']:.1f}% porosity")
    axes[0, 2].axis('off')
    
    # Detected pores
    axes[1, 0].imshow(image, cmap='gray')
    if analysis_results and 'pore_mask' in analysis_results:
        axes[1, 0].contour(analysis_results['pore_mask'], colors='cyan', linewidths=1)
    axes[1, 0].contour(fiber_mask, colors='red', linewidths=2)
    detected_porosity = analysis_results['porosity_percent'] if analysis_results else 0
    axes[1, 0].set_title(f"Detected Pores\\n{detected_porosity:.1f}% porosity")
    axes[1, 0].axis('off')
    
    # Size distribution
    if analysis_results and analysis_results['pore_areas']:
        axes[1, 1].hist(analysis_results['pore_areas'], bins=15, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Pore Area (pixels)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Detected Pore Sizes')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No pores detected', ha='center', va='center', 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Pore Size Distribution')
    
    # Results comparison
    if analysis_results:
        porosity_error = abs(ground_truth['porosity_percent'] - analysis_results['porosity_percent'])
        count_error = abs(ground_truth['pore_count'] - analysis_results['pore_count'])
        
        results_text = f"""COMPARISON RESULTS:

Ground Truth:
  Porosity: {ground_truth['porosity_percent']:.2f}%
  Pore Count: {ground_truth['pore_count']}

Detected:
  Porosity: {analysis_results['porosity_percent']:.2f}%
  Pore Count: {analysis_results['pore_count']}

Errors:
  Porosity Error: {porosity_error:.2f}%
  Count Error: {count_error}
  
Relative Error: {(porosity_error/ground_truth['porosity_percent']*100):.1f}%"""
    else:
        results_text = "Analysis failed - check imports and dependencies"
    
    axes[1, 2].text(0.05, 0.95, results_text, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontsize=9, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 2].set_title('Results Summary')
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def run_porosity_test():
    """Main test function."""
    print("="*60)
    print("POROSITY ANALYSIS TESTING")
    print("="*60)
    
    # Test imports
    print("\\n1. Testing imports...")
    imports_ok, PorosityAnalyzer = test_imports()
    
    # Generate test data
    print("\\n2. Generating synthetic test data...")
    np.random.seed(42)  # For reproducible results
    
    image, fiber_mask, ground_truth = create_synthetic_fiber_data(
        shape=(512, 512),
        fiber_type='hollow_fiber',
        porosity_percent=15.0,
        pore_count=35,
        noise_level=0.08
    )
    
    print(f"✅ Generated synthetic data:")
    print(f"   - Image shape: {image.shape}")
    print(f"   - Fiber area: {ground_truth['total_fiber_area']} pixels")
    print(f"   - Ground truth porosity: {ground_truth['porosity_percent']:.2f}%")
    print(f"   - Ground truth pore count: {ground_truth['pore_count']}")
    
    # Run analysis
    print("\\n3. Running porosity analysis...")
    
    if imports_ok and PorosityAnalyzer:
        try:
            # Use the full porosity analyzer
            analyzer = PorosityAnalyzer()
            results = analyzer.analyze_porosity(
                image=image,
                fiber_mask=fiber_mask,
                scale_factor=0.1,  # 0.1 μm/pixel
                fiber_type='hollow_fiber'
            )
            analysis_results = results['porosity_metrics']
            analysis_results['pore_mask'] = results['pore_mask']
            analysis_results['pore_areas'] = [p['area_pixels'] for p in results['pore_properties']]
            print("✅ Used full PorosityAnalyzer")
            
        except Exception as e:
            print(f"❌ Error with PorosityAnalyzer: {e}")
            print("Falling back to simple analysis...")
            analysis_results = simple_porosity_analysis(image, fiber_mask)
    else:
        # Use simple analysis
        analysis_results = simple_porosity_analysis(image, fiber_mask)
        print("✅ Used simple porosity analysis")
    
    # Display results
    if analysis_results:
        print("\\n4. Analysis Results:")
        print(f"   - Detected porosity: {analysis_results['porosity_percent']:.2f}%")
        print(f"   - Detected pore count: {analysis_results['pore_count']}")
        print(f"   - Porosity error: {abs(ground_truth['porosity_percent'] - analysis_results['porosity_percent']):.2f}%")
        print(f"   - Count error: {abs(ground_truth['pore_count'] - analysis_results['pore_count'])}")
        
        # Calculate accuracy metrics
        relative_error = abs(ground_truth['porosity_percent'] - analysis_results['porosity_percent']) / ground_truth['porosity_percent'] * 100
        print(f"   - Relative error: {relative_error:.1f}%")
        
        # Visualize results
        print("\\n5. Creating visualization...")
        visualize_test_results(image, fiber_mask, ground_truth, analysis_results)
        
    else:
        print("❌ Analysis failed")
    
    # Run multiple test cases
    print("\\n6. Running multiple test cases...")
    run_accuracy_evaluation()
    
    print("\\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)

def run_accuracy_evaluation(num_tests=5):
    """Run multiple tests to evaluate accuracy."""
    print(f"Testing accuracy with {num_tests} different synthetic samples...")
    
    errors = []
    count_errors = []
    
    for i in range(num_tests):
        # Generate test case with random parameters
        target_porosity = np.random.uniform(8, 25)
        target_count = int(np.random.uniform(20, 60))
        noise = np.random.uniform(0.05, 0.15)
        
        # Generate data
        image, fiber_mask, ground_truth = create_synthetic_fiber_data(
            porosity_percent=target_porosity,
            pore_count=target_count,
            noise_level=noise
        )
        
        # Analyze
        results = simple_porosity_analysis(image, fiber_mask)
        
        if results:
            porosity_error = abs(ground_truth['porosity_percent'] - results['porosity_percent'])
            count_error = abs(ground_truth['pore_count'] - results['pore_count'])
            
            errors.append(porosity_error)
            count_errors.append(count_error)
            
            print(f"  Test {i+1}: GT={ground_truth['porosity_percent']:.1f}%, "
                  f"Detected={results['porosity_percent']:.1f}%, "
                  f"Error={porosity_error:.1f}%")
    
    if errors:
        print(f"\\nAccuracy Summary:")
        print(f"  Mean porosity error: {np.mean(errors):.2f}%")
        print(f"  Max porosity error: {np.max(errors):.2f}%")
        print(f"  Mean count error: {np.mean(count_errors):.1f} pores")
        print(f"  Success rate (error < 5%): {(np.array(errors) < 5).mean()*100:.0f}%")

if __name__ == "__main__":
    # Ensure we can create plots
    plt.ion()  # Turn on interactive mode
    
    # Run the test
    run_porosity_test()