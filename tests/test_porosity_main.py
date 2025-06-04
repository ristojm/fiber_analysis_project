#!/usr/bin/env python3
"""
Standalone Porosity Analysis Testing Script

This script demonstrates and tests the porosity analysis functionality
without requiring Jupyter notebooks. It can be run directly from the command line.

Usage:
    python test_porosity_main.py

Author: Fiber Analysis Project
Date: 2025
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add the modules directory to Python path if needed
current_dir = Path(__file__).parent
modules_dir = current_dir / 'modules'
if modules_dir.exists():
    sys.path.insert(0, str(modules_dir))

def test_imports():
    """Test if all required modules can be imported."""
    try:
        # These would be your actual module imports
        # from porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity, quick_porosity_check
        # from image_preprocessing import load_and_preprocess
        # from fiber_type_detection import detect_fiber_type
        # from scale_detection import detect_scale_bar
        print("✅ Module imports test passed (commented out for demo)")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all modules are in the correct path")
        return False

def create_synthetic_test_data():
    """Create synthetic SEM-like data for testing porosity analysis."""
    print("Generating synthetic test data...")
    
    # Image parameters
    height, width = 512, 512
    center_y, center_x = height // 2, width // 2
    
    # Create hollow fiber mask
    y, x = np.ogrid[:height, :width]
    
    # Outer fiber boundary
    outer_radius_y = height * 0.35
    outer_radius_x = width * 0.35
    fiber_outer = ((x - center_x) ** 2 / outer_radius_x ** 2 + 
                   (y - center_y) ** 2 / outer_radius_y ** 2) <= 1
    
    # Inner lumen
    lumen_radius_y = outer_radius_y * 0.3
    lumen_radius_x = outer_radius_x * 0.3
    lumen_mask = ((x - center_x) ** 2 / lumen_radius_x ** 2 + 
                  (y - center_y) ** 2 / lumen_radius_y ** 2) <= 1
    
    # Fiber wall (excluding lumen)
    fiber_mask = fiber_outer & ~lumen_mask
    
    # Create base image
    image = np.ones((height, width), dtype=np.float32) * 0.7  # Background
    image[fiber_mask] = 0.9  # Fiber material
    image[lumen_mask] = 0.2  # Lumen
    
    # Add synthetic pores
    pore_mask = np.zeros_like(fiber_mask, dtype=bool)
    pore_count = 0
    target_pores = 30
    
    # Generate random pores
    for _ in range(target_pores * 3):  # Try more times to get target count
        # Random position within fiber
        fiber_coords = np.where(fiber_mask)
        if len(fiber_coords[0]) == 0:
            break
            
        idx = np.random.randint(len(fiber_coords[0]))
        pore_y, pore_x = fiber_coords[0][idx], fiber_coords[1][idx]
        
        # Random pore size
        radius = np.random.randint(3, 12)
        
        # Create circular pore
        pore_candidate = ((x - pore_x) ** 2 + (y - pore_y) ** 2) <= radius ** 2
        pore_candidate = pore_candidate & fiber_mask
        
        # Check overlap with existing pores
        overlap = np.sum(pore_candidate & pore_mask) / np.sum(pore_candidate) if np.sum(pore_candidate) > 0 else 1
        
        if overlap < 0.3 and np.sum(pore_candidate) > 4:
            pore_mask |= pore_candidate
            image[pore_candidate] = np.random.uniform(0.1, 0.3)
            pore_count += 1
            
            if pore_count >= target_pores:
                break
    
    # Add noise and SEM-like effects
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Calculate ground truth
    total_pore_area = np.sum(pore_mask)
    total_fiber_area = np.sum(fiber_mask)
    actual_porosity = (total_pore_area / total_fiber_area) * 100 if total_fiber_area > 0 else 0
    
    ground_truth = {
        'porosity_percent': actual_porosity,
        'pore_count': pore_count,
        'pore_mask': pore_mask,
        'total_pore_area': total_pore_area,
        'total_fiber_area': total_fiber_area
    }
    
    print(f"✅ Generated synthetic data:")
    print(f"   - Image size: {image_uint8.shape}")
    print(f"   - Actual porosity: {actual_porosity:.2f}%")
    print(f"   - Pore count: {pore_count}")
    
    return image_uint8, fiber_mask, ground_truth

def basic_porosity_analysis(image, fiber_mask):
    """
    Basic porosity analysis using simple thresholding.
    This is a simplified version for demonstration purposes.
    """
    print("Running basic porosity analysis...")
    
    # Restrict analysis to fiber regions
    fiber_region = image * fiber_mask
    
    # Basic Otsu thresholding for pore detection
    fiber_pixels = fiber_region[fiber_mask > 0]
    if len(fiber_pixels) == 0:
        return {'porosity_percent': 0, 'pore_count': 0}
    
    # Calculate threshold
    from skimage import filters
    threshold = filters.threshold_otsu(fiber_pixels)
    
    # More aggressive threshold for pore detection
    pore_threshold = threshold * 0.7
    pore_mask = (fiber_region < pore_threshold) & fiber_mask
    
    # Remove small objects
    from skimage.morphology import remove_small_objects
    pore_mask = remove_small_objects(pore_mask, min_size=5)
    
    # Label connected components
    from skimage import measure
    pore_labels = measure.label(pore_mask)
    
    # Calculate metrics
    total_pore_area = np.sum(pore_mask)
    total_fiber_area = np.sum(fiber_mask)
    porosity_percent = (total_pore_area / total_fiber_area) * 100 if total_fiber_area > 0 else 0
    pore_count = np.max(pore_labels)
    
    # Extract pore properties
    props = measure.regionprops(pore_labels)
    pore_areas = [prop.area for prop in props]
    pore_diameters = [prop.equivalent_diameter for prop in props]
    
    results = {
        'porosity_percent': porosity_percent,
        'pore_count': pore_count,
        'pore_mask': pore_mask,
        'pore_areas': pore_areas,
        'pore_diameters': pore_diameters,
        'total_pore_area': total_pore_area,
        'total_fiber_area': total_fiber_area,
        'mean_pore_area': np.mean(pore_areas) if pore_areas else 0,
        'std_pore_area': np.std(pore_areas) if pore_areas else 0
    }
    
    return results

def visualize_results(image, fiber_mask, ground_truth, analysis_results, save_path=None):
    """Create comprehensive visualization of porosity analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original SEM Image')
    axes[0, 0].axis('off')
    
    # 2. Fiber mask
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].contour(fiber_mask, colors='red', linewidths=2)
    axes[0, 1].set_title('Fiber Region')
    axes[0, 1].axis('off')
    
    # 3. Ground truth pores
    axes[0, 2].imshow(image, cmap='gray')
    if 'pore_mask' in ground_truth:
        axes[0, 2].contour(ground_truth['pore_mask'], colors='yellow', linewidths=1)
    axes[0, 2].contour(fiber_mask, colors='red', linewidths=2)
    axes[0, 2].set_title(f"Ground Truth Pores\\n{ground_truth['porosity_percent']:.1f}% porosity")
    axes[0, 2].axis('off')
    
    # 4. Detected pores
    axes[1, 0].imshow(image, cmap='gray')
    if 'pore_mask' in analysis_results:
        axes[1, 0].contour(analysis_results['pore_mask'], colors='cyan', linewidths=1)
    axes[1, 0].contour(fiber_mask, colors='red', linewidths=2)
    axes[1, 0].set_title(f"Detected Pores\\n{analysis_results['porosity_percent']:.1f}% porosity")
    axes[1, 0].axis('off')
    
    # 5. Pore size distribution
    if analysis_results['pore_areas']:
        axes[1, 1].hist(analysis_results['pore_areas'], bins=15, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Pore Area (pixels)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Pore Size Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No pores detected', ha='center', va='center', 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Pore Size Distribution')
    
    # 6. Comparison table
    comparison_data = [
        ['Metric', 'Ground Truth', 'Detected', 'Error'],
        ['Porosity (%)', f"{ground_truth['porosity_percent']:.2f}", 
         f"{analysis_results['porosity_percent']:.2f}", 
         f"{abs(ground_truth['porosity_percent'] - analysis_results['porosity_percent']):.2f}"],
        ['Pore Count', str(ground_truth['pore_count']), 
         str(analysis_results['pore_count']), 
         str(abs(ground_truth['pore_count'] - analysis_results['pore_count']))],
        ['Mean Area (px)', 
         f"{ground_truth['total_pore_area']/ground_truth['pore_count']:.1f}" if ground_truth['pore_count'] > 0 else "0",
         f"{analysis_results['mean_pore_area']:.1f}",
         "N/A"]
    ]
    
    table = axes[1, 2].table(cellText=comparison_data[1:], 
                            colLabels=comparison_data[0],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[1, 2].set_title('Results Comparison')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def run_accuracy_test(num_tests=5):
    """Run multiple tests to evaluate accuracy."""
    print(f"\\nRunning accuracy test with {num_tests} synthetic samples...")
    
    errors = []
    count_errors = []
    
    for i in range(num_tests):
        print(f"Test {i+1}/{num_tests}")
        
        # Generate test data with varying parameters
        target_porosity = np.random.uniform(5, 25)  # Random porosity between 5-25%
        
        # Create synthetic data (simplified version)
        image, fiber_mask, ground_truth = create_synthetic_test_data()
        
        # Run analysis
        results = basic_porosity_analysis(image, fiber_mask)
        
        # Calculate errors
        porosity_error = abs(ground_truth['porosity_percent'] - results['porosity_percent'])
        count_error = abs(ground_truth['pore_count'] - results['pore_count'])
        
        errors.append(porosity_error)
        count_errors.append(count_error)
        
        print(f"  Ground truth: {ground_truth['porosity_percent']:.2f}%, "
              f"Detected: {results['porosity_percent']:.2f}%, "
              f"Error: {porosity_error:.2f}%")
    
    # Summary statistics
    print("\\n" + "="*50)
    print("ACCURACY TEST SUMMARY")
    print("="*50)
    print(f"Mean porosity error: {np.mean(errors):.2f}%")
    print(f"Std porosity error: {np.std(errors):.2f}%")
    print(f"Max porosity error: {np.max(errors):.2f}%")
    print(f"Mean count error: {np.mean(count_errors):.1f} pores")
    print(f"Success rate (error < 5%): {np.sum(np.array(errors) < 5)/len(errors)*100:.1f}%")
    
    return errors, count_errors

def test_real_image_workflow(image_path=None):
    """Test workflow with a real SEM image if available."""
    if image_path and Path(image_path).exists():
        print(f"\\nTesting with real image: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("❌ Could not load image")
            return
        
        print(f"✅ Loaded image: {image.shape}")
        
        # For demo purposes, create a simple circular mask
        # In reality, you'd use your fiber detection module
        height, width = image.shape
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        radius = min(height, width) * 0.4
        fiber_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        
        # Run analysis
        results = basic_porosity_analysis(image, fiber_mask)
        
        print(f"Real image analysis results:")
        print(f"  Detected porosity: {results['porosity_percent']:.2f}%")
        print(f"  Pore count: {results['pore_count']}")
        print(f"  Mean pore area: {results['mean_pore_area']:.1f} pixels")
        
        # Visualize
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(image, cmap='gray')
        plt.contour(fiber_mask, colors='red', linewidths=2)
        plt.title('Fiber Region')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(image, cmap='gray')
        plt.contour(results['pore_mask'], colors='yellow', linewidths=1)
        plt.contour(fiber_mask, colors='red', linewidths=2)
        plt.title(f"Detected Pores\\n{results['porosity_percent']:.1f}%")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("\\nNo real image provided for testing")
        print("To test with real data, provide image_path parameter")

def main():
    """Main testing function."""
    print("="*60)
    print("SEM FIBER POROSITY ANALYSIS - TESTING SCRIPT")
    print("="*60)
    
    # Test 1: Check imports
    print("\\n1. Testing module imports...")
    import_success = test_imports()
    
    # Test 2: Generate and analyze synthetic data
    print("\\n2. Testing with synthetic data...")
    image, fiber_mask, ground_truth = create_synthetic_test_data()
    
    # Run basic analysis
    results = basic_porosity_analysis(image, fiber_mask)
    
    print("\\nAnalysis Results:")
    print(f"  Ground truth porosity: {ground_truth['porosity_percent']:.2f}%")
    print(f"  Detected porosity: {results['porosity_percent']:.2f}%")
    print(f"  Error: {abs(ground_truth['porosity_percent'] - results['porosity_percent']):.2f}%")
    print(f"  Ground truth pore count: {ground_truth['pore_count']}")
    print(f"  Detected pore count: {results['pore_count']}")
    
    # Test 3: Visualization
    print("\\n3. Creating visualization...")
    visualize_results(image, fiber_mask, ground_truth, results)
    
    # Test 4: Accuracy evaluation
    print("\\n4. Running accuracy evaluation...")
    errors, count_errors = run_accuracy_test(5)
    
    # Test 5: Real image workflow (if image available)
    print("\\n5. Real image workflow test...")
    # Uncomment and provide path to test with real image
    # test_real_image_workflow("path/to/your/sem_image.tif")
    test_real_image_workflow()
    
    print("\\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\\nNext steps:")
    print("1. Add real SEM images to test with actual data")
    print("2. Integrate with your full fiber analysis pipeline")
    print("3. Tune parameters based on your specific samples")
    print("4. Validate results against manual measurements")

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run the main testing function
    main()