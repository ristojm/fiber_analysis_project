#!/usr/bin/env python3
"""
Debug Porosity Analysis Script
Specifically investigates porosity calculation issues
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

# Import modules
from modules.scale_detection import detect_scale_bar
from modules.fiber_type_detection import FiberTypeDetector
from modules.image_preprocessing import load_image
from modules.porosity_analysis import EnhancedPorosityAnalyzer

def debug_porosity_analysis(image_path):
    """Debug the porosity analysis step by step"""
    
    print(f"🔍 DEBUGGING POROSITY ANALYSIS")
    print(f"Image: {Path(image_path).name}")
    print("=" * 60)
    
    # Load image
    print("📸 Step 1: Loading image...")
    image = load_image(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"   ✅ Image loaded: {image.shape}")
    
    # Get scale factor
    print("📏 Step 2: Getting scale factor...")
    scale_result = detect_scale_bar(image, debug=False)
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_result.get('scale_detected') else 1.0
    print(f"   Scale factor: {scale_factor:.4f} μm/pixel")
    
    # Detect fibers
    print("🧬 Step 3: Detecting fibers...")
    detector = FiberTypeDetector()
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
    print(f"   Fiber type: {fiber_type} (confidence: {confidence:.3f})")
    print(f"   Total fibers: {analysis_data.get('total_fibers', 0)}")
    
    # Get fiber mask
    fiber_mask = analysis_data.get('fiber_mask', np.zeros_like(image))
    if fiber_mask is not None:
        print(f"   Fiber mask shape: {fiber_mask.shape}")
        print(f"   Fiber mask total area: {np.sum(fiber_mask > 0):,} pixels")
        print(f"   Fiber mask area in μm²: {np.sum(fiber_mask > 0) * (scale_factor**2):,.0f} μm²")
    
    # Debug individual fibers
    individual_results = analysis_data.get('individual_results', [])
    print(f"\n🔍 Individual Fiber Analysis:")
    for i, result in enumerate(individual_results):
        fiber_props = result.get('fiber_properties', {})
        has_lumen = result.get('has_lumen', False)
        lumen_props = result.get('lumen_properties', {})
        
        print(f"\n   Fiber {i+1}:")
        print(f"     Has lumen: {has_lumen}")
        print(f"     Fiber area: {fiber_props.get('area', 0):,} pixels")
        print(f"     Fiber area: {fiber_props.get('area', 0) * (scale_factor**2):,.0f} μm²")
        
        if has_lumen and lumen_props:
            lumen_area = lumen_props.get('area', 0)
            print(f"     Lumen area: {lumen_area:,} pixels")
            print(f"     Lumen area: {lumen_area * (scale_factor**2):,.0f} μm²")
            print(f"     Lumen ratio: {lumen_props.get('area_ratio', 0):.3f}")
            
            # Calculate wall area
            wall_area_pixels = fiber_props.get('area', 0) - lumen_area
            wall_area_um2 = wall_area_pixels * (scale_factor**2)
            print(f"     Wall area: {wall_area_pixels:,} pixels")
            print(f"     Wall area: {wall_area_um2:,.0f} μm²")
    
    # Run porosity analysis
    print(f"\n🕳️ Step 4: Running porosity analysis...")
    analyzer = EnhancedPorosityAnalyzer()
    
    try:
        porosity_result = analyzer.analyze_fiber_porosity(
            image, 
            fiber_mask.astype(np.uint8), 
            scale_factor, 
            fiber_type,
            analysis_data
        )
        
        # Extract detailed results
        pm = porosity_result.get('porosity_metrics', {})
        
        print(f"\n📊 POROSITY RESULTS:")
        print(f"   Total porosity: {pm.get('total_porosity_percent', 0):.2f}%")
        print(f"   Pore count: {pm.get('pore_count', 0)}")
        print(f"   Total pore area: {pm.get('total_pore_area_um2', 0):.2f} μm²")
        print(f"   Fiber area used: {pm.get('fiber_area_um2', 0):.2f} μm²")
        print(f"   Average pore size: {pm.get('average_pore_size_um2', 0):.2f} μm²")
        print(f"   Min pore size: {pm.get('min_pore_size_um2', 0):.2f} μm²")
        print(f"   Max pore size: {pm.get('max_pore_size_um2', 0):.2f} μm²")
        
        # Check individual pore data
        pore_data = analyzer.get_pore_dataframe()
        if not pore_data.empty:
            print(f"\n🔍 INDIVIDUAL PORE ANALYSIS:")
            print(f"   Number of pores detected: {len(pore_data)}")
            print(f"   Pore size statistics (μm²):")
            print(f"     Mean: {pore_data['area_um2'].mean():.2f}")
            print(f"     Median: {pore_data['area_um2'].median():.2f}")
            print(f"     Std: {pore_data['area_um2'].std():.2f}")
            print(f"     Min: {pore_data['area_um2'].min():.2f}")
            print(f"     Max: {pore_data['area_um2'].max():.2f}")
            
            # Show size distribution
            print(f"\n   Pore size distribution:")
            bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
            labels = ['<10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '>5000']
            
            for i, (bin_start, bin_end, label) in enumerate(zip(bins[:-1], bins[1:], labels)):
                count = len(pore_data[(pore_data['area_um2'] >= bin_start) & (pore_data['area_um2'] < bin_end)])
                percentage = count / len(pore_data) * 100 if len(pore_data) > 0 else 0
                print(f"     {label} μm²: {count} pores ({percentage:.1f}%)")
        
        # Create debug visualization
        create_debug_visualization(image, fiber_mask, porosity_result, scale_factor, image_path)
        
        # Calculate manual porosity for comparison
        print(f"\n🧮 MANUAL CALCULATION CHECK:")
        
        # Get visible pore statistics
        if not pore_data.empty:
            large_pores = pore_data[pore_data['area_um2'] > 100]  # Pores larger than 100 μm²
            print(f"   Large pores (>100 μm²): {len(large_pores)}")
            print(f"   Large pore total area: {large_pores['area_um2'].sum():.2f} μm²")
            
            if len(large_pores) > 0:
                large_pore_porosity = large_pores['area_um2'].sum() / pm.get('fiber_area_um2', 1) * 100
                print(f"   Large pore porosity: {large_pore_porosity:.2f}%")
        
        # Visual estimation helper
        total_pixels = image.shape[0] * image.shape[1]
        dark_pixels = np.sum(image < 100)  # Very rough estimate of dark areas
        visual_estimate = dark_pixels / total_pixels * 100
        print(f"   Visual dark area estimate: {visual_estimate:.1f}% of total image")
        
    except Exception as e:
        print(f"❌ Porosity analysis failed: {e}")
        import traceback
        traceback.print_exc()

def create_debug_visualization(image, fiber_mask, porosity_result, scale_factor, image_path):
    """Create detailed debug visualization"""
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0,0].imshow(image, cmap='gray')
        axes[0,0].set_title('Original SEM Image')
        axes[0,0].axis('off')
        
        # Fiber mask
        axes[0,1].imshow(fiber_mask, cmap='gray')
        axes[0,1].set_title('Fiber Mask')
        axes[0,1].axis('off')
        
        # Overlay
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Draw fiber boundaries in green
        contours, _ = cv2.findContours(fiber_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
        
        axes[0,2].imshow(overlay)
        axes[0,2].set_title('Fiber Boundaries')
        axes[0,2].axis('off')
        
        # Pore detection visualization
        if 'pore_results' in porosity_result:
            pore_results = porosity_result['pore_results']
            pore_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            for pore in pore_results:
                if 'bbox' in pore:
                    x1, y1, x2, y2 = pore['bbox']
                    cv2.rectangle(pore_overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            axes[1,0].imshow(pore_overlay)
            axes[1,0].set_title(f'Detected Pores ({len(pore_results)})')
            axes[1,0].axis('off')
        
        # Histogram of image intensities
        axes[1,1].hist(image.flatten(), bins=50, alpha=0.7)
        axes[1,1].axvline(x=100, color='red', linestyle='--', label='Dark threshold')
        axes[1,1].set_title('Image Intensity Distribution')
        axes[1,1].set_xlabel('Pixel Intensity')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        # Results summary
        pm = porosity_result.get('porosity_metrics', {})
        summary_text = f"Porosity Analysis Results:\n\n"
        summary_text += f"Total Porosity: {pm.get('total_porosity_percent', 0):.2f}%\n"
        summary_text += f"Pore Count: {pm.get('pore_count', 0)}\n"
        summary_text += f"Total Pore Area: {pm.get('total_pore_area_um2', 0):.0f} μm²\n"
        summary_text += f"Fiber Area: {pm.get('fiber_area_um2', 0):.0f} μm²\n"
        summary_text += f"Avg Pore Size: {pm.get('average_pore_size_um2', 0):.1f} μm²\n"
        summary_text += f"Scale Factor: {scale_factor:.4f} μm/pixel\n"
        
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1,2].set_title('Analysis Summary')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        
        # Save debug image
        debug_name = f"debug_porosity_{Path(image_path).stem}.png"
        plt.savefig(debug_name, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"📸 Debug visualization saved: {debug_name}")
        
    except Exception as e:
        print(f"⚠️ Could not create debug visualization: {e}")

def main():
    """Main debug function"""
    
    # Look for your real image
    image_paths = [
        "sample_images/28d_001.jpg",
        "28d_001.jpg",
        "sample_images/28d_001.tif",
        "28d_001.tif"
    ]
    
    image_path = None
    for path in image_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path:
        debug_porosity_analysis(image_path)
    else:
        print("❌ Could not find 28d_001.jpg")
        print("Available image paths to try:")
        for path in image_paths:
            print(f"   {path}")
        
        # List available images
        sample_dir = Path("sample_images")
        if sample_dir.exists():
            images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.tif"))
            if images:
                print(f"\nFound images in sample_images/:")
                for img in images:
                    print(f"   {img}")
                print(f"\nTrying first available image...")
                debug_porosity_analysis(str(images[0]))

if __name__ == "__main__":
    main()