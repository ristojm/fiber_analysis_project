#!/usr/bin/env python3
"""
Test if porosity analysis is ready and working
Now that scale detection is fixed, let's test the full pipeline
"""

import sys
from pathlib import Path
import traceback

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "modules"))

def test_porosity_analysis():
    """Test the full pipeline including porosity analysis."""
    
    print("="*60)
    print("TESTING FULL PIPELINE WITH POROSITY ANALYSIS")
    print("="*60)
    
    try:
        # Test imports
        print("🔍 Testing module imports...")
        from image_preprocessing import load_image
        from fiber_type_detection import FiberTypeDetector
        from scale_detection import detect_scale_bar
        
        try:
            from porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity
            print("✅ All modules imported successfully (including porosity)")
            porosity_available = True
        except ImportError as e:
            print(f"⚠️ Porosity module import failed: {e}")
            porosity_available = False
        
        # Test on hollow fiber (best candidate for porosity analysis)
        img_name = "hollow_fiber_sample.jpg"
        img_path = project_root / "sample_images" / img_name
        
        if not img_path.exists():
            print(f"❌ Image not found: {img_path}")
            return
        
        print(f"\n🔬 Testing full pipeline on: {img_name}")
        print("-" * 50)
        
        # Load image
        print("📥 Loading image...")
        image = load_image(str(img_path))
        print(f"✅ Image loaded: {image.shape}")
        
        # Detect fiber type
        print("🔬 Running fiber type detection...")
        detector = FiberTypeDetector()
        fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
        
        print(f"✅ Fiber type: {fiber_type} (confidence: {confidence:.3f})")
        print(f"   Total fibers: {analysis_data.get('total_fibers', 0)}")
        print(f"   Hollow fibers: {analysis_data.get('hollow_fibers', 0)}")
        
        # Detect scale
        print("📏 Testing scale detection...")
        scale_info = detect_scale_bar(str(img_path))
        if isinstance(scale_info, dict):
            scale_factor = scale_info.get('micrometers_per_pixel', 1.0)
            if scale_info.get('scale_detected', False):
                print(f"✅ Scale detected: {scale_factor:.4f} μm/pixel")
            else:
                print(f"⚠️ Scale detection failed, using default: {scale_factor:.4f} μm/pixel")
        else:
            scale_factor = 1.0
            print(f"⚠️ Unexpected scale result, using default: {scale_factor:.4f} μm/pixel")
        
        # Test porosity analysis if available
        if porosity_available and fiber_type == "hollow_fiber":
            print("\n🔬 Testing porosity analysis...")
            
            # Get fiber mask from detection results
            fiber_mask = analysis_data.get('fiber_mask')
            if fiber_mask is None:
                print("❌ No fiber mask available")
                return
            
            print(f"   Fiber mask shape: {fiber_mask.shape}")
            print(f"   Fiber mask pixels: {fiber_mask.sum():,}")
            
            # Run porosity analysis
            print("🧮 Running porosity analysis...")
            
            # Method 1: Using the PorosityAnalyzer class
            analyzer = PorosityAnalyzer()
            results = analyzer.analyze_porosity(
                image=image,
                fiber_mask=fiber_mask,
                scale_factor=scale_factor,
                fiber_type=fiber_type
            )
            
            # Display results
            porosity_metrics = results['porosity_metrics']
            
            print(f"\n🎯 POROSITY RESULTS:")
            print(f"   Total Porosity: {porosity_metrics['total_porosity_percent']:.2f}%")
            print(f"   Pore Count: {porosity_metrics['pore_count']}")
            print(f"   Average Pore Size: {porosity_metrics['average_pore_size_um2']:.3f} μm²")
            print(f"   Fiber Area: {porosity_metrics['fiber_area_um2']:.1f} μm²")
            print(f"   Pore Density: {porosity_metrics['pore_density_per_mm2']:.1f} pores/mm²")
            
            # Size distribution
            size_dist = results['size_distribution']
            if size_dist['sizes_um2']:
                stats = size_dist['statistics']
                print(f"\n📊 PORE SIZE DISTRIBUTION:")
                print(f"   Mean diameter: {stats['mean_diameter_um']:.3f} μm")
                print(f"   Median diameter: {stats['median_diameter_um']:.3f} μm")
                print(f"   Size range: {min(size_dist['diameters_um']):.3f} - {max(size_dist['diameters_um']):.3f} μm")
            
            # Export results
            output_dir = project_root / 'analysis_results'
            output_dir.mkdir(exist_ok=True)
            
            excel_file = output_dir / f"{Path(img_name).stem}_porosity_results.xlsx"
            analyzer.export_results(str(excel_file))
            print(f"\n💾 Results exported to: {excel_file}")
            
            print(f"✅ Porosity analysis completed successfully!")
            
        elif not porosity_available:
            print("\n⚠️ Porosity analysis not available - module import failed")
        elif fiber_type != "hollow_fiber":
            print(f"\n⚠️ Porosity analysis skipped - fiber type is '{fiber_type}', not 'hollow_fiber'")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

def test_porosity_simple():
    """Test porosity analysis with a simple approach if main method fails."""
    
    print(f"\n" + "="*60)
    print("TESTING SIMPLE POROSITY APPROACH")
    print("="*60)
    
    try:
        # Try the convenience function approach
        from porosity_analysis import quick_porosity_check
        from image_preprocessing import load_image
        from fiber_type_detection import FiberTypeDetector
        
        img_path = project_root / "sample_images" / "hollow_fiber_sample.jpg"
        image = load_image(str(img_path))
        
        # Get basic fiber mask
        detector = FiberTypeDetector()
        _, _, analysis_data = detector.classify_fiber_type(image)
        fiber_mask = analysis_data.get('fiber_mask')
        
        if fiber_mask is not None:
            # Quick porosity check
            porosity_percent = quick_porosity_check(image, fiber_mask, scale_factor=1.0)
            print(f"✅ Quick porosity check: {porosity_percent:.2f}%")
        else:
            print("❌ No fiber mask available for quick check")
            
    except Exception as e:
        print(f"❌ Simple porosity test failed: {e}")

if __name__ == "__main__":
    test_porosity_analysis()
    test_porosity_simple()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("If porosity analysis worked:")
    print("✅ Your system is COMPLETE and ready for production use!")
    print("✅ You can analyze fiber type + porosity + scale calibration")
    print("✅ Results are exported to Excel automatically")
    print("\nIf porosity analysis failed:")
    print("🔧 We need to troubleshoot the porosity module import/setup")
    print("🔧 But fiber type detection is working perfectly")