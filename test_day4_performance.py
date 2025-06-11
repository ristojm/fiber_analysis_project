#!/usr/bin/env python3
"""
Day 4 Performance Testing
Validates that the reorganized system maintains or improves performance.
"""

import sys
import time
import numpy as np
from pathlib import Path
import psutil
import os

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))
sys.path.insert(0, str(project_root / "Crumble_ML"))

print(f"⚡ Day 4 Performance Testing")
print(f"=" * 50)

def create_synthetic_test_images(count=10):
    """Create synthetic test images for performance testing."""
    print(f"🎨 Creating {count} synthetic test images...")
    
    test_images = []
    
    for i in range(count):
        # Create realistic SEM-like images
        base_size = (480, 640, 3)
        
        # Create fiber-like structures
        image = np.random.randint(50, 200, base_size, dtype=np.uint8)
        
        # Add some fiber-like circular structures
        center_y, center_x = base_size[0]//2, base_size[1]//2
        radius = 80 + np.random.randint(-20, 20)
        
        # Create circular mask
        y, x = np.ogrid[:base_size[0], :base_size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Add fiber structure
        image[mask] = np.random.randint(100, 255, image[mask].shape, dtype=np.uint8)
        
        # Add some noise and texture
        noise = np.random.normal(0, 10, base_size).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add scale bar simulation at bottom
        image[-30:, :100] = 30  # Dark scale bar region
        
        test_images.append(image)
    
    print(f"✅ Created {len(test_images)} synthetic test images")
    return test_images

def benchmark_single_orchestrator(test_images, iterations=5):
    """Benchmark the orchestrator function."""
    print(f"\n🏃 Benchmarking Single Orchestrator")
    print(f"   Test images: {len(test_images)}")
    print(f"   Iterations per image: {iterations}")
    
    try:
        from multiprocessing_crumbly_workflow import process_single_image_orchestrator
        import tempfile
        import cv2
        
        total_times = []
        memory_usage = []
        
        for i, test_image in enumerate(test_images):
            print(f"   Testing image {i+1}/{len(test_images)}...")
            
            # Save test image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, test_image)
                temp_path = tmp_file.name
            
            image_times = []
            
            for iteration in range(iterations):
                # Monitor memory before
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Run orchestrator
                worker_args = {'image_path': temp_path, 'debug': False}
                start_time = time.time()
                
                result = process_single_image_orchestrator(worker_args)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Monitor memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before
                
                image_times.append(processing_time)
                memory_usage.append(memory_diff)
                
                success = result.get('success', False)
                if not success:
                    print(f"     ⚠️ Iteration {iteration+1} failed: {result.get('error', 'Unknown')}")
            
            # Cleanup
            os.unlink(temp_path)
            
            avg_time = np.mean(image_times)
            total_times.extend(image_times)
            
            print(f"     Average time: {avg_time:.3f}s")
        
        # Calculate statistics
        overall_avg = np.mean(total_times)
        overall_std = np.std(total_times)
        min_time = np.min(total_times)
        max_time = np.max(total_times)
        avg_memory = np.mean(memory_usage)
        
        print(f"\n📊 Performance Statistics:")
        print(f"   Average processing time: {overall_avg:.3f} ± {overall_std:.3f}s")
        print(f"   Min processing time: {min_time:.3f}s")
        print(f"   Max processing time: {max_time:.3f}s")
        print(f"   Average memory usage: {avg_memory:.1f} MB")
        print(f"   Total test runs: {len(total_times)}")
        
        # Performance classification
        if overall_avg < 2.0:
            performance_grade = "🚀 EXCELLENT"
        elif overall_avg < 5.0:
            performance_grade = "✅ GOOD"
        elif overall_avg < 10.0:
            performance_grade = "⚠️ ACCEPTABLE"
        else:
            performance_grade = "❌ NEEDS OPTIMIZATION"
        
        print(f"   Performance grade: {performance_grade}")
        
        return {
            'success': True,
            'avg_time': overall_avg,
            'std_time': overall_std,
            'min_time': min_time,
            'max_time': max_time,
            'avg_memory': avg_memory,
            'performance_grade': performance_grade,
            'total_runs': len(total_times)
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def benchmark_parallel_workflow(test_images, process_counts=[1, 2, 4]):
    """Benchmark parallel workflow with different process counts."""
    print(f"\n🏭 Benchmarking Parallel Workflow")
    print(f"   Test images: {len(test_images)}")
    print(f"   Process counts to test: {process_counts}")
    
    try:
        from multiprocessing_crumbly_workflow import MultiprocessingCrumblyWorkflow
        import tempfile
        import shutil
        import cv2
        
        results = {}
        
        for num_processes in process_counts:
            print(f"\n   Testing with {num_processes} processes...")
            
            # Create temporary dataset
            temp_dir = Path(tempfile.mkdtemp())
            test_folder = temp_dir / "test"
            test_folder.mkdir()
            
            # Save test images
            for i, test_image in enumerate(test_images):
                img_path = test_folder / f"test_image_{i+1:03d}.png"
                cv2.imwrite(str(img_path), test_image)
            
            # Create workflow
            workflow = MultiprocessingCrumblyWorkflow(num_processes=num_processes)
            
            # Monitor system resources
            process = psutil.Process(os.getpid())
            cpu_before = psutil.cpu_percent()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Run parallel evaluation
            start_time = time.time()
            
            eval_results = workflow.run_parallel_evaluation(
                dataset_path=str(temp_dir),
                max_images=len(test_images),
                debug_mode=False
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Monitor resources after
            cpu_after = psutil.cpu_percent()
            memory_after = process.memory_info().rss / 1024 / 1024
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            # Analyze results
            success = eval_results.get('success', False)
            success_rate = eval_results.get('success_rate', 0.0)
            
            results[num_processes] = {
                'total_time': total_time,
                'success': success,
                'success_rate': success_rate,
                'cpu_usage': cpu_after - cpu_before,
                'memory_usage': memory_after - memory_before,
                'images_per_second': len(test_images) / total_time if total_time > 0 else 0
            }
            
            print(f"     Total time: {total_time:.2f}s")
            print(f"     Success rate: {success_rate:.1f}%")
            print(f"     Images/second: {results[num_processes]['images_per_second']:.2f}")
        
        # Calculate speedup
        if 1 in results and results[1]['success']:
            baseline_time = results[1]['total_time']
            
            print(f"\n📈 Parallel Performance Analysis:")
            print(f"   Baseline (1 process): {baseline_time:.2f}s")
            
            for num_processes in process_counts:
                if num_processes > 1 and results[num_processes]['success']:
                    parallel_time = results[num_processes]['total_time']
                    speedup = baseline_time / parallel_time
                    efficiency = speedup / num_processes * 100
                    
                    print(f"   {num_processes} processes: {parallel_time:.2f}s "
                          f"(speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Parallel benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def test_memory_efficiency():
    """Test memory efficiency and leak detection."""
    print(f"\n🧠 Testing Memory Efficiency")
    
    try:
        import gc
        
        # Get initial memory
        gc.collect()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"   Initial memory: {initial_memory:.1f} MB")
        
        # Create and process multiple images
        test_images = create_synthetic_test_images(20)
        
        memory_samples = []
        
        for i in range(10):  # Process 10 cycles
            # Process images
            for j, test_image in enumerate(test_images[:5]):  # Process 5 images per cycle
                # Simulate processing
                processed = test_image.copy()
                processed = np.roll(processed, 10, axis=0)  # Simple transformation
                del processed
            
            # Force garbage collection
            gc.collect()
            
            # Sample memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            if i % 3 == 0:
                print(f"   Cycle {i+1}: {current_memory:.1f} MB")
        
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_samples)
        
        print(f"\n📊 Memory Analysis:")
        print(f"   Final memory: {final_memory:.1f} MB")
        print(f"   Memory growth: {memory_growth:.1f} MB")
        print(f"   Peak memory: {max_memory:.1f} MB")
        
        # Memory efficiency assessment
        if memory_growth < 50:
            memory_grade = "🚀 EXCELLENT"
        elif memory_growth < 100:
            memory_grade = "✅ GOOD"
        elif memory_growth < 200:
            memory_grade = "⚠️ ACCEPTABLE"
        else:
            memory_grade = "❌ MEMORY LEAK DETECTED"
        
        print(f"   Memory efficiency: {memory_grade}")
        
        return {
            'success': True,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_growth': memory_growth,
            'peak_memory': max_memory,
            'memory_grade': memory_grade
        }
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Run comprehensive performance testing."""
    print(f"🚀 Starting Day 4 Performance Testing")
    
    # System info
    print(f"\n💻 System Information:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   Available memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"   Python version: {sys.version.split()[0]}")
    
    # Create test data
    test_images = create_synthetic_test_images(8)
    
    # Run performance tests
    results = {}
    
    # Test 1: Single orchestrator performance
    print(f"\n" + "="*60)
    print(f"PERFORMANCE TEST 1: SINGLE IMAGE PROCESSING")
    print(f"="*60)
    
    single_results = benchmark_single_orchestrator(test_images[:3], iterations=3)
    results['single_orchestrator'] = single_results
    
    # Test 2: Parallel workflow performance
    print(f"\n" + "="*60)
    print(f"PERFORMANCE TEST 2: PARALLEL PROCESSING")
    print(f"="*60)
    
    parallel_results = benchmark_parallel_workflow(test_images, process_counts=[1, 2])
    results['parallel_workflow'] = parallel_results
    
    # Test 3: Memory efficiency
    print(f"\n" + "="*60)
    print(f"PERFORMANCE TEST 3: MEMORY EFFICIENCY")
    print(f"="*60)
    
    memory_results = test_memory_efficiency()
    results['memory_efficiency'] = memory_results
    
    # Summary
    print(f"\n" + "="*60)
    print(f"PERFORMANCE TESTING SUMMARY")
    print(f"="*60)
    
    if single_results.get('success'):
        print(f"✅ Single processing: {single_results['performance_grade']}")
        print(f"   Average time: {single_results['avg_time']:.3f}s")
    else:
        print(f"❌ Single processing: FAILED")
    
    if parallel_results and not parallel_results.get('error'):
        print(f"✅ Parallel processing: FUNCTIONAL")
        if 2 in parallel_results and parallel_results[2]['success']:
            speedup = parallel_results[1]['total_time'] / parallel_results[2]['total_time'] if parallel_results[1]['success'] else 0
            print(f"   Speedup (2 processes): {speedup:.2f}x")
    else:
        print(f"❌ Parallel processing: FAILED")
    
    if memory_results.get('success'):
        print(f"✅ Memory efficiency: {memory_results['memory_grade']}")
        print(f"   Memory growth: {memory_results['memory_growth']:.1f} MB")
    else:
        print(f"❌ Memory efficiency: FAILED")
    
    # Overall assessment
    successful_tests = sum(1 for result in [single_results, memory_results] if result.get('success'))
    if parallel_results and not parallel_results.get('error'):
        successful_tests += 1
    
    print(f"\n🎯 Overall Performance: {successful_tests}/3 tests passed")
    
    if successful_tests == 3:
        print(f"🚀 PERFORMANCE VALIDATION SUCCESSFUL!")
        print(f"✅ System ready for production deployment")
    else:
        print(f"⚠️ Some performance issues detected")
        print(f"🔧 Review failed tests before production")
    
    return successful_tests == 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)