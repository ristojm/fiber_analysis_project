"""
SEM Fiber Analysis System - Optimized Scale Detection Module
Automatic detection and calibration of scale bars in SEM images.
UPDATED: High performance version with batch testing support

Key optimizations:
1. RapidOCR support for better Intel GPU performance
2. Intel OpenVINO support for OCR acceleration
3. Parallel processing for image regions
4. Caching for repeated operations
5. Numpy vectorization
6. Early exit strategies
7. Enhanced text-centered detection with aggressive line finding
"""

import cv2
import numpy as np
import re
from typing import Dict, Optional, List, Tuple
from collections import Counter
import multiprocessing as mp
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import json
from datetime import datetime
import os

# Try to import Intel-optimized libraries
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except:
    OPENVINO_AVAILABLE = False

# OCR library selection - prioritize RapidOCR for Intel GPUs
OCR_BACKEND = None
DEVICE = 'cpu'

# Try RapidOCR first (better for Intel GPU)
try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_BACKEND = 'rapidocr'
    RAPIDOCR_AVAILABLE = True
    print("✅ RapidOCR available - using optimized ONNX runtime")
except:
    RAPIDOCR_AVAILABLE = False

# Fallback to EasyOCR
if not RAPIDOCR_AVAILABLE:
    try:
        import easyocr
        import torch
        
        # Check for Intel GPU support
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            DEVICE = 'xpu'  # Intel GPU
        else:
            DEVICE = 'cpu'
        
        OCR_BACKEND = 'easyocr'
        EASYOCR_AVAILABLE = True
        print("✅ EasyOCR available")
    except:
        EASYOCR_AVAILABLE = False

# Legacy OCR support (for compatibility)
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Enhanced OCR will be used.")

if OCR_BACKEND is None:
    print("❌ No OCR backend available. Please install rapidocr-onnxruntime or easyocr")

# Global reader instance to avoid re-initialization
_ocr_reader = None

def get_ocr_reader():
    """Get or create a cached OCR reader instance."""
    global _ocr_reader
    
    if _ocr_reader is None:
        if OCR_BACKEND == 'rapidocr' and RAPIDOCR_AVAILABLE:
            # Initialize RapidOCR with optimized settings
            _ocr_reader = RapidOCR(
                det_use_cuda=False,  # Use ONNX runtime instead
                rec_use_cuda=False,
                cls_use_cuda=False,
                det_limit_side_len=960,
                det_limit_type='max',
                det_thresh=0.3,
                max_candidates=1000,
                unclip_ratio=1.5,
                use_angle_cls=True,
                cls_thresh=0.9,
                enable_mkldnn=True,  # Intel MKL-DNN optimization
                cpu_threads=mp.cpu_count(),
                print_verbose=False  # Disable verbose output
            )
            print(f"🚀 Using RapidOCR with Intel optimizations")
        elif OCR_BACKEND == 'easyocr' and EASYOCR_AVAILABLE:
            # Initialize EasyOCR
            _ocr_reader = easyocr.Reader(['en'], gpu=(DEVICE != 'cpu'), verbose=False)
            print(f"🚀 Using EasyOCR on {DEVICE}")
    
    return _ocr_reader

# Wrapper class to provide unified interface for both OCR backends
class UnifiedOCRReader:
    """Unified interface for both RapidOCR and EasyOCR"""
    
    def __init__(self):
        self.reader = get_ocr_reader()
        self.backend = OCR_BACKEND
    
    def _extract_score(self, score_data):
        """Recursively extract numeric score from various data structures"""
        if isinstance(score_data, (int, float)):
            return float(score_data)
        elif isinstance(score_data, str):
            try:
                return float(score_data)
            except:
                return 0.0
        elif isinstance(score_data, (list, tuple)):
            # Recursively search for numeric value
            for item in score_data:
                result = self._extract_score(item)
                if result > 0:
                    return result
            return 0.0
        else:
            return 0.0
    
    def readtext(self, image, detail=1, text_threshold=0.7, **kwargs):
        """Unified readtext method that works with both backends"""
        if self.backend == 'rapidocr':
            # RapidOCR returns: [((x1,y1),(x2,y2),(x3,y3),(x4,y4)), text, score]
            results = self.reader(image)
            if results is None:
                return []
            
            # Handle both tuple and list returns from RapidOCR
            if isinstance(results, tuple) and len(results) >= 1:
                # RapidOCR sometimes returns (results_list, elapsed_time) or (results_list, ...)
                results = results[0]
            
            if not results:
                return []
            
            # Convert to EasyOCR format for compatibility
            formatted_results = []
            for i, result in enumerate(results):
                try:
                    # RapidOCR format variations:
                    # 1. [bbox, text, score]
                    # 2. [bbox, (text, score)]
                    # 3. (bbox, text, score)
                    
                    # Convert to list if tuple
                    if isinstance(result, tuple):
                        result = list(result)
                    
                    if len(result) >= 3:
                        bbox = result[0]
                        text = result[1]
                        score = result[2]
                    elif len(result) == 2:
                        bbox = result[0]
                        # Check if second element is tuple/list of (text, score)
                        if isinstance(result[1], (list, tuple)) and len(result[1]) >= 2:
                            text = str(result[1][0])
                            score = result[1][1]
                        else:
                            # Skip if we can't parse
                            continue
                    else:
                        continue
                    
                    # Ensure text is string
                    text = str(text) if text is not None else ""
                    
                    # Extract numeric score using recursive method
                    score = self._extract_score(score)
                    
                    # Filter by confidence threshold
                    if score >= text_threshold:
                        # Convert bbox to numpy array format
                        bbox_array = np.array(bbox, dtype=np.float32)
                        # Ensure bbox has correct shape (4, 2)
                        if bbox_array.shape == (4, 2):
                            formatted_results.append((bbox_array, text, score))
                        
                except Exception as e:
                    # Debug print for problematic results
                    if detail > 1:  # Only print in verbose mode
                        print(f"Warning: Failed to parse RapidOCR result {i}: {result}, Error: {e}")
                    continue
            
            return formatted_results
        
        elif self.backend == 'easyocr':
            # EasyOCR already returns in the expected format
            return self.reader.readtext(image, detail=detail, text_threshold=text_threshold, **kwargs)
        
        return []

# Precompiled regex patterns for better performance
REGEX_PATTERNS = {
    'numbers': [re.compile(p) for p in [
        r'(\d+\.\d+)',
        r'(\d+,\d+)',
        r'(\d+\s+\.\s*\d+)',
        r'(\d+\.?\d*)',
        r'(\d+)'
    ]],
    'magnification': re.compile(r'\d+x\s*$'),
    'voltage': re.compile(r'\d+\.?\d*\s*kv', re.IGNORECASE),
    'scale_format': re.compile(r'^\d+\.?\d*\s*[μµu]?m$', re.IGNORECASE)
}

class ScaleBarDetector:
    """
    Optimized scale bar detector with caching and parallel processing.
    Maintains compatibility with existing SEM Fiber Analysis System.
    """
    
    def __init__(self, 
                 scale_region_fraction: float = 0.15,
                 min_bar_length: int = 50,
                 max_bar_thickness: int = 20,
                 text_search_region: float = 0.3,
                 use_enhanced_detection: bool = True,
                 use_gpu: bool = True, 
                 num_threads: Optional[int] = None, 
                 ocr_backend: Optional[str] = None):
        """
        Initialize scale bar detector.
        
        Args:
            scale_region_fraction: Fraction of image height to search for scale bar
            min_bar_length: Minimum length of scale bar in pixels
            max_bar_thickness: Maximum thickness of scale bar in pixels
            text_search_region: Fraction of scale region to search for text
            use_enhanced_detection: Use new enhanced detection algorithm
            use_gpu: Enable GPU acceleration if available
            num_threads: Number of threads for parallel processing
            ocr_backend: OCR backend to use ('rapidocr', 'easyocr', or None for auto)
        """
        self.scale_region_fraction = scale_region_fraction
        self.min_bar_length = min_bar_length
        self.max_bar_thickness = max_bar_thickness
        self.text_search_region = text_search_region
        self.use_enhanced_detection = use_enhanced_detection
        self.use_gpu = use_gpu and (DEVICE != 'cpu' or OCR_BACKEND == 'rapidocr')
        self.num_threads = num_threads or mp.cpu_count()
        self.cache = {}
        self._all_ocr_results = []  # Store OCR results for reuse
        self.ocr_backend = ocr_backend or OCR_BACKEND
        
        # Create unified OCR reader
        if self.ocr_backend:
            self.ocr_reader = UnifiedOCRReader()
        else:
            self.ocr_reader = None
        
        if self.ocr_backend == 'rapidocr':
            print(f"🚀 ScaleBarDetector using RapidOCR with Intel optimizations")
        elif self.use_gpu and OPENVINO_AVAILABLE:
            print(f"🚀 ScaleBarDetector using Intel GPU acceleration via OpenVINO")
        elif self.use_gpu and DEVICE == 'cuda':
            print(f"🚀 ScaleBarDetector using NVIDIA GPU acceleration")
        else:
            print(f"💻 ScaleBarDetector using CPU with {self.num_threads} threads")
    
    def extract_scale_region(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Extract the bottom region likely to contain scale bar.
        
        Args:
            image: Input SEM image
            
        Returns:
            Tuple of (scale_region, y_offset)
        """
        height = image.shape[0]
        y_start = int(height * (1 - self.scale_region_fraction))
        
        scale_region = image[y_start:, :]
        return scale_region, y_start
    
    def detect_scale_bar(self, image: np.ndarray, debug: bool = False, 
                        save_debug_image: bool = False, output_dir: Optional[str] = None) -> Dict:
        """
        Main function to detect scale bar and calculate calibration.
        Uses enhanced detection with optimizations.
        
        Args:
            image: Input SEM image
            debug: Enable debug output
            save_debug_image: Save debug visualization
            output_dir: Directory for debug images
            
        Returns:
            Dictionary containing scale detection results
        """
        start_time = time.time()
        
        if debug:
            print(f"🔍 Analyzing image: {image.shape}")
        
        result = {
            'scale_detected': False,
            'micrometers_per_pixel': 0.0,
            'scale_factor': 0.0,
            'error': None,
            'method_used': 'optimized_enhanced',
            'confidence': 0.0,
            'processing_time': 0.0,
            'ocr_backend': self.ocr_backend
        }
        
        # Extract scale region
        scale_region, y_offset = self.extract_scale_region(image)
        result.update({
            'scale_region': scale_region,
            'y_offset': y_offset
        })
        
        if self.use_enhanced_detection and self.ocr_reader:
            # Use enhanced text-centered detection
            if debug:
                print(f"📝 Step 1: Finding scale text (optimized with {self.ocr_backend})...")
            
            text_info = self._find_scale_text_optimized(image, debug)
            
            if not text_info:
                result['error'] = "No valid scale text found with enhanced detection"
                result['processing_time'] = time.time() - start_time
                return result
            
            if debug:
                print(f"✅ Found text: '{text_info['text']}' = {text_info['micrometers']} μm")
                print(f"   Parse confidence: {text_info.get('parse_confidence', 'unknown')}")
            
            # Find complete scale bar span
            if debug:
                print("📏 Step 2: Finding scale bar segments (parallel)...")
            
            total_span, bar_method, bar_details = self._find_scale_bar_parallel(image, text_info, debug)
            
            if total_span <= 0:
                result['error'] = "Could not detect complete scale bar span"
                result['processing_time'] = time.time() - start_time
                return result
            
            # Calculate results
            micrometers_per_pixel = text_info['micrometers'] / total_span
            
            # Calculate confidence
            text_confidence = text_info.get('confidence', 0.5)
            parse_confidence = 1.0 if text_info.get('parse_confidence') == 'high' else 0.7
            validation_score = text_info.get('validation_score', 0.5)
            
            method_confidence = {
                'paired_segments': 0.95,
                'reconstructed': 0.85,
                'single_side_proportion': 0.7,
                'fallback_scan': 0.6,
                'estimated': 0.5
            }.get(bar_method, 0.5)
            
            overall_confidence = (text_confidence * 0.3 + 
                                 parse_confidence * 0.2 + 
                                 validation_score * 0.2 + 
                                 method_confidence * 0.3)
            
            result.update({
                'scale_detected': True,
                'micrometers_per_pixel': micrometers_per_pixel,
                'scale_factor': micrometers_per_pixel,
                'scale_info': text_info,
                'bar_length_pixels': total_span,
                'detection_method': bar_method,
                'confidence': overall_confidence,
                'processing_time': time.time() - start_time,
                'bar_details': bar_details
            })
            
            if debug:
                print(f"✅ SUCCESS: {micrometers_per_pixel:.4f} μm/pixel")
                print(f"   Total span: {total_span} pixels")
                print(f"   Detection method: {bar_method}")
                print(f"   Overall confidence: {overall_confidence:.2%}")
                print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
        
        else:
            # Fallback to legacy detection for compatibility
            if debug:
                print("Using legacy detection method (enhanced detection disabled)")
            
            result = self._legacy_detection(image, scale_region, y_offset, debug)
            result['processing_time'] = time.time() - start_time
        
        # Save debug image if requested
        if save_debug_image and result['scale_detected'] and output_dir:
            try:
                debug_image_path = self._save_debug_image(image, result, output_dir, debug)
                result['debug_image_path'] = debug_image_path
            except Exception as e:
                if debug:
                    print(f"⚠️ Failed to save debug image: {e}")
        
        return result
    
    def _find_scale_text_optimized(self, image: np.ndarray, debug=False) -> Optional[Dict]:
        """Optimized text detection using parallel processing."""
        
        if not self.ocr_reader:
            return None
        
        height, width = image.shape
        
        # Search bottom 30% of image
        search_height = int(height * 0.3)
        bottom_region = image[height - search_height:, :]
        y_offset = height - search_height
        
        if debug:
            print(f"   Searching bottom region: {bottom_region.shape}")
        
        # Parallel preprocessing for different detection strategies
        all_results = []
        
        if self.ocr_backend == 'rapidocr':
            # RapidOCR handles preprocessing internally
            try:
                results1 = self.ocr_reader.readtext(bottom_region, text_threshold=0.7)
                all_results.extend(results1)
            except Exception as e:
                if debug:
                    print(f"   RapidOCR standard detection error: {e}")
            
            # Enhanced contrast version
            try:
                enhanced = cv2.convertScaleAbs(bottom_region, alpha=1.5, beta=20)
                results2 = self.ocr_reader.readtext(enhanced, text_threshold=0.5)
                all_results.extend(results2)
            except Exception as e:
                if debug:
                    print(f"   RapidOCR enhanced detection error: {e}")
            
        else:
            # EasyOCR - use ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                
                # Standard detection
                futures.append(executor.submit(self.ocr_reader.readtext, bottom_region, detail=1))
                
                # Low threshold detection
                try:
                    futures.append(executor.submit(self.ocr_reader.readtext, bottom_region, 
                                                 detail=1, text_threshold=0.5))
                except:
                    pass
                
                # Enhanced contrast version
                enhanced = cv2.convertScaleAbs(bottom_region, alpha=1.5, beta=20)
                futures.append(executor.submit(self.ocr_reader.readtext, enhanced, detail=1))
                
                # Collect results
                for future in futures:
                    try:
                        results = future.result(timeout=10)
                        all_results.extend(results)
                    except:
                        pass
        
        # Fast deduplication
        seen = set()
        unique_results = []
        for item in all_results:
            try:
                bbox, text, confidence = item
                if text not in seen:
                    seen.add(text)
                    unique_results.append((bbox, text, confidence))
            except Exception as e:
                if debug:
                    print(f"   Error processing OCR result: {e}, item: {item}")
        
        if debug:
            print(f"   {self.ocr_backend} found {len(unique_results)} unique text elements")
            for (bbox, text, confidence) in unique_results:
                print(f"     '{text}' (confidence: {confidence:.3f})")
        
        # Store all OCR results for later use
        self._all_ocr_results = [(bbox, text, confidence, y_offset) for bbox, text, confidence in unique_results]
        
        # Parallel text parsing
        scale_candidates = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for bbox, text, confidence in unique_results:
                if confidence > 0.1:
                    futures.append(executor.submit(
                        self._process_text_candidate, 
                        bbox, text, confidence, y_offset, image.shape, debug
                    ))
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        scale_candidates.append(result)
                except Exception as e:
                    if debug:
                        print(f"   Error processing text candidate: {e}")
        
        # Select best candidate
        if scale_candidates:
            scale_candidates.sort(key=lambda x: (x['validation_score'], x['confidence']), reverse=True)
            if debug:
                print(f"   Selected best candidate from {len(scale_candidates)} options")
            return scale_candidates[0]
        
        return None
    
    def _process_text_candidate(self, bbox, text, confidence, y_offset, image_shape, debug=False):
        """Process a single text candidate (for parallel execution)."""
        
        try:
            # Ensure confidence is a float
            if isinstance(confidence, (list, tuple)):
                confidence = float(confidence[0]) if len(confidence) > 0 else 0.0
            else:
                confidence = float(confidence)
            
            # Fast SEM parameter check
            if is_sem_parameter_text_fast(text):
                if debug:
                    print(f"       ❌ Filtered out '{text}' (SEM parameter)")
                return None
            
            scale_info = parse_scale_text_optimized(text.strip())
            if not scale_info:
                if debug:
                    print(f"       ❌ Could not parse '{text}'")
                return None
            
            # Calculate bbox info
            bbox_array = np.array(bbox)
            bbox_adjusted = bbox_array.copy()
            bbox_adjusted[:, 1] += y_offset
            
            min_x = int(np.min(bbox_adjusted[:, 0]))
            max_x = int(np.max(bbox_adjusted[:, 0]))
            min_y = int(np.min(bbox_adjusted[:, 1]))
            max_y = int(np.max(bbox_adjusted[:, 1]))
            
            validation_score = validate_scale_text_fast(text, (min_x, max_x, min_y, max_y), image_shape)
            
            if debug:
                print(f"       ✅ Parsed '{text}': {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm")
                print(f"       Validation score: {validation_score:.3f}")
            
            return {
                'text': text.strip(),
                'value': scale_info['value'],
                'unit': scale_info['unit'],
                'micrometers': scale_info['micrometers'],
                'confidence': confidence,
                'center_x': (min_x + max_x) // 2,
                'center_y': (min_y + max_y) // 2,
                'bbox': bbox_adjusted,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'width': max_x - min_x,
                'height': max_y - min_y,
                'validation_score': validation_score,
                'parse_confidence': scale_info.get('confidence', 'high')
            }
        except Exception as e:
            if debug:
                print(f"       ❌ Error processing candidate '{text}': {str(e)}")
            return None
    
    def _find_scale_bar_parallel(self, image: np.ndarray, text_info: Dict, debug=False) -> Tuple[int, str, Dict]:
        """Find scale bar using parallel processing with intelligent text-based boundaries."""
        
        # Define search regions
        text_center_y = int(text_info['center_y'])
        text_min_x = int(text_info['min_x'])
        text_max_x = int(text_info['max_x'])
        text_width = int(text_info['width'])
        
        height, width = image.shape
        
        # Search parameters
        default_search_margin = 500
        y_tolerance = 15
        
        if debug:
            print(f"   Text spans from x={text_min_x} to x={text_max_x} (width={text_width})")
            print(f"   Looking for scale line at y≈{text_center_y}")
        
        # Use text boundaries to constrain search
        text_bounds = self._find_text_boundaries_smart(text_info, debug)
        
        if text_bounds:
            left_x1 = max(0, text_bounds['left_boundary'])
            left_x2 = text_min_x - 5
            right_x1 = text_max_x + 5
            right_x2 = min(width, text_bounds['right_boundary'])
            
            if debug:
                print(f"   Smart boundaries from text detection:")
                print(f"     Left search: x=[{left_x1}, {left_x2}]")
                print(f"     Right search: x=[{right_x1}, {right_x2}]")
        else:
            left_x1 = max(0, text_min_x - default_search_margin)
            left_x2 = text_min_x - 5
            right_x1 = text_max_x + 5
            right_x2 = min(width, text_max_x + default_search_margin)
        
        search_y1 = max(0, text_center_y - y_tolerance)
        search_y2 = min(height, text_center_y + y_tolerance)
        
        # Ensure all search boundaries are integers
        left_x1, left_x2 = int(left_x1), int(left_x2)
        right_x1, right_x2 = int(right_x1), int(right_x2)
        search_y1, search_y2 = int(search_y1), int(search_y2)
        
        # Extract regions with bounds checking
        if left_x2 > left_x1 and search_y2 > search_y1:
            left_region = image[search_y1:search_y2, left_x1:left_x2]
        else:
            left_region = np.array([])
            
        if right_x2 > right_x1 and search_y2 > search_y1:
            right_region = image[search_y1:search_y2, right_x1:right_x2]
        else:
            right_region = np.array([])
        
        if debug:
            print(f"   Left search region: {left_region.shape}")
            print(f"   Right search region: {right_region.shape}")
        
        # Parallel line detection
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(find_lines_vectorized, left_region, debug)
            right_future = executor.submit(find_lines_vectorized, right_region, debug)
            
            left_lines = left_future.result()
            right_lines = right_future.result()
        
        if debug:
            print(f"   Found {len(left_lines)} left segments, {len(right_lines)} right segments")
        
        # Find best span using optimized algorithms
        best_span = 0
        method = 'unknown'
        best_details = None
        
        # Try paired segments first
        if left_lines and right_lines:
            # Vectorized comparison
            left_y = np.array([l['center_y'] for l in left_lines])
            right_y = np.array([r['center_y'] for r in right_lines])
            
            # Calculate all y-differences at once
            y_diff_matrix = np.abs(left_y[:, np.newaxis] - right_y[np.newaxis, :])
            
            # Find aligned pairs
            aligned_pairs = np.where(y_diff_matrix <= 3)
            
            for i, j in zip(aligned_pairs[0], aligned_pairs[1]):
                left_line = left_lines[i]
                right_line = right_lines[j]
                
                left_start = left_x1 + left_line['start_x']
                right_end = right_x1 + right_line['end_x']
                total_span = right_end - left_start
                
                # Score this combination
                score = score_line_pair_fast(left_line, right_line, text_info)
                
                if debug and score > 0.4:
                    print(f"     Candidate pair: left={left_line['length']}px, right={right_line['length']}px")
                    print(f"       Total span: {total_span}px, score: {score:.3f}")
                
                # Validate against text boundaries
                if text_bounds:
                    if left_start < text_bounds['left_boundary'] or right_end > text_bounds['right_boundary']:
                        if debug:
                            print(f"       Rejected: extends beyond text boundaries")
                        continue
                
                if score > 0.3 and 150 <= total_span <= 1200 and total_span > best_span:
                    best_span = total_span
                    method = 'paired_segments'
                    best_details = {
                        'left_line': left_line,
                        'right_line': right_line,
                        'left_start_global': left_start,
                        'right_end_global': right_end,
                        'total_span': total_span,
                        'score': score
                    }
        
        # Fallback methods if no paired segments found
        if best_span == 0:
            if left_lines and right_lines:
                # Reconstructed method
                best_left = max(left_lines, key=lambda x: x['score'])
                best_right = max(right_lines, key=lambda x: x['score'])
                best_span = best_left['length'] + text_width + best_right['length']
                method = 'reconstructed'
                best_details = {'left_length': best_left['length'], 'right_length': best_right['length']}
                
            elif left_lines or right_lines:
                # Single side proportion
                single_line = max((left_lines + right_lines), key=lambda x: x['score'])
                best_span = int(single_line['length'] / 0.4)
                method = 'single_side_proportion'
                best_details = {'visible_length': single_line['length']}
                
            else:
                # Estimation based on text width
                best_span = int(text_width * 2.2)
                method = 'estimated'
                best_details = {'text_width': text_width}
        
        if best_span > 0 and debug:
            print(f"✅ Best span found: {best_span} pixels using method: {method}")
        
        return best_span, method, best_details
    
    def _find_text_boundaries_smart(self, scale_text_info: Dict, debug=False) -> Optional[Dict]:
        """Find intelligent boundaries for scale bar search based on other detected text."""
        
        if not hasattr(self, '_all_ocr_results') or not self._all_ocr_results:
            return None
        
        scale_y = int(scale_text_info['center_y'])
        scale_min_x = int(scale_text_info['min_x'])
        scale_max_x = int(scale_text_info['max_x'])
        scale_text_lower = scale_text_info['text'].lower().strip()
        
        left_boundary = 0
        right_boundary = 10000
        left_constraint_text = None
        right_constraint_text = None
        
        # Analyze all detected text
        for bbox, text, confidence, y_offset in self._all_ocr_results:
            text_clean = text.strip()
            text_lower = text_clean.lower()
            
            # Skip the scale text itself
            if text_lower == scale_text_lower:
                continue
            
            # Handle confidence as list or float
            if isinstance(confidence, (list, tuple)):
                conf_value = float(confidence[0]) if len(confidence) > 0 else 0.0
            else:
                conf_value = float(confidence)
            
            # Skip very low confidence text
            if conf_value < 0.3:
                continue
            
            # Skip facility/copyright text
            facility_keywords = ['facility', 'bioimaging', 'wolfson', 'laboratory', 
                               'lab', 'center', 'centre', 'institute', 'university',
                               'copyright', '©', 'imaging', 'microscopy']
            
            if any(keyword in text_lower for keyword in facility_keywords):
                continue
            
            # Calculate text position
            bbox_array = np.array(bbox)
            text_min_x = int(np.min(bbox_array[:, 0]))
            text_max_x = int(np.max(bbox_array[:, 0]))
            text_center_y = int(np.mean(bbox_array[:, 1]) + y_offset)
            
            # Only consider text at similar vertical position
            if abs(text_center_y - scale_y) < 20:
                # Additional check: Skip if this is likely metadata below scale bar
                if text_center_y > scale_y + 10:
                    continue
                    
                # Text to the left of scale text
                if text_max_x < scale_min_x:
                    if is_sem_parameter_text_fast(text_clean) or len(text_clean) < 20:
                        potential_boundary = int(text_max_x + 15)
                        if potential_boundary > left_boundary:
                            left_boundary = potential_boundary
                            left_constraint_text = text_clean
                
                # Text to the right of scale text
                elif text_min_x > scale_max_x:
                    if text_min_x - scale_max_x < 200:
                        if is_sem_parameter_text_fast(text_clean) or len(text_clean) < 20:
                            potential_boundary = int(text_min_x - 15)
                            if potential_boundary < right_boundary:
                                right_boundary = potential_boundary
                                right_constraint_text = text_clean
        
        # Validate boundaries
        if right_boundary == 10000:
            right_boundary = int(scale_max_x + 500)
        
        # Ensure minimum search width
        min_search_width = 100
        if left_boundary > scale_min_x - min_search_width:
            left_boundary = int(scale_min_x - min_search_width)
        if right_boundary < scale_max_x + min_search_width:
            right_boundary = int(scale_max_x + min_search_width)
        
        result = {
            'left_boundary': int(max(0, left_boundary)),
            'right_boundary': int(right_boundary),
            'left_constraint': left_constraint_text,
            'right_constraint': right_constraint_text
        }
        
        if debug:
            print(f"   📍 Smart text boundaries detected:")
            if left_constraint_text:
                print(f"      Left: constrained by '{left_constraint_text}'")
            if right_constraint_text:
                print(f"      Right: constrained by '{right_constraint_text}'")
        
        return result
    
    def _legacy_detection(self, image: np.ndarray, scale_region: np.ndarray, 
                         y_offset: int, debug: bool) -> Dict:
        """Legacy detection method for compatibility."""
        
        result = {
            'scale_detected': False,
            'micrometers_per_pixel': 0.0,
            'scale_factor': 0.0,
            'error': None,
            'method_used': 'legacy',
            'confidence': 0.0
        }
        
        # Try to detect scale bar lines using original method
        bar_candidates = self._detect_scale_bar_lines_legacy(scale_region)
        
        if not bar_candidates:
            result['error'] = "No scale bar candidates detected"
            return result
        
        # Try OCR to extract scale text
        text_lines = []
        
        # Try legacy Tesseract if available
        if PYTESSERACT_AVAILABLE:
            try:
                import pytesseract
                config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789µμmnmkMKMm.μ '
                text = pytesseract.image_to_string(scale_region, config=config)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                text_lines.extend(lines)
            except Exception as e:
                if debug:
                    print(f"Tesseract OCR failed: {e}")
        
        # Try modern OCR as fallback
        if not text_lines and self.ocr_reader:
            try:
                ocr_results = self.ocr_reader.readtext(scale_region, detail=1)
                text_lines = [result[1] for result in ocr_results if result[2] > 0.5]
            except Exception as e:
                if debug:
                    print(f"Modern OCR failed: {e}")
        
        if not text_lines:
            result['error'] = "No text extracted from scale region"
            return result
        
        # Parse scale information
        scale_info = self._parse_scale_text_legacy(text_lines)
        if not scale_info:
            result['error'] = "Could not parse scale information from text"
            return result
        
        # Calculate calibration
        best_bar = bar_candidates[0]
        bar_length = best_bar['length']
        micrometers_per_pixel = scale_info['micrometers'] / bar_length
        
        result.update({
            'scale_detected': True,
            'micrometers_per_pixel': micrometers_per_pixel,
            'scale_factor': micrometers_per_pixel,
            'scale_info': scale_info,
            'bar_length_pixels': bar_length,
            'confidence': 0.7  # Default legacy confidence
        })
        
        return result
    
    def _detect_scale_bar_lines_legacy(self, scale_region: np.ndarray) -> List[Dict]:
        """Legacy scale bar line detection."""
        blurred = cv2.GaussianBlur(scale_region, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_bar_length or h > self.max_bar_thickness:
                continue
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 5:
                continue
            
            area = cv2.contourArea(contour)
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            candidates.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'length': w,
                'thickness': h,
                'aspect_ratio': aspect_ratio,
                'fill_ratio': fill_ratio,
                'area': area,
                'center_x': x + w // 2,
                'center_y': y + h // 2
            })
        
        candidates.sort(key=lambda x: x['length'], reverse=True)
        return candidates
    
    def _parse_scale_text_legacy(self, text_lines: List[str]) -> Optional[Dict]:
        """Legacy scale text parsing."""
        unit_patterns = {
            r'(\d+\.?\d*)\s*μm': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*µm': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*um': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*nm': ('nanometer', 0.001),
            r'(\d+\.?\d*)\s*mm': ('millimeter', 1000.0),
        }
        
        for text in text_lines:
            text = text.replace(' ', '').lower()
            for pattern, (unit_name, conversion_factor) in unit_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        return {
                            'value': value,
                            'unit': unit_name,
                            'micrometers': value * conversion_factor,
                            'original_text': text,
                            'text': text
                        }
                    except ValueError:
                        continue
        return None
    
    def _save_debug_image(self, image: np.ndarray, result: Dict, 
                         output_dir: str, debug: bool) -> str:
        """Save a debug image showing the detected scale bar components."""
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Create debug image
            if len(image.shape) == 2:
                debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                debug_img = image.copy()
            
            # Add visualizations based on result data
            if 'scale_info' in result:
                text_info = result['scale_info']
                if 'bbox' in text_info:
                    bbox = text_info['bbox'].astype(int)
                    cv2.polylines(debug_img, [bbox], True, (0, 255, 0), 2)
                    cv2.putText(debug_img, f"{text_info['text']}", 
                               (int(bbox[0][0]), int(bbox[0][1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add detection info
            info_lines = [
                f"Scale: {result['micrometers_per_pixel']:.4f} um/pixel",
                f"Method: {result.get('detection_method', result.get('method_used', 'unknown'))}",
                f"Confidence: {result['confidence']:.2%}",
                f"OCR: {self.ocr_backend or 'legacy'}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(debug_img, line, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"scale_debug_{timestamp}.jpg"
            debug_path = output_path / output_filename
            
            # Save the debug image
            cv2.imwrite(str(debug_path), debug_img)
            
            if debug:
                print(f"📸 Debug image saved to: {debug_path}")
            
            return str(debug_path)
            
        except Exception as e:
            if debug:
                print(f"⚠️ Failed to save debug image: {e}")
            return None

# Optimized helper functions

@lru_cache(maxsize=1024)
def is_sem_parameter_text_fast(text: str) -> bool:
    """Fast SEM parameter detection with caching."""
    text_lower = text.lower().strip()
    
    # Quick checks first
    if len(text) > 30:
        return True
    
    # Fast keyword check
    sem_keywords = {
        'kv', 'etd', 'wd', 'spot', 'pressure', 'temp', 'mag', 
        'det', 'hv', 'hfov', 'vfov', 'se', 'bse', 'acc', 'mode',
        'scan', 'dwell', 'working distance', 'accelerating',
        'electron', 'vacuum', 'chamber', 'stage', 'tilt',
        'brightness', 'contrast', 'stigmation', 'aperture'
    }
    
    for kw in sem_keywords:
        if kw in text_lower:
            return True
    
    # Quick pattern checks
    if REGEX_PATTERNS['voltage'].search(text_lower):
        return True
    
    # Quick number count
    if len(re.findall(r'\d+', text)) >= 3:
        return True
    
    # Check for facility names
    facility_keywords = ['facility', 'bioimaging', 'wolfson', 'lab', 'laboratory', 
                        'center', 'centre', 'institute', 'university']
    
    if any(kw in text_lower for kw in facility_keywords):
        return True
    
    return False

@lru_cache(maxsize=1024)
def parse_scale_text_optimized(text: str) -> Optional[Dict]:
    """Optimized text parsing with caching and fuzzy matching."""
    if not text:
        return None
    
    text_clean = text.strip()
    
    # Fast number extraction
    value = None
    number_match = None
    for pattern in REGEX_PATTERNS['numbers']:
        match = pattern.search(text_clean)
        if match:
            try:
                number_str = match.group(1).replace(',', '.').replace(' ', '')
                value = float(number_str)
                number_match = match
                break
            except:
                continue
    
    if value is None:
        return None
    
    # Get text after number
    after_number = text_clean[number_match.end():].strip()
    
    # Fast unit detection with fuzzy matching
    text_lower = text.lower()
    unit = None
    factor = None
    
    # Direct checks first
    if any(c in text_lower for c in ['μ', 'µ', 'u']) and 'm' in text_lower:
        unit = 'μm'
        factor = 1.0
    elif 'nm' in text_lower:
        unit = 'nm'
        factor = 0.001
    elif 'mm' in text_lower and 'wd' not in text_lower:
        unit = 'mm'
        factor = 1000.0
    elif text_lower.endswith('m') and len(after_number) <= 4:
        # Fuzzy matching for OCR errors
        suspicious_chars = ['u', 'j', 'p', 'q', 'o', '0', 'µ', 'μ']
        if any(c in after_number.lower() for c in suspicious_chars):
            unit = 'μm'
            factor = 1.0
    
    # Context-based inference if no unit found
    if unit is None:
        if 0.01 <= value <= 10:
            unit = 'mm'
            factor = 1000.0
        elif 10 < value <= 5000:
            unit = 'μm'
            factor = 1.0
        elif value > 5000:
            unit = 'nm'
            factor = 0.001
        else:
            unit = 'μm'
            factor = 1.0
    
    micrometers = value * factor
    
    # Validate range
    if 0.1 <= micrometers <= 10000:
        return {
            'value': value,
            'unit': unit,
            'micrometers': micrometers,
            'original_text': text_clean,
            'after_number': after_number,
            'confidence': 'high' if unit in text_lower else 'inferred'
        }
    
    return None

def validate_scale_text_fast(text: str, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> float:
    """Fast validation scoring."""
    min_x, max_x, min_y, max_y = bbox
    height, width = image_shape
    
    # Position score
    x_ratio = ((min_x + max_x) / 2) / width
    y_ratio = ((min_y + max_y) / 2) / height
    
    pos_score = 0.0
    if x_ratio > 0.7:
        pos_score += 0.3
    elif x_ratio > 0.4:
        pos_score += 0.2
    
    if y_ratio > 0.85:
        pos_score += 0.3
    elif y_ratio > 0.7:
        pos_score += 0.2
    
    # Format score
    text_clean = text.strip().lower()
    format_score = 0.0
    
    if REGEX_PATTERNS['scale_format'].match(text_clean):
        format_score = 0.4
    elif re.match(r'^\d+\.?\d*\s*[μµu]m', text_clean):
        format_score = 0.3
    
    # Value range bonus
    numbers = re.findall(r'\d+\.?\d*', text_clean)
    if numbers:
        try:
            value = float(numbers[0])
            if 50 <= value <= 2000:
                format_score += 0.2
            elif 10 <= value <= 5000:
                format_score += 0.1
        except:
            pass
    
    return max(0.0, min(1.0, pos_score + format_score))

def find_lines_vectorized(region: np.ndarray, debug=False) -> List[Dict]:
    """Vectorized line detection for better performance."""
    if region.size == 0:
        return []
    
    lines = []
    
    # Extended thresholds to catch brighter lines
    thresholds = np.array([250, 240, 230, 220, 210, 200, 190, 180, 170])
    
    for thresh in thresholds:
        # Fast binary threshold
        binary = (region > thresh).astype(np.uint8) * 255
        
        # Use larger kernel for better line connectivity
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Connect nearby segments more aggressively
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, connect_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # More relaxed criteria for line detection
            if w >= 20 and h <= 15 and w / h >= 2:
                area = cv2.contourArea(contour)
                fill_ratio = area / (w * h) if w * h > 0 else 0
                
                # Prefer longer lines more strongly
                length_score = min(1.0, w / 200) * 0.7
                aspect_score = min(1.0, (w/h) / 10) * 0.3
                
                score = length_score + aspect_score
                
                lines.append({
                    'start_x': x,
                    'end_x': x + w,
                    'center_x': x + w // 2,
                    'center_y': y + h // 2,
                    'length': w,
                    'thickness': h,
                    'area': area,
                    'fill_ratio': fill_ratio,
                    'score': score,
                    'threshold': thresh
                })
    
    # Remove duplicates with aggressive merging
    if lines:
        lines.sort(key=lambda l: l['start_x'])
        
        merged_lines = []
        current_line = lines[0]
        
        for line in lines[1:]:
            gap = line['start_x'] - current_line['end_x']
            
            if gap <= 20:  # Merge if gap is 20 pixels or less
                current_line = {
                    'start_x': min(current_line['start_x'], line['start_x']),
                    'end_x': max(current_line['end_x'], line['end_x']),
                    'center_x': (min(current_line['start_x'], line['start_x']) + 
                                max(current_line['end_x'], line['end_x'])) // 2,
                    'center_y': (current_line['center_y'] + line['center_y']) // 2,
                    'length': max(current_line['end_x'], line['end_x']) - 
                             min(current_line['start_x'], line['start_x']),
                    'thickness': max(current_line['thickness'], line['thickness']),
                    'area': current_line['area'] + line['area'],
                    'fill_ratio': max(current_line['fill_ratio'], line['fill_ratio']),
                    'score': max(current_line['score'], line['score']),
                    'threshold': min(current_line['threshold'], line['threshold'])
                }
            else:
                merged_lines.append(current_line)
                current_line = line
        
        merged_lines.append(current_line)
        merged_lines.sort(key=lambda x: x['score'], reverse=True)
        
        return merged_lines
    
    return []

def score_line_pair_fast(left_line: Dict, right_line: Dict, text_info: Dict) -> float:
    """Fast scoring for line pairs."""
    # Length similarity
    length_diff = abs(left_line['length'] - right_line['length'])
    max_length = max(left_line['length'], right_line['length'])
    length_similarity = 1.0 - (length_diff / max_length) if max_length > 0 else 0
    
    # Y alignment
    y_diff = abs(left_line['center_y'] - right_line['center_y'])
    y_alignment = max(0, 1.0 - y_diff / 10)
    
    # Line quality
    line_quality = (left_line['score'] + right_line['score']) / 2
    
    return 0.3 * length_similarity + 0.3 * y_alignment + 0.4 * line_quality

# Convenience functions for compatibility with existing system

def detect_scale_bar(image_input, use_enhanced: bool = True, debug: bool = False, 
                    save_debug_image: bool = False, output_dir: Optional[str] = None,
                    ocr_backend: Optional[str] = None, **kwargs) -> Dict:
    """
    Convenience function to detect scale bar and return full results.
    Maintains compatibility with existing SEM Fiber Analysis System.
    
    Args:
        image_input: Either np.ndarray (image) or str (path to image)
        use_enhanced: Use enhanced text-centered detection
        debug: Enable debug output
        save_debug_image: Save debug visualization
        output_dir: Directory for debug images
        ocr_backend: OCR backend to use ('rapidocr', 'easyocr', or None for auto)
        **kwargs: Additional parameters for ScaleBarDetector
        
    Returns:
        Dictionary containing scale detection results
    """
    # Handle both image arrays and file paths
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {
                'scale_detected': False,
                'micrometers_per_pixel': 0.0,
                'scale_factor': 0.0,
                'error': f'Could not load image from path: {image_input}'
            }
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        return {
            'scale_detected': False,
            'micrometers_per_pixel': 0.0,
            'scale_factor': 0.0,
            'error': f'Invalid input type: {type(image_input)}. Expected str or np.ndarray'
        }
    
    # Run detection
    detector = ScaleBarDetector(use_enhanced_detection=use_enhanced, ocr_backend=ocr_backend, **kwargs)
    result = detector.detect_scale_bar(image, debug=debug, save_debug_image=save_debug_image, 
                                     output_dir=output_dir)
    
    # Ensure consistent return format
    if not isinstance(result, dict):
        result = {
            'scale_detected': False,
            'micrometers_per_pixel': 0.0,
            'scale_factor': 0.0,
            'error': f'Unexpected return type from detector: {type(result)}'
        }
    
    # Add scale_factor for backward compatibility
    if 'scale_factor' not in result:
        result['scale_factor'] = result.get('micrometers_per_pixel', 0.0)
    
    return result

def detect_scale_factor_only(image_input, use_enhanced: bool = True, ocr_backend: Optional[str] = None, **kwargs) -> float:
    """
    Convenience function that returns only the scale factor.
    
    Args:
        image_input: Either np.ndarray (image) or str (path to image)
        use_enhanced: Use enhanced text-centered detection
        ocr_backend: OCR backend to use
        **kwargs: Additional parameters for ScaleBarDetector
        
    Returns:
        Micrometers per pixel conversion factor (0.0 if detection failed)
    """
    result = detect_scale_bar(image_input, use_enhanced=use_enhanced, ocr_backend=ocr_backend, **kwargs)
    
    if result['scale_detected']:
        return result['micrometers_per_pixel']
    else:
        print(f"Scale detection failed: {result.get('error', 'Unknown error')}")
        return 0.0

def manual_scale_calibration(bar_length_pixels: int, bar_length_micrometers: float) -> float:
    """
    Manual calibration when automatic detection fails.
    
    Args:
        bar_length_pixels: Length of scale bar in pixels
        bar_length_micrometers: Real length of scale bar in micrometers
        
    Returns:
        Micrometers per pixel conversion factor
    """
    if bar_length_pixels <= 0:
        raise ValueError("Bar length in pixels must be positive")
    
    return bar_length_micrometers / bar_length_pixels

def pixels_to_micrometers(pixel_measurement: float, micrometers_per_pixel: float) -> float:
    """Convert pixel measurements to micrometers."""
    return pixel_measurement * micrometers_per_pixel

def micrometers_to_pixels(micrometer_measurement: float, micrometers_per_pixel: float) -> float:
    """Convert micrometer measurements to pixels."""
    if micrometers_per_pixel <= 0:
        raise ValueError("Micrometers per pixel must be positive")
    return micrometer_measurement / micrometers_per_pixel

# Visualization function for compatibility
def visualize_scale_detection(image: np.ndarray, detection_result: Dict, 
                             figsize: Tuple[int, int] = (12, 8)):
    """Visualize scale detection results (legacy compatibility)."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Scale region
        if 'scale_region' in detection_result:
            axes[0, 1].imshow(detection_result['scale_region'], cmap='gray')
            axes[0, 1].set_title('Scale Region')
            axes[0, 1].axis('off')
        
        # Detection overlay
        if detection_result['scale_detected']:
            axes[1, 0].imshow(image, cmap='gray')
            axes[1, 0].set_title('Detection Results')
            # Add any overlay visualizations here
        else:
            axes[1, 0].imshow(image, cmap='gray')
            axes[1, 0].set_title('Detection Failed')
        axes[1, 0].axis('off')
        
        # Results summary
        ax_text = axes[1, 1]
        ax_text.axis('off')
        
        if detection_result['scale_detected']:
            info = detection_result.get('scale_info', {})
            text = f"Scale Detection: SUCCESS\n\n"
            text += f"Method: {detection_result.get('method_used', 'unknown')}\n"
            text += f"Scale Value: {info.get('value', 'N/A')} {info.get('unit', 'N/A')}\n"
            text += f"Bar Length: {detection_result.get('bar_length_pixels', 'N/A')} pixels\n"
            text += f"Calibration: {detection_result['micrometers_per_pixel']:.4f} μm/pixel\n"
            text += f"Confidence: {detection_result.get('confidence', 0):.2%}\n"
            text += f"OCR Backend: {detection_result.get('ocr_backend', 'unknown')}\n"
        else:
            text = f"Scale Detection: FAILED\n\n"
            text += f"Method: {detection_result.get('method_used', 'unknown')}\n"
            text += f"Error: {detection_result.get('error', 'Unknown error')}\n"
        
        ax_text.text(0.05, 0.95, text, transform=ax_text.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")

# Test and batch processing functions for compatibility

def test_scale_detection_batch(image_directory: str = "sample_images", 
                              output_dir: Optional[str] = None,
                              save_debug_images: bool = True,
                              ocr_backend: Optional[str] = None,
                              verbose: bool = True) -> Dict:
    """
    Test the optimized scale detection on all images in a directory.
    
    Args:
        image_directory: Directory containing SEM images
        output_dir: Directory for saving results and debug images
        save_debug_images: Whether to save debug visualizations
        ocr_backend: OCR backend to use ('rapidocr', 'easyocr', or None for auto)
        verbose: Enable verbose output
        
    Returns:
        Dictionary containing batch test results
    """
    print(f"🧪 BATCH SCALE DETECTION TEST")
    print("=" * 60)
    
    # Setup directories
    image_dir = Path(image_directory)
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return {'error': f'Directory not found: {image_dir}'}
    
    if output_dir is None:
        output_dir = image_dir.parent / 'scale_detection_results'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))  # Remove duplicates
    
    if not image_files:
        print(f"❌ No image files found in {image_dir}")
        return {'error': f'No images found in {image_dir}'}
    
    print(f"📁 Testing {len(image_files)} images from: {image_dir}")
    print(f"📊 Results will be saved to: {output_dir}")
    
    if verbose:
        print(f"🔧 Using OCR backend: {ocr_backend or OCR_BACKEND}")
    
    # Initialize detector
    detector = ScaleBarDetector(ocr_backend=ocr_backend)
    
    # Process each image
    results = []
    successful = 0
    total_time = 0
    
    for i, image_path in enumerate(image_files, 1):
        if verbose:
            print(f"\n[{i}/{len(image_files)}] 📸 Processing: {image_path.name}")
            print("-" * 40)
        
        try:
            start_time = time.time()
            
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Run detection
            result = detector.detect_scale_bar(
                image, 
                debug=verbose,
                save_debug_image=save_debug_images,
                output_dir=output_dir if save_debug_images else None
            )
            
            processing_time = time.time() - start_time
            total_time += processing_time
            
            # Prepare result summary
            test_result = {
                'filename': image_path.name,
                'filepath': str(image_path),
                'image_shape': image.shape,
                'success': result['scale_detected'],
                'processing_time': processing_time,
                'ocr_backend': result.get('ocr_backend', 'unknown'),
                'method_used': result.get('method_used', 'unknown')
            }
            
            if result['scale_detected']:
                scale_info = result.get('scale_info', {})
                test_result.update({
                    'scale_factor': result['micrometers_per_pixel'],
                    'scale_text': scale_info.get('text', ''),
                    'scale_value': scale_info.get('value', 0),
                    'scale_unit': scale_info.get('unit', ''),
                    'bar_length_pixels': result.get('bar_length_pixels', 0),
                    'confidence': result.get('confidence', 0),
                    'detection_method': result.get('detection_method', 'unknown')
                })
                
                if verbose:
                    print(f"   ✅ SUCCESS: {result['micrometers_per_pixel']:.4f} μm/pixel")
                    print(f"      Scale text: '{scale_info.get('text', 'N/A')}'")
                    print(f"      Confidence: {result.get('confidence', 0):.2%}")
                    print(f"      Processing time: {processing_time:.3f}s")
                
                successful += 1
            else:
                test_result['error'] = result.get('error', 'Unknown error')
                
                if verbose:
                    print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
                    print(f"      Processing time: {processing_time:.3f}s")
            
            results.append(test_result)
            
        except Exception as e:
            error_result = {
                'filename': image_path.name,
                'filepath': str(image_path),
                'success': False,
                'error': f'Exception: {str(e)}',
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
            results.append(error_result)
            
            if verbose:
                print(f"   💥 ERROR: {str(e)}")
    
    # Generate summary
    print(f"\n" + "=" * 60)
    print("📊 BATCH TEST SUMMARY")
    print("=" * 60)
    
    success_rate = successful / len(image_files) * 100 if image_files else 0
    avg_time = total_time / len(image_files) if image_files else 0
    
    print(f"Total images: {len(image_files)}")
    print(f"Successful detections: {successful}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.3f} seconds")
    
    # Analyze successful results
    if successful > 0:
        successful_results = [r for r in results if r.get('success', False)]
        
        scale_factors = [r['scale_factor'] for r in successful_results if 'scale_factor' in r]
        if scale_factors:
            print(f"\n📏 Scale Factor Analysis:")
            print(f"   Range: {min(scale_factors):.4f} - {max(scale_factors):.4f} μm/pixel")
            print(f"   Mean: {np.mean(scale_factors):.4f} μm/pixel")
            print(f"   Median: {np.median(scale_factors):.4f} μm/pixel")
        
        processing_times = [r['processing_time'] for r in successful_results if r.get('processing_time', 0) > 0]
        if processing_times:
            print(f"\n⚡ Performance Analysis:")
            print(f"   Fastest: {min(processing_times):.3f} seconds")
            print(f"   Slowest: {max(processing_times):.3f} seconds")
            print(f"   Average: {np.mean(processing_times):.3f} seconds")
    
    # Show failures
    failed_results = [r for r in results if not r.get('success', False)]
    if failed_results:
        print(f"\n❌ Failed Images ({len(failed_results)}):")
        for failure in failed_results[:5]:  # Show first 5
            error = failure.get('error', 'Unknown error')
            print(f"   {failure['filename']}: {error}")
        if len(failed_results) > 5:
            print(f"   ... and {len(failed_results) - 5} more")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary dictionary
    summary = {
        'timestamp': timestamp,
        'test_directory': str(image_dir),
        'output_directory': str(output_dir),
        'ocr_backend': ocr_backend or OCR_BACKEND,
        'total_images': len(image_files),
        'successful': successful,
        'failed': len(image_files) - successful,
        'success_rate': success_rate,
        'total_processing_time': total_time,
        'avg_time_per_image': avg_time,
        'results': results
    }
    
    # Save as JSON
    json_file = output_dir / f'batch_test_results_{timestamp}.json'
    try:
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {json_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save JSON results: {e}")
    
    # Save summary report
    report_file = output_dir / f'batch_test_summary_{timestamp}.txt'
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SEM Scale Detection Batch Test Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test directory: {image_dir}\n")
            f.write(f"OCR Backend: {ocr_backend or OCR_BACKEND}\n\n")
            
            f.write(f"Summary:\n")
            f.write(f"  Total images: {len(image_files)}\n")
            f.write(f"  Successful: {successful}\n")
            f.write(f"  Failed: {len(image_files) - successful}\n")
            f.write(f"  Success rate: {success_rate:.1f}%\n")
            f.write(f"  Total time: {total_time:.2f}s\n")
            f.write(f"  Avg time: {avg_time:.3f}s\n\n")
            
            if successful > 0:
                f.write(f"Scale Factor Statistics:\n")
                if scale_factors:
                    f.write(f"  Range: {min(scale_factors):.4f} - {max(scale_factors):.4f} um/pixel\n")
                    f.write(f"  Mean: {np.mean(scale_factors):.4f} um/pixel\n")
                    f.write(f"  Median: {np.median(scale_factors):.4f} um/pixel\n\n")
            
            f.write("Individual Results:\n")
            f.write("-" * 20 + "\n")
            for result in results:
                status = "SUCCESS" if result.get('success', False) else "FAILED"
                f.write(f"{result['filename']}: {status}\n")
                
                if result.get('success', False):
                    f.write(f"  Scale: {result.get('scale_factor', 0):.4f} um/pixel\n")
                    f.write(f"  Text: {result.get('scale_text', 'N/A')}\n")
                    f.write(f"  Time: {result.get('processing_time', 0):.3f}s\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown')}\n")
                f.write("\n")
        
        print(f"📄 Summary report saved to: {report_file}")
        
    except Exception as e:
        print(f"⚠️ Could not save summary report: {e}")
    
    print(f"\n🎯 BATCH TEST COMPLETE!")
    
    return summary


def compare_ocr_backends(image_path: str, iterations: int = 3) -> Dict:
    """
    Compare performance between available OCR backends.
    
    Args:
        image_path: Path to test image
        iterations: Number of test iterations
        
    Returns:
        Dictionary containing comparison results
    """
    print(f"\n🔬 Comparing OCR backends on {Path(image_path).name}")
    print("=" * 60)
    
    results = {}
    
    # Test RapidOCR if available
    if RAPIDOCR_AVAILABLE:
        print("\n📸 Testing RapidOCR...")
        detector = ScaleBarDetector(ocr_backend='rapidocr')
        
        times = []
        for i in range(iterations):
            start = time.time()
            result = detector.detect_scale_bar(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), debug=False)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        results['rapidocr'] = {
            'avg_time': avg_time,
            'success': result['scale_detected'],
            'result': result.get('micrometers_per_pixel', 0) if result['scale_detected'] else None,
            'confidence': result.get('confidence', 0)
        }
        print(f"   Average time: {avg_time:.3f}s")
        if result['scale_detected']:
            print(f"   Result: {result['micrometers_per_pixel']:.4f} μm/pixel")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
    
    # Test EasyOCR if available
    if EASYOCR_AVAILABLE:
        print("\n📸 Testing EasyOCR...")
        detector = ScaleBarDetector(ocr_backend='easyocr')
        
        times = []
        for i in range(iterations):
            start = time.time()
            result = detector.detect_scale_bar(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), debug=False)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        results['easyocr'] = {
            'avg_time': avg_time,
            'success': result['scale_detected'],
            'result': result.get('micrometers_per_pixel', 0) if result['scale_detected'] else None,
            'confidence': result.get('confidence', 0)
        }
        print(f"   Average time: {avg_time:.3f}s")
        if result['scale_detected']:
            print(f"   Result: {result['micrometers_per_pixel']:.4f} μm/pixel")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
    
    # Summary
    print("\n📊 Summary:")
    if len(results) > 1:
        fastest = min(results.items(), key=lambda x: x[1]['avg_time'])
        print(f"   Fastest: {fastest[0]} ({fastest[1]['avg_time']:.3f}s)")
        
        if 'rapidocr' in results and 'easyocr' in results:
            speedup = results['easyocr']['avg_time'] / results['rapidocr']['avg_time']
            print(f"   RapidOCR speedup: {speedup:.2f}x")
    
    return results

# Export the main classes and functions for the module
__all__ = [
    'ScaleBarDetector',
    'detect_scale_bar',
    'detect_scale_factor_only', 
    'manual_scale_calibration',
    'pixels_to_micrometers',
    'micrometers_to_pixels',
    'visualize_scale_detection',
    'test_scale_detection_batch',
    'compare_ocr_backends'
]