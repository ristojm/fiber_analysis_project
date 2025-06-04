"""
SEM Fiber Analysis System - Enhanced Scale Detection Module
Automatic detection and calibration of scale bars in SEM images.
UPDATED: Incorporates improved text-centered detection with aggressive line finding.
"""

import cv2
import numpy as np
import re
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
from skimage import morphology, measure
from scipy import ndimage

# Optional OCR imports - will handle gracefully if not available
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Scale text detection will be limited.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not available. Scale text detection will be limited.")

# If no OCR is available, we'll use manual calibration
if not PYTESSERACT_AVAILABLE and not EASYOCR_AVAILABLE:
    print("No OCR engines available. Using manual scale calibration mode.")

class ScaleBarDetector:
    """
    Enhanced scale bar detector with improved text-centered detection.
    """
    
    def __init__(self, 
                 scale_region_fraction: float = 0.15,
                 min_bar_length: int = 50,
                 max_bar_thickness: int = 20,
                 text_search_region: float = 0.3,
                 use_enhanced_detection: bool = True):
        """
        Initialize scale bar detector.
        
        Args:
            scale_region_fraction: Fraction of image height to search for scale bar
            min_bar_length: Minimum length of scale bar in pixels
            max_bar_thickness: Maximum thickness of scale bar in pixels
            text_search_region: Fraction of scale region to search for text
            use_enhanced_detection: Use new enhanced detection algorithm
        """
        self.scale_region_fraction = scale_region_fraction
        self.min_bar_length = min_bar_length
        self.max_bar_thickness = max_bar_thickness
        self.text_search_region = text_search_region
        self.use_enhanced_detection = use_enhanced_detection
        
        # Initialize OCR reader if available
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
            except:
                pass
    
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
    
    def find_scale_text_enhanced(self, scale_region: np.ndarray) -> List[Dict]:
        """
        Enhanced scale text detection with SEM-specific filtering and OCR error handling.
        """
        
        scale_candidates = []
        
        # Try EasyOCR first
        if self.easyocr_reader:
            try:
                results = self.easyocr_reader.readtext(scale_region, detail=1, width_ths=0.4, height_ths=0.4)
                
                for (bbox, text, confidence) in results:
                    text_clean = text.strip()
                    scale_info = self._parse_scale_text_enhanced(text_clean)
                    
                    if scale_info and confidence > 0.2:
                        bbox_array = np.array(bbox)
                        center_x = int(np.mean(bbox_array[:, 0]))
                        center_y = int(np.mean(bbox_array[:, 1]))
                        
                        bbox_width = np.max(bbox_array[:, 0]) - np.min(bbox_array[:, 0])
                        bbox_height = np.max(bbox_array[:, 1]) - np.min(bbox_array[:, 1])
                        
                        # SEM scale preference scoring
                        micrometers = scale_info['micrometers']
                        if 50 <= micrometers <= 1000:  # Most common SEM range
                            preference_score = 1.0
                        elif 10 <= micrometers < 50 or 1000 < micrometers <= 3000:
                            preference_score = 0.8
                        elif 1 <= micrometers < 10:
                            preference_score = 0.6
                        else:
                            preference_score = 0.3
                        
                        combined_score = confidence * 0.7 + preference_score * 0.3
                        
                        scale_candidates.append({
                            'text': text_clean,
                            'value': scale_info['value'],
                            'unit': scale_info['unit'],
                            'micrometers': micrometers,
                            'center_x': center_x,
                            'center_y': center_y,
                            'bbox': bbox_array,
                            'bbox_width': bbox_width,
                            'bbox_height': bbox_height,
                            'confidence': confidence,
                            'method': 'EasyOCR',
                            'preference_score': preference_score,
                            'combined_score': combined_score
                        })
            
            except Exception as e:
                print(f"EasyOCR failed: {e}")
        
        # Try Tesseract as backup
        if PYTESSERACT_AVAILABLE and len(scale_candidates) == 0:
            try:
                # Enhanced preprocessing for Tesseract
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                enhanced = clahe.apply(scale_region)
                
                config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.μµumnmkMm '
                data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT, config=config)
                
                for i, text in enumerate(data['text']):
                    if text.strip() and int(data['conf'][i]) > 20:
                        scale_info = self._parse_scale_text_enhanced(text.strip())
                        
                        if scale_info:
                            x = data['left'][i]
                            y = data['top'][i]
                            w = data['width'][i]
                            h = data['height'][i]
                            
                            center_x = x + w // 2
                            center_y = y + h // 2
                            confidence = int(data['conf'][i]) / 100.0
                            
                            # SEM preference scoring
                            micrometers = scale_info['micrometers']
                            if 50 <= micrometers <= 1000:
                                preference_score = 1.0
                            elif 10 <= micrometers < 50 or 1000 < micrometers <= 3000:
                                preference_score = 0.8
                            else:
                                preference_score = 0.6
                            
                            combined_score = confidence * 0.7 + preference_score * 0.3
                            
                            scale_candidates.append({
                                'text': text.strip(),
                                'value': scale_info['value'],
                                'unit': scale_info['unit'],
                                'micrometers': micrometers,
                                'center_x': center_x,
                                'center_y': center_y,
                                'bbox': np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]),
                                'bbox_width': w,
                                'bbox_height': h,
                                'confidence': confidence,
                                'method': 'Tesseract',
                                'preference_score': preference_score,
                                'combined_score': combined_score
                            })
            
            except Exception as e:
                print(f"Tesseract failed: {e}")
        
        # Remove duplicates and sort by combined score
        unique_candidates = self._remove_duplicate_text_candidates(scale_candidates)
        unique_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return unique_candidates
    
    def _parse_scale_text_enhanced(self, text: str) -> Optional[Dict]:
        """
        Enhanced scale text parsing with comprehensive OCR error handling.
        """
        
        # Comprehensive patterns including common OCR errors
        patterns = [
            # Standard patterns
            (r'(\d+\.?\d*)\s*μm', 'μm', 1.0),
            (r'(\d+\.?\d*)\s*µm', 'μm', 1.0),
            (r'(\d+\.?\d*)\s*um', 'μm', 1.0),
            
            # Common OCR errors for μm
            (r'(\d+\.?\d*)\s*jm', 'μm', 1.0),       # jm instead of μm (very common!)
            (r'(\d+\.?\d*)\s*[|l1]m', 'μm', 1.0),   # |m, lm, 1m instead of μm
            (r'(\d+\.?\d*)\s*[Oo]m', 'μm', 1.0),    # Om instead of μm
            (r'(\d+\.?\d*)\s*[Oo]jm', 'μm', 1.0),   # Ojm instead of μm
            (r'(\d+\.?\d*)\s*jim', 'μm', 1.0),      # jim instead of μm
            (r'(\d+\.?\d*)\s*μrn', 'μm', 1.0),      # μrn instead of μm
            (r'(\d+\.?\d*)\s*urn', 'μm', 1.0),      # urn instead of μm
            (r'(\d+\.?\d*)\s*jun', 'μm', 1.0),      # jun instead of μm
            (r'(\d+\.?\d*)\s*μn', 'μm', 1.0),       # μn instead of μm
            (r'(\d+\.?\d*)\s*pm', 'μm', 1.0),       # pm instead of μm
            
            # Other units
            (r'(\d+\.?\d*)\s*nm', 'nm', 0.001),
            (r'(\d+\.?\d*)\s*mm', 'mm', 1000.0),
            (r'(\d+\.?\d*)\s*cm', 'cm', 10000.0),
            
            # Decimal variations
            (r'(\d+)[,.](\d+)\s*μm', 'μm', 1.0),
            (r'(\d+)[,.](\d+)\s*jm', 'μm', 1.0),
            (r'(\d+)[,.](\d+)\s*[|l1]m', 'μm', 1.0),
            (r'(\d+)[,.](\d+)\s*mm', 'mm', 1000.0),
        ]
        
        text_clean = text.lower().replace(' ', '').replace('\n', '')
        
        for pattern, unit, conversion in patterns:
            match = re.search(pattern, text_clean)
            if match:
                try:
                    if len(match.groups()) > 1:
                        # Handle decimal patterns
                        value = float(f"{match.group(1)}.{match.group(2)}")
                    else:
                        value = float(match.group(1))
                    
                    micrometers = value * conversion
                    
                    # SEM scale filtering: 0.05μm to 3000μm (50nm to 3mm)
                    if 0.05 <= micrometers <= 3000:
                        return {
                            'value': value,
                            'unit': unit,
                            'micrometers': micrometers,
                            'pattern_matched': pattern,
                            'original_text': text
                        }
                
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _remove_duplicate_text_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate text candidates."""
        
        unique = []
        seen_values = set()
        
        for candidate in candidates:
            key = (candidate['value'], candidate['unit'])
            if key not in seen_values:
                seen_values.add(key)
                unique.append(candidate)
        
        return unique
    
    def find_complete_scale_bar_span(self, scale_region: np.ndarray, text_center_x: int, 
                                   text_center_y: int, text_bbox: np.ndarray) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Find the complete scale bar span using enhanced detection methods.
        """
        
        height, width = scale_region.shape
        
        # Search parameters
        y_search_radius = 15
        search_y_min = max(0, text_center_y - y_search_radius)
        search_y_max = min(height, text_center_y + y_search_radius)
        search_region = scale_region[search_y_min:search_y_max, :]
        
        # Generate white line detection masks
        white_masks = self._get_white_line_masks(search_region)
        
        # Extract segments from all masks
        all_segments = []
        for mask_name, white_mask in white_masks:
            segments = self._extract_horizontal_segments(white_mask, mask_name, search_y_min)
            all_segments.extend(segments)
        
        # Fallback detection if no segments found
        if len(all_segments) == 0:
            fallback_segments = self._try_fallback_detection(search_region, search_y_min)
            all_segments.extend(fallback_segments)
        
        if len(all_segments) == 0:
            return None, []
        
        # Group segments by Y level and find best span
        y_groups = self._group_segments_by_y_level(all_segments, text_center_y)
        best_span = self._find_best_complete_span(y_groups, text_center_x, text_center_y)
        
        return best_span, all_segments
    
    def _get_white_line_masks(self, search_region: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Generate multiple white line detection masks."""
        
        white_masks = []
        
        # Multiple threshold strategies
        thresholds = [240, 200, 150, 100]
        for thresh in thresholds:
            _, white_mask = cv2.threshold(search_region, thresh, 255, cv2.THRESH_BINARY)
            white_masks.append((f"thresh_{thresh}", white_mask))
        
        # Adaptive threshold
        try:
            adaptive = cv2.adaptiveThreshold(search_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 21, -10)
            white_masks.append(("adaptive", adaptive))
        except:
            pass
        
        # Otsu threshold
        _, otsu = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_masks.append(("otsu", otsu))
        
        # Percentile-based thresholds
        for p in [90, 95, 99]:
            thresh_val = np.percentile(search_region, p)
            if thresh_val > 50:
                _, p_mask = cv2.threshold(search_region, int(thresh_val * 0.9), 255, cv2.THRESH_BINARY)
                white_masks.append((f"p{p}", p_mask))
        
        # Edge detection
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(search_region)
            edges = cv2.Canny(enhanced, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            white_masks.append(("edges", edges_dilated))
        except:
            pass
        
        return white_masks
    
    def _extract_horizontal_segments(self, white_mask: np.ndarray, mask_name: str, y_offset: int) -> List[Dict]:
        """Extract horizontal segments from a white mask."""
        
        segments_all = []
        
        # Multiple morphological approaches
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        ]
        
        for i, kernel in enumerate(kernels):
            # Closing operation
            connected = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            segments = self._find_segments_in_mask(connected, f"{mask_name}_morph{i}", y_offset)
            segments_all.extend(segments)
        
        # Direct contour finding
        segments_direct = self._find_segments_in_mask(white_mask, f"{mask_name}_direct", y_offset)
        segments_all.extend(segments_direct)
        
        # Remove duplicates
        unique_segments = self._remove_duplicate_segments(segments_all)
        
        return unique_segments
    
    def _find_segments_in_mask(self, mask: np.ndarray, approach_name: str, y_offset: int) -> List[Dict]:
        """Find line segments in a binary mask."""
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segments = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Relaxed criteria for segments
            if aspect_ratio > 1.0 and w > 8 and h <= 12:
                full_y = y + y_offset
                
                segments.append({
                    'x': x,
                    'y': full_y,
                    'width': w,
                    'height': h,
                    'center_x': x + w // 2,
                    'center_y': full_y + h // 2,
                    'end_x': x + w,
                    'aspect_ratio': aspect_ratio,
                    'area': cv2.contourArea(contour),
                    'approach': approach_name
                })
        
        return segments
    
    def _remove_duplicate_segments(self, segments: List[Dict]) -> List[Dict]:
        """Remove duplicate segments."""
        
        if len(segments) <= 1:
            return segments
        
        unique = []
        for seg in segments:
            is_duplicate = False
            for existing in unique:
                x_overlap = max(0, min(seg['end_x'], existing['end_x']) - max(seg['x'], existing['x']))
                y_overlap = max(0, min(seg['y'] + seg['height'], existing['y'] + existing['height']) - 
                              max(seg['y'], existing['y']))
                
                if x_overlap > 0.7 * min(seg['width'], existing['width']) and y_overlap > 0:
                    if seg['aspect_ratio'] > existing['aspect_ratio']:
                        unique.remove(existing)
                        unique.append(seg)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(seg)
        
        return unique
    
    def _try_fallback_detection(self, search_region: np.ndarray, y_offset: int) -> List[Dict]:
        """Fallback detection for very faint scale bars."""
        
        fallback_segments = []
        
        try:
            # Look for brightness variations row by row
            row_means = np.mean(search_region, axis=1)
            overall_mean = np.mean(row_means)
            
            for y, row_mean in enumerate(row_means):
                if row_mean > overall_mean * 1.1:  # 10% brighter than average
                    row = search_region[y, :]
                    bright_threshold = np.mean(row) * 1.2
                    bright_mask = row > bright_threshold
                    
                    # Find continuous bright regions
                    diff = np.diff(np.concatenate(([False], bright_mask, [False])).astype(int))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    
                    for start, end in zip(starts, ends):
                        width = end - start
                        if width > 15:
                            fallback_segments.append({
                                'x': start,
                                'y': y + y_offset,
                                'width': width,
                                'height': 1,
                                'center_x': start + width // 2,
                                'center_y': y + y_offset,
                                'end_x': end,
                                'aspect_ratio': width,
                                'area': width,
                                'approach': 'fallback_brightness'
                            })
        
        except Exception:
            pass
        
        return fallback_segments
    
    def _group_segments_by_y_level(self, all_segments: List[Dict], text_center_y: int) -> List[List[Dict]]:
        """Group segments by Y level."""
        
        # Filter segments near text Y level
        relevant_segments = [seg for seg in all_segments 
                           if abs(seg['center_y'] - text_center_y) <= 20]
        
        relevant_segments.sort(key=lambda x: x['center_y'])
        
        groups = []
        y_tolerance = 10
        i = 0
        
        while i < len(relevant_segments):
            current_y = relevant_segments[i]['center_y']
            group = []
            
            for j in range(i, len(relevant_segments)):
                if abs(relevant_segments[j]['center_y'] - current_y) <= y_tolerance:
                    group.append(relevant_segments[j])
                else:
                    break
            
            if len(group) >= 1:
                groups.append(group)
            
            i += len(group)
        
        return groups
    
    def _find_best_complete_span(self, y_groups: List[List[Dict]], text_center_x: int, text_center_y: int) -> Optional[Dict]:
        """Find the best complete span from grouped segments."""
        
        if not y_groups:
            return None
        
        span_candidates = []
        
        for group in y_groups:
            group.sort(key=lambda x: x['x'])
            
            leftmost_x = min(seg['x'] for seg in group)
            rightmost_x = max(seg['end_x'] for seg in group)
            total_span = rightmost_x - leftmost_x
            
            text_relative_pos = (text_center_x - leftmost_x) / total_span if total_span > 0 else 0.5
            
            # Score the span
            score = self._score_complete_span(group, total_span, text_relative_pos)
            
            span_candidates.append({
                'segments': group,
                'leftmost_x': leftmost_x,
                'rightmost_x': rightmost_x,
                'total_span': total_span,
                'segment_count': len(group),
                'text_relative_pos': text_relative_pos,
                'score': score,
                'average_y': sum(seg['center_y'] for seg in group) / len(group),
                'text_centrality_score': 1.0 - 2 * abs(text_relative_pos - 0.5)
            })
        
        span_candidates.sort(key=lambda x: x['score'], reverse=True)
        return span_candidates[0] if span_candidates else None
    
    def _score_complete_span(self, segments: List[Dict], total_span: float, text_relative_pos: float) -> float:
        """Score a complete span based on multiple criteria."""
        
        score = 0.0
        
        # Span length score
        if 100 <= total_span <= 600:
            span_score = 1.0
        elif 50 <= total_span < 100 or 600 < total_span <= 800:
            span_score = 0.7
        elif total_span < 50:
            span_score = 0.3
        else:
            span_score = 0.5
        
        score += span_score * 0.4
        
        # Text centrality score
        centrality = 1.0 - 2 * abs(text_relative_pos - 0.5)
        score += max(0, centrality) * 0.3
        
        # Segment structure score
        if len(segments) == 2:
            structure_score = 1.0  # Ideal: two segments
        elif len(segments) == 1:
            structure_score = 0.6  # Single segment
        elif 2 <= len(segments) <= 4:
            structure_score = 0.8  # Multiple segments
        else:
            structure_score = 0.3  # Too many segments
        
        score += structure_score * 0.3
        
        return score
    
    def detect_scale_bar(self, image: np.ndarray) -> Dict:
        """
        Main function to detect scale bar and calculate calibration.
        Uses enhanced detection if enabled, falls back to original method.
        
        Args:
            image: Input SEM image
            
        Returns:
            Dictionary containing scale detection results
        """
        # Extract scale region
        scale_region, y_offset = self.extract_scale_region(image)
        
        result = {
            'scale_detected': False,
            'scale_region': scale_region,
            'y_offset': y_offset,
            'micrometers_per_pixel': None,
            'scale_info': None,
            'error': None,
            'method_used': 'enhanced' if self.use_enhanced_detection else 'original'
        }
        
        if self.use_enhanced_detection:
            # Use enhanced text-centered detection
            text_candidates = self.find_scale_text_enhanced(scale_region)
            
            if not text_candidates:
                result['error'] = "No valid scale text found with enhanced detection"
                return result
            
            best_text = text_candidates[0]
            result['scale_info'] = {
                'value': best_text['value'],
                'unit': best_text['unit'],
                'micrometers': best_text['micrometers'],
                'original_text': best_text['text'],
                'confidence': best_text['confidence'],
                'method': best_text['method']
            }
            
            # Find complete scale bar span
            best_span, all_segments = self.find_complete_scale_bar_span(
                scale_region, 
                best_text['center_x'], 
                best_text['center_y'], 
                best_text['bbox']
            )
            
            if best_span:
                micrometers_per_pixel = best_text['micrometers'] / best_span['total_span']
                
                result.update({
                    'scale_detected': True,
                    'micrometers_per_pixel': micrometers_per_pixel,
                    'bar_length_pixels': best_span['total_span'],
                    'segments_found': len(all_segments),
                    'segments_in_span': best_span['segment_count'],
                    'text_centrality': best_span['text_relative_pos'],
                    'detection_score': best_span['score']
                })
            else:
                result['error'] = "Scale bar lines not detected with enhanced method"
        
        else:
            # Fall back to original detection method
            bar_candidates = self.detect_scale_bar_line(scale_region)
            
            if not bar_candidates:
                result['error'] = "No scale bar candidates detected"
                return result
            
            # Try OCR to extract scale text
            text_lines = []
            if self.easyocr_reader:
                text_lines.extend(self.extract_scale_text_easyocr(scale_region, bar_candidates))
            if not text_lines and PYTESSERACT_AVAILABLE:
                text_lines.extend(self.extract_scale_text_pytesseract(scale_region, bar_candidates))
            
            if not text_lines:
                result['error'] = "No text extracted from scale region"
                return result
            
            # Parse scale information
            scale_info = self.parse_scale_text(text_lines)
            if not scale_info:
                result['error'] = "Could not parse scale information from text"
                return result
            
            # Calculate calibration
            best_bar = bar_candidates[0]
            bar_length = best_bar['length']
            micrometers_per_pixel = self.calculate_pixel_scale(scale_info, bar_length)
            
            result.update({
                'scale_detected': True,
                'micrometers_per_pixel': micrometers_per_pixel,
                'scale_info': scale_info,
                'best_bar': best_bar,
                'bar_length_pixels': bar_length,
                'bar_candidates': bar_candidates,
                'extracted_text': text_lines
            })
        
        return result
    
    # Keep original methods for backward compatibility
    def detect_scale_bar_line(self, scale_region: np.ndarray) -> List[Dict]:
        """Original scale bar line detection method."""
        # [Original implementation from your existing code]
        # This is kept for backward compatibility
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
        
        candidates.sort(key=self._calculate_bar_confidence, reverse=True)
        return candidates
    
    def _calculate_bar_confidence(self, candidate: Dict) -> float:
        """Calculate confidence score for scale bar candidate."""
        score = 0.0
        length_score = min(1.0, candidate['length'] / 200)
        score += 0.3 * length_score
        aspect_score = min(1.0, candidate['aspect_ratio'] / 20)
        score += 0.3 * aspect_score
        fill_score = min(1.0, candidate['fill_ratio'] / 0.7)
        score += 0.2 * fill_score
        relative_x = candidate['center_x'] / 640
        position_score = 1.0 - 2 * abs(relative_x - 0.5)
        score += 0.2 * max(0, position_score)
        return score
    
    def extract_scale_text_pytesseract(self, scale_region: np.ndarray, bar_candidates: List[Dict]) -> List[str]:
        """Extract text using Tesseract (original method)."""
        if not PYTESSERACT_AVAILABLE:
            return []
        
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(scale_region)
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789µμmnmkMKMm.μ '
            
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, enhanced)
                temp_path = temp_file.name
            
            try:
                text = pytesseract.image_to_string(temp_path, config=config)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                return lines
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")
            return []
    
    def extract_scale_text_easyocr(self, scale_region: np.ndarray, bar_candidates: List[Dict]) -> List[str]:
        """Extract text using EasyOCR (original method)."""
        if not self.easyocr_reader:
            return []
        
        try:
            results = self.easyocr_reader.readtext(scale_region)
            texts = [result[1] for result in results if result[2] > 0.5]
            return texts
        except Exception as e:
            print(f"EasyOCR failed: {e}")
            return []
    
    def parse_scale_text(self, text_lines: List[str]) -> Optional[Dict]:
        """Parse scale text (original method)."""
        unit_patterns = {
            r'(\d+\.?\d*)\s*μm': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*µm': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*um': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*nm': ('nanometer', 0.001),
            r'(\d+\.?\d*)\s*mm': ('millimeter', 1000.0),
            r'(\d+\.?\d*)\s*cm': ('centimeter', 10000.0),
            r'(\d+\.?\d*)\s*m(?!m)': ('meter', 1000000.0),
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
                            'original_text': text
                        }
                    except ValueError:
                        continue
        return None
    
    def calculate_pixel_scale(self, scale_info: Dict, bar_length_pixels: int) -> float:
        """Calculate micrometers per pixel conversion factor."""
        if bar_length_pixels <= 0:
            return 0.0
        micrometers = scale_info['micrometers']
        return micrometers / bar_length_pixels
    
    def visualize_scale_detection(self, image: np.ndarray, detection_result: Dict, figsize: Tuple[int, int] = (12, 8)):
        """Visualize scale detection results."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Scale region
        scale_region = detection_result['scale_region']
        axes[0, 1].imshow(scale_region, cmap='gray')
        axes[0, 1].set_title('Scale Region')
        axes[0, 1].axis('off')
        
        # Detection visualization
        if detection_result['scale_detected']:
            overlay = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
            
            if detection_result['method_used'] == 'enhanced':
                # Show enhanced detection results
                axes[1, 0].imshow(overlay)
                axes[1, 0].set_title('Enhanced Detection (Text-Centered)')
            else:
                # Show original detection results
                if 'bar_candidates' in detection_result:
                    for i, candidate in enumerate(detection_result['bar_candidates'][:3]):
                        x, y, w, h = candidate['bbox']
                        color = (0, 255, 0) if i == 0 else (255, 255, 0)
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                axes[1, 0].imshow(overlay)
                axes[1, 0].set_title('Original Detection')
        else:
            axes[1, 0].imshow(scale_region, cmap='gray')
            axes[1, 0].set_title('Detection Failed')
        axes[1, 0].axis('off')
        
        # Results summary
        ax_text = axes[1, 1]
        ax_text.axis('off')
        
        if detection_result['scale_detected']:
            info = detection_result['scale_info']
            text = f"Scale Detection: SUCCESS\n\n"
            text += f"Method: {detection_result['method_used']}\n"
            text += f"Scale Value: {info.get('value', 'N/A')} {info.get('unit', 'N/A')}\n"
            text += f"Bar Length: {detection_result.get('bar_length_pixels', 'N/A')} pixels\n"
            text += f"Calibration: {detection_result['micrometers_per_pixel']:.4f} μm/pixel\n\n"
            
            if detection_result['method_used'] == 'enhanced':
                text += f"Segments Found: {detection_result.get('segments_found', 'N/A')}\n"
                text += f"Text Centrality: {detection_result.get('text_centrality', 'N/A'):.3f}\n"
                text += f"Detection Score: {detection_result.get('detection_score', 'N/A'):.3f}\n"
        else:
            text = f"Scale Detection: FAILED\n\n"
            text += f"Method: {detection_result['method_used']}\n"
            text += f"Error: {detection_result.get('error', 'Unknown error')}\n"
        
        ax_text.text(0.05, 0.95, text, transform=ax_text.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()

# Convenience functions
def detect_scale_bar(image_input, use_enhanced: bool = True, **kwargs) -> Dict:
    """
    Convenience function to detect scale bar and return full results.
    
    Args:
        image_input: Either np.ndarray (image) or str (path to image)
        use_enhanced: Use enhanced text-centered detection
        **kwargs: Additional parameters for ScaleBarDetector
        
    Returns:
        Dictionary containing scale detection results
    """
    # Handle both image arrays and file paths
    if isinstance(image_input, str):
        import cv2
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
    detector = ScaleBarDetector(use_enhanced_detection=use_enhanced, **kwargs)
    result = detector.detect_scale_bar(image)
    
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

def detect_scale_factor_only(image_input, use_enhanced: bool = True, **kwargs) -> float:
    """
    Convenience function that returns only the scale factor.
    
    Args:
        image_input: Either np.ndarray (image) or str (path to image)
        use_enhanced: Use enhanced text-centered detection
        **kwargs: Additional parameters for ScaleBarDetector
        
    Returns:
        Micrometers per pixel conversion factor (0.0 if detection failed)
    """
    result = detect_scale_bar(image_input, use_enhanced=use_enhanced, **kwargs)
    
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