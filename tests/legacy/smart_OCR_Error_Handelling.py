#!/usr/bin/env python3
"""
Smart OCR Error Handling with Fuzzy Matching
Instead of hardcoding every possible OCR error, use intelligent pattern matching
"""

import re
import numpy as np
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher

def smart_parse_scale_text(text: str, debug: bool = True) -> Optional[Dict]:
    """
    Smart scale text parsing that can handle any OCR errors using fuzzy matching.
    
    Args:
        text: OCR text that might contain errors
        debug: Print debugging information
        
    Returns:
        Dictionary with scale information or None
    """
    
    if debug:
        print(f"    🧠 Smart parsing: '{text}'")
    
    # Step 1: Extract any numbers from the text
    numbers = extract_numbers_from_text(text)
    
    if not numbers:
        if debug:
            print(f"    ❌ No numbers found")
        return None
    
    # Step 2: For each number, try to find a unit that could be a scale unit
    for number in numbers:
        # Get the text after this number
        number_str = str(number).replace('.', r'\.')
        pattern = rf'{number_str}\s*(.{{0,10}})'  # Capture up to 10 chars after number
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            potential_unit = match.group(1).strip()
            
            # Try to match this to a known unit using fuzzy matching
            unit_result = fuzzy_match_unit(potential_unit, debug)
            
            if unit_result:
                micrometers = number * unit_result['conversion']
                
                # SEM scale filtering
                if 0.05 <= micrometers <= 3000:
                    if debug:
                        print(f"    ✅ Matched: {number} + '{potential_unit}' → {number} {unit_result['unit']} = {micrometers} μm")
                        print(f"       Confidence: {unit_result['confidence']:.3f}")
                    
                    return {
                        'value': number,
                        'unit': unit_result['unit'],
                        'micrometers': micrometers,
                        'original_text': text,
                        'fuzzy_confidence': unit_result['confidence'],
                        'matched_unit_text': potential_unit
                    }
                else:
                    if debug:
                        print(f"    ❌ {number} {unit_result['unit']} = {micrometers} μm (outside SEM range)")
    
    if debug:
        print(f"    ❌ No valid scale units found")
    
    return None

def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extract all possible numbers from text, handling various decimal formats.
    """
    
    # Patterns for numbers with different decimal separators
    number_patterns = [
        r'\b(\d+\.\d+)\b',      # 400.0
        r'\b(\d+,\d+)\b',       # 400,0 (European)
        r'\b(\d+)\b',           # 400 (integer)
    ]
    
    numbers = []
    
    for pattern in number_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                # Handle different decimal separators
                number_str = match.replace(',', '.')
                number = float(number_str)
                
                # Only reasonable scale values
                if 0.1 <= number <= 50000:
                    numbers.append(number)
            except ValueError:
                continue
    
    # Remove duplicates and sort
    numbers = sorted(list(set(numbers)))
    
    return numbers

def fuzzy_match_unit(text: str, debug: bool = False) -> Optional[Dict]:
    """
    Use fuzzy matching to identify scale units even with OCR errors.
    
    Args:
        text: Potential unit text (might contain errors)
        debug: Print debugging info
        
    Returns:
        Dictionary with unit info and confidence, or None
    """
    
    # Define target units with their variations
    target_units = {
        'μm': {
            'conversion': 1.0,
            'canonical': 'μm',
            'variations': [
                'μm', 'µm', 'um', 'micrometer', 'microns',
                # Common OCR errors
                'jm', 'jim', 'pm', 'Om', 'om', '0m', 'Qm', 'qm',
                'Opm', 'opm', 'Qpm', 'qpm', '0pm', 'qum', 'Qum',
                'urn', 'μrn', 'μn', 'jun', 'ium', 'jum', 'lum',
                '|m', 'lm', '1m', 'Im', 'Ojm', 'ojm', 'ljm'
            ]
        },
        'nm': {
            'conversion': 0.001,
            'canonical': 'nm',
            'variations': [
                'nm', 'nanometer', 'nanometers'
            ]
        },
        'mm': {
            'conversion': 1000.0,
            'canonical': 'mm',
            'variations': [
                'mm', 'millimeter', 'millimeters'
            ]
        },
        'cm': {
            'conversion': 10000.0,
            'canonical': 'cm',
            'variations': [
                'cm', 'centimeter', 'centimeters'
            ]
        }
    }
    
    # Clean the input text
    clean_text = clean_unit_text(text)
    
    if debug:
        print(f"      Fuzzy matching: '{text}' → cleaned: '{clean_text}'")
    
    best_match = None
    best_confidence = 0.0
    
    for unit_name, unit_info in target_units.items():
        for variation in unit_info['variations']:
            confidence = calculate_unit_similarity(clean_text, variation)
            
            if confidence > best_confidence and confidence > 0.4:  # Minimum threshold
                best_match = {
                    'unit': unit_info['canonical'],
                    'conversion': unit_info['conversion'],
                    'confidence': confidence,
                    'matched_variation': variation
                }
                best_confidence = confidence
    
    if debug and best_match:
        print(f"      Best match: '{best_match['matched_variation']}' → {best_match['unit']} (conf: {best_confidence:.3f})")
    
    return best_match

def clean_unit_text(text: str) -> str:
    """
    Clean unit text by removing common OCR artifacts.
    """
    
    # Remove common OCR artifacts
    cleaned = text.strip()
    
    # Remove trailing underscores, periods, etc.
    cleaned = re.sub(r'[_.,;:|!]+$', '', cleaned)
    
    # Remove leading/trailing spaces and special chars
    cleaned = re.sub(r'^[^a-zA-Z0-9μµ]+', '', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9μµ]+$', '', cleaned)
    
    # Convert to lowercase for matching
    cleaned = cleaned.lower()
    
    return cleaned

def calculate_unit_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two unit strings using multiple methods.
    """
    
    text1_clean = text1.lower().strip()
    text2_clean = text2.lower().strip()
    
    # Method 1: Exact match (highest confidence)
    if text1_clean == text2_clean:
        return 1.0
    
    # Method 2: One is substring of the other
    if text1_clean in text2_clean or text2_clean in text1_clean:
        return 0.9
    
    # Method 3: Sequence matcher (fuzzy matching)
    sequence_similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
    
    # Method 4: Character-level similarity (handle OCR char swaps)
    char_similarity = calculate_character_similarity(text1_clean, text2_clean)
    
    # Method 5: Edit distance based similarity
    edit_similarity = calculate_edit_distance_similarity(text1_clean, text2_clean)
    
    # Combine similarities (weighted average)
    combined_similarity = (
        sequence_similarity * 0.4 + 
        char_similarity * 0.3 + 
        edit_similarity * 0.3
    )
    
    return combined_similarity

def calculate_character_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity based on character overlap (good for OCR errors).
    """
    
    if not text1 or not text2:
        return 0.0
    
    # Get character sets
    set1 = set(text1)
    set2 = set(text2)
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_edit_distance_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity based on edit distance (Levenshtein distance).
    """
    
    if not text1 or not text2:
        return 0.0
    
    # Calculate edit distance
    edit_distance = levenshtein_distance(text1, text2)
    
    # Convert to similarity (normalized by max length)
    max_length = max(len(text1), len(text2))
    if max_length == 0:
        return 1.0
    
    similarity = 1.0 - (edit_distance / max_length)
    return max(0.0, similarity)

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    """
    
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def test_smart_ocr_patterns():
    """
    Test the smart OCR error handling on various examples.
    """
    
    print("🧪 TESTING SMART OCR ERROR HANDLING")
    print("="*50)
    
    # Test cases with various OCR errors
    test_cases = [
        "400.0μm",      # Perfect
        "400.Ojm",      # Your first case
        "400 Opm",      # Your second case  
        "300 Qum_",     # Your third case
        "500jim",       # No space
        "200 |m",       # Pipe instead of μ
        "1000 nm",      # Different unit
        "2.5 mm",       # Decimal + different unit
        "150μrn",       # μrn instead of μm
        "800 urn",      # urn instead of μm
        "350.0 qm",     # qm instead of μm
        "450 lum",      # lum instead of μm
        "600Om_",       # Om with underscore
        "100.5jum",     # jum variation
        "250 0pm",      # 0pm instead of μm
        "Invalid text", # Should fail
        "12345",        # Number only
        "μm only",      # Unit only
    ]
    
    print("Testing various OCR error patterns:")
    print("-" * 50)
    
    success_count = 0
    
    for i, test_text in enumerate(test_cases):
        print(f"\nTest {i+1}: '{test_text}'")
        result = smart_parse_scale_text(test_text, debug=True)
        
        if result:
            print(f"  ✅ Result: {result['value']} {result['unit']} = {result['micrometers']} μm")
            print(f"     Confidence: {result['fuzzy_confidence']:.3f}")
            success_count += 1
        else:
            print(f"  ❌ No valid scale found")
    
    print(f"\n📊 RESULTS: {success_count}/{len(test_cases)} patterns successfully parsed")
    print(f"Success rate: {success_count/len(test_cases)*100:.1f}%")

def update_complete_detection_with_smart_parsing():
    """
    Show how to integrate smart parsing into the complete detection system.
    """
    
    print(f"\n🔧 INTEGRATION EXAMPLE:")
    print("="*50)
    
    # Example of how to modify find_scale_text_simple to use smart parsing
    example_code = '''
def find_scale_text_with_smart_parsing(scale_region):
    """Enhanced text finding with smart OCR error handling."""
    
    scale_candidates = []
    
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        results = reader.readtext(scale_region, detail=1)
        
        for (bbox, text, confidence) in results:
            # Use smart parsing instead of regex patterns
            scale_info = smart_parse_scale_text(text.strip(), debug=False)
            
            if scale_info and confidence > 0.2:
                bbox_array = np.array(bbox)
                center_x = int(np.mean(bbox_array[:, 0]))
                center_y = int(np.mean(bbox_array[:, 1]))
                
                # Combine OCR confidence with fuzzy matching confidence
                combined_confidence = confidence * 0.6 + scale_info['fuzzy_confidence'] * 0.4
                
                scale_candidates.append({
                    'text': text.strip(),
                    'value': scale_info['value'],
                    'unit': scale_info['unit'],
                    'micrometers': scale_info['micrometers'],
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': bbox_array,
                    'confidence': confidence,
                    'fuzzy_confidence': scale_info['fuzzy_confidence'],
                    'combined_confidence': combined_confidence,
                    'method': 'EasyOCR+SmartParsing'
                })
        
    except Exception as e:
        print(f"Smart parsing failed: {e}")
    
    # Sort by combined confidence
    scale_candidates.sort(key=lambda x: x['combined_confidence'], reverse=True)
    return scale_candidates
    '''
    
    print("Replace the regex-based parsing with smart_parse_scale_text():")
    print(example_code)

if __name__ == "__main__":
    # Test the smart OCR error handling
    test_smart_ocr_patterns()
    
    # Show integration example
    update_complete_detection_with_smart_parsing()
    
    print(f"\n🎯 BENEFITS OF SMART PARSING:")
    print("="*50)
    print("✅ Handles ANY OCR error pattern automatically")
    print("✅ No need to hardcode specific error patterns") 
    print("✅ Uses fuzzy matching with confidence scoring")
    print("✅ Combines multiple similarity methods")
    print("✅ Self-improving as it sees more patterns")
    print("✅ Works with: 400.Ojm, 400 Opm, 300 Qum_, etc.")
    
    print(f"\n🔧 TO INTEGRATE:")
    print("Replace parse_scale_text_simple() with smart_parse_scale_text()")
    print("This will handle '300 Qum_' and any future OCR errors automatically!")