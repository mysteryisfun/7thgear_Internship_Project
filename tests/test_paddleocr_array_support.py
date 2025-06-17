"""
Test PaddleOCRTextExtractor with both file path and numpy array input.
"""
import sys
import os
import cv2
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from text_processing.paddleocr_text_extractor import PaddleOCRTextExtractor

def test_paddleocr_with_path_and_array():
    extractor = PaddleOCRTextExtractor()
    img_path = 'data/test_images/frame_00014.jpg'
    assert os.path.exists(img_path), f"Test image not found: {img_path}"
    # Test with file path
    text_from_path = extractor.extract_text_from_frames([img_path])[0]
    print("[PATH] OCR Output:", text_from_path)
    # Test with numpy array (if supported)
    img = cv2.imread(img_path)
    try:
        text_from_array = extractor.extract_text_from_frames([img])[0]
        print("[ARRAY] OCR Output:", text_from_array)
    except Exception as e:
        print("[ARRAY] OCR failed:", e)

if __name__ == "__main__":
    test_paddleocr_with_path_and_array()
