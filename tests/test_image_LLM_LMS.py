"""
Test Script for LM Studio Image Context Extraction

This script tests the `extract_image_context_lmstudio` function from `lmstudio_image_context.py`.

Usage (PowerShell):
    conda activate pygpu
    python tests/test_lmstudio_image_context.py
"""

import os
import sys
from pprint import pprint

# Import the LM Studio image context extraction function
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.image_processing.LMS_img_LLM import extract_image_context_lmstudio

IMAGE_PATH = 'data/test_images/frame_00014.jpg'  # Update path as needed

def test_lmstudio_image_context():
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Could not find image: {IMAGE_PATH}")
        return
    print(f"[INFO] Testing LM Studio image context extraction for: {IMAGE_PATH}")
    try:
        result = extract_image_context_lmstudio(IMAGE_PATH)
        print("[SUCCESS] LM Studio structured response:")
        pprint(result, indent=2, width=120, compact=False)
    except Exception as e:
        print(f"[ERROR] LM Studio request failed: {e}")

if __name__ == "__main__":
    test_lmstudio_image_context()
