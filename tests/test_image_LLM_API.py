"""
Gemini API Image Analysis Test Script (modular)

This script tests sending an image to the Gemini API (flash-2.0 model) for analysis using the extract_image_context_gemini function.

Usage (PowerShell):
    $env:GEMINI_API_KEY="<your_api_key>"
    conda activate pygpu
    python tests/test_image_LLM_API.py
"""

import os
import sys
from pprint import pprint

# Import the Gemini image context extraction function
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.image_processing.API_img_LLM import extract_image_context_gemini

IMAGE_PATH = 'data/test_images/frame_00014.jpg'  # Update path as needed

def test_image_llm_api():
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Could not find image: {IMAGE_PATH}")
        return
    print(f"[INFO] Testing Gemini API image context extraction for: {IMAGE_PATH}")
    try:
        result = extract_image_context_gemini(IMAGE_PATH)
        print("[SUCCESS] Gemini API structured response:")
        pprint(result, indent=2, width=120, compact=True)
    except Exception as e:
        print(f"[ERROR] Gemini API request failed: {e}")

if __name__ == "__main__":
    test_image_llm_api()