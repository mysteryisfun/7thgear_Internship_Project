"""
Gemini API Image Analysis Test Script

This script tests sending an image to the Gemini API (flash-2.0 model) for analysis.
It demonstrates how to send a JPEG image and receive a response from the Gemini API.

Usage (PowerShell):
    $env:GEMINI_API_KEY="<your_api_key>"
    conda activate pygpu
    python tests/test_image_LLM_API.py
"""

import requests
import cv2
import base64
import time
import os

API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    raise EnvironmentError("Please set the GEMINI_API_KEY environment variable.")
MODEL = 'gemini-2.0-flash'
API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}'

IMAGE_PATH = 'data/test_images/frame_00014.jpg'  # Update path as needed

def test_image_llm_api():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"[ERROR] Could not load image: {IMAGE_PATH}")
        return
    # Encode image as JPEG and then base64
    _, buffer = cv2.imencode('.jpg', image)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    payload = {
        'contents': [
            {
                'parts': [
                    {
                        'inline_data': {
                            'mime_type': 'image/jpeg',
                            'data': img_b64
                        }
                    }
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    print("[INFO] Sending image to Gemini API...")
    start = time.perf_counter()
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    try:
        response.raise_for_status()
        results = response.json()
        print("[SUCCESS] Gemini API response:")
        print(results)
    except Exception as e:
        print(f"[ERROR] Gemini API request failed: {e}")
        print(response.text)
    print(f"[INFO] Gemini API image analysis time: {elapsed:.3f} seconds")

if __name__ == "__main__":
    test_image_llm_api()