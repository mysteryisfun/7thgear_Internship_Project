"""
Gemma Model Speed Test Script

This script tests the response time of the Gemma context extraction model for different OCR text inputs.
It is intended for benchmarking and optimizing KV cache and inference options in the Gemma backend.

Usage (PowerShell):
    conda activate pygpu
    python tests/test_gemma_speed.py
"""
import time
from src.text_processing.gemma_2B_context_model import GemmaContextExtractor

# Example OCR text (replace or extend with your own test cases)
OCR_TEXTS = [
    "This is a test slide about project updates and next steps for Q3.",
    "Renewal Summary: Medical Benefits, Dental Benefits. Organizations: Cigna, The City, AON. Plan Year Renewal 2020/2021. Total Annual Premium: $12,005,106.2.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nec odio. Praesent libero. Sed cursus ante dapibus diam.",
]

def main():
    gemma = GemmaContextExtractor()
    for idx, ocr_text in enumerate(OCR_TEXTS):
        print(f"\n--- Test Case {idx+1} ---")
        print(f"Input OCR Text: {ocr_text}")
        start = time.perf_counter()
        result = gemma.extract_context(ocr_text)
        elapsed = time.perf_counter() - start
        print(f"Gemma Output: {result}")
        print(f"[INFO] Gemma model response time: {elapsed:.3f} seconds")
        print("[NOTE] Use this script to compare timings for different KV cache or backend optimization settings.")

if __name__ == "__main__":
    main()
