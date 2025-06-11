"""
Test script for people masking module (Haar Cascade)

Usage (PowerShell):
    conda activate pygpu
    python tests/test_people_masking.py --img_path <path_to_image>
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import argparse
from src.image_processing.classifier2_workflow.people_masking import mask_people_in_image

def main():
    parser = argparse.ArgumentParser(description="Test people masking using Haar Cascade.")
    parser.add_argument('--img_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save masked image (optional)')
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(f"Image not found: {args.img_path}")
        return
    img = cv2.imread(args.img_path)
    if img is None:
        print("Failed to load image.")
        return
    masked = mask_people_in_image(img)
    cv2.imshow('Original', img)
    cv2.imshow('Masked', masked)
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if args.output:
        cv2.imwrite(args.output, masked)
        print(f"Masked image saved to: {args.output}")

if __name__ == "__main__":
    main()
