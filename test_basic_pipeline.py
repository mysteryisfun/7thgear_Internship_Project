#!/usr/bin/env python3
"""
Test basic video processing pipeline without complex dependencies.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.video_processing.frame_extractor import FrameExtractor
from src.video_processing.paddleocr_text_extractor import PaddleOCRTextExtractor

def test_basic_pipeline():
    """Test the basic video processing pipeline."""
    video_path = "data/faces_and_text.mp4"
    output_dir = "test_basic_output"
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    print(f"Testing basic pipeline with: {video_path}")
    
    # Step 1: Extract frames
    print("1. Extracting frames...")
    frame_extractor = FrameExtractor(video_path, os.path.join(output_dir, "frames"), fps=1.0)
    
    try:
        frame_paths_with_timestamps = frame_extractor.extract_frames()
        frame_paths = [path for path, _ in frame_paths_with_timestamps]
        print(f"   Extracted {len(frame_paths)} frames")
        
        # Display first few frame info
        for i, (path, timestamp) in enumerate(frame_paths_with_timestamps[:3]):
            print(f"   Frame {i}: {os.path.basename(path)} at {timestamp:.2f}s")
        
    except Exception as e:
        print(f"   Error in frame extraction: {e}")
        return
    
    # Step 2: Extract text using PaddleOCR
    print("2. Extracting text with PaddleOCR...")
    try:
        text_extractor = PaddleOCRTextExtractor()
        
        # Test on first few frames only
        test_frames = frame_paths[:3]
        extracted_texts = text_extractor.extract_text_from_frames(test_frames)
        
        print(f"   Processed {len(extracted_texts)} frames for text")
        
        # Display extracted text
        for i, text in enumerate(extracted_texts):
            print(f"   Frame {i} text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
    except Exception as e:
        print(f"   Error in text extraction: {e}")
        return
    
    print("Basic pipeline test completed successfully!")

if __name__ == "__main__":
    test_basic_pipeline()
