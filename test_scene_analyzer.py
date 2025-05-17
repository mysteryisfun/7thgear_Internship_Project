"""
Test script for the Scene Analyzer module.

This script tests the SceneAnalyzer functionality using sample video files.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.video_processing.frame_extractor import FrameExtractor
from src.video_processing.scene_analyzer import SceneAnalyzer
import importlib.util
import subprocess

def test_scene_analyzer_with_video(video_path, output_base_dir, similarity_threshold=0.8, debug=False):
    """
    Test the SceneAnalyzer with a video file.
    
    Args:
        video_path: Path to the video file
        output_base_dir: Base directory for output
        similarity_threshold: Threshold for text similarity
        debug: Enable debug output
        
    Returns:
        Number of detected scene changes
    """
    print(f"Testing SceneAnalyzer with video: {video_path}")
    
    # Check if tesseract is available on system
    tesseract_found = False
    try:
        # Try to import pytesseract
        spec = importlib.util.find_spec("pytesseract")
        if spec is not None:
            # Try to check if tesseract is installed
            import subprocess
            try:
                result = subprocess.run(['tesseract', '--version'], 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, 
                                       check=False)
                if result.returncode == 0:
                    tesseract_found = True
                    print(f"Tesseract found: {result.stdout.decode('utf-8').splitlines()[0]}")
            except Exception:
                pass
    except Exception:
        pass
        
    if not tesseract_found:
        print("Tesseract not available.")
        print("Please install Tesseract OCR and make sure it's in your PATH.")
        print("Visit: https://github.com/UB-Mannheim/tesseract/wiki")
        print("Using image-based similarity fallback instead of OCR.")
    
    # Create output directories
    frames_dir = os.path.join(output_base_dir, "frames")
    keyframes_dir = os.path.join(output_base_dir, "keyframes")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(keyframes_dir, exist_ok=True)
    
    # Step 1: Extract frames
    print("Extracting frames...")
    frame_extractor = FrameExtractor(video_path, frames_dir)
    frame_paths_with_timestamps = frame_extractor.extract_frames()
    frame_paths = [path for path, _ in frame_paths_with_timestamps]
    print(f"Extracted {len(frame_paths)} frames")
    
    # Create output directories
    frames_dir = os.path.join(output_base_dir, "frames")
    keyframes_dir = os.path.join(output_base_dir, "keyframes")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(keyframes_dir, exist_ok=True)
    
    # Step 1: Extract frames
    print("Extracting frames...")
    frame_extractor = FrameExtractor(video_path, frames_dir)
    frame_paths_with_timestamps = frame_extractor.extract_frames()
    frame_paths = [path for path, _ in frame_paths_with_timestamps]
    print(f"Extracted {len(frame_paths)} frames")
    
    # Step 2: Analyze frames and print unique processed texts
    print("Analyzing frames for scene changes...")
    
    # Create SceneAnalyzer (will automatically use image-based similarity if OCR not available)
    scene_analyzer = SceneAnalyzer(similarity_threshold=similarity_threshold)
    
    # Extract and analyze frames
    unique_texts = scene_analyzer.analyze_frames(frame_paths, debug=debug)
    
    print(f"\n=== Unique Preprocessed Texts/Images Detected ===")
    for idx, txt in enumerate(unique_texts, 1):
        print(f"Scene {idx}: {txt}")
    print(f"Total unique scenes detected: {len(unique_texts)}")
    
    # If we have OCR, also extract keyframes for visualization
    if scene_analyzer.use_ocr:
        keyframe_paths = scene_analyzer.extract_keyframes(frame_paths, keyframes_dir)
        print(f"Keyframes saved to: {keyframes_dir}")
    
    return len(unique_texts)

def main():
    # Path to the test video file
    video_path = os.path.join(project_root, "data", "faces_start.mp4")
    
    # Output directory for extracted frames and keyframes
    output_dir = os.path.join(project_root, "data", "scene_analysis_output")
    
    # Enable debug mode for detailed analysis
    debug_mode = True
    
    # Test with different similarity thresholds
    # Lower thresholds are less sensitive to changes (more permissive)
    # Higher thresholds are more sensitive to changes (more strict)
    thresholds = [0.6, 0.5, 0.4]
    
    for threshold in thresholds:
        print(f"\nTesting with similarity threshold: {threshold}")
        threshold_dir = os.path.join(output_dir, f"threshold_{threshold}")
        num_scenes = test_scene_analyzer_with_video(
            video_path, 
            threshold_dir,
            similarity_threshold=threshold,
            debug=debug_mode
        )
        print(f"Result: Detected {num_scenes} scenes with threshold {threshold}")

if __name__ == "__main__":
    main()
