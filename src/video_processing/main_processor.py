"""
Main processing script for video analysis.

This script integrates frame extraction and scene analysis to process videos
and extract key frames containing distinct text content.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple

from src.video_processing.frame_extractor import FrameExtractor
from src.video_processing.scene_analyzer import SceneAnalyzer

def process_video(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    similarity_threshold: float = 0.8,
    tesseract_path: str = None
) -> Tuple[List[str], List[str]]:
    """
    Process a video to extract frames and detect scenes based on text changes.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save outputs
        fps: Frame rate for extraction
        similarity_threshold: Threshold for text similarity
        tesseract_path: Path to tesseract executable
    
    Returns:
        Tuple of (all_frame_paths, keyframe_paths)
    """
    # Create output directories
    frames_dir = os.path.join(output_dir, "frames")
    keyframes_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(keyframes_dir, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Extract frames
    print("Extracting frames...")
    frame_extractor = FrameExtractor(video_path, frames_dir, fps)
    frame_paths_with_timestamps = frame_extractor.extract_frames()
    frame_paths = [path for path, _ in frame_paths_with_timestamps]
    print(f"Extracted {len(frame_paths)} frames at {fps} fps")
    
    # Step 2: Analyze frames for scene changes
    print("Analyzing frames for scene changes...")
    scene_analyzer = SceneAnalyzer(
        similarity_threshold=similarity_threshold,
        tesseract_path=tesseract_path
    )
    
    # Extract keyframes
    keyframe_paths = scene_analyzer.extract_keyframes(frame_paths, keyframes_dir)
    print(f"Detected {len(keyframe_paths)} scenes")
    
    # Print scene information
    for i, path in enumerate(keyframe_paths):
        print(f"  Scene {i+1}: {os.path.basename(path)}")
    
    print(f"Processing complete. Key frames saved to: {keyframes_dir}")
    return frame_paths, keyframe_paths

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process video to extract frames and detect scenes"
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save outputs",
        default="output"
    )
    parser.add_argument(
        "--fps", "-f",
        help="Frame rate for extraction",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--similarity-threshold", "-s",
        help="Threshold for text similarity (0.0 to 1.0)",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--tesseract-path", "-t",
        help="Path to tesseract executable",
        default=None
    )
    
    args = parser.parse_args()
    
    # Process the video
    process_video(
        args.video_path,
        args.output_dir,
        args.fps,
        args.similarity_threshold,
        args.tesseract_path
    )

if __name__ == "__main__":
    main()
