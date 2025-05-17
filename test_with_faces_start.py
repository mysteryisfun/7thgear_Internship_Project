"""
Test script to evaluate the Scene Detector with faces_start.mp4
"""
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.video_processing.scene_detector import SceneDetector

def main():
    # Path to the test video file
    test_video_path = os.path.join(project_root, "data", "faces_start.mp4")
    
    # Create output directory for keyframes
    output_dir = os.path.join(project_root, "data", "keyframes_faces_start")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing Scene Detector with: {test_video_path}")
    
    # Initialize Scene Detector
    detector = SceneDetector(test_video_path)
    
    # 1. Test threshold calibration
    calibrated_threshold = detector.calibrate_threshold()
    print(f"Calibrated threshold: {calibrated_threshold}")
    
    # 2. Test scene detection
    scenes = detector.detect_scenes()
    print(f"Detected {len(scenes)} scenes")
    for i, (start, end) in enumerate(scenes):
        print(f"  Scene {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    # 3. Test keyframe extraction
    keyframe_paths = detector.extract_keyframes(output_dir, frames_per_scene=2)
    print(f"Extracted {len(keyframe_paths)} keyframes")
    print(f"Keyframes saved to: {output_dir}")
    
    # 4. Adjust the threshold for testing
    detector.threshold = 20.0  # Lower threshold to detect more scenes
    print(f"Adjusted threshold: {detector.threshold}")

    # Re-run scene detection with the adjusted threshold
    scenes = detector.detect_scenes()
    print(f"Detected {len(scenes)} scenes with adjusted threshold")
    for i, (start, end) in enumerate(scenes):
        print(f"  Scene {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    # 5. Return the number of keyframes extracted for verification
    return len(keyframe_paths)

if __name__ == "__main__":
    main()
