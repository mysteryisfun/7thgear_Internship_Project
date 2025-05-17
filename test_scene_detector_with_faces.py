import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.video_processing.scene_detector import SceneDetector

def main():
    # Path to the test video file
    video_path = os.path.join(project_root, "data", "faces_start.mp4")
    
    # Create output directory for keyframes
    output_dir = os.path.join(project_root, "data", "keyframes_faces_start")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize scene detector
    print(f"Initializing scene detector for {video_path}")
    detector = SceneDetector(video_path)
    
    # Calibrate threshold
    threshold = detector.calibrate_threshold()
    print(f"Calibrated threshold: {threshold}")
    
    # Detect scenes
    print("Detecting scenes...")
    scenes = detector.detect_scenes(min_scene_duration=0.5)  # Lower min duration to catch more scenes
    
    print(f"Detected {len(scenes)} scenes:")
    for i, (start, end) in enumerate(scenes):
        print(f"  Scene {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    # Extract keyframes
    print(f"Extracting keyframes to {output_dir}...")
    keyframes = detector.extract_keyframes(output_dir, frames_per_scene=2)  # Extract 2 frames per scene for better coverage
    
    print(f"Extracted {len(keyframes)} keyframes:")
    for i, frame_path in enumerate(keyframes):
        print(f"  {i+1}: {os.path.basename(frame_path)}")

if __name__ == "__main__":
    main()
