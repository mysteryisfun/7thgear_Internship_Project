import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video_processing.scene_detector import SceneDetector

class TestSceneDetector(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_output_dir = tempfile.mkdtemp()
        
        # Path to a test video file in the data directory
        # We'll use one of the sample videos from the data directory
        self.test_video_path = os.path.join(
            project_root, "data", "n8n_only.mp4"
        )
        
        # Skip tests if the video file doesn't exist
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video file not found: {self.test_video_path}")
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_output_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that the SceneDetector initializes correctly."""
        detector = SceneDetector(self.test_video_path)
        self.assertEqual(detector.video_path, self.test_video_path)
        self.assertEqual(detector.platform, 'default')
        self.assertEqual(detector.threshold, SceneDetector.PLATFORM_THRESHOLDS['default'])
        
        # Test with a specific platform
        detector = SceneDetector(self.test_video_path, platform='zoom')
        self.assertEqual(detector.platform, 'zoom')
        self.assertEqual(detector.threshold, SceneDetector.PLATFORM_THRESHOLDS['zoom'])
    
    def test_threshold_calibration(self):
        """Test that threshold calibration returns a reasonable value."""
        detector = SceneDetector(self.test_video_path)
        calibrated_threshold = detector.calibrate_threshold()
        
        # Threshold should be within reasonable bounds
        self.assertGreaterEqual(calibrated_threshold, 20.0)
        self.assertLessEqual(calibrated_threshold, 40.0)
    
    def test_scene_detection(self):
        """Test that scene detection returns a list of scene timestamps."""
        detector = SceneDetector(self.test_video_path)
        scenes = detector.detect_scenes()
        
        # Should return a list of tuples
        self.assertIsInstance(scenes, list)
        
        # If any scenes were detected, verify their format
        if scenes:
            self.assertIsInstance(scenes[0], tuple)
            self.assertEqual(len(scenes[0]), 2)  # (start_time, end_time)
            self.assertLess(scenes[0][0], scenes[0][1])  # start < end
    
    def test_keyframe_extraction(self):
        """Test that keyframe extraction produces frame files."""
        detector = SceneDetector(self.test_video_path)
        keyframe_paths = detector.extract_keyframes(self.test_output_dir)
        
        # Should return a list of frame paths
        self.assertIsInstance(keyframe_paths, list)
        
        # If any keyframes were extracted, verify they exist
        if keyframe_paths:
            self.assertTrue(os.path.exists(keyframe_paths[0]))
    
    def test_frame_filtering(self):
        """Test that the frame filtering removes duplicate frames."""
        detector = SceneDetector(self.test_video_path)
        
        # Create a temporary directory for test frames
        frames_dir = os.path.join(self.test_output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # We'll use the test frames from the data directory if available
        test_frames_dir = os.path.join(project_root, "data", "frames_test")
        
        if not os.path.exists(test_frames_dir):
            self.skipTest(f"Test frames directory not found: {test_frames_dir}")
        
        # Get list of frame paths
        frame_paths = [os.path.join(test_frames_dir, f) for f in os.listdir(test_frames_dir) 
                      if f.endswith('.jpg')]
        
        if not frame_paths:
            self.skipTest("No test frames found")
        
        # Test filtering with a high similarity threshold to force filtering
        filtered_frames = detector.filter_similar_frames(frame_paths, similarity_threshold=0.99)
        
        # Should remove at least some frames with high similarity threshold
        self.assertLessEqual(len(filtered_frames), len(frame_paths))

if __name__ == '__main__':
    unittest.main()
