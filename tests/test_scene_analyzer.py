"""
Unit tests for the Scene Analyzer module.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video_processing.scene_analyzer import SceneAnalyzer

class TestSceneAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_output_dir = tempfile.mkdtemp()
        
        # Path to test frames in the data directory
        self.test_frames_dir = os.path.join(
            project_root, "data", "frames_test"
        )
        
        # Skip tests if the frames directory doesn't exist
        if not os.path.exists(self.test_frames_dir):
            self.skipTest(f"Test frames directory not found: {self.test_frames_dir}")
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_output_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that the SceneAnalyzer initializes correctly."""
        analyzer = SceneAnalyzer()
        self.assertEqual(analyzer.similarity_threshold, 0.8)
        self.assertIsNone(analyzer.text_cache)
        self.assertIsNone(analyzer.normalized_cache)
        
        # Test with custom threshold
        analyzer = SceneAnalyzer(similarity_threshold=0.9)
        self.assertEqual(analyzer.similarity_threshold, 0.9)
    
    def test_normalize_text(self):
        """Test that text normalization works correctly."""
        analyzer = SceneAnalyzer()
        
        # Test with various text inputs
        test_cases = [
            # Input text, Expected normalized text
            ("Hello World!", "hello world"),
            ("  Extra  Spaces  ", "extra spaces"),
            ("Mixed CASE text", "mixed case text"),
            ("Special @#$%^ Characters", "special characters"),
            ("Numbers 123 and punctuation...", "numbers 123 and punctuation..."),
            ("Line\nBreaks\nRemoved", "line breaks removed"),
            ("", ""),  # Empty string
            (None, ""),  # None value
        ]
        
        for input_text, expected in test_cases:
            normalized = analyzer.normalize_text(input_text)
            self.assertEqual(normalized, expected)
    
    def test_compare_text(self):
        """Test text comparison functionality."""
        analyzer = SceneAnalyzer()
        
        # Test with various text pairs
        test_cases = [
            # Text1, Text2, Expected similarity >= threshold?
            ("Hello World", "Hello World", True),  # Identical
            ("Hello World", "hello world", True),  # Case difference
            ("Hello  World", "Hello World", True),  # Extra spaces
            ("Hello World", "Hello Universe", False),  # Different content
            ("Short", "Very long different text", False),  # Length difference
            ("", "", True),  # Both empty
            ("Text", "", False),  # One empty
            ("", "Text", False),  # One empty
        ]
        
        threshold = 0.8
        for text1, text2, expected in test_cases:
            similarity = analyzer.compare_text(text1, text2)
            result = similarity >= threshold
            self.assertEqual(result, expected, 
                f"Failed for '{text1}' vs '{text2}': "
                f"got similarity {similarity}, expected {expected}")
    
    def test_analyze_frame(self):
        """Test frame analysis functionality."""
        # Get list of frame paths
        frame_paths = [os.path.join(self.test_frames_dir, f) 
                      for f in os.listdir(self.test_frames_dir) 
                      if f.endswith('.jpg')]
        
        if not frame_paths:
            self.skipTest("No test frames found")
        
        frame_paths.sort()  # Ensure frames are in order
        
        analyzer = SceneAnalyzer(similarity_threshold=0.8)
        
        # First frame should always be a new scene
        result1 = analyzer.analyze_frame(frame_paths[0])
        self.assertTrue(result1["is_new_scene"])
        self.assertGreaterEqual(len(result1["text"]), 0)
        
        # Analyze the second frame
        result2 = analyzer.analyze_frame(frame_paths[1])
        
        # We don't know if it's a new scene without looking at the content,
        # but we can check that the result contains the expected fields
        self.assertIn("is_new_scene", result2)
        self.assertIn("similarity", result2)
        self.assertIn("text", result2)
        self.assertIn("normalized_text", result2)
    
    def test_extract_keyframes(self):
        """Test keyframe extraction functionality."""
        # Get list of frame paths
        frame_paths = [os.path.join(self.test_frames_dir, f) 
                      for f in os.listdir(self.test_frames_dir) 
                      if f.endswith('.jpg')]
        
        if not frame_paths:
            self.skipTest("No test frames found")
        
        frame_paths.sort()  # Ensure frames are in order
        
        # Test with high similarity threshold to ensure some keyframes
        analyzer = SceneAnalyzer(similarity_threshold=0.95)
        
        keyframe_paths = analyzer.extract_keyframes(
            frame_paths, 
            os.path.join(self.test_output_dir, "keyframes")
        )
        
        # Check that at least the first frame was extracted as a keyframe
        self.assertGreaterEqual(len(keyframe_paths), 1)
        
        # Check that the keyframe files exist
        for path in keyframe_paths:
            self.assertTrue(os.path.exists(path))

if __name__ == '__main__':
    unittest.main()
