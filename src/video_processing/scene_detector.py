import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import scenedetect
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

class SceneDetector:
    """
    A class for detecting scene transitions in video files.
    Uses PySceneDetect for slide transition detection with threshold calibration
    for different meeting platforms and filtering for duplicate frames.
    """
    
    # Default threshold values for different platforms
    PLATFORM_THRESHOLDS = {
        'zoom': 30.0,    # Zoom tends to have smoother transitions
        'teams': 27.0,   # Teams has medium compression artifacts
        'meet': 25.0,    # Google Meet has higher compression
        'default': 30.0  # Default value for unknown platforms
    }
    
    def __init__(self, video_path: str, platform: str = 'default'):
        """
        Initialize the SceneDetector.
        
        Args:
            video_path: Path to the video file
            platform: Meeting platform ('zoom', 'teams', 'meet', or 'default')
        """
        self.video_path = video_path
        self.platform = platform.lower()
        self.threshold = self._get_platform_threshold()
        
    def _get_platform_threshold(self) -> float:
        """Get the appropriate threshold value for the specified platform."""
        return self.PLATFORM_THRESHOLDS.get(
            self.platform, 
            self.PLATFORM_THRESHOLDS['default']
        )
    
    def calibrate_threshold(self, sample_scenes: int = 5) -> float:
        """
        Calibrate the threshold based on the video content.
        
        Args:
            sample_scenes: Number of sample scenes to analyze for calibration
            
        Returns:
            Calibrated threshold value
        """
        video = open_video(self.video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=15.0))

        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        if len(scene_list) < sample_scenes:
            return self._get_platform_threshold()

        stats = scene_manager.get_metrics()
        content_values = [metrics['content_val'] for metrics in stats.values() if 'content_val' in metrics]

        if not content_values:
            return self._get_platform_threshold()

        sorted_values = sorted(content_values)
        calibrated_threshold = sorted_values[int(len(sorted_values) * 0.8)]

        platform_factor = {
            'zoom': 1.1,
            'teams': 1.0,
            'meet': 0.9,
            'default': 1.0
        }.get(self.platform, 1.0)

        adjusted_threshold = calibrated_threshold * platform_factor
        return max(20.0, min(40.0, adjusted_threshold))
    
    def detect_scenes(self, min_scene_duration: float = 1.0) -> List[Tuple[float, float]]:
        """
        Detect scenes in the video file.
        
        Args:
            min_scene_duration: Minimum duration of a scene in seconds
            
        Returns:
            List of tuples containing (start_time, end_time) for each scene
        """
        video = open_video(self.video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.calibrate_threshold()))

        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            if end_time - start_time >= min_scene_duration:
                scenes.append((start_time, end_time))

        return scenes
        
    def filter_similar_frames(self, frame_paths: List[str], similarity_threshold: float = 0.9) -> List[str]:
        """
        Filter out duplicate or very similar frames.
        
        Args:
            frame_paths: List of paths to frames
            similarity_threshold: Threshold for considering frames as similar (0.0 to 1.0)
            
        Returns:
            List of frame paths with duplicates removed
        """
        if not frame_paths:
            return []
        
        # Keep the first frame always
        filtered_frames = [frame_paths[0]]
        last_kept_frame = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
        
        # For efficient comparison, resize frames to a smaller size
        height, width = 200, 300
        
        if last_kept_frame is None:
            return frame_paths  # Can't process the frames
            
        last_kept_frame = cv2.resize(last_kept_frame, (width, height))
        
        for frame_path in frame_paths[1:]:
            current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            
            if current_frame is None:
                continue  # Skip unreadable frames
                
            current_frame = cv2.resize(current_frame, (width, height))
            
            # Calculate structural similarity index (SSIM) between the frames
            # Higher values indicate more similarity
            try:
                # Calculate mean squared error (MSE) between the frames
                # Lower values indicate more similarity
                mse = np.mean((last_kept_frame.astype(np.float32) - current_frame.astype(np.float32)) ** 2)
                max_mse = 255.0 ** 2
                similarity = 1 - (mse / max_mse)  # Convert to similarity score (0 to 1)
                
                # If frames are not too similar, keep the current frame
                if similarity < similarity_threshold:
                    filtered_frames.append(frame_path)
                    last_kept_frame = current_frame
            except Exception as e:
                # If comparison fails, keep the frame to be safe
                filtered_frames.append(frame_path)
                last_kept_frame = current_frame
                
        return filtered_frames
        
    def extract_keyframes(self, output_dir: str, frames_per_scene: int = 1) -> List[str]:
        """
        Extract key frames from each detected scene.
        
        Args:
            output_dir: Directory to save extracted key frames
            frames_per_scene: Number of frames to extract per scene
            
        Returns:
            List of paths to the extracted key frames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect scenes
        scenes = self.detect_scenes()
        
        # Open video file
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        fps = video.get(cv2.CAP_PROP_FPS)
        keyframe_paths = []
        
        for scene_idx, (start_time, end_time) in enumerate(scenes):
            # Calculate frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Calculate which frames to extract
            frames_to_extract = []
            if frames_per_scene == 1:
                # Just take the middle frame of the scene
                frames_to_extract = [start_frame + (end_frame - start_frame) // 2]
            else:
                # Distribute frames evenly across the scene
                for i in range(frames_per_scene):
                    frame_idx = start_frame + i * (end_frame - start_frame) // max(1, frames_per_scene - 1)
                    frames_to_extract.append(frame_idx)
            
            # Extract the selected frames
            for frame_idx in frames_to_extract:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video.read()
                
                if ret:
                    frame_path = os.path.join(output_dir, f"scene_{scene_idx:03d}_frame_{frame_idx:05d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    keyframe_paths.append(frame_path)
        
        video.release()
        
        # Apply similarity filtering to remove any highly similar keyframes
        return self.filter_similar_frames(keyframe_paths)
