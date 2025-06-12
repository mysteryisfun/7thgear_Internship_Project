import os
import sys
import subprocess
from typing import List, Tuple

class FrameExtractor:
    def __init__(self, video_path: str, output_dir: str, fps: float = 1.0):
        self.video_path = video_path
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(self.output_dir, exist_ok=True)
        
    def check_ffmpeg_installed(self):
        """
        Check if FFmpeg is installed and available in the system PATH.
        Raises FileNotFoundError if not found.
        """
        try:
            # Try to run ffmpeg -version to check if it's installed
            subprocess.run(
                ['ffmpeg', '-version'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            raise FileNotFoundError(
                "FFmpeg is not installed or not found in your system PATH. "
                "Please install FFmpeg and make sure it's in your PATH."
            )

    def extract_frames(self) -> List[Tuple[str, float]]:
        """
        Extract frames from the video at the specified frame rate (fps).
        Returns a list of tuples: (frame_path, timestamp)
        
        Raises:
            FileNotFoundError: If FFmpeg is not installed
            FileNotFoundError: If the video file doesn't exist
            RuntimeError: If frame extraction fails
        """
        # Check if ffmpeg is installed
        self.check_ffmpeg_installed()
        
        # Check if video file exists
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        try:
            import ffmpeg
            
            # Get video duration
            probe = ffmpeg.probe(self.video_path)
            duration = float(probe['format']['duration'])
            
            frame_paths = []
            for i in range(int(duration * self.fps)):
                timestamp = i / self.fps
                frame_file = os.path.join(self.output_dir, f"frame_{i:05d}.jpg")
                
                # Extract frame at the specified timestamp
                (
                    ffmpeg
                    .input(self.video_path, ss=timestamp)
                    .output(frame_file, vframes=1)
                    .overwrite_output()
                    .run(quiet=True)
                )
                frame_paths.append((frame_file, timestamp))
            
            # Return the list of frame paths with their timestamps
            return frame_paths
            
        except ImportError:
            raise ImportError("ffmpeg-python package not installed. Run: pip install ffmpeg-python")
        except Exception as e:
            raise RuntimeError(f"FFmpeg error: {str(e)}")
