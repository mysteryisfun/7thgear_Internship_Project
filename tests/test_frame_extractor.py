import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.video_processing.frame_extractor import FrameExtractor


def test_frame_extraction():
    """Test frame extraction functionality"""
    try:
        # Paths to video file and output directory for frames
        video_path = os.path.join('data', 'faces_start.mp4')
        output_dir = os.path.join('data', 'frames_test')
        
        # Verify video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Looking for: {os.path.abspath(video_path)}")
            return
        
        # Set frame rate to 1 fps
        fps = 1.0  
        
        # Create frame extractor instance
        extractor = FrameExtractor(video_path, output_dir, fps)
        
        try:
            # Check if FFmpeg is installed
            extractor.check_ffmpeg_installed()
            print("FFmpeg is installed and available.")
            
            # Extract frames
            frames = extractor.extract_frames()
            print(f"Successfully extracted {len(frames)} frames.")
            
            # Print information about the first few frames
            for i, (frame_path, timestamp) in enumerate(frames[:5]):
                print(f"Frame {i+1}: {os.path.basename(frame_path)}, Timestamp: {timestamp:.2f}s")
            
            # Count total frames extracted
            total_frames = len(os.listdir(output_dir))
            print(f"Total frames in output directory: {total_frames}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nTo install FFmpeg:")
            print("1. Windows: Download from https://ffmpeg.org/download.html or use Chocolatey: choco install ffmpeg")
            print("2. macOS: Use Homebrew: brew install ffmpeg")
            print("3. Linux: Use apt-get: sudo apt-get install ffmpeg")
            
        except Exception as e:
            print(f"Error during frame extraction: {e}")
    
    except Exception as e:
        print(f"Test error: {e}")


if __name__ == "__main__":
    test_frame_extraction()
