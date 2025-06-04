"""
Frame Loader Module

Loads video frames one at a time at 1 FPS, yielding each frame for processing.
Does not load all frames into memory. Designed for stepwise/parallel workflows.

Usage Example (PowerShell):
    conda activate pygpu
    python -c "from src.main_workflow.frame_loader import frame_generator; for idx, frame, ts in frame_generator('data/video.mp4'): print(idx, ts)"
"""

import cv2
import os
from typing import Generator, Tuple


def frame_generator(video_path: str, fps: float = 1.0) -> Generator[Tuple[int, any, float], None, None]:
    """
    Generator that yields one frame at a time at the specified FPS.
    Each yield: (frame_index, frame_image, timestamp_seconds)
    Only loads the next frame after the previous is processed.
    Args:
        video_path (str): Path to the video file.
        fps (float): Frames per second to extract (default 1.0).
    Yields:
        Tuple[int, frame, float]: (frame_index, frame_image, timestamp_seconds)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps else 0
    frame_interval = int(round(video_fps / fps)) if video_fps and fps else 1
    frame_idx = 0
    output_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps if video_fps else 0
            yield (output_idx, frame, timestamp)
            output_idx += 1
        frame_idx += 1
    cap.release()

# Example usage (for testing):
if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/faces_and_text.mp4"
    for idx, frame, ts in frame_generator(video, fps=1.0):
        print(f"Frame {idx} at {ts:.2f}s")
        # Insert frame processing logic here
