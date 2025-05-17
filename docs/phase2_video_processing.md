
# Phase 2: Video Processing Module - Frame Extraction Implementation

This document outlines the implementation of the Frame Extraction component of the Intelligent Data Extraction System's Video Processing Module.

## Overview

The Frame Extraction component is responsible for extracting individual frames from meeting videos at specified intervals. This is the first step in our video processing pipeline and provides the foundation for subsequent analysis.

## Features Implemented

1. **FFmpeg Integration for Video Decoding**
   - Integrated with FFmpeg via the `ffmpeg-python` wrapper
   - Added system checks to ensure FFmpeg is properly installed
   - Error handling for missing dependencies

2. **Configurable Frame Rate Reduction**
   - Default extraction at 1 fps (frame per second)
   - Variable frame rate support through constructor parameter
   - Automatic calculation of total frames based on video duration

3. **Frame Storage with Timestamp Preservation**
   - Organized frame storage in configurable output directory
   - Consistent file naming convention (frame_00001.jpg, frame_00002.jpg, etc.)
   - Timestamp metadata preserved with each extracted frame

## Technical Implementation

The implementation consists of a `FrameExtractor` class in `src/video_processing/frame_extractor.py` with the following key methods:

- `__init__(video_path, output_dir, fps)`: Initializes the extractor with video path, output directory, and desired frame rate
- `check_ffmpeg_installed()`: Verifies FFmpeg availability on the system
- `extract_frames()`: Extracts frames at the specified rate and returns frame paths with timestamps

## Requirements

- Python 3.6+
- FFmpeg installed and available in system PATH
- `ffmpeg-python` package

## Installation Instructions

1. **Install FFmpeg**:
   - **Windows**: 
     - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
     - Or install via Chocolatey: `choco install ffmpeg`
     - Add FFmpeg to system PATH

   - **macOS**:
     - Install via Homebrew: `brew install ffmpeg`

   - **Linux**:
     - Install via package manager: `sudo apt-get install ffmpeg`

2. **Install Python Dependencies**:
   ```powershell
   # Activate the virtual environment
   .\.venv\Scripts\Activate
   
   # Install required packages
   pip install ffmpeg-python
   ```

## Usage Example

```python
from src.video_processing.frame_extractor import FrameExtractor

# Initialize the frame extractor
extractor = FrameExtractor(
    video_path='data/meeting_video.mp4',
    output_dir='data/extracted_frames',
    fps=1.0  # Extract 1 frame per second
)

# Extract frames
frames = extractor.extract_frames()

# frames is now a list of tuples: (frame_path, timestamp)
for frame_path, timestamp in frames:
    print(f"Frame at {timestamp}s: {frame_path}")
```

## Testing

A test script is available at `tests/test_frame_extractor.py`. Run it to verify the frame extraction functionality:

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate

# Run the test script
python tests/test_frame_extractor.py
```

## Next Steps

- Implement scene detection to identify slide transitions
- Add support for parallel processing to improve performance
- Develop presentation area detection within extracted frames

## Integration Notes

This module serves as the foundation for the video processing pipeline. The extracted frames will be used by:

1. Scene detection module to identify slide transitions
2. Text extraction module for OCR processing
3. Image classification module for visual analysis

When integrating, ensure the frame extraction configurations (especially fps) are appropriate for the specific use case and video content type.