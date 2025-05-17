# Documentation for Updated Video Processing Pipeline

## Overview
This document outlines the updates made to the video processing pipeline, including the implementation of custom scene detection logic, text pre-processing, and NLP application for scene-specific frames.

## Key Updates

### 1. Scene Detection Logic
- Replaced external scene detection libraries with a custom logic that compares consecutive frame texts.
- Texts are pre-processed by removing spaces and concatenating into a single string before comparison.
- Scene changes are detected based on a similarity threshold using `SequenceMatcher`.

### 2. Text Pre-Processing
- Extracted texts from frames are pre-processed to remove spaces and form a single string.
- This ensures consistent and accurate comparison for scene detection.

### 3. NLP Application
- NLP techniques (lemmatization, spell correction, etc.) are applied only to frames where scene changes are detected.
- This optimization reduces unnecessary processing and improves efficiency.

### 4. PaddleOCR Integration
- The pipeline supports text extraction using PaddleOCR with enhanced configuration for better accuracy.
- Parameters such as `use_angle_cls` and `det_db_box_thresh` are configured for optimal performance.

## Implementation Details

### Scene Detection
- Implemented in `main_processor.py`.
- Uses `SequenceMatcher` to calculate similarity between consecutive frame texts.
- Frames with similarity below the threshold are marked as scene changes.

### Text Pre-Processing
- Texts are pre-processed in `main_processor.py` by removing spaces and concatenating into a single string.

### NLP Processing
- NLP is applied only to the texts of frames where scene changes are detected.
- Implemented in `main_processor.py` using the `NLPProcessor` class.

### PaddleOCR
- Text extraction is handled by `PaddleOCRTextExtractor` in `paddleocr_text_extractor.py`.
- Configured with parameters for angle classification and detection box threshold.

## Testing
- The updated pipeline was tested with the video file `data/faces_and_text.mp4`.
- Detected scenes and processed texts were verified for accuracy.

## Usage
1. Run the `main_processor.py` script with the desired video file.
2. Use the `--use-paddleocr` flag to enable PaddleOCR for text extraction.
3. Specify the similarity threshold using the `--similarity-threshold` flag.

Example:
```bash
python src/video_processing/main_processor.py data/faces_and_text.mp4 --use-paddleocr --similarity-threshold 0.8
```

## Future Work
- Validate the pipeline with additional video files to ensure robustness.
- Optimize the similarity threshold for better scene detection accuracy.
- Document additional test cases and results.

## References
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- Python `difflib.SequenceMatcher` for text similarity.
