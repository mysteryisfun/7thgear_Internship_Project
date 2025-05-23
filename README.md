# Intelligent Data Extraction System
## Running the Project
To run the project, use the following command:
```bash
python src/video_processing/main_processor.py <input_video_path> --output-dir <output_directory> --fps <frames_per_second> --similarity-threshold <threshold> --use-paddleocr
```
eg:python src\video_processing\main_processor.py data\faces_and_text.mp4 --output-dir output --fps 1.0 --similarity-threshold 0.8 --use-paddleocr

### Arguments:
- `<input_video_path>`: Path to the input video file.
- `--output-dir`: Directory to store the output files.
- `--fps`: Frames per second for video processing (default: 1.0).
- `--similarity-threshold`: Threshold for scene similarity (default: 0.8).
- `--use-paddleocr`: Use PaddleOCR for text extraction (optional).

### Data Storage:
- Store input videos in the `data/` directory.
- Output files will be saved in the specified `--output-dir`.

## Purpose of Files in `video_processing` Directory
- `main_processor.py`: Main pipeline for video processing and data extraction.
- `bert_processor.py`: Handles text summarization using Pegasus.
- `ocr_processor.py`: Performs OCR on extracted frames.
- `scene_detector.py`: Detects scene changes in videos.
- `image_processor.py`: Processes images and diagrams from slides.

## Installation Requirements
1. Install Python 3.9.
2. Install FFmpeg:
   - On Ubuntu: `sudo apt install ffmpeg`
   - On Windows: Download from [FFmpeg.org](https://ffmpeg.org/).
3. Install PaddleOCR:
   - Install PaddleOCR via pip: `pip install paddleocr`
4. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Technical Stack
The system leverages a robust suite of tools tailored to each component:
- **Video Processing**: OpenCV (frame analysis), FFmpeg (video handling), PySceneDetect (scene detection).
- **Text Extraction**: PaddleOCR (text extraction), spaCy (NLP postprocessing).
- **Image and Diagram Handling**: ResNet (image classification), YOLO (object detection), BLIP (image captioning), ChartOCR (diagram parsing).
- **Information Merging**: BART (text summarization), LLM APIs (e.g., OpenAI 4.0/4.1).

## Phases
### Completed Phases
1. **Video Processing**:
   - Implemented frame extraction using OpenCV and PySceneDetect.
   - Reduced frame rate to optimize processing.
2. **Text Extraction**:
   - Integrated Tesseract and Google Cloud Vision API for OCR.
   - Postprocessed text using spaCy for error correction.
3. **Model Testing**:
   - Tested multiple summarization models (DistilBERT, BART, T5, Pegasus).
   - Finalized Pegasus (`google/pegasus-xsum`) for summarization.
4. **Pipeline Updates**:
   - Removed BERT integration from the main pipeline.
   - Updated `bert_processor.py` to use Pegasus with tailored prompts.

### Pending Phases
1. **Image and Diagram Handling**:
   - Implement ResNet and YOLO for image classification and object detection.
   - Integrate ChartOCR for diagram parsing.
2. **Information Merging**:
   - Develop a unified pipeline to merge text, images, and diagrams into timestamped summaries.
3. **Integration and Scalability**:
   - Connect to platforms like Zoom and Microsoft Teams for real-time processing.
   - Deploy on cloud infrastructure for scalability.

