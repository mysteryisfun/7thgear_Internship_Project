# Intelligent Data Extraction System

## Environment Setup (Recommended)
Before running the project, set up your environment and install dependencies using Conda for best compatibility:

```powershell
# Create and activate a new Conda environment (Python 3.10 recommended)
conda create -n py310 python=3.10
conda activate py310

# (Optional, for GPU support) Install CUDA and cuDNN
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Install all project dependencies using requirements.txt
pip install -r requirements.txt
```

## Running the Project
To run the project, use the following command:
```bash
python src/text_processing/main_processor.py <input_video_path> --output-dir <output_directory> --fps <frames_per_second> --similarity-threshold <threshold> --use-paddleocr
```
eg:python src\text_processing\main_processor.py data\faces_and_text.mp4 --output-dir output --fps 1.0 --similarity-threshold 0.8 --use-paddleocr

### Arguments:
- `<input_video_path>`: Path to the input video file.
- `--output-dir`: Directory to store the output files.
- `--fps`: Frames per second for video processing (default: 1.0).
- `--similarity-threshold`: Threshold for scene similarity (default: 0.8).
- `--use-paddleocr`: Use PaddleOCR for text extraction (optional).

### Data Storage:
- Store input videos in the `data/` directory.
- Output files will be saved in the specified `--output-dir`.
- **Structured results** saved in organized analysis folders within `output/results/` directory.
- **Optimized JSON format** with timestamped data and reduced file size.
- **Summary reports** generated automatically for human-readable analysis.

### Output Structure:
```
output/
├── frames/              # All extracted frames
├── keyframes/           # Scene change frames  
└── results/             # Analysis results organized by video and timestamp
    └── {video_name}_analysis_{timestamp}/
        ├── {video_name}_analysis_{timestamp}.json  # Structured analysis data
        └── {video_name}_analysis_{timestamp}.txt   # Human-readable summary
```

## Purpose of Files in `video_processing` Directory
- `main_processor.py`: Main pipeline for video processing and data extraction.
- `output_manager.py`: Structured data storage and JSON export management.
- `paddleocr_text_extractor.py`: PaddleOCR-based text extraction from frames.
- `nlp_processing.py`: Advanced NLP processing with spaCy.
- `frame_extractor.py`: FFmpeg-based frame extraction from videos.
- `text_processor.py`: Text preprocessing and similarity detection.
- `bert_processor.py`: Handles text summarization using Pegasus.

## Installation Requirements
1. Install Python 3.9 or 3.10.
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

