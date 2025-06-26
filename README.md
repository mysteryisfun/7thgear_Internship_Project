<iframe width="768" height="432" src="https://miro.com/app/live-embed/uXjVItwHIT4=/?embedMode=view_only_without_ui&moveToViewport=-1575,21,4698,1410&embedId=394864167890" frameborder="0" scrolling="no" allow="fullscreen; clipboard-read; clipboard-write" allowfullscreen></iframe>
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
## Environment Variables and API Key Security
- Example variable: `GEMINI_API_KEY` (used for Google Gemini API access)
- Set your API key in `.env` 
- To set the variable in PowerShell for a session:
  ```powershell
  $env:GEMINI_API_KEY="<your_api_key>"
  conda activate pygpu
  python ...
  ```
## Running the Project
To run the project, use the following command:
```bash
python src/main_workflow/main_pipeline.py --video <input_video_path> --output_dir <output_directory> --fps <frames_per_second> --text_llm_backend <LMS_or_API> --image_llm_backend <LMS_or_API>
```
Example:
```powershell
conda activate pygpu; $env:PYTHONPATH="."; python src/main_workflow/main_pipeline.py --video data/test_files/test-1.mp4 --output_dir output --fps 1.0 --text_llm_backend LMS --image_llm_backend API
```

### Arguments:
- `<input_video_path>`: Path to the input video file.
- `--output_dir`: Directory to store the output files (default: `output/`).
- `--fps`: Frames per second for video processing (default: 1.0).
- `--text_llm_backend`: Choose the LLM backend for text context extraction. Options: `LMS` (for local Gemma model via LM Studio) or `API` (for Google Gemini API). Default: `API`.
- `--image_llm_backend`:Choose the LLM backend for image context extraction. Options: `LMS` (for local Gemma model via LM Studio) or `API` (for Google Gemini API). Default: `API`.

### Data Storage:
- Store input videos in the `data/` directory.
- Output files will be saved in the specified `--output_dir`.
- **Structured results** saved in organized analysis folders within `output/results/` directory.
- **Optimized JSON format** with timestamped data, including `text_processor_backend` used.
- **Summary reports** generated automatically for human-readable analysis.

### Output Structure:
```
output/
├── frames/              # All extracted frames
├── keyframes/           # Scene change frames (deduplicated)
└── results/             # Analysis results organized by video and timestamp
    └── {video_name}_analysis_{timestamp}/
        ├── {video_name}_analysis_{timestamp}.json  # Structured analysis data
        └── {video_name}_analysis_{timestamp}.txt   # Human-readable summary (if generated)
```

## Purpose of Key Files
- `src/main_workflow/main_pipeline.py`: Main pipeline for video processing, classification, and data extraction.
- `src/main_workflow/frame_comparator.py`: Handles frame deduplication using vector embeddings.
- `src/image_processing/classifier2_models/clip_classifier.py`: Classifies frames as 'text', 'image', or 'other' using CLIP.
- `src/text_processing/gemma_2B_context_model.py`: Extracts context from text using a local Gemma model (via LM Studio).
- `src/text_processing/API_text_LLM.py`: Extracts context from text using the Google Gemini API.
- `src/text_processing/ocr_processor.py`: Handles OCR for text extraction from frames.
- `output_manager.py`: Structured data storage and JSON export management.
- `frame_extractor.py`: FFmpeg-based frame extraction from videos.

## Benchmarking LLMs
Scripts to benchmark the LLM response times are available in the `tests/` directory:
- `tests/test_gemma_speed.py`: Benchmarks the local Gemma model.
- `tests/test_gemini_api_speed.py`: Benchmarks the Google Gemini API.

Run them directly using Python:
```bash
python tests/test_gemma_speed.py
python tests/test_gemini_api_speed.py
```

## Installation Requirements
1. Install Python 3.9 or 3.10.
2. Install FFmpeg:
   - On Ubuntu: `sudo apt install ffmpeg`
   - On Windows: Download from [FFmpeg.org](https://ffmpeg.org/).
3. Install PaddleOCR (or your preferred OCR engine if modifying the code):
   - Install PaddleOCR via pip: `pip install paddleocr paddlepaddle` (CPU) or `pip install paddleocr paddlepaddle-gpu` (GPU, ensure CUDA compatibility).
4. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. For local LLM (Gemma via LM Studio):
   - Download and run LM Studio.
   - Download the Gemma 2B model (or preferred compatible model) within LM Studio.
   - Ensure the LM Studio server is running and accessible (default: `http://localhost:1234/v1`).

## Technical Stack
The system leverages a robust suite of tools tailored to each component:
- **Video Processing**: OpenCV (frame analysis), FFmpeg (video handling).
- **Frame Classification**: OpenAI CLIP (image/text slide classification).
- **Text Extraction**: PaddleOCR (OCR), other OCR engines can be integrated.
- **Text Context Extraction**: Local LLMs (e.g., Gemma via LM Studio), API-based LLMs (e.g., Google Gemini).
- **Image and Diagram Handling**: (Future work) ResNet (image classification), YOLO (object detection), BLIP (image captioning), ChartOCR (diagram parsing).
- **Information Merging**: (Future work) Advanced LLMs for comprehensive summarization.

## Deduplication Strategy
- The pipeline now uses vector-based deduplication for identifying unique frames, replacing the previous pHash-based approach. This provides more robust and semantically aware deduplication.

## Phases
### Completed Phases
1. **Video Processing**:
   - Implemented frame extraction using OpenCV and FFmpeg.
   - Optimized frame rate for processing.
2. **Text Extraction**:
   - Integrated PaddleOCR for text extraction.
3. **Frame Classification & Deduplication**:
   - Integrated CLIP for classifying frames as text, image, or other.
   - Implemented vector-based deduplication for unique frame identification.
4. **Text Context Extraction**:
   - Integrated support for local LLMs (Gemma via LM Studio) and API-based LLMs (Google Gemini) for context extraction from OCR text.
   - Added command-line argument to select LLM backend.
5. **Pipeline Updates**:
   - Refined main pipeline logic for efficiency and clarity.
   - Updated output JSON structure.
   - Added LLM benchmarking scripts.

### Pending Phases
1. **Image and Diagram Handling**:
   - Implement ResNet and YOLO for image classification and object detection.
   - Integrate ChartOCR for diagram parsing.
2. **Information Merging**:
   - Develop a unified pipeline to merge text, images, and diagrams into timestamped summaries using advanced LLMs.
3. **Enhanced Output & Reporting**:
   - Generate more comprehensive human-readable summaries.
   - Visualize pipeline results (e.g., timeline of classified frames).

