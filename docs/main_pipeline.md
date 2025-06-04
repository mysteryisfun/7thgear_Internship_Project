# Main Pipeline for Frame-by-Frame Video Analysis

## How to Run the Main Pipeline

**Prerequisites:**
- Ensure you have the Conda environment `pygpu` set up with all dependencies installed (see `requirements.txt`).
- Place your input video in the `data/` directory or provide the correct path.

**Run the pipeline using PowerShell:**
```powershell
# Activate the correct Conda environment
conda activate pygpu
# Run the main pipeline script
python src/main_workflow/main_pipeline.py --video data/faces_and_text.mp4 --model CNN --fps 1.0
```
- `--video`: Path to the input video file
- `--model`: Classifier model to use (`CNN` or `EFF`)
- `--fps`: Frames per second to extract (default: 1.0)

## Pipeline Steps and Implementation Details

### 1. Frame Extraction
- **File:** `src/main_workflow/frame_loader.py`
- **Function:** `frame_generator`
- **Description:** Extracts frames from the input video at the specified FPS, yielding each frame and its timestamp.

### 2. Frame Classification
- **Files:**
  - `src/image_processing/classifier1_models/custom_cnn_classifier.py` (for CNN)
  - `src/image_processing/classifier1_models/efficientnet_functional.py` (for EfficientNet)
- **Description:**
  - Loads the selected TensorFlow model.
  - Each frame is classified as either `people` or `presentation`.
  - Only `presentation` frames proceed to further analysis; `people` frames are immediately discarded.

### 3. Duplicate Detection (Presentation Frames Only)
- **File:** `src/main_workflow/frame_comparator.py`
- **Class:** `FrameComparator`
- **Description:**
  - Compares each presentation frame to the previous one using:
    - **Perceptual Hash (phash):** Detects near-duplicate images.
    - **OCR + Text Similarity:** Extracts text (via PaddleOCR) and computes cosine similarity using TensorFlow BERT embeddings.
  - Frames are categorized as `unique_image`, `unique_text`, or `duplicate` based on thresholds.

### 4. JSON Export of Results
- **Location:** `output/main_pipeline_res/`
- **Description:**
  - Results are exported as a structured JSON file containing:
    - `metadata`: Video info, processing parameters, output info
    - `summary`: Frame and classification statistics
    - `frames`: List of per-frame results with all relevant fields (classification, duplicate status, phash, text similarity, OCR text, timing, etc.)
  - The output filename includes the video name and a timestamp for traceability.

## Files Used in the Pipeline
- `src/main_workflow/main_pipeline.py` — Main entry point and workflow logic
- `src/main_workflow/frame_loader.py` — Frame extraction utility
- `src/main_workflow/frame_comparator.py` — Duplicate detection logic
- `src/image_processing/classifier1_models/custom_cnn_classifier.py` — CNN classifier
- `src/image_processing/classifier1_models/efficientnet_functional.py` — EfficientNet classifier
- `src/text_processing/paddleocr_text_extractor.py` — OCR extraction (used by comparator)
- `src/text_processing/bert_processor.py` — TensorFlow BERT for text similarity

## Output
- The main pipeline creates a JSON file in `output/main_pipeline_res/` with all results and metadata for each run.
- Each run is timestamped for easy tracking and reproducibility.

## Notes
- All code is designed to run in the `pygpu` Conda environment.
- For further details on the output structure, see the JSON file generated after running the pipeline.
