# Main Pipeline for Frame-by-Frame Video Analysis

## How to Run the Main Pipeline

**Prerequisites:**
- Ensure you have the Conda environment `pygpu` set up with all dependencies installed (see `requirements.txt`).
- Download the Gemma 2B model weights locally (see below for instructions).
- Place your input video in the `data/` directory or provide the correct path.

**Download and Setup Gemma 2B Model (for local GPU inference):**
```powershell
# Install Hugging Face CLI if not already
pip install "huggingface_hub[cli]"
# Login to Hugging Face
huggingface-cli login
# Download model weights
huggingface-cli download google/gemma-2-2b-it --local-dir ./gemma-2b-it
```

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
    - **DINOv2 Embedding Cosine Similarity:** Detects near-duplicate images using deep visual features.
    - **OCR + Text Similarity:** Extracts text (via PaddleOCR) and computes cosine similarity using BERT embeddings.
  - Frames are categorized as `unique_image`, `unique_text`, or `duplicate` based on thresholds.

### 4. Text Extraction and Context Analysis (NEW)
- **File:** `src/text_processing/paddleocr_text_extractor.py`, `src/text_processing/gemma_2B_context_transformers.py`
- **Description:**
  - PaddleOCR is used for robust text extraction from frames.
  - The extracted text is passed to the Gemma 2B model (via HuggingFace transformers, local GPU) for context extraction and scene understanding.
  - The Gemma extractor uses direct model.generate() and tokenizer.decode() for maximum compatibility and GPU support on Windows.
  - All context extraction is performed locally, no LM Studio or cloud API required.

### 5. Slide Type Classification (NEW)
- **File:** `src/image_processing/classifier2_models/clip_classifier.py`
- **Description:**
  - Uses OpenAI CLIP (via Hugging Face Transformers) to classify each presentation frame as either `text` (text-only slide) or `image` (diagram/visual-rich slide).
  - Robust prompt set for real-world slide variations.
  - Only unique presentation frames are classified; result is used to route frames to the appropriate LLM (text or image context extraction).

### 6. Image LLM Context Extraction (NEW)
- **Files:**
  - `src/image_processing/API_img_LLM.py` (Gemini API, Google)
  - `src/image_processing/LMS_img_LLM.py` (LM Studio, Gemma 3-4b)
- **Description:**
  - For frames classified as `image`, sends the frame to the selected image LLM backend (Gemini API or LM Studio) for structured context extraction.
  - Both modules accept file path or numpy array input, return structured JSON (topics, subtopics, entities, numerical_values, descriptive_explanation, tasks_identified, key_findings).
  - Robust error handling and output parsing.

### 7. JSON Export of Results
- **Location:** `output/main_pipeline_res/`
- **Description:**
  - Results are exported as a structured JSON file containing:
    - `metadata`: Video info, processing parameters, output info
    - `summary`: Frame and classification statistics
    - `frames`: List of per-frame results with all relevant fields (classification, duplicate status, embedding similarity, text similarity, OCR text, timing, Gemma context, etc.)
  - The output filename includes the video name and a timestamp for traceability.

## Files Used in the Pipeline
- `src/main_workflow/main_pipeline.py` — Main entry point and workflow logic
- `src/main_workflow/frame_loader.py` — Frame extraction utility
- `src/main_workflow/frame_comparator.py` — Duplicate detection logic
- `src/image_processing/classifier1_models/custom_cnn_classifier.py` — CNN classifier
- `src/image_processing/classifier1_models/efficientnet_functional.py` — EfficientNet classifier
- `src/text_processing/paddleocr_text_extractor.py` — OCR extraction (used by comparator)
- `src/text_processing/gemma_2B_context_transformers.py` — Gemma 2B context extraction (local GPU, transformers)
- `src/text_processing/bert_processor.py` — TensorFlow BERT for text similarity
- `src/image_processing/classifier2_models/clip_classifier.py` — CLIP-based text/image slide classifier
- `src/image_processing/API_img_LLM.py` — Gemini API image LLM integration
- `src/image_processing/LMS_img_LLM.py` — LM Studio (Gemma) image LLM integration

## Output
- The main pipeline creates a JSON file in `output/main_pipeline_res/` with all results and metadata for each run.
- Each run is timestamped for easy tracking and reproducibility.
- For each unique presentation frame, both text and image LLM context extraction results are included in the per-frame and scene-level JSON outputs, depending on the classifier2 result.

## Notes
- All code is designed to run in the `pygpu` Conda environment.
- For further details on the output structure, see the JSON file generated after running the pipeline.
- For full GPU support on Windows, the pipeline uses direct model.generate() and tokenizer.decode() for Gemma 2B, bypassing Triton and TorchInductor.
- If you encounter GPU issues, see the troubleshooting section in `docs/text_extraction&processing.md`.
