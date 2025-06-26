# Image Analysis Classifier 2: CLIP-based Text/Image Slide Classification

## Overview
This document describes the implementation, usage, and evaluation of the second image classification module (Classifier 2) for distinguishing between "text" and "image/diagram" slides in presentation frames. This module is designed for integration into the Intelligent Data Extraction System and leverages the OpenAI CLIP model via Hugging Face Transformers.

## Implemented Model

### CLIP-based Classifier
- **File:** `src/image_processing/classifier2_models/clip_classifier.py`
- **Model:** `openai/clip-vit-large-patch14` (Hugging Face Transformers)
- **Input:** Numpy array (BGR image, as from OpenCV)
- **Output:** 'text' or 'image' (with confidence and inference time)
- **Prompts:** Robust set of 12 prompts (6 for text, 6 for image/diagram-rich slides)
- **Device:** Automatically uses CUDA if available, else CPU
- **Batching:** Processes one frame at a time (designed for pipeline integration)

## Usage

### Inference (PowerShell, in `pygpu` conda env)
```powershell
conda activate pygpu
python src/image_processing/classifier2_models/clip_classifier.py path/to/frame.jpg
```

### Integration (Python)
```python
from src.image_processing.classifier2_models.clip_classifier import classify_presentation_frame
result, prob, elapsed = classify_presentation_frame(frame)
# result: 'text' or 'image', prob: confidence, elapsed: seconds
```

## Evaluation & Testing
- **Test Script:** `tests/test_clip_classifier2.py` (classifies images in test folders, prints accuracy)
- **Metrics:** Accuracy, confidence, inference time
- **Prompts:** Designed for robustness to real-world slide variations

## Integration Notes
- Model and processor are loaded globally (only once per process)
- Designed for direct use in the main pipeline after Classifier 1
- Compatible with the `pygpu` conda environment

## Files Added/Updated
- `src/image_processing/classifier2_models/clip_classifier.py`
- `tests/test_clip_classifier2.py`

## See Also
- `docs/Image analysis classifier 1.md` for people/presentation classifier
- `docs/execution_guide.md` for pipeline integration

## API and LM Studio Image LLM Integration

### Gemini API (Google)
- **File:** `src/image_processing/API_img_LLM.py`
- **Function:** `extract_image_context_gemini(image, api_key=None)`
- **Input:** Image file path or numpy array (BGR)
- **Output:** Structured JSON with keys: topics, subtopics, entities, numerical_values, descriptive_explanation, tasks_identified, key_findings
- **API:** Uses Gemini API (requires `GEMINI_API_KEY` in environment or .env)
- **Error Handling:** Raises `RuntimeError` on API or parsing errors; robust JSON extraction from response
- **Usage Example:**
    ```python
    from src.image_processing.API_img_LLM import extract_image_context_gemini
    result = extract_image_context_gemini(frame)
    ```

### LM Studio (Gemma 3-4b)
- **File:** `src/image_processing/LMS_img_LLM.py`
- **Function:** `extract_image_context_lmstudio(image)`
- **Input:** Image file path or numpy array (BGR)
- **Output:** Structured JSON with keys: topics, subtopics, entities, numerical_values, descriptive_explanation, tasks_identified, key_findings
- **API:** Sends request to local LM Studio server (default: `http://localhost:1234/v1/chat/completions`)
- **Error Handling:** Raises `RuntimeError` on API or parsing errors; robust JSON extraction from response
- **Usage Example:**
    ```python
    from src.image_processing.LMS_img_LLM import extract_image_context_lmstudio
    result = extract_image_context_lmstudio(frame)
    ```

Both modules are designed for direct integration in the main pipeline and support both file path and numpy array inputs for maximum flexibility. They ensure robust error handling and structured output for downstream processing.
