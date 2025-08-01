# Project Execution Guide and Current Implementation Status

## Overview
This document provides a comprehensive guide to the current implementation status of the Intelligent Data Extraction System, including detailed directions for running the video processing pipeline, file descriptions, and execution sequences.

## Current Implementation Status

### Completed Components

#### 1. Video Processing Pipeline
- **Status**: ✅ Fully Implemented and Tested
- **Location**: `src/video_processing/`
- **Key Features**:
  - Frame extraction from videos using FFmpeg
  - PaddleOCR-based text extraction
  - Custom scene detection using text similarity
  - NLP processing for text enhancement
  - Configurable parameters for fine-tuning

#### 2. Text Processing and NLP
- **Status**: ✅ Implemented
- **Components**:
  - Text preprocessing and normalization
  - Spell correction and word segmentation
  - Named Entity Recognition (NER)
  - Sentence boundary detection

#### 3. Summarization Model Integration
- **Status**: ✅ Implemented (Pegasus Model)
- **Model**: `google/pegasus-xsum`
- **Purpose**: Context analysis for meeting slide information

#### 4. Image Classification Module (Classifier 1)
- **Status**: ✅ Implemented and Evaluated (June 2025)
- **Location**: `src/image_processing/classifier1_models/`
- **Key Features**:
  - Custom CNN and EfficientNetV2B0-based classifiers for binary image classification (people vs. presentation)
  - Training and evaluation scripts with robust serialization and reproducibility
  - Jupyter notebook for side-by-side model comparison (accuracy, precision, loss, time, size)
  - Models saved in both H5 and SavedModel formats
- **Key Files**:
  - `custom_cnn_classifier.py` (custom CNN, H5 model)
  - `efficientnet_functional.py` (EfficientNetV2B0, H5 & SavedModel)
  - `test_custom_cnn_classifier_eval.ipynb` (evaluation & comparison)
  - `custom_cnn_classifier_model.h5`, `efficientnet_functional_model.h5`, `efficientnet_savedmodel/`
- **Usage**:
  - Train: `python src/image_processing/classifier1_models/custom_cnn_classifier.py --train`
  - Train: `python src/image_processing/classifier1_models/efficientnet_functional.py --train`
  - Evaluate: `jupyter notebook src/image_processing/classifier1_models/test_custom_cnn_classifier_eval.ipynb`
- **Integration**: Models can be imported for downstream prediction in the main pipeline.

#### 5. Slide Type Classification (Classifier 2)
**Purpose**: Classify presentation frames as 'text' or 'image/diagram' using CLIP

**Key Features**:
- Uses OpenAI CLIP (via Hugging Face Transformers)
- Robust prompt set for real-world slide variations
- Only unique presentation frames are classified
- Result is used to route frames to the appropriate LLM (text or image context extraction)

**File**: `src/image_processing/classifier2_models/clip_classifier.py`
**Function**: `classify_presentation_frame(frame)`

#### 6. Image LLM Context Extraction
**Purpose**: Extract structured context from image/diagram slides using an image LLM

**Key Features**:
- Supports both Gemini API (Google) and LM Studio (Gemma 3-4b)
- Accepts file path or numpy array input
- Returns structured JSON (topics, subtopics, entities, numerical_values, descriptive_explanation, tasks_identified, key_findings)
- Robust error handling and output parsing

**Files**:
- `src/image_processing/API_img_LLM.py` (Gemini API)
- `src/image_processing/LMS_img_LLM.py` (LM Studio, Gemma)

## File Descriptions and Usage

### Main Processing Files

#### 1. `main_processor.py`
**Purpose**: Central orchestrator for the entire video processing pipeline

**Key Functions**:
- `process_video()`: Main processing function that coordinates all components
- Frame extraction coordination
- Text extraction using PaddleOCR
- Scene detection based on text similarity
- NLP processing application

**Usage**:
```powershell
python src\video_processing\main_processor.py <video_path> [options]
```

**Arguments**:
- `video_path`: Path to input video file (required)
- `--output-dir`: Output directory (default: "output")
- `--fps`: Frame extraction rate (default: 1.0)
- `--similarity-threshold`: Text similarity threshold (default: 0.8)
- `--use-paddleocr`: Enable PaddleOCR (flag)

**Example**:
```powershell
python src\video_processing\main_processor.py data\faces_and_text.mp4 --output-dir output --fps 1.0 --similarity-threshold 0.8 --use-paddleocr
```

#### 2. `frame_extractor.py`
**Purpose**: Extract frames from video files at specified intervals

**Key Features**:
- FFmpeg integration for video decoding
- Configurable frame rate extraction
- Timestamp preservation
- Error handling for missing dependencies

**Class**: `FrameExtractor`
**Methods**:
- `__init__(video_path, output_dir, fps)`: Initialize extractor
- `check_ffmpeg_installed()`: Verify FFmpeg availability
- `extract_frames()`: Extract frames and return paths with timestamps

**Dependencies**: FFmpeg, ffmpeg-python

#### 3. `paddleocr_text_extractor.py`
**Purpose**: Extract text from image frames using PaddleOCR

**Key Features**:
- High-accuracy text extraction
- Configurable OCR parameters
- Multi-language support (configured for English)
- Robust error handling

**Class**: `PaddleOCRTextExtractor`
**Methods**:
- `__init__()`: Initialize OCR with optimized parameters
- `extract_text_from_frames(frame_paths)`: Extract text from multiple frames

**Configuration**:
- `use_angle_cls=True`: Enable text angle classification
- `lang='en'`: English language model
- `det_db_box_thresh=0.3`: Detection box threshold

#### 4. `nlp_processing.py`
**Purpose**: Apply advanced NLP techniques to enhance extracted text

**Key Features**:
- Word segmentation using WordNinja
- Spell correction with autocorrect
- Named Entity Recognition (NER)
- Sentence boundary detection
- Noise removal and text normalization

**Class**: `NLPProcessor`
**Dependencies**: spaCy (en_core_web_sm), wordninja, autocorrect

**Processing Steps**:
1. Word segmentation for compound words
2. Spell correction
3. Sentence boundary detection
4. Named entity recognition
5. Case normalization
6. Noise filtering

#### 5. `text_processor.py`
**Purpose**: Basic text preprocessing and similarity detection

**Key Features**:
- Text normalization and cleaning
- Duplicate detection based on similarity
- Image preprocessing for OCR enhancement

**Class**: `TextProcessor`
**Methods**:
- `process_text(texts)`: Clean and deduplicate text list
- `preprocess_image(image_path)`: Enhance image for better OCR

## Execution Sequences

### Standard Processing Workflow

1. **Video Input**: Load video file from `data/` directory
2. **Frame Extraction**: Extract frames at specified fps using FFmpeg
3. **Frame Classification**: Classify each frame as 'people' or 'presentation' (Classifier 1)
4. **Duplicate Detection**: Detect unique presentation frames
5. **Slide Type Classification**: Classify unique presentation frames as 'text' or 'image' (Classifier 2)
6. **Text/Image LLM Context Extraction**: Route to Gemma (text) or Gemini/LM Studio (image) for context extraction
7. **Output Generation**: Save processed results to output directory

### Scene Detection Logic

The system uses a custom scene detection algorithm:

```python
# Text preprocessing
preprocessed_texts = ["".join(text.split()) for text in extracted_texts]

# Similarity comparison
for i, text in enumerate(preprocessed_texts):
    if previous_text:
        similarity = SequenceMatcher(None, previous_text, text).ratio()
        if similarity < similarity_threshold:
            scenes.append(i)  # Mark as scene change
```

**Threshold Interpretation**:
- `0.8` (default): Detects significant text changes
- Lower values: More sensitive to minor changes
- Higher values: Only major content changes trigger detection

## Testing and Validation

### Tested Video Files
Located in `data/` directory:
- `faces_and_text.mp4`: Primary test video
- `faces_start.mp4`: Alternative test case
- `n8n_and_faces_and_text.mp4`: Complex content test
- `image_text_website_faces.mp4`: Mixed content test

### Output Structure
```
output/
├── frames/           # All extracted frames
│   ├── frame_00000.jpg
│   ├── frame_00001.jpg
│   └── ...
└── keyframes/        # Frames where scene changes detected
```

## Dependencies and Installation

### Required Software
1. **Python 3.9**
2. **FFmpeg**: Video processing backend
3. **PaddleOCR**: Text extraction engine

### Python Packages
Install via `pip install -r requirements.txt`:
- `tensorflow==2.10.0`
- `opencv-python==4.5.5.64`
- `ffmpeg-python==0.2.0`
- `paddleocr==2.6.0.1`
- `spacy==3.7.2`
- Additional NLP and utility packages

### SpaCy Model
Download English language model:
```powershell
python -m spacy download en_core_web_sm
```

## Performance Characteristics

### Processing Times (Approximate)
- **Frame Extraction**: ~1-2 seconds per minute of video
- **Text Extraction**: ~0.5-1 seconds per frame
- **Scene Detection**: ~0.1 seconds per frame comparison
- **NLP Processing**: ~1-2 seconds per scene text

### Resource Usage
- **Memory**: 2-4 GB during processing
- **Storage**: ~100-200 KB per extracted frame
- **CPU**: Moderate usage, GPU acceleration available for OCR

## Troubleshooting

### Common Issues

1. **FFmpeg Not Found**
   - Solution: Install FFmpeg and add to system PATH
   - Verification: `ffmpeg -version` in terminal

2. **PaddleOCR Installation Issues**
   - Solution: Use conda environment or specific pip version
   - Alternative: `conda install paddlepaddle paddleocr`

3. **SpaCy Model Missing**
   - Solution: `python -m spacy download en_core_web_sm`

4. **Memory Issues**
   - Solution: Reduce fps or process shorter video segments
   - Recommendation: Process videos in chunks

### Debug Mode
Enable verbose output by modifying print statements in `main_processor.py`

## Future Enhancements

### Planned Features
1. **Image Classification**: ResNet integration for visual content analysis
2. **Diagram Processing**: ChartOCR integration for chart/graph analysis  
3. **Audio Processing**: Speech-to-text integration
4. **Real-time Processing**: Live meeting integration
5. **Cloud Deployment**: Scalable processing infrastructure

### Integration Points
- Zoom/Teams API integration
- Task management system connections
- Cloud storage integration
- Real-time dashboard

## Execution Examples

### Basic Usage
```powershell
# Process video with default settings
python src\video_processing\main_processor.py data\faces_and_text.mp4

# Custom output directory and frame rate
python src\video_processing\main_processor.py data\faces_and_text.mp4 --output-dir my_output --fps 0.5

# High sensitivity scene detection
python src\video_processing\main_processor.py data\faces_and_text.mp4 --similarity-threshold 0.9

# Complete example with all options
python src\video_processing\main_processor.py data\faces_and_text.mp4 --output-dir output --fps 1.0 --similarity-threshold 0.8 --use-paddleocr
```

### Batch Processing
For multiple videos, create a PowerShell script:
```powershell
# process_all_videos.ps1
$videos = Get-ChildItem -Path "data\*.mp4"
foreach ($video in $videos) {
    python src\video_processing\main_processor.py $video.FullName --output-dir "output\$($video.BaseName)"
}
```

This documentation provides a complete guide for understanding, running, and troubleshooting the current video processing implementation.

## [UPDATED June 2025] Scene Text Context Extraction Pipeline

### Major Pipeline Change
- The pipeline now uses the **Gemma LLM (via LM Studio API)** for extracting meaningful context from scene text, replacing the previous BERT/Pegasus-based approach.
- Scene-level context is extracted using the `GemmaContextExtractor` (see `src/video_processing/text_context_extraction.py`).
- The output is a structured JSON file per video, with:
    - **Comprehensive metadata**
    - **Scene-level context** (Gemma fields: topics, subtopics, entities, numerical_values, descriptive explanation, tasks identified, timestamp_range, etc.)
    - **Summary text file** for each analysis
- The output directory and file structure remain unchanged, but the context extraction and scene data fields are now LLM-based.

### Output File Structure
```
output/
├── frames/              # All extracted frames
├── keyframes/           # Scene change frames
└── results/             # Analysis results organized by video and timestamp
    └── {video_name}_analysis_{timestamp}/
        ├── {video_name}_analysis_{timestamp}.json  # Structured analysis data
        └── {video_name}_analysis_{timestamp}.txt   # Human-readable summary
```

### JSON Output Structure (Gemma-based)
Each scene entry now includes:
- `frame_id`, `frame_timestamp`, `scene_range`, `ocr_text`
- `text_model`, `model_endpoint`
- `topics`, `subtopics`, `entities`, `numerical_values`
- `descriptive explanation`, `tasks identified`
- `timestamp_range` (start, end, duration)

#### Example Scene Entry
```json
{
  "frame_id": "frame_00000",
  "frame_timestamp": "0.0",
  "scene_range": "0.0 - END",
  "ocr_text": "...",
  "text_model": "gemma-2-2b-it",
  "model_endpoint": "http://localhost:1234",
  "topics": ["..."],
  "subtopics": ["..."],
  "entities": {"persons": ["..."], "organizations": ["..."], "events": [], "dates": []},
  "numerical_values": [],
  "descriptive explanation": "...",
  "tasks identified": ["..."],
  "timestamp_range": {"start_seconds": 0.0, "end_seconds": 21.0, "duration_seconds": 21.0}
}
```

### Usage and Integration
- Run the pipeline as before using `main_processor.py` (see usage examples below).
- All context extraction and scene data are now LLM-based and more descriptive.
- See `docs/text_extraction&processing.md` for further details on the new output structure.

### [LEGACY] BERT/Pegasus-based context extraction is deprecated and removed.
