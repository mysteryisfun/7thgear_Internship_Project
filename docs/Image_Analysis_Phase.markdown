# Image Analysis Phase Execution Plan

This document details the execution plan for the Image Analysis Component of the Intelligent Data Extraction System, as outlined in the internship project updates. The component consists of three phases: **Image Detection**, **Image Comparison**, and **Image Analysis**. The goal is to identify, compare, and extract meaningful context from visual elements (e.g., images, charts, diagrams) within presentation slides from meeting videos, aligning with the project's aim to create actionable, timestamped summaries. Per the project requirements, YOLOv5 is used exclusively for detecting images and obtaining their bounding boxes in slides for comparison purposes, not for classification.

## Phase 1: Image Detection

### Objective
Identify visual elements (e.g., images, charts, diagrams) within slide frames and extract their bounding boxes for subsequent comparison and analysis.

### Approach
- **Model Selection**: YOLOv5, a lightweight and efficient object detection model, is chosen for its speed and accuracy in detecting visual elements. YOLOv5 will not classify visuals (e.g., as "chart" or "photo") but will detect regions containing visual content and provide bounding box coordinates.
- **Input**: Frames extracted from meeting videos at 1 frame per second (FPS), as established in the completed Video Processing Pipeline (using FFmpeg and OpenCV).
- **Process**:
  1. **Preprocessing**: Convert frames to a format suitable for YOLOv5 (e.g., RGB, resized to 640x640 for optimal performance).
  2. **Detection**: Feed frames into YOLOv5 to detect regions containing visual elements. YOLOv5 outputs bounding box coordinates (x_min, y_min, x_max, y_max) for each detected visual.
  3. **Storage**: Store bounding box coordinates along with frame timestamps and frame identifiers in a structured format (e.g., JSON) for downstream processing.
- **Output**: A list of detected visual regions per frame, each with:
  - Bounding box coordinates.
  - Timestamp of the frame.
  - Frame identifier (e.g., frame number or file path).

### Implementation Details
- **YOLOv5 Setup**:
  - Use a pre-trained YOLOv5 model (e.g., YOLOv5s for efficiency) from the Ultralytics repository.
  - Fine-tune on a dataset of presentation slides (e.g., SlideShare dataset or custom annotated slides) to improve detection accuracy for slide-specific visuals.
  - Configure YOLOv5 to detect a single class ("visual") to focus on identifying regions without classifying them.
- **Preprocessing with OpenCV**:
  - Convert frames to grayscale and apply Gaussian blur to reduce noise, as described in the development approach.
  - Optionally, use edge detection (Canny) to enhance detection in cluttered slides.
- **Storage Format**:
  ```json
  [
    {
      "frame_id": "frame_001",
      "timestamp": "00:01.000",
      "bounding_boxes": [
        {"x_min": 100, "y_min": 150, "x_max": 300, "y_max": 400},
        {"x_min": 500, "y_min": 200, "x_max": 700, "y_max": 450}
      ]
    },
    ...
  ]
  ```
- **Challenges**:
  - **Cluttered Slides**: Slides with overlays or animations may confuse detection.
  - **Solution**: Use temporal analysis (frame differencing) to focus on stable regions, as suggested in the development approach.
  - **Low-Resolution Visuals**: Small or blurry visuals may be missed.
  - **Solution**: Enhance images with OpenCV (e.g., sharpening) before detection.
- **Why It Matters**: Accurate detection of visual regions ensures only relevant content is processed, optimizing resource usage and aligning with the project's efficiency goals.

## Phase 2: Image Comparison

### Objective
Avoid duplicate processing by identifying unchanged visuals across slide frames using perceptual hashing.

### Approach
- **Technique**: Use perceptual hashing (pHash) to generate a hash for each detected visual, enabling efficient comparison to detect duplicates.
- **Input**: Bounding box regions from Phase 1, cropped from the original frames.
- **Process**:
  1. **Cropping**: Extract the visual region from the frame using bounding box coordinates.
  2. **Hash Computation**: Compute the pHash for each cropped visual using a library like `imagehash`.
  3. **Cache Mechanism**: Store hashes in an in-memory cache (e.g., Redis) with frame timestamps for quick comparison.
  4. **Comparison**: Compare the pHash of the current visual with hashes from the previous frame’s visuals. If the Hamming distance is below a threshold (e.g., 5), mark the visual as a duplicate and skip further analysis.
  5. **Output**: Flag unique visuals (non-duplicates) for analysis in Phase 3, along with their bounding box coordinates and timestamps.
- **Output**: A filtered list of unique visuals with:
  - Bounding box coordinates.
  - Timestamp and frame identifier.
  - pHash value (for reference).

### Implementation Details
- **pHash Computation**:
  - Use the `imagehash` Python library to compute perceptual hashes.
  - Resize cropped visuals to a fixed size (e.g., 64x64) for consistent hashing.
  - Example:
    ```python
    from PIL import Image
    import imagehash

    def compute_phash(image_path):
        img = Image.open(image_path)
        return imagehash.phash(img)
    ```
- **Cache Setup**:
  - Use Redis for in-memory storage of hashes.
  - Structure: Key = `frame_id:visual_index`, Value = `pHash`.
  - Example:
    ```python
    import redis

    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.set(f"frame_001:visual_1", str(phash))
    ```
- **Comparison Logic**:
  - Compute Hamming distance between pHashes:
    ```python
    def is_duplicate(hash1, hash2, threshold=5):
        return hash1 - hash2 <= threshold
    ```
  - If duplicate, skip the visual; otherwise, pass to Phase 3.
- **Challenges**:
  - **Minor Visual Changes**: Small changes (e.g., text overlays on charts) may result in different hashes.
  - **Solution**: Adjust the Hamming distance threshold based on testing to balance sensitivity and efficiency.
  - **Cache Scalability**: Large meetings with many visuals may strain memory.
  - **Solution**: Implement a sliding window cache (e.g., store hashes for the last 10 frames).
- **Why It Matters**: Skipping duplicate visuals reduces computational overhead, aligning with the project’s resource efficiency goals.

## Phase 3: Image Analysis

### Objective
Extract meaningful context from new or changed visuals (e.g., "Pie chart: 50% R&D") to integrate with text summaries.

### Approach
- **Options for Analysis**:
  1. **API Services**: Use Vision APIs (e.g., GPT-4 or Google Gemini) for rapid prototyping and testing due to their ease of integration.
  2. **Self-Deployed Models**: Transition to models like DeepSeek VL2 or Gemma for production to ensure scalability and cost efficiency.
- **Decision**: Start with GPT-4 API for initial testing to validate the pipeline quickly, then evaluate self-hosted models for production.
- **Input**: Cropped unique visuals from Phase 2.
- **Process**:
  1. **API Integration**: Send cropped visuals to the GPT-4 Vision API for context extraction.
  2. **Context Extraction**: Generate descriptive text (e.g., “Bar chart showing Q1 sales: 60% domestic, 40% international”).
  3. **Alignment**: Associate extracted context with timestamps from the original frame.
  4. **Storage**: Store context in a structured format alongside text data for integration.
- **Output**: Structured data combining visual context, bounding box coordinates, and timestamps:
  ```json
  [
    {
      "frame_id": "frame_001",
      "timestamp": "00:01.000",
      "bounding_box": {"x_min": 100, "y_min": 150, "x_max": 300, "y_max": 400},
      "context": "Pie chart: 50% R&D, 30% Marketing, 20% Operations"
    },
    ...
  ]
  ```

### Implementation Details
- **GPT-4 API Integration**:
  - Use the OpenAI API client:
    ```python
    from openai import OpenAI

    client = OpenAI(api_key="your_api_key")
    def analyze_image(image_path):
        with open(image_path, "rb") as image_file:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the visual content."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.encodebytes(image_file.read()).decode('utf-8')}"}}
                        ]
                    }
                ]
            )
        return response.choices[0].message.content
    ```
  - Input: Base64-encoded cropped visual.
  - Output: Descriptive text.
- **Self-Hosted Model Plan**:
  - Fine-tune DeepSeek VL2 on a custom dataset (e.g., 200 annotated slides from internal meetings).
  - Deploy on cloud infrastructure (e.g., AWS EC2 with GPU support).
  - Optimize for latency using batch processing.
- **Challenges**:
  - **Complex Visuals**: Charts with overlapping text or low contrast may confuse APIs.
  - **Solution**: Preprocess visuals with OpenCV (e.g., contrast enhancement) and fine-tune models on diverse slide data.
  - **API Costs**: High usage of GPT-4 may be expensive during testing.
  - **Solution**: Limit API calls to unique visuals and transition to self-hosted models early.
- **Why It Matters**: Extracting meaningful context from visuals ensures the system captures critical insights, enhancing the quality of meeting summaries.

## Integration with Existing Pipeline
- **Input**: Frames from the Video Processing Pipeline (1 FPS, timestamped).
- **Flow**:
  1. **Detection**: YOLOv5 identifies visual regions and outputs bounding boxes.
  2. **Comparison**: pHash filters out duplicates, passing unique visuals to analysis.
  3. **Analysis**: GPT-4 API (or self-hosted model) generates context for unique visuals.
  4. **Output Integration**: Combine visual context with extracted text (from PaddleOCR) using timestamps for alignment, feeding into the summarization module (e.g., BART).
- **Storage**: Use a unified JSON structure to store text and visual data, ensuring compatibility with the Information Merging module.

## Testing and Validation
- **Datasets**:
  - **SlideShare Dataset**: Test detection and analysis on diverse public slides.
  - **Custom Dataset**: Annotate 200 internal meeting slides to validate performance on real-world data.
- **Testing Strategies**:
  - **Unit Testing**: Verify YOLOv5 bounding box accuracy (>90% IoU with ground truth).
  - **Integration Testing**: Ensure end-to-end flow (frame → bounding box → hash → context) produces accurate outputs.
  - **Performance Testing**: Measure latency for processing 1,000 slides/hour, targeting <5% latency increase under load.
- **Tools**: Use pytest for unit tests and GitHub Actions for CI/CD.

## Scalability Considerations
- **Batch Processing**: Process multiple visuals in parallel to reduce latency.
- **Caching**: Use Redis to store pHashes and avoid recomputation.
- **Cloud Deployment**: Deploy YOLOv5 and self-hosted models on scalable cloud infrastructure (e.g., AWS).

## Why This Component Matters
The Image Analysis Component is critical to capturing the full context of meeting slides, as visuals often convey key insights (e.g., charts, diagrams). By using YOLOv5 for precise bounding box detection, pHash for efficient comparison, and Vision APIs for context extraction, this pipeline ensures resource efficiency and high-quality outputs. The approach aligns with the project’s goals of automation, scalability, and actionable insights, paving the way for seamless integration with text processing and summarization modules.