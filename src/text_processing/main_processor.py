"""
Main processing script for video analysis.

This script integrates frame extraction and scene analysis to process videos
and extract key frames containing distinct text content.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import sys
import cv2
from difflib import SequenceMatcher

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.text_processing.frame_extractor import FrameExtractor
from src.text_processing.paddleocr_text_extractor import PaddleOCRTextExtractor
from src.text_processing.enhanced_text_processor import EnhancedTextProcessor
from src.text_processing.output_manager import OutputManager
from src.text_processing.gemma_2B_context_model import GemmaContextExtractor


def process_video(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    similarity_threshold: float = 0.95,
    tesseract_path: str = None,
    use_paddleocr: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Process a video to extract frames and detect scenes based on text changes.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save outputs
        fps: Frame rate for extraction
        similarity_threshold: Threshold for text similarity
        tesseract_path: Path to tesseract executable
        use_paddleocr: Use PaddleOCR instead of Tesseract
    
    Returns:
        Tuple of (all_frame_paths, keyframe_paths)
    """
    # Create output directories
    frames_dir = os.path.join(output_dir, "frames")
    keyframes_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(keyframes_dir, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Extract frames
    print("Extracting frames...")
    frame_extractor = FrameExtractor(video_path, frames_dir, fps)
    frame_paths_with_timestamps = frame_extractor.extract_frames()
    frame_paths = [path for path, _ in frame_paths_with_timestamps]
    print(f"Extracted {len(frame_paths)} frames at {fps} fps")
    
    # Step 2: Extract text from frames using PaddleOCR
    print("Extracting text from frames using PaddleOCR...")
    text_extractor = PaddleOCRTextExtractor()

    extracted_texts = text_extractor.extract_text_from_frames(frame_paths)

    # Print original extracted text
    for i, text in enumerate(extracted_texts):
        print(f"Original Text {i+1}:\n{text}\n")

    # Initialize EnhancedTextProcessor
    text_processor = EnhancedTextProcessor()

    # Step 3: Pre-process extracted text by removing spaces and making it a single string
    print("Pre-processing extracted text...")
    preprocessed_texts = ["".join(text.split()) for text in extracted_texts]

    # Step 4: Scene Detection based on semantic similarity
    print("Detecting scenes based on semantic changes...")
    scenes = [0]  # Always start with the first frame as a scene
    previous_text = preprocessed_texts[0] if preprocessed_texts else ""

    for i, text in enumerate(preprocessed_texts[1:], start=1):
        similarity = text_processor.compute_text_similarity(previous_text, text)
        print(f"Similarity between frame {i-1} and frame {i}: {similarity}")  # Debug log
        if similarity < similarity_threshold:
            scenes.append(i)  # Mark this frame as a scene change
        previous_text = text

    print(f"Detected {len(scenes)} scenes.")
    for i, scene_index in enumerate(scenes):
        print(f"Scene {i+1}: Frame {scene_index}")

    # Process text of frames where scene changes were detected
    scene_texts = [extracted_texts[scene_index] for scene_index in scenes]
    # Get structured, meaningful context for each scene
    scene_contexts = text_processor.process_texts(scene_texts)

    # Initialize Gemma context extractor
    gemma_extractor = GemmaContextExtractor()

    # For each scene, send the NLP-processed text to Gemma and collect results for saving
    gemma_scene_results = []
    for i, (context, scene_index) in enumerate(zip(scene_contexts, scenes)):
        # Join the body_content, headers, and metadata for NLP-processed text
        nlp_processed_text = " ".join(context.get('headers', []) + context.get('metadata', []) + context.get('body_content', []))
        print(f"\n=== Scene {i+1} NLP-Processed Text ===")
        print(nlp_processed_text)
        gemma_result = gemma_extractor.extract_context(nlp_processed_text)
        print(f"\n=== Scene {i+1} Gemma Output ===")
        print(gemma_result)

        # Prepare new structure for saving
        frame_id = f"frame_{scene_index:05d}"
        frame_timestamp = str(frame_paths_with_timestamps[scene_index][1]) if scene_index < len(frame_paths_with_timestamps) else ""
        # Scene range: current frame to next scene or end
        if i + 1 < len(scenes):
            next_scene_index = scenes[i+1]
            scene_range = f"{frame_timestamp} - {str(frame_paths_with_timestamps[next_scene_index][1])}"
        else:
            scene_range = f"{frame_timestamp} - END"
        ocr_text = extracted_texts[scene_index]
        # Parse Gemma output
        topics = gemma_result.get('topics', [])
        subtopics = gemma_result.get('subtopics', [])
        entities = gemma_result.get('entities', {"persons": [], "organizations": [], "events": [], "dates": []})
        numerical_values = gemma_result.get('numerical_values', [])
        # Fix: descriptive explanation is a top-level key, not nested
        desc_explanation = gemma_result.get('descriptive explanation', "")
        tasks_identified = gemma_result.get('tasks identified', [])
        # Add timestamp_range to the scene dictionary
        timestamp_range = {
            "start_seconds": float(frame_timestamp) if frame_timestamp else 0.0,
            "end_seconds": float(frame_paths_with_timestamps[next_scene_index][1]) if i + 1 < len(scenes) else float(frame_paths_with_timestamps[-1][1]),
            "duration_seconds": (float(frame_paths_with_timestamps[next_scene_index][1]) - float(frame_timestamp)) if i + 1 < len(scenes) else (float(frame_paths_with_timestamps[-1][1]) - float(frame_timestamp))
        }

        gemma_scene_results.append({
            "frame_id": frame_id,
            "frame_timestamp": frame_timestamp,
            "scene_range": scene_range,
            "ocr_text": ocr_text,
            "text_model": "gemma-2-2b-it",
            "model_endpoint": "http://localhost:1234",
            "topics": topics,
            "subtopics": subtopics,
            "entities": entities,
            "numerical_values": numerical_values,
            "descriptive explanation": desc_explanation,
            "tasks identified": tasks_identified,
            "timestamp_range": timestamp_range
        })

    # Save new structured results to JSON and metadata text file
    import json
    from datetime import datetime

    # Generate unique analysis folder and filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_folder = f"{Path(video_path).stem}_analysis_{timestamp}"
    analysis_dir = os.path.join(output_dir, "results", analysis_folder)
    os.makedirs(analysis_dir, exist_ok=True)

    json_filename = f"{Path(video_path).stem}_analysis_{timestamp}.json"
    json_path = os.path.join(analysis_dir, json_filename)

    txt_filename = f"{Path(video_path).stem}_analysis_{timestamp}.txt"
    txt_path = os.path.join(analysis_dir, txt_filename)

    # Build metadata and summary
    metadata = {
        "video_file": {
            "path": video_path,
            "filename": Path(video_path).name,
            "size_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else None
        },
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "processing_date": datetime.now().strftime("%Y-%m-%d"),
            "processing_time": datetime.now().strftime("%H:%M:%S"),
            "total_frames_extracted": len(frame_paths),
            "total_scenes_detected": len(scenes),
            "video_duration_seconds": frame_paths_with_timestamps[-1][1] if frame_paths_with_timestamps else 0
        },
        "parameters": {
            "fps": fps,
            "similarity_threshold": similarity_threshold,
            "use_paddleocr": use_paddleocr,
            "tesseract_path": tesseract_path,
            "nlp_processing": True,
            "scene_detection_method": "text_similarity"
        },
        "output_info": {
            "output_directory": output_dir,
            "analysis_folder": analysis_folder,
            "results_file": json_filename,
            "frames_directory": os.path.join(output_dir, "frames"),
            "keyframes_directory": os.path.join(output_dir, "keyframes")
        }
    }

    summary = {
        "total_scenes": len(scenes),
        "total_frames": len(frame_paths),
        "video_duration": frame_paths_with_timestamps[-1][1] if frame_paths_with_timestamps else 0,
        "average_scene_duration": sum(scene["timestamp_range"]["duration_seconds"] for scene in gemma_scene_results) / len(gemma_scene_results) if gemma_scene_results else 0,
        "total_text_length": sum(len(scene["ocr_text"]) for scene in gemma_scene_results),
        "scenes_with_text": len([scene for scene in gemma_scene_results if scene["ocr_text"]])
    }

    # Combine metadata, summary, and scenes
    results = {
        "metadata": metadata,
        "summary": summary,
        "scenes": gemma_scene_results
    }

    # Save JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save metadata text file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Video Analysis Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Video File: {metadata['video_file']['filename']}\n")
        f.write(f"Processing Date: {metadata['processing_info']['processing_date']}\n")
        f.write(f"Processing Time: {metadata['processing_info']['processing_time']}\n")
        f.write(f"Analysis Folder: {metadata['output_info']['analysis_folder']}\n\n")
        f.write("Summary Statistics:\n")
        f.write(f"- Total Scenes: {summary['total_scenes']}\n")
        f.write(f"- Total Frames: {summary['total_frames']}\n")
        f.write(f"- Video Duration: {summary['video_duration']:.2f} seconds\n")
        f.write(f"- Average Scene Duration: {summary['average_scene_duration']:.2f} seconds\n")
        f.write(f"- Total Text Length: {summary['total_text_length']} characters\n")
        f.write(f"- Scenes with Text: {summary['scenes_with_text']}\n\n")
        f.write("Scene Details:\n")
        f.write("=" * 30 + "\n")
        for scene in gemma_scene_results:
            f.write(f"\nScene {scene['frame_id']}:\n")
            f.write(f"  Time Range: {scene['scene_range']}\n")
            f.write(f"  Text Length: {len(scene['ocr_text'])} characters\n")
            f.write(f"  Processed Text: {scene['ocr_text'][:100]}{'...' if len(scene['ocr_text']) > 100 else ''}\n")

    print(f"Analysis results saved to: {json_path}")
    print(f"Summary report saved to: {txt_path}")

    print("Text processing complete.")

    return frame_paths, scenes

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process video to extract frames and detect scenes"
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save outputs",
        default="output"
    )
    parser.add_argument(
        "--fps", "-f",
        help="Frame rate for extraction",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--similarity-threshold", "-s",
        help="Threshold for text similarity (0.0 to 1.0)",
        type=float,
        default=0.95
    )
    parser.add_argument(
        "--tesseract-path", "-t",
        help="Path to tesseract executable",
        default=None
    )
    parser.add_argument(
        "--use-paddleocr", "-p",
        help="Use PaddleOCR instead of Tesseract",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Process the video
    process_video(
        args.video_path,
        args.output_dir,
        args.fps,
        args.similarity_threshold,
        args.tesseract_path,
        args.use_paddleocr
    )

if __name__ == "__main__":
    main()
