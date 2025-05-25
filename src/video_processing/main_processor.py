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

from src.video_processing.frame_extractor import FrameExtractor
from src.video_processing.paddleocr_text_extractor import PaddleOCRTextExtractor
from src.video_processing.text_processor import TextProcessor
from src.video_processing.nlp_processing import NLPProcessor
from src.video_processing.output_manager import OutputManager


def process_video(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    similarity_threshold: float = 0.8,
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

    # Initialize NLPProcessor
    nlp_processor = NLPProcessor()

    # Step 3: Pre-process extracted text by removing spaces and making it a single string
    print("Pre-processing extracted text...")
    preprocessed_texts = ["".join(text.split()) for text in extracted_texts]

    # Step 4: Scene Detection based on text similarity
    print("Detecting scenes based on text changes...")
    scenes = []
    previous_text = ""

    for i, text in enumerate(preprocessed_texts):
        if previous_text:
            similarity = SequenceMatcher(None, previous_text, text).ratio()
            if similarity < similarity_threshold:
                scenes.append(i)  # Mark this frame as a scene change
        previous_text = text

    print(f"Detected {len(scenes)} scenes.")
    for i, scene_index in enumerate(scenes):
        print(f"Scene {i+1}: Frame {scene_index}")

    # Process text of frames where scene changes were detected
    scene_texts = [extracted_texts[scene_index] for scene_index in scenes]
    nlp_processed_scene_texts = nlp_processor.process_texts(scene_texts)

    # Print NLP-processed text for scenes
    for i, text in enumerate(nlp_processed_scene_texts):
        print(f"NLP Processed Scene Text {i+1}:\n{text}\n")

    # Save structured results to JSON
    print("Saving structured results...")
    output_manager = OutputManager(output_dir)
    
    # Prepare processing parameters for metadata
    processing_params = {
        "fps": fps,
        "similarity_threshold": similarity_threshold,
        "use_paddleocr": use_paddleocr,
        "tesseract_path": tesseract_path,
        "nlp_processing": True,
        "scene_detection_method": "text_similarity"
    }
    
    # Save complete results
    json_path = output_manager.save_processing_results(
        video_path=video_path,
        frame_data=frame_paths_with_timestamps,
        scene_indices=scenes,
        extracted_texts=extracted_texts,
        processed_texts=nlp_processed_scene_texts,
        processing_params=processing_params
    )
    
    # Create summary report
    summary_path = output_manager.create_summary_report(json_path)
    
    print(f"Results saved to: {json_path}")
    print(f"Summary report: {summary_path}")
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
        default=0.8
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
