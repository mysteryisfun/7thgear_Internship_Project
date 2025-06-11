"""
Main Pipeline for Frame-by-Frame Video Analysis

Loads frames one at a time using frame_loader, classifies each frame as 'people' or 'presentation' using the selected model (CNN or EfficientNet), and prints the result. Only 'presentation' frames continue to next stages (not implemented here).

Usage (PowerShell):
    conda activate pygpu
    python src/main_workflow/main_pipeline.py --video data/faces_and_text.mp4 --model CNN
    python src/main_workflow/main_pipeline.py --video data/faces_and_text.mp4 --model EFF
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import time
from src.main_workflow.frame_loader import frame_generator
from src.main_workflow.frame_comparator import FrameComparator

# Import classifier model loader and predict functions
from src.image_processing.classifier1_models.custom_cnn_classifier import load_model as cnn_load_model, predict_frame_class as cnn_predict
from src.image_processing.classifier1_models.efficientnet_functional import load_model as eff_load_model, predict_frame_class as eff_predict

CLASSIFIER_MODELS = {
    'CNN': (cnn_load_model, cnn_predict),
    'EFF': (eff_load_model, eff_predict)
}

def main(video_path: str, model_type: str, fps: float = 1.0):
    if model_type not in CLASSIFIER_MODELS:
        raise ValueError(f"Invalid model_type: {model_type}. Use 'CNN' or 'EFF'.")
    load_model, classifier = CLASSIFIER_MODELS[model_type]
    print(f"[INFO] Using {model_type} model for classification.")
    print(f"[INFO] Processing video: {video_path}")
    model = load_model()
    comparator = FrameComparator(phash_threshold=5, text_threshold=0.85)
    prev_frame = None
    prev_ocr_text = None
    frame_results = []
    start_time = time.time()
    for idx, frame, ts in frame_generator(video_path, fps=fps):
        start_pred = time.perf_counter()
        label, prob = classifier(model, frame)
        pred_time = time.perf_counter() - start_pred
        print(f"Frame {idx} at {ts:.2f}s: Classification: {label} (probability: {prob:.2f}) | predict_time: {pred_time:.4f} sec")
        frame_entry = {
            "frame_id": f"frame_{idx:05d}",
            "frame_timestamp": ts,
            "classification": label,
            "classification_probability": float(prob),
            "predict_time_sec": float(pred_time),
            "duplicate_status": None,
            "phash": None,
            "phash_diff": None,
            "text_similarity": None,
            "ocr_text": None,
            "processing_type": None,
            "notes": ""
        }
        if label == 'people':
            frame_entry["processing_type"] = "discarded"
            frame_entry["notes"] = "Frame classified as people, discarded."
            frame_results.append(frame_entry)
            continue
        # Only 'presentation' frames go to duplicate/unique checks
        if prev_frame is not None:
            is_img_unique, is_text_unique, ocr_text, phash_diff, text_sim = comparator.is_unique(frame, prev_frame, prev_ocr_text)
            frame_entry["phash_diff"] = int(phash_diff) if phash_diff is not None else None
            frame_entry["text_similarity"] = float(text_sim) if text_sim is not None else None
            frame_entry["ocr_text"] = ocr_text
            if is_img_unique:
                frame_entry["duplicate_status"] = "unique_image"
                frame_entry["processing_type"] = "image_processing"
                print(f"  UNIQUE IMAGE (phash diff: {phash_diff}) -> send to image processing")
            elif is_text_unique:
                frame_entry["duplicate_status"] = "unique_text"
                frame_entry["processing_type"] = "text_processing"
                print(f"  UNIQUE TEXT (sim: {text_sim:.3f}) -> send to text processing")
            else:
                frame_entry["duplicate_status"] = "duplicate"
                frame_entry["processing_type"] = "discarded"
                frame_entry["notes"] = "Duplicate frame, discarded."
                print(f"  DUPLICATE (phash diff: {phash_diff}, sim: {text_sim:.3f}) -> discard")
                frame_results.append(frame_entry)
                prev_frame = frame
                prev_ocr_text = ocr_text
                continue
            prev_ocr_text = ocr_text
        else:
            # Always keep the first presentation frame
            frame_entry["duplicate_status"] = "first_presentation"
            frame_entry["processing_type"] = "image_processing"
            frame_entry["notes"] = "First presentation frame."
            prev_ocr_text = comparator.compute_text(frame)
        prev_frame = frame
        frame_results.append(frame_entry)
    # Save results to JSON
    output_dir = os.path.join("output", "main_pipeline_res")
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    json_filename = f"{video_name}_image_analysis_{timestamp_str}.json"
    json_path = os.path.join(output_dir, json_filename)
    metadata = {
        "video_file": {
            "path": video_path,
            "filename": os.path.basename(video_path),
            "size_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else None
        },
        "processing_info": {
            "timestamp": timestamp_str,
            "total_frames_extracted": len(frame_results),
            "video_duration_seconds": frame_results[-1]["frame_timestamp"] if frame_results else 0
        },
        "parameters": {
            "fps": fps,
            "phash_threshold": comparator.phash_threshold,
            "text_similarity_threshold": comparator.text_threshold,
            "classifier_model": model_type
        },
        "output_info": {
            "output_directory": output_dir,
            "results_file": json_filename
        }
    }
    summary = {
        "total_frames": len(frame_results),
        "people_frames": sum(1 for f in frame_results if f["classification"] == 'people'),
        "presentation_frames": sum(1 for f in frame_results if f["classification"] == 'presentation'),
        "unique_images": sum(1 for f in frame_results if f["duplicate_status"] == 'unique_image'),
        "unique_texts": sum(1 for f in frame_results if f["duplicate_status"] == 'unique_text'),
        "duplicates": sum(1 for f in frame_results if f["duplicate_status"] == 'duplicate'),
        "first_presentations": sum(1 for f in frame_results if f["duplicate_status"] == 'first_presentation')
    }
    export = {
        "metadata": metadata,
        "summary": summary,
        "frames": frame_results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Frame-by-frame JSON results exported to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Pipeline for Frame-by-Frame Video Analysis")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, choices=['CNN', 'EFF'], required=True, help='Classifier model to use (CNN or EFF)')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second to extract (default 1.0)')
    args = parser.parse_args()
    main(args.video, args.model, args.fps)