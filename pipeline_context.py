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
    for idx, frame, ts in frame_generator(video_path, fps=fps):
        # Always classify first
        start_time = time.perf_counter()
        label, prob = classifier(model, frame)
        pred_time = time.perf_counter() - start_time
        print(f"Frame {idx} at {ts:.2f}s: Classification: {label} (probability: {prob:.2f}) | predict_time: {pred_time:.4f} sec")
        if label == 'people':
            continue  # Discard people frames immediately
        # Only 'presentation' frames go to duplicate/unique checks
        if prev_frame is not None:
            is_img_unique, is_text_unique, ocr_text, phash_diff, text_sim = comparator.is_unique(frame, prev_frame, prev_ocr_text)
            if is_img_unique:
                print(f"  UNIQUE IMAGE (phash diff: {phash_diff}) -> send to image processing")
            elif is_text_unique:
                print(f"  UNIQUE TEXT (sim: {text_sim:.3f}) -> send to text processing")
            else:
                print(f"  DUPLICATE (phash diff: {phash_diff}, sim: {text_sim:.3f}) -> discard")
                prev_frame = frame
                prev_ocr_text = ocr_text
                continue
            prev_ocr_text = ocr_text
        else:
            # Always keep the first presentation frame
            print(f"  FIRST PRESENTATION FRAME -> keep")
            prev_ocr_text = comparator.compute_text(frame)
        prev_frame = frame
        # ... further processing for unique presentation frames ...
        #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Pipeline for Frame-by-Frame Video Analysis")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, choices=['CNN', 'EFF'], required=True, help='Classifier model to use (CNN or EFF)')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second to extract (default 1.0)')
    args = parser.parse_args()
    main(args.video, args.model, args.fps)