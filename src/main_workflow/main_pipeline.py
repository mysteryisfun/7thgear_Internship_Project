"""
Main Pipeline for Frame-by-Frame Video Analysis

Loads frames one at a time using frame_loader, classifies each frame as 'people' or 'presentation' using the selected model (CNN or EfficientNet), and prints the result. Only 'presentation' frames continue to next stages (not implemented here).

Usage (PowerShell):
    conda activate pygpu
    python src/main_workflow/main_pipeline.py --video data/faces_and_text.mp4 --model CNN
    python src/main_workflow/main_pipeline.py --video data/faces_and_text.mp4 --model EFF
"""
import torch
import os
import argparse
import cv2
import numpy as np
#import tensorflow as tf
import time
from datetime import datetime
from src.main_workflow.frame_loader import frame_generator
from src.main_workflow.frame_comparator import FrameComparator
from src.text_processing.gemma_2B_context_model import GemmaContextExtractor
from src.text_processing.API_text_LLM import GeminiAPIContextExtractor
from src.image_processing.API_img_LLM import extract_image_context_gemini
from src.image_processing.LMS_img_LLM import extract_image_context_lmstudio

# Import classifier model loader and predict functions
from src.image_processing.classifier1_models.custom_cnn_classifier import load_model as cnn_load_model, predict_frame_class as cnn_predict
from src.image_processing.classifier1_models.efficientnet_functional import load_model as eff_load_model, predict_frame_class as eff_predict
from src.image_processing.classifier2_models.clip_classifier import classify_presentation_frame

CLASSIFIER_MODELS = {
    'CNN': (cnn_load_model, cnn_predict),
    'EFF': (eff_load_model, eff_predict)
}

def save_frame_and_json(frame, frame_entry, output_dir):
    """
    Save a unique frame and its corresponding JSON entry to the specified directory.

    Args:
        frame (np.ndarray): The frame image to save.
        frame_entry (dict): The JSON entry corresponding to the frame.
        output_dir (str): The directory to save the frame and JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the frame
    frame_filename = f"{frame_entry['frame_id']}.jpg"
    frame_path = os.path.join(output_dir, frame_filename)
    cv2.imwrite(frame_path, frame)

    # Save the JSON file
    json_filename = f"{os.path.basename(output_dir)}.json"
    json_path = os.path.join(output_dir, json_filename)
    # Convert all numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif hasattr(obj, 'item') and callable(obj.item):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        else:
            return obj
    frame_entry = convert_types(frame_entry)
    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(frame_entry, f, indent=2, ensure_ascii=False)

def main(video_path: str, model_type: str, fps: float = 1.0, text_llm_backend: str = 'LMS', image_llm_backend: str = 'API'):
    main_start=time.time()
    if model_type not in CLASSIFIER_MODELS:
        raise ValueError(f"Invalid model_type: {model_type}. Use 'CNN' or 'EFF'.")
    load_model, classifier = CLASSIFIER_MODELS[model_type]
    print(f"[INFO] Using {model_type} model for classification.")
    print(f"[INFO] Processing video: {video_path}")
    model = load_model()
    comparator = FrameComparator(dino_similarity_threshold=0.98, text_threshold=0.85)
    if text_llm_backend == 'API':
        text_llm_extractor = GeminiAPIContextExtractor()
    else:
        text_llm_extractor = GemmaContextExtractor()
    prev_frame = None
    prev_ocr_text = None
    frame_results = []
    start_time = time.time()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", "main_pipeline_res", f"{video_name}_{timestamp_str}")

    os.makedirs(output_dir, exist_ok=True)

    for idx, frame, ts in frame_generator(video_path, fps=fps):
        start_pred = time.perf_counter()
        label, prob = classifier(model, frame)
        pred_time = time.perf_counter() - start_pred
        #print(f"{'='*59}")
        #print(f"Frame {idx} at {ts:.2f}s: {label}")
        #print(f"  Classification: {label} | Model: {model_type} | Confidence: {prob:.2f} | Time: {pred_time:.4f}s")
        if label == 'people':
            #print(f"  Result: PEOPLE -> Discard")
            #print(f"{'='*59}")
            frame_entry = {
                "frame_id": f"frame_{idx:05d}",
                "frame_timestamp": ts,
                "classification": label,
                "classification_probability": float(prob),
                "predict_time_sec": float(pred_time),
                "duplicate_status": None,
                "embedding_time": None,
                "text_similarity": None,
                "ocr_text": None,
                "processing_type": "discarded",
                "notes": "Frame classified as people, discarded."
            }
            frame_results.append(frame_entry)
            continue
        # Only 'presentation' frames go to duplicate/unique checks
        if prev_frame is not None:
            is_img_unique, is_text_unique, ocr_text, embedding_time, text_sim, cosine_sim = comparator.is_unique(frame, prev_frame, prev_ocr_text)
            frame_entry = {
                "frame_id": f"frame_{idx:05d}",
                "frame_timestamp": ts,
                "classification": label,
                "classification_probability": float(prob),
                "predict_time_sec": float(pred_time),
                "duplicate_status": None,
                "embedding_time": float(embedding_time),
                "embedding_similarity": cosine_sim,
                "text_similarity": float(text_sim) if text_sim is not None else None,
                "ocr_text": ocr_text,
                "processing_type": None,
                "notes": ""
            }
            # Print frame analysis summary ONCE per frame, after all processing
            print(f"{'='*59}")
            print(f"Frame {idx} at {ts:.2f}s: {label}")
            print(f"  Classification: {label} | Model: {model_type} | Confidence: {prob:.2f} | Time: {pred_time:.4f}s")
            print(f"  Embedding Similarity: {cosine_sim:.5f}")
            print(f"  Embedding Time: {embedding_time:.4f}s")
            print(f"  Text Similarity: {text_sim:.4f}")
            print(f"  Text Processing Time: {embedding_time:.4f}s")
            if is_img_unique:
                frame_entry["duplicate_status"] = "unique_image"
                print(f"  Result: UNIQUE IMAGE -> Image Processing")
                # --- Classifier 2: CLIP-based text/image classifier ---
                classifier2_start = time.perf_counter()
                classifier2_result, classifier2_prob, classifier2_time = classify_presentation_frame(frame)
                classifier2_elapsed = time.perf_counter() - classifier2_start
                print(f"  classifier2: {classifier2_result} | Confidence: {classifier2_prob:.3f} | Time: {classifier2_time:.3f}s (total: {classifier2_elapsed:.3f}s)")
                frame_entry["classifier2_result"] = classifier2_result
                frame_entry["classifier2_confidence"] = classifier2_prob
                frame_entry["classifier2_time_sec"] = classifier2_time
                # --- Route based on classifier2 output ---
                if classifier2_result == 'text':
                    # Use OCR text from deduplication, do not run OCR again
                    gemma_start = time.perf_counter()
                    gemma_output = text_llm_extractor.extract_context(ocr_text)
                    gemma_time = time.perf_counter() - gemma_start
                    frame_entry["gemma_context"] = gemma_output
                    frame_entry["gemma_time_sec"] = gemma_time
                    if text_llm_backend == 'LMS':
                        print("Text Processor : Gemma 2-2B-it")
                        print(f"  gemma processing: {gemma_output}, time: {gemma_time:.4f}s")
                    else:
                        print("Text Processor : Google gemini-2.0-flash API")
                        print(f"  Gemini Processing: {gemma_output}, time: {gemma_time:.4f}s")
                    frame_entry["processing_type"] = "text_processing"
                elif classifier2_result == 'image':
                    frame_entry["processing_type"] = "image_processing"
                    # --- Image LLM integration ---
                    print("[INFO] Running Image LLM for image frame...")
                    if image_llm_backend == 'API':
                        img_llm_result = extract_image_context_gemini(frame)
                        print("[Gemini API Image LLM Result]:", img_llm_result)
                    else:
                        img_llm_result = extract_image_context_lmstudio(frame)
                        print("[Gemma LM Studio Image LLM Result]:", img_llm_result)
            elif is_text_unique:
                frame_entry["duplicate_status"] = "unique_text"
                frame_entry["processing_type"] = "text_processing"
                # Call Gemma only for unique_text frames
                gemma_start = time.perf_counter()
                gemma_output = text_llm_extractor.extract_context(ocr_text)
                gemma_time = time.perf_counter() - gemma_start
                frame_entry["gemma_context"] = gemma_output
                frame_entry["gemma_time_sec"] = gemma_time
                print(f"  Result: UNIQUE TEXT -> Text Processing (Gemma time: {gemma_time:.4f}s)")
            else:
                frame_entry["duplicate_status"] = "duplicate"
                frame_entry["processing_type"] = "discarded"
                frame_entry["notes"] = "Duplicate frame, discarded."
                print(f"  Result: DUPLICATE -> Discard")
                print(f"{'='*59}")
                frame_results.append(frame_entry)
                prev_frame = frame
                prev_ocr_text = ocr_text
                continue
            prev_ocr_text = ocr_text
        else:
            # Treat the first frame as distinct
            is_img_unique, is_text_unique, ocr_text, embedding_time, text_sim, cosine_sim = comparator.is_unique(frame, frame, None)
            frame_entry = {
                "frame_id": f"frame_{idx:05d}",
                "frame_timestamp": ts,
                "classification": label,
                "classification_probability": float(prob),
                "predict_time_sec": float(pred_time),
                "duplicate_status": "first_presentation",
                "embedding_time": float(embedding_time),
                "embedding_similarity": cosine_sim,
                "text_similarity": float(text_sim) if text_sim is not None else None,
                "ocr_text": ocr_text,
                "processing_type": None,
                "notes": "First frame processed as distinct."
            }
            # Save the frame as distinct
            save_frame_and_json(frame, frame_entry, output_dir)
            # Send to classifier2
            classifier2_start = time.perf_counter()
            classifier2_result, classifier2_prob, classifier2_time = classify_presentation_frame(frame)
            classifier2_elapsed = time.perf_counter() - classifier2_start
            frame_entry["classifier2_result"] = classifier2_result
            frame_entry["classifier2_confidence"] = classifier2_prob
            frame_entry["classifier2_time_sec"] = classifier2_time
            print(f"{'='*59}")
            print(f"Frame {idx} at {ts:.2f}s: {label}")
            print(f"  Classification: {label} | Model: {model_type} | Confidence: {prob:.2f} | Time: {pred_time:.4f}s")
            print(f"  Embedding Similarity: {cosine_sim:.5f}")
            print(f"  Embedding Time: {embedding_time:.4f}s")
            print(f"  Text Similarity: {text_sim:.4f}")
            print(f"  Text Processing Time: {embedding_time:.4f}s")
            print(f"  classifier2: {classifier2_result} | Confidence: {classifier2_prob:.3f} | Time: {classifier2_time:.3f}s (total: {classifier2_elapsed:.3f}s)")
            if classifier2_result == 'text':
                print(f"  Result: UNIQUE TEXT -> Text Processing")
                frame_entry["processing_type"] = "text_processing"
                # Add LLM context extraction for first presentation frame
                gemma_start = time.perf_counter()
                gemma_output = text_llm_extractor.extract_context(ocr_text)
                gemma_time = time.perf_counter() - gemma_start
                frame_entry["gemma_context"] = gemma_output
                frame_entry["gemma_time_sec"] = gemma_time
            elif classifier2_result == 'image':
                print(f"  Result: UNIQUE IMAGE -> Image Processing")
                frame_entry["processing_type"] = "image_processing"
                # --- Image LLM integration for first frame ---
                print("[INFO] Running Image LLM for image frame...")
                if image_llm_backend == 'API':
                    img_llm_result = extract_image_context_gemini(frame)
                    print("[Gemini API Image LLM Result]:", img_llm_result)
                else:
                    img_llm_result = extract_image_context_lmstudio(frame)
                    print("[Gemma LM Studio Image LLM Result]:", img_llm_result)
            print(f"{'='*59}")
            frame_results.append(frame_entry)
            prev_frame = frame
            prev_ocr_text = ocr_text
            continue
        prev_frame = frame
        frame_results.append(frame_entry)
        # Save every unique/distinct frame (image or text)
        if frame_entry["duplicate_status"] in ["unique_image", "unique_text", "first_presentation"]:
            save_frame_and_json(frame, frame_entry, output_dir)

    # Save results to JSON
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
            "text_similarity_threshold": comparator.text_threshold,
            "classifier_model": model_type,
            "text_processor_backend": text_llm_backend if text_llm_backend == 'Google Gemini API' else 'Gemma LM Studio'
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
    # Convert all numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif hasattr(obj, 'item') and callable(obj.item):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        else:
            return obj
    export = convert_types(export)
    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Frame-by-frame JSON results exported to: {json_path}")
    print(f"[INFO] All unique frames and JSON results saved to: {output_dir}")
    main_end = time.time()
    print(f"[INFO] Main pipeline completed in {main_end - main_start:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Pipeline for Frame-by-Frame Video Analysis")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, choices=['CNN', 'EFF'], required=True, help='Classifier model to use (CNN or EFF)')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second to extract (default 1.0)')
    parser.add_argument('--text_llm_backend', type=str, choices=['LMS', 'API'], default='LMS', help='Text LLM backend: LMS (Gemma) or API (Gemini)')
    parser.add_argument('--image_llm_backend', type=str, choices=['API', 'LMS'], default='API', help='Image LLM backend: API (Gemini) or LMS (Gemma LM Studio)')
    args = parser.parse_args()
    main(args.video, args.model, args.fps, args.text_llm_backend, args.image_llm_backend)