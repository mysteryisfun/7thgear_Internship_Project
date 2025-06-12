"""
Test script for DINOv2-based frame deduplication using HuggingFace Transformers.
Extracts frames from a test video, computes DINOv2 embeddings, compares with cosine similarity,
saves distinct frames, and prints timing for each embedding extraction.

Requirements:
- torch
- transformers
- opencv-python
- numpy
- PIL

Run in Conda 'pygpu' environment.

Usage (PowerShell):
    conda activate pygpu
    python tests/test_dinov2_deduplication.py
"""
import sys, os
print(f"[DIAG] sys.executable: {sys.executable}")
print(f"[DIAG] PATH: {os.environ.get('PATH', 'Not Set')}")
print(f"[DIAG] CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '')}")
print(f"[DIAG] CUDA_PATH: {os.environ.get('CUDA_PATH', '')}")
import os
import cv2
import numpy as np
import time
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

def extract_frames(video_path, fps=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def save_all_frames(frames, output_dir):
    """
    Save all extracted frames to a specified directory for reference or debugging.
    Args:
        frames: List of (idx, frame) tuples.
        output_dir: Directory to save all frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in frames:
        out_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(out_path, frame)

def main():
    """
    Extract frames from video, save all frames, then deduplicate using DINOv2 embeddings.
    """
    video_path = os.path.join("data", "test_files", "test-1.mp4")
    all_frames_dir = os.path.join("output", "frames")
    output_dir = os.path.join("output", "dinov2_dedup_test")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Loading HuggingFace DINOv2 model and processor (PyTorch)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    print("[INFO] DINOv2 model loaded.")
    frames = extract_frames(video_path, fps=1.0)
    # Save all frames first
    print(f"[INFO] Saving all extracted frames to {all_frames_dir} ...")
    save_all_frames(frames, all_frames_dir)
    print(f"[INFO] All frames saved.")
    embedding_cache = []  # Store all unique embeddings
    last_distinct_embedding = None
    last_distinct_idx = None
    threshold_scene_change = 0.94  # Cosine similarity threshold for scene change (compare to last distinct)
    threshold_duplicate = 0.99     # Cosine similarity threshold for duplicate (compare to all stored)
    distinct_count = 0
    for idx, frame in frames:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        start = time.perf_counter()
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # DINOv2: use last_hidden_state mean as embedding
            embedding = outputs.last_hidden_state.mean(dim=1)
        end = time.perf_counter()
        embedding_np = embedding.cpu().numpy().flatten()
        is_scene_change = False
        is_distinct = False
        max_sim_last = None
        max_sim_cache = None
        # 1. Compare to last distinct embedding to detect scene change
        if last_distinct_embedding is None:
            is_scene_change = True
            max_sim_last = None
        else:
            sim_last = cosine_similarity(last_distinct_embedding, embedding_np)
            max_sim_last = sim_last
            if sim_last < threshold_scene_change:
                is_scene_change = True
        # 2. If scene changed, check if embedding is already present in cache (duplicate)
        if is_scene_change:
            found_duplicate = False
            for cached_emb in embedding_cache:
                sim = cosine_similarity(cached_emb, embedding_np)
                if max_sim_cache is None or sim > max_sim_cache:
                    max_sim_cache = sim
                if sim > threshold_duplicate:
                    found_duplicate = True
                    break
            if not found_duplicate:
                # New distinct frame
                out_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
                cv2.imwrite(out_path, frame)
                embedding_cache.append(embedding_np)
                last_distinct_embedding = embedding_np
                last_distinct_idx = idx
                distinct_count += 1
                print(f"[INFO] Frame {idx:05d}: Scene change detected, new embedding added.")
                is_distinct = True
            else:
                print(f"[INFO] Frame {idx:05d}: Scene change detected, but embedding already present (duplicate).")
        else:
            print(f"[INFO] Frame {idx:05d}: No scene change (similar to last distinct frame).")
        print(f"Frame {idx:05d}: DINOv2 time: {end-start:.4f}s | Sim to last distinct: {max_sim_last if max_sim_last is not None else 'N/A'} | Max sim in cache: {max_sim_cache if max_sim_cache is not None else 'N/A'} | Distinct: {is_distinct}")
    print(f"[RESULT] Total distinct frames saved: {distinct_count}")

if __name__ == "__main__":
    main()
