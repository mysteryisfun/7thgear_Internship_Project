"""
Frame Comparator Module

Compares frames for visual (DINOv2 embedding) and textual (OCR cosine similarity) duplication.
If embedding is different (below threshold), frame is unique for image processing.
If embedding is similar but text similarity is below threshold, frame is unique for text processing.

Usage:
    from src.main_workflow.frame_comparator import FrameComparator
    comparator = FrameComparator(dino_similarity_threshold=0.98, text_threshold=0.85)
    is_img_unique, is_text_unique = comparator.is_unique(frame, prev_frame, ocr_text, prev_ocr_text)
"""
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from src.text_processing.paddleocr_text_extractor import PaddleOCRTextExtractor
from src.text_processing.enhanced_text_processor import EnhancedTextProcessor
import cv2
import time
import collections
import imagehash
#phash threshold is used but the original comparator is Dino embedding with cosine similarity
class FrameComparator:
    def __init__(self, dino_similarity_threshold=0.94, text_threshold=0.85, cache_size=100):
        """
        Compares frames for visual (DINOv2 embedding) and textual (OCR cosine similarity) duplication.
        dino_similarity_threshold: Cosine similarity threshold for DINOv2 embeddings (0.98 = near-duplicate)
        text_threshold: Cosine similarity threshold for BERT text similarity
        cache_size: Embedding cache size
        """
        self.dino_similarity_threshold = dino_similarity_threshold
        self.text_threshold = text_threshold
        self.ocr = PaddleOCRTextExtractor()
        self.text_processor = EnhancedTextProcessor()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        self.model.eval()
        self.embedding_cache = collections.OrderedDict()
        self.cache_size = cache_size

    def compute_embedding(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.cpu().numpy().flatten()

    def compute_text(self, frame: np.ndarray) -> str:
        # Use PaddleOCRTextExtractor's array support to avoid temp file I/O
        try:
            result = self.ocr.extract_text_from_frames([frame])
            if isinstance(result, list) and len(result) > 0:
                text = result[0] if isinstance(result[0], str) else ''
            else:
                text = ''
        except Exception as e:
            text = ''
        return text

    def get_cached_embedding(self, frame: np.ndarray) -> np.ndarray:
        frame_hash = imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if frame_hash in self.embedding_cache:
            return self.embedding_cache[frame_hash]
        embedding = self.compute_embedding(frame)
        if len(self.embedding_cache) >= self.cache_size:
            self.embedding_cache.popitem(last=False)  # Remove the oldest item
        self.embedding_cache[frame_hash] = embedding
        return embedding

    def is_unique(self, frame, prev_frame, prev_ocr_text=None):
        """
        Returns (is_img_unique, is_text_unique, ocr_text, embedding_time, text_sim, dino_cosine_sim)
        """
        start_embed = time.perf_counter()
        embedding1 = self.get_cached_embedding(prev_frame)
        embedding2 = self.get_cached_embedding(frame)
        embedding_time = time.perf_counter() - start_embed

        cosine_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        ocr_text = self.compute_text(frame)
        text_sim = 0.0
        if prev_ocr_text is not None:
            text_sim = self.text_processor.compute_text_similarity(prev_ocr_text, ocr_text)
        is_img_unique = cosine_sim < self.dino_similarity_threshold
        is_text_unique = (cosine_sim >= self.dino_similarity_threshold) and (text_sim < self.text_threshold)
        return is_img_unique, is_text_unique, ocr_text, embedding_time, text_sim, cosine_sim
