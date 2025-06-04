"""
Frame Comparator Module

Compares frames for visual (phash) and textual (OCR cosine similarity) duplication.
If phash is different (above threshold), frame is unique for image processing.
If phash is similar but text similarity is below threshold, frame is unique for text processing.

Usage:
    from src.main_workflow.frame_comparator import FrameComparator
    comparator = FrameComparator(phash_threshold=5, text_threshold=0.85)
    is_img_unique, is_text_unique = comparator.is_unique(frame, prev_frame, ocr_text, prev_ocr_text)
"""
import imagehash
from PIL import Image
import numpy as np
from src.text_processing.paddleocr_text_extractor import PaddleOCRTextExtractor
from src.text_processing.enhanced_text_processor import EnhancedTextProcessor
import cv2

class FrameComparator:
    def __init__(self, phash_threshold=5, text_threshold=0.85):
        self.phash_threshold = phash_threshold
        self.text_threshold = text_threshold
        self.ocr = PaddleOCRTextExtractor()
        self.text_processor = EnhancedTextProcessor()

    def compute_phash(self, frame: np.ndarray) -> imagehash.ImageHash:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return imagehash.phash(pil_img)

    def compute_text(self, frame: np.ndarray) -> str:
        # Save frame to temp file for OCR
        import tempfile
        import os
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_file.name, frame)
        temp_file.close()  # Ensure file is closed before OCR
        try:
            result = self.ocr.extract_text_from_frames([temp_file.name])
            if isinstance(result, list) and len(result) > 0:
                text = result[0] if isinstance(result[0], str) else ''
            else:
                text = ''
        except Exception as e:
            print(f"Error processing frame {temp_file.name}: {e}")
            text = ''
        finally:
            try:
                os.remove(temp_file.name)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_file.name}: {e}")
        return text

    def is_unique(self, frame, prev_frame, prev_ocr_text=None):
        """
        Returns (is_img_unique, is_text_unique, ocr_text, phash_diff, text_sim)
        """
        phash1 = self.compute_phash(prev_frame)
        phash2 = self.compute_phash(frame)
        phash_diff = phash1 - phash2
        ocr_text = self.compute_text(frame)
        text_sim = 0.0
        if prev_ocr_text is not None:
            text_sim = self.text_processor.compute_text_similarity(prev_ocr_text, ocr_text)
        is_img_unique = phash_diff > self.phash_threshold
        is_text_unique = (phash_diff <= self.phash_threshold) and (text_sim < self.text_threshold)
        return is_img_unique, is_text_unique, ocr_text, phash_diff, text_sim
