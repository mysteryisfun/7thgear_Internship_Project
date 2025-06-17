from paddleocr import PaddleOCR
from typing import List
import logging
import os
import cv2
import numpy as np

class PaddleOCRTextExtractor:
    def __init__(self):
        import paddleocr
        import paddle
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en',det_db_box_thresh=0.3)

    def extract_text_from_frames(self, frames: List) -> List[str]:
        """
        Extract text from a list of frame image paths or numpy arrays using PaddleOCR.

        Args:
            frames: List of paths to frame images or numpy arrays

        Returns:
            List of extracted text for each frame
        """
        extracted_texts = []
        for frame in frames:
            try:
                img = None
                # If input is a numpy array, use it directly
                if isinstance(frame, str):
                    if not os.path.exists(frame):
                        extracted_texts.append("")
                        continue
                    img = cv2.imread(frame)
                    if img is None:
                        extracted_texts.append("")
                        continue
                    ocr_result = self.ocr.ocr(frame, det=True, rec=True)
                elif isinstance(frame, (np.ndarray,)):
                    img = frame
                    ocr_result = self.ocr.ocr(img, det=True, rec=True)
                else:
                    extracted_texts.append("")
                    continue
                # Robustly extract all recognized text lines from the OCR result
                text_lines = []
                if isinstance(ocr_result, list) and len(ocr_result) > 0:
                    for block in ocr_result:
                        if isinstance(block, list):
                            for line in block:
                                # Skip box-only lines (list of coordinates)
                                if isinstance(line, list) and all(isinstance(x, (int, float)) or (isinstance(x, list) and all(isinstance(y, (int, float)) for y in x)) for x in line):
                                    continue
                                # Handle tuple (text, confidence)
                                if isinstance(line, tuple) and len(line) > 0 and isinstance(line[0], str):
                                    text_lines.append(line[0])
                                # Handle [box, (text, confidence)]
                                elif isinstance(line, list) and len(line) > 1 and isinstance(line[1], tuple) and len(line[1]) > 0 and isinstance(line[1][0], str):
                                    text_lines.append(line[1][0])
                text = '\n'.join(text_lines)
                extracted_texts.append(text)
            except Exception:
                extracted_texts.append("")
        return extracted_texts
