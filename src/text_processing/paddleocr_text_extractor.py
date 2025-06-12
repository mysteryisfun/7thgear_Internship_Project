from paddleocr import PaddleOCR
from typing import List
import logging
import os
import cv2

class PaddleOCRTextExtractor:
    def __init__(self):
        import paddleocr
        import paddle
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en',det_db_box_thresh=0.3)

    def extract_text_from_frames(self, frame_paths: List[str]) -> List[str]:
        """
        Extract text from a list of frame image paths using PaddleOCR.

        Args:
            frame_paths: List of paths to frame images

        Returns:
            List of extracted text for each frame
        """
        extracted_texts = []
        for frame_path in frame_paths:
            try:
                if not os.path.exists(frame_path):
                    extracted_texts.append("")
                    continue
                img = cv2.imread(frame_path)
                if img is None:
                    extracted_texts.append("")
                    continue
                try:
                    result = self.ocr.ocr(frame_path, det=True, rec=True)
                except Exception:
                    extracted_texts.append("")
                    continue
                # Robustly extract all recognized text lines from the OCR result
                text_lines = []
                if isinstance(result, list) and len(result) > 0:
                    for block in result:
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
