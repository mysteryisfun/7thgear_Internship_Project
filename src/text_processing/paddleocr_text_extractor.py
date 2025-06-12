from paddleocr import PaddleOCR
from typing import List
import logging

class PaddleOCRTextExtractor:
    def __init__(self):
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
                result = self.ocr.ocr(frame_path, det=True, rec=True)
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                    text = '\n'.join([line[1][0] for line in result[0] if isinstance(line, list) and len(line) > 1 and isinstance(line[1], tuple) and len(line[1]) > 0 and isinstance(line[1][0], str)])
                else:
                    raise ValueError("Unexpected OCR result format")
                extracted_texts.append(text)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                extracted_texts.append("")
        return extracted_texts
