from paddleocr import PaddleOCR
from typing import List

class PaddleOCRTextExtractor:
    def __init__(self):
        self.ocr = PaddleOCR()

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
                text = '\n'.join([line[1][0] for line in result[0]])
                extracted_texts.append(text)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                extracted_texts.append("")
        return extracted_texts
