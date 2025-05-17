import pytesseract
from PIL import Image
from typing import List

class TesseractTextExtractor:
    def __init__(self, tesseract_path: str = None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def extract_text_from_frames(self, frame_paths: List[str]) -> List[str]:
        """
        Extract text from a list of frame image paths using Tesseract OCR.

        Args:
            frame_paths: List of paths to frame images

        Returns:
            List of extracted text for each frame
        """
        extracted_texts = []
        for frame_path in frame_paths:
            try:
                image = Image.open(frame_path)
                text = pytesseract.image_to_string(image)
                extracted_texts.append(text)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                extracted_texts.append("")
        return extracted_texts
