import re
from typing import List
from difflib import SequenceMatcher
import cv2
import numpy as np

class TextProcessor:
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize the TextProcessor with a similarity threshold.

        Args:
            similarity_threshold: Threshold for text similarity (0.0 to 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.cache_text = ""

    def process_text(self, texts: List[str]) -> List[str]:
        """
        Process a list of extracted texts to remove duplicates based on similarity.

        Args:
            texts: List of extracted texts

        Returns:
            List of processed and unique texts
        """
        processed_texts = []

        for text in texts:
            # Step 1: Remove special characters
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

            # Step 2: Convert distorted words to correct (placeholder for actual implementation)
            # For now, assume text is already correct

            # Step 3: Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()

            # Step 4: Join together to a single string (already done in previous steps)

            # Step 5: Convert to lowercase
            text = text.lower()

            # Check similarity with the cached text
            if self.cache_text:
                similarity = SequenceMatcher(None, self.cache_text, text).ratio()
                if similarity >= self.similarity_threshold:
                    continue  # Discard similar text

            # Add unique text to the list and update the cache
            processed_texts.append(text)
            self.cache_text = text

        return processed_texts

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess an image to enhance text clarity.

        Args:
            image_path: Path to the input image

        Returns:
            Preprocessed image as a numpy array
        """
        # Read the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Enhance contrast
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

        # Reduce noise
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Binarize the image
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return image
