import re
from typing import List
import wordninja
from autocorrect import Speller

class NLPProcessor:
    def __init__(self):
        """
        Initialize the NLPProcessor with lightweight NLP tools.
        """
        self.spell = Speller()

    def process_texts(self, texts: List[str]) -> List[str]:
        """
        Apply lightweight NLP techniques to a list of texts.

        Args:
            texts: List of original texts to process

        Returns:
            List of processed texts with NLP enhancements
        """
        processed_texts = []
        for text in texts:
            # Word Segmentation
            segmented_text = " ".join(wordninja.split(text))

            # Spell Correction
            corrected_text = " ".join([self.spell(word) for word in segmented_text.split()])

            # Basic Sentence Normalization
            normalized_text = re.sub(r'[^a-zA-Z0-9\s]', '', corrected_text).strip()

            # Capitalize first letter of each sentence
            final_text = '. '.join([sentence.capitalize() for sentence in normalized_text.split('. ')]).strip()

            processed_texts.append(final_text)

        return processed_texts
