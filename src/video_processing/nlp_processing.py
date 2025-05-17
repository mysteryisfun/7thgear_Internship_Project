import spacy
from typing import List

class NLPProcessor:
    def __init__(self):
        """
        Initialize the NLPProcessor with spaCy's language model.
        """
        self.nlp = spacy.load("en_core_web_sm")

    def process_texts(self, texts: List[str]) -> List[str]:
        """
        Apply NLP techniques to a list of texts.

        Args:
            texts: List of original texts to process

        Returns:
            List of processed texts with NLP enhancements
        """
        processed_texts = []
        for text in texts:
            doc = self.nlp(text)

            # Example NLP techniques:
            # 1. Lemmatization
            lemmatized_text = " ".join([token.lemma_ for token in doc])

            # 2. Remove stop words
            filtered_text = " ".join([token.text for token in doc if not token.is_stop])

            # Combine or choose one of the techniques (here, lemmatized text is used)
            processed_texts.append(lemmatized_text)

        return processed_texts
