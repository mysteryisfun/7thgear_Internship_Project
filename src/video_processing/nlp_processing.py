import spacy
from typing import List
import wordninja
from autocorrect import Speller

class NLPProcessor:
    def __init__(self):
        """
        Initialize the NLPProcessor with spaCy's language model and other NLP tools.
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.spell = Speller()

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
            # Word Segmentation
            segmented_text = " ".join(wordninja.split(text))

            # Spell Correction
            corrected_text = " ".join([self.spell(word) for word in segmented_text.split()])

            # Sentence Boundary Detection
            doc = self.nlp(corrected_text)
            sentences = [sent.text for sent in doc.sents]
            sentence_restored_text = " ".join(sentences)

            # Named Entity Recognition (NER) and Case Normalization
            ner_corrected_text = []
            for token in self.nlp(sentence_restored_text):
                if token.ent_type_:
                    ner_corrected_text.append(token.text.title())
                else:
                    ner_corrected_text.append(token.text.capitalize() if token.i == 0 else token.text)

            final_text = " ".join(ner_corrected_text)

            # Remove Noise
            filtered_text = " ".join([word for word in final_text.split() if word.isalpha() or word in [ent.text for ent in doc.ents]])

            processed_texts.append(filtered_text)

        return processed_texts
