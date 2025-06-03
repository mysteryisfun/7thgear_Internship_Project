import re
from typing import List, Dict, Any
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from src.text_processing.bert_processor import BERTProcessor

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class EnhancedTextProcessor:
    """
    Enhanced text processor for extracting meaningful context from OCR text.
    """
    def __init__(self):
        self.bert_processor = BERTProcessor()
        self.stop_words = set(stopwords.words('english'))

    def process_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a list of texts and extract meaningful context.
        Returns a list of dicts with categorized and reconstructed content.
        """
        results = []
        for text in texts:
            context = self.extract_meaningful_context(text)
            results.append(context)
        return results

    def extract_meaningful_context(self, text: str) -> Dict[str, Any]:
        """
        Extracts categorized and contextually reconstructed content from text.
        """
        # Tokenize sentences and words
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        # Categorize content
        headers = self.extract_headers(sentences)
        metadata = self.extract_metadata(sentences)
        action_items = self.extract_action_items(sentences)
        technical_terms = self.extract_technical_terms(words)
        body_content = self.extract_body_content(sentences, headers, metadata, action_items)
        # Extract topics and key points
        topics = self.extract_topics(sentences)
        key_points = self.extract_key_points(sentences)
        # Extract named entities (simple regex for demo)
        people_orgs = self.extract_named_entities(text)
        numbers = self.extract_numbers(text)
        return {
            "headers": headers,
            "metadata": metadata,
            "body_content": body_content,
            "action_items": action_items,
            "technical_terms": technical_terms,
            "topics": topics,
            "key_points": key_points,
            "numbers": numbers,
            "people_organizations": people_orgs
        }

    def extract_headers(self, sentences: List[str]) -> List[str]:
        # Heuristic: lines in all caps or short lines at the top
        return [s for s in sentences[:3] if s.isupper() or (len(s.split()) <= 6 and s == s.title())]

    def extract_metadata(self, sentences: List[str]) -> List[str]:
        # Look for dates, authors, page numbers
        meta = []
        for s in sentences:
            if re.search(r'\b\d{4}\b', s) or re.search(r'page \d+', s, re.I) or re.search(r'author', s, re.I):
                meta.append(s)
        return meta

    def extract_action_items(self, sentences: List[str]) -> List[str]:
        # Bullet points, numbered lists
        return [s for s in sentences if re.match(r'^(\d+\.|\-|\*)\s', s.strip())]

    def extract_technical_terms(self, words: List[str]) -> List[str]:
        # Heuristic: long words, camel case, or all caps
        return [w for w in words if (len(w) > 8 or w.isupper() or re.match(r'[A-Z][a-z]+[A-Z][a-z]+', w)) and w.lower() not in self.stop_words]

    def extract_body_content(self, sentences: List[str], headers, metadata, action_items) -> List[str]:
        # Exclude headers, metadata, and action items
        exclude = set(headers + metadata + action_items)
        return [s for s in sentences if s not in exclude]

    def extract_topics(self, sentences: List[str]) -> List[str]:
        # Use most common nouns as topics (simple heuristic)
        words = [w for s in sentences for w in word_tokenize(s) if w.isalpha() and w.lower() not in self.stop_words]
        freq = Counter(words)
        return [w for w, _ in freq.most_common(5)]

    def extract_key_points(self, sentences: List[str]) -> List[str]:
        # Sentences with numbers or strong verbs
        return [s for s in sentences if re.search(r'\d', s) or re.search(r'\b(should|must|will|require|increase|decrease|improve|reduce)\b', s, re.I)]

    def extract_named_entities(self, text: str) -> List[str]:
        # Simple regex for capitalized words (names/orgs)
        return re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)

    def extract_numbers(self, text: str) -> List[str]:
        return re.findall(r'\b\d+(?:\.\d+)?\b', text)

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using BERT embeddings.
        """
        return self.bert_processor.compute_similarity(text1, text2)
