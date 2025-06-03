import unittest
from src.video_processing.text_context_extraction import GemmaContextExtractor
import os

class TestGemmaContextExtractor(unittest.TestCase):
    def setUp(self):
        # Use default localhost:1234 and gemma-2-2b-it
        self.extractor = GemmaContextExtractor()

    def test_extract_context_basic(self):
        ocr_text = "Quarterly Results: Revenue increased by 15%. Presenters: Alice Smith, Bob Lee. Event: Q1 2025 Review. Date: March 10, 2025. Company: Acme Corp."
        result = self.extractor.extract_context(ocr_text)
        print("\nGemma Model Output:")
        print(result)
        # Check that the result is a dict and has at least the top-level keys
        self.assertIsInstance(result, dict)
        for key in ["topics", "subtopics", "entities", "numerical_values", "descriptive_context"]:
            self.assertIn(key, result)

    def test_extract_context_empty(self):
        ocr_text = ""
        result = self.extractor.extract_context(ocr_text)
        print("\nGemma Model Output (Empty):")
        print(result)
        self.assertIsInstance(result, dict)

if __name__ == "__main__":
    unittest.main()

