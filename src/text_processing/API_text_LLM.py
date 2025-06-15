import requests
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Ensure GEMINI_API_KEY is loaded from .env
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file or environment variables.")

class GeminiAPIContextExtractor:
    """
    Extracts meaningful context from OCR text using the Gemini API (flash-2.0 model).
    """
    def __init__(self, api_key: str = API_KEY, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        self.system_prompt = (
            "Extract meaningful context out of this OCR text extracted from a meeting presentation in format of {\"topics\": [],"
            "  \"subtopics\": [],"
            "  \"entities\": {"
            "    \"persons\": [],"
            "    \"organizations\": [],"
            "    \"events\": [],"
            "    \"dates\": []"
            "  },"
            "  \"numerical_values\": [],"
            "    \"descriptive explanation\": \"\""
            "    \"tasks identified\": [],"
            "  }"
            "}"
        )

    def extract_context(self, ocr_text: str, temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Sends OCR text to the Gemini API and returns the structured context.
        Handles code block formatting and extracts JSON from the response.
        Tries to robustly parse even if the JSON is malformed or incomplete.
        """
        import re
        import json
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {"role": "user", "parts": [
                    {"text": f"{self.system_prompt}\n{ocr_text}"}
                ]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return {"error": str(e), "raw_response": None}

        # Try to extract JSON from the content
        json_str = None
        content = re.sub(r'```(json)?', '', content).strip()
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end+1]
        else:
            json_str = content
        try:
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if close_braces < open_braces:
                json_str += '}' * (open_braces - close_braces)
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            if close_brackets < open_brackets:
                json_str += ']' * (open_brackets - close_brackets)
            parsed = json.loads(json_str)
            return parsed
        except Exception as e:
            return {"error": f"JSON parsing error: {e}", "raw_response": json_str}
