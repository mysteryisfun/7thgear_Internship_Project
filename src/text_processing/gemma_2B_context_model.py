import requests
import json
from typing import Dict, Any

class GemmaContextExtractor:
    """
    Extracts meaningful context from OCR text using the Gemma model via LM Studio API.
    """
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions", model: str = "gemma-2-2b-it"):
        self.api_url = api_url
        self.model = model
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
        Sends OCR text to the Gemma model and returns the structured context.
        Handles code block formatting and extracts JSON from the response.
        Tries to robustly parse even if the JSON is malformed or incomplete.
        """
        import re
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": ocr_text}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
        except Exception as e:
            return {"error": str(e), "raw_response": None}

        # Try to extract JSON from the content
        json_str = None
        # Remove code block markers if present
        content = re.sub(r'```(json)?', '', content).strip()
        # Find the first and last curly braces
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end+1]
        else:
            json_str = content
        # Try to fix common JSON issues
        try:
            # Remove trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            # Add missing closing brackets if possible
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
            # Return raw string for debugging if parsing fails
            return {"error": f"JSON parsing error: {e}", "raw_response": json_str}
