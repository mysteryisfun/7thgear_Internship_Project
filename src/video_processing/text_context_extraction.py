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
        """
        import re
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": ocr_text}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        # Remove code block markers if present
        if content.strip().startswith('```'):
            content = content.strip().lstrip('`').split('\n', 1)[-1]
            if content.strip().startswith('json'):
                content = content.strip()[4:]
            content = content.strip('`').strip()
        # Extract JSON block from the response
        json_start = content.find('{')
        json_end = content.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = content[json_start:json_end+1]
            # Remove trailing commas before closing braces/brackets
            json_str = re.sub(r',([\s\n]*[}\]])', r'\1', json_str)
            try:
                return json.loads(json_str)
            except Exception as e:
                print(f"[GemmaContextExtractor] JSON parsing error: {e}\nRaw JSON string: {json_str}")
        # Fallback: return raw response
        return {"raw_response": content}
