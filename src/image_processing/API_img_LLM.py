"""
API_img_LLM.py
Structured image context extraction using Gemini API for the main pipeline.

- Uses GEMINI_API_KEY from environment variables (supports .env loading)
- Returns structured JSON with: topics, subtopics, entities, numerical_values, descriptive_explanation, tasks_identified, key_findings
- Designed for import into main pipeline
"""
import os
import requests
import base64
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import cv2
import numpy as np

load_dotenv()

# Ensure GEMINI_API_KEY is loaded from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file or environment variables.")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def encode_image_to_base64(image: "str|np.ndarray") -> str:
    """Encode image from file path or numpy array as base64 string."""
    if isinstance(image, str):
        with open(image, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    elif isinstance(image, np.ndarray):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode("utf-8")
    else:
        raise ValueError("Input must be a file path or numpy array.")


def build_gemini_image_prompt() -> str:
    """Prompt for structured image context extraction."""
    return (
        "Extract meaningful context out of this image frame from a meeting presentation without losing any information in format of {\"topics\": [],"
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
            "    \"Key findings\": [],"
            "  }"
            "}"
    )


def extract_image_context_gemini(
    image: "str|np.ndarray",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send image (file path or numpy array) to Gemini API and return structured context extraction.
    Args:
        image_path: Path to image file or numpy array
        api_key: Optionally override GEMINI_API_KEY
    Returns:
        Dict with keys: topics, subtopics, entities, numerical_values, descriptive_explanation, tasks_identified, key_findings
    Raises:
        RuntimeError on API or parsing error
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in environment or .env file.")
    img_b64 = encode_image_to_base64(image)
    prompt = build_gemini_image_prompt()
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}
            ]}
        ]
    }
    url = f"{GEMINI_API_URL}?key={key}"
    start=time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    end=time.time()
    print(f"[INFO] Gemini API request took {end - start:.2f} seconds")
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error: {resp.status_code} {resp.text}")
    try:
        candidates = resp.json()["candidates"]
        text = candidates[0]["content"]["parts"][0]["text"]
        # Remove Markdown code block if present
        if text.strip().startswith("```json"):
            text = text.strip()
            text = text[text.find("\n")+1:]  # Remove the first line (```json)
            if text.endswith("```"):
                text = text[:text.rfind("```")].strip()
        # Extract embedded JSON if present
        embedded_json_start = text.find("{")
        embedded_json_end = text.rfind("}") + 1
        if embedded_json_start != -1 and embedded_json_end != -1:
            text = text[embedded_json_start:embedded_json_end]
        # Try to parse JSON from response
        import json
        try:
            result = json.loads(text)
            return result
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Gemini response: {e}\nRaw: {resp.text}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while parsing Gemini response: {e}\nRaw: {resp.text}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse Gemini response: {e}\nRaw: {resp.text}")


# Example usage (for testing only)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python API_img_LLM.py <image_path>")
        exit(1)
    image_path = sys.argv[1]
    result = extract_image_context_gemini(image_path)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
