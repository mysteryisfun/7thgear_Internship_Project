"""
LM Studio Image Context Extraction

This script provides a function to send an image to the LM Studio API for processing using the google/gemma-3-4b model.

- Encodes the image in base64 format.
- Sends the image to the LM Studio API.
- Returns structured JSON with keys: topics, subtopics, entities, numerical_values, descriptive_explanation, tasks_identified, key_findings.

"""

import os
import base64
import requests
import cv2
import numpy as np
from typing import Dict, Any

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

def extract_image_context_lmstudio(image: "str|np.ndarray") -> Dict[str, Any]:
    """
    Send image (file path or numpy array) to the LM Studio API for context extraction using the google/gemma-3-4b model.

    Args:
        image: Image file path or numpy array.

    Returns:
        A dictionary with keys: topics, subtopics, entities, numerical_values, descriptive_explanation, tasks_identified, key_findings.

    Raises:
        RuntimeError: If the API request fails or the response cannot be parsed.
    """
    img_b64 = encode_image_to_base64(image)
    payload = {
        "model": "google/gemma-3-4b",
        "messages": [
            {"role": "system", "content": build_gemini_image_prompt()},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                }
            ]}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }

    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    import time
    start = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    end = time.time()
    print(f"LM Studio API request took {end - start:.2f} seconds")

    if response.status_code != 200:
        raise RuntimeError(f"LM Studio API error: {response.status_code} {response.text}")

    try:
        result = response.json()["choices"][0]["message"]["content"]
        # Extract JSON block from the content
        if "```json" in result:
            start_index = result.find("```json") + len("```json")
            end_index = result.rfind("```")
            result = result[start_index:end_index].strip()
        else:
            raise RuntimeError("No JSON block found in the response content.")
        # Parse the JSON response
        import json
        parsed_result = json.loads(result)
        # Ensure all required fields are present
        default_fields = {
            "topics": [],
            "subtopics": [],
            "entities": {"persons": [], "organizations": [], "events": [], "dates": []},
            "numerical_values": [],
            "descriptive_explanation": "",
            "tasks_identified": [],
            "key_findings": []
        }
        for field, default_value in default_fields.items():
            if field not in parsed_result:
                parsed_result[field] = default_value
        return parsed_result
    except Exception as e:
        raise RuntimeError(f"Failed to parse LM Studio response: {e}\nRaw: {response.text}")

# Example usage (for testing only)
if __name__ == "__main__":
    test_image_path = "data/test_images/frame_00014.jpg"  # Update path as needed
    try:
        result = extract_image_context_lmstudio(test_image_path)
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")
