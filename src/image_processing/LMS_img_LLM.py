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
def encode_image_to_base64(image_path: str) -> str:
    """Read image and encode as base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def extract_image_context_lmstudio(image_path: str) -> Dict[str, Any]:
    """
    Send an image to the LM Studio API for context extraction using the google/gemma-3-4b model.

    Args:
        image_path: Path to the image file.

    Returns:
        A dictionary with keys: topics, subtopics, entities, numerical_values, descriptive_explanation, tasks_identified, key_findings.

    Raises:
        RuntimeError: If the API request fails or the response cannot be parsed.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")


    img_b64 = encode_image_to_base64(image_path)
    payload = {
        "model": "google/gemma-3-4b",
        "messages": [
            {"role": "system", "content": build_gemini_image_prompt()},
            {"role": "user", "content": [
                {
                    "type":"image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                }
            ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }

    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    import time
    start= time.time()
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
