"""
Classifier 2: CLIP-based text/image slide classifier for presentation frames.

- Uses openai/clip-vit-large-patch14 via transformers
- Returns 'text' or 'image' for a given frame (numpy array, BGR)
- Designed for easy import and use in the main pipeline

Usage:
    from src.image_processing.classifier2_models.clip_classifier import classify_presentation_frame
    result, prob, elapsed = classify_presentation_frame(frame)
    # result: 'text' or 'image', prob: confidence, elapsed: seconds
"""
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time
import cv2

# Load model and processor once (global)
clip_model = None
clip_processor = None
clip_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_clip():
    global clip_model, clip_processor
    if clip_model is None or clip_processor is None:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(clip_device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",use_fast=True)
        print("Loading CLIP model")

# Prompts (robust, as discussed)
prompts = [
    # Text-only
    "a presentation slide that is mostly text, possibly with small images, logos, or colored backgrounds, but no significant visual content",
    "a slide where the main content is written words, even if there are small icons, decorative elements, or profile pictures of attendees",
    "a slide with paragraphs, bullet points, or lists, and any images present are not important for understanding the content",
    "a document-like slide filled with text, possibly with minor graphics, logos, or people profile images, but no meaningful diagrams or charts",
    "a text-based slide that may include attendee photos or company branding, but the focus is on textual information in the presentation area",
    "a slide that looks like an agenda, minutes, or text document, with no significant visual explanation or diagram",
    # Image/diagram-rich
    "a presentation slide where images, diagrams, charts, or visual elements are the main focus, even if there is supporting text",
    "a slide with large graphics, illustrations, mindmaps, or infographics, possibly with some text, but the images are essential for understanding",
    "a slide that explains concepts visually, using pictures, diagrams, flows, or visual metaphors, not just text",
    "a slide with screenshots, product images, or visual examples in the presentation area, not just decorative images",
    "a slide where the most important information is shown in a chart, graph, flow diagram, or figure, possibly with text annotations",
    "a slide with both text and images, but the images, diagrams, or visual content in the presentation area are more important than the text, including profile images if they are part of the main content"
]
text_indices = set(range(6))
image_indices = set(range(6, 12))

def classify_presentation_frame(frame: np.ndarray):
    """
    Classify a presentation frame as 'text' or 'image' using CLIP.
    Args:
        frame (np.ndarray): BGR image (OpenCV format)
    Returns:
        result (str): 'text' or 'image'
        prob (float): confidence (softmax probability)
        elapsed (float): time taken in seconds
    """
    _load_clip()
    start = time.perf_counter()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)
    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to(clip_device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image.softmax(dim=1)
        pred_idx = logits_per_image.argmax(dim=1).item()
        prob = logits_per_image.max().item()
        result = 'text' if pred_idx in text_indices else 'image'
    elapsed = time.perf_counter() - start
    return result, prob, elapsed

# For direct test
if __name__ == "__main__":
    import cv2
    import sys
    img_path = sys.argv[1]
    frame = cv2.imread(img_path)
    res, prob, t = classify_presentation_frame(frame)
    print(f"{img_path}: {res} (prob={prob:.3f}, time={t:.3f}s)")
