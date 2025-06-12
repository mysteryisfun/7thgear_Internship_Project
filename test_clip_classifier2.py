"""
Test script for classifying presentation frames as 'text-only' or 'visual/diagram' using CLIP with text prompts.

Usage (in pygpu conda env):
    conda activate pygpu
    python test_clip_classifier2.py --mode text
    python test_clip_classifier2.py --mode images

--mode text   : Classifies images in data/classifier2/test_CLIP/text
--mode images : Classifies images in data/classifier2/test_CLIP/images

Prints only the image name and predicted label, then prints summary accuracy for the expected class.
"""
import os
import argparse
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['text', 'images'], required=True, help='Which set to test: text or images')
    args = parser.parse_args()

    if args.mode == 'text':
        img_dir = os.path.join('data', 'classifier2', 'test_CLIP', 'text')
        expected_label = "a slide with mostly text ignoring small insignificant images and people"
    else:
        img_dir = os.path.join('data', 'classifier2', 'test_CLIP', 'images')
        expected_label = "a slide with diagrams or images"

    assert os.path.isdir(img_dir), f"Image directory not found: {img_dir}"
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompts = [
        # Text-only prompts
        "a presentation slide with mostly text and no meaningful images",
        "a slide that contains only written information without diagrams or pictures",
        "a slide with full text and no visual content of importance",
        "a document-like slide filled with text, ignoring logos and UI elements",
        "a text-based slide without illustrations, charts, or diagrams",
        "a text-based slide with people profile who are attendees",
        # Image/diagram-rich prompts
        "a slide that contains charts, diagrams, or visual illustrations",
        "a presentation slide with graphical elements, images, or infographics",
        "a visual slide with diagrams or picture-based content",
        "a slide with illustrations explaining concepts visually may include text but image holds significance",
        "a content slide that includes meaningful images or drawings",
        "a content slide that includes meaningful images and also text but significantly image"
    ]

    text_indices = set(range(6))
    image_indices = set(range(6, 12))

    total = 0
    correct = 0
    for fname in image_files:
        img_path = os.path.join(img_dir, fname)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image.softmax(dim=1)
            pred_idx = logits_per_image.argmax(dim=1).item()
            # For output, print only 'text' or 'images'
            if pred_idx in text_indices:
                print(f"{fname} text")
            else:
                print(f"{fname} images")
            total += 1
            if args.mode == 'text' and pred_idx in text_indices:
                correct += 1
            elif args.mode == 'images' and pred_idx in image_indices:
                correct += 1
    print(f"{correct}/{total}")

if __name__ == "__main__":
    main()
