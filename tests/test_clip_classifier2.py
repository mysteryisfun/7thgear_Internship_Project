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
        #img_dir = os.path.join('data', 'classifier2', 'test_CLIP', 'text')
        img_dir = os.path.join('output', 'frames')
        expected_label = "a slide with mostly text ignoring small insignificant images and people"
    else:
        img_dir = os.path.join('data', 'classifier2', 'test_CLIP', 'images')
        expected_label = "a slide with diagrams or images"

    assert os.path.isdir(img_dir), f"Image directory not found: {img_dir}"
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompts = [
        # Text-only (robust, allows small images, logos, colors, and people profiles)
        "a presentation slide that is mostly text, possibly with small images, logos, or colored backgrounds, but no significant visual content",
        "a slide where the main content is written words, even if there are small icons, decorative elements, or profile pictures of attendees",
        "a slide with paragraphs, bullet points, or lists, and any images present are not important for understanding the content",
        "a document-like slide filled with text, possibly with minor graphics, logos, or people profile images, but no meaningful diagrams or charts",
        "a text-based slide that may include attendee photos or company branding, but the focus is on textual information in the presentation area",
        "a slide that looks like an agenda, minutes, or text document, with no significant visual explanation or diagram may contain tables with only text",
        # Image/diagram-rich (robust, allows mixed content, but images/diagrams are key)
        "a presentation slide where images, diagrams, charts, or visual elements are the main focus, even if there is supporting text",
        "a slide with large graphics, illustrations, mindmaps, or infographics, possibly with some text, but the images are essential for understanding",
        "a slide that explains concepts visually, using pictures, diagrams, flows, or visual metaphors, not just text",
        "a slide with screenshots, product images, or visual examples in the presentation area, not just decorative images",
        "a slide where the most important information is shown in a chart, graph, flow diagram, or figure, possibly with text annotations",
        "a slide with both text and images, but the images, diagrams, or visual content in the presentation area are more important than the text, including profile images if they are part of the main content"
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
