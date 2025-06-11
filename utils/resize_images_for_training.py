"""
Resize Images for Model Training

This script resizes all images in data/classifier1/people and data/classifier1/presentation to 224x224 (EfficientNet-B0/CNN standard).
Resized images are saved in data/classifier1/resized/<class_name>/

Usage (PowerShell, inside pygpu env):
    conda activate pygpu
    python utils/resize_images_for_training.py
"""

import os
import cv2
from tqdm import tqdm

INPUT_ROOT = os.path.join('data', 'classifier2','augmented')
OUTPUT_ROOT = os.path.join('data', 'classifier2', 'resized')
CLASSES = ['text', 'image']
"""INPUT_ROOT = os.path.join('data', 'classifier1','augmented')
OUTPUT_ROOT = os.path.join('data', 'classifier1', 'resized')
CLASSES = ['people', 'presentation']"""
TARGET_SIZE = (224, 224)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

for cls in CLASSES:
    input_dir = os.path.join(INPUT_ROOT, cls)
    output_dir = os.path.join(OUTPUT_ROOT, cls)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in tqdm(images, desc=f'Resizing {cls}'):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, resized)
