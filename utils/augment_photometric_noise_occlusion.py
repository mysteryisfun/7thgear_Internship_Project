"""
Image Augmentation Utilities for Classifier1 Dataset

Applies photometric, noise/blur, and occlusion/distortion augmentations using Albumentations.
Augmented images are saved in data/classifier1/augmented/<class_name>/

Usage (PowerShell, inside pygpu env):
    conda activate pygpu
    python utils/augment_photometric_noise_occlusion.py
"""

import os
import cv2
import albumentations as A
from tqdm import tqdm

# Augmentation pipeline
AUGMENT = A.Compose([
    # Photometric
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    # Noise & Blur
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
    # Occlusion & Distortion
    A.CoarseDropout(max_holes=4, max_height=32, max_width=32, min_holes=1, fill_value=0, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
    # Resize to EfficientNet-B0 input size
    A.Resize(224, 224)
])

INPUT_ROOT = os.path.join('data', 'classifier1')
OUTPUT_ROOT = os.path.join('data', 'classifier1', 'augmented')
CLASSES = ['people', 'presentation']
AUGS_PER_IMAGE = 5  # Number of augmentations per original image

os.makedirs(OUTPUT_ROOT, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

for cls in CLASSES:
    input_dir = os.path.join(INPUT_ROOT, cls)
    output_dir = os.path.join(OUTPUT_ROOT, cls)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in tqdm(images, desc=f'Augmenting {cls}'):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        for i in range(AUGS_PER_IMAGE):
            augmented = AUGMENT(image=img)['image']
            out_name = f"{os.path.splitext(img_name)[0]}_aug{i+1}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, augmented)
