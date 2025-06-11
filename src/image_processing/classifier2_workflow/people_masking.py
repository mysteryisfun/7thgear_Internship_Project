"""
People Masking Module using Haar Cascade

This module provides a function to mask faces/people in an image using OpenCV's Haar Cascade.
Designed for direct integration in the main workflow for Classifier 2 preprocessing.

Usage:
    from src.image_processing.classifier2_workflow.people_masking import mask_people_in_image
    masked_img = mask_people_in_image(img)
"""
import cv2
import numpy as np
import os

# Default Haar cascade path (OpenCV ships with this file)
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Load the cascade once
_face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def mask_people_in_image(image, mask_color=(0, 0, 0), scaleFactor=1.1, minNeighbors=5):
    """
    Detects faces in the input image and masks them with a solid color.

    Args:
        image (np.ndarray): Input BGR image (as loaded by cv2.imread or from video frame)
        mask_color (tuple): BGR color to use for masking faces (default: black)
        scaleFactor (float): Haar cascade scale factor
        minNeighbors (int): Haar cascade minNeighbors
    Returns:
        np.ndarray: Image with faces masked
    """
    if image is None or not hasattr(image, 'shape'):
        raise ValueError("Input image must be a valid numpy array.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    masked = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(masked, (x, y), (x + w, y + h), mask_color, thickness=-1)
    return masked

# For batch processing

def mask_people_in_images(images, mask_color=(0, 0, 0), scaleFactor=1.1, minNeighbors=5):
    """
    Masks faces in a batch of images.
    Args:
        images (List[np.ndarray]): List of BGR images
    Returns:
        List[np.ndarray]: List of masked images
    """
    return [mask_people_in_image(img, mask_color, scaleFactor, minNeighbors) for img in images]
