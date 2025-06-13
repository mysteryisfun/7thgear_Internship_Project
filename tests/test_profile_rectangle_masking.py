"""
Test script to detect rectangles in images, check if they contain faces, and mask those rectangles.
Input:  All images in data/rec_test/
Output: Masked images saved in output/rect_test/

Usage (in pygpu conda env):
    conda activate pygpu
    python tests/test_profile_rectangle_masking.py
"""
import os
import cv2

# Ensure output directory exists
in_dir = os.path.join('data', 'rec_test')
out_dir = os.path.join('output', 'rect_test')
os.makedirs(out_dir, exist_ok=True)

# Load face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for fname in os.listdir(in_dir):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(in_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. Detect rectangles (contours)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 2. Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # 3. For each rectangle, check if it contains a face
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = (x, y, x+w, y+h)
        for (fx, fy, fw, fh) in faces:
            face_center = (fx + fw//2, fy + fh//2)
            if rect[0] <= face_center[0] <= rect[2] and rect[1] <= face_center[1] <= rect[3]:
                # 4. Mask the rectangle
                cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), -1)
                break  # Only need to mask once per rectangle
    # Save masked image
    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, img)
print(f"Masked images saved to {out_dir}")
