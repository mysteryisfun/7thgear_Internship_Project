"""
Scene Analyzer Module

This module is responsible for analyzing frames from videos to detect scene changes 
based on text content. It uses OCR to extract text from frames and compares 
consecutive frames to identify meaningful changes in text content.

Key features:
- OCR-based text extraction from frames
- Text normalization to reduce false positives
- Caching mechanism to avoid reprocessing identical frames
- Text comparison using difflib to catch subtle changes
- Fallback to image-based similarity when OCR is unavailable or fails
- Support for both skimage.metrics SSIM and simple CV2 image difference comparison
"""

import os
import cv2
import difflib
import re
import numpy as np
import subprocess
from typing import List, Tuple, Dict, Optional

# Check if pytesseract and PIL are available
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Check if skimage is available for SSIM calculations
try:
    from skimage.metrics import structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Check if tesseract executable is available
TESSERACT_EXECUTABLE_AVAILABLE = False
try:
    result = subprocess.run(['tesseract', '--version'], 
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, 
                           check=False)
    if result.returncode == 0:
        TESSERACT_EXECUTABLE_AVAILABLE = True
except Exception:
    pass

class SceneAnalyzer:
    """
    A class for analyzing scenes in video frames based on text content changes.
    Uses OCR to extract text and detects meaningful changes between frames.
    If OCR is not available, falls back to image similarity methods.
    """
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 tesseract_path: Optional[str] = None,
                 use_ocr: bool = True,
                 enable_fallback: bool = True):
        """
        Initialize the SceneAnalyzer.
        
        Args:
            similarity_threshold: Threshold for text similarity (0.0 to 1.0)
            tesseract_path: Path to tesseract executable (optional)
            use_ocr: Whether to use OCR for text extraction (if available)
            enable_fallback: Whether to use image similarity as fallback when OCR fails
        """
        self.similarity_threshold = similarity_threshold
        self.text_cache = None
        self.normalized_cache = None
        self.last_frame_cache = None
        self.enable_fallback = enable_fallback
        
        # Define common filler words to filter out
        self.filler_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by'}
          # Check if we can use OCR
        self.use_ocr = use_ocr and TESSERACT_AVAILABLE and TESSERACT_EXECUTABLE_AVAILABLE
        
        # Log OCR availability
        if self.use_ocr:
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print("OCR enabled for scene analysis")
        else:
            print("OCR not available. Falling back to image similarity methods")
            if not TESSERACT_AVAILABLE:
                print("  - pytesseract module not found")
            if not TESSERACT_EXECUTABLE_AVAILABLE:
                print("  - tesseract executable not found")
            if not use_ocr:
                print("  - OCR disabled by configuration")
                
        # Log fallback availability
        if self.enable_fallback:
            if SKIMAGE_AVAILABLE:
                print("Image-based fallback using skimage.metrics is available")
            else:
                print("skimage.metrics not available, using simple image difference as fallback")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess the image to improve OCR quality.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return denoised
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        if not self.use_ocr:
            # OCR not available, return placeholder
            return f"image:{os.path.basename(image_path)}"
            
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image_path)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(processed_img)
            
            return text
        except Exception as e:
            print(f"Error extracting text from {image_path}: {str(e)}")
            return ""
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text to reduce false positives from OCR inconsistencies.
        Steps: lowercase, remove extra spaces, remove special chars, filter stopwords, spell correction.
        """
        if not text:
            return ""
        # Convert to lowercase
        normalized = text.lower()
        # Remove special characters (keep alphanum and spaces)
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        # Remove common filler words (stopwords)
        stopwords = set(['the', 'and', 'a', 'an', 'of', 'to', 'in', 'on', 'for', 'with', 'at', 'by', 'from', 'as', 'is', 'it', 'this', 'that', 'these', 'those', 'be', 'or', 'are'])
        normalized = ' '.join([w for w in normalized.split() if w not in stopwords])
        # Spell correction (very basic, only for common OCR errors)
        corrections = {'0': 'o', '1': 'i', '5': 's', '8': 'b', '6': 'g', '9': 'g', 'l': 'i', '|': 'i'}
        normalized = ''.join([corrections.get(c, c) for c in normalized])
        return normalized

    def holistic_text(self, text: str) -> str:
        """
        Combine all text into a single string for holistic comparison.
        """
        return ' '.join(self.normalize_text(text).split())

    def compare_text(self, text1: str, text2: str) -> float:
        """
        Compare two text strings and return similarity ratio.
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        norm1 = self.holistic_text(text1)
        norm2 = self.holistic_text(text2)
        return difflib.SequenceMatcher(None, norm1, norm2).ratio()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase, remove special chars, remove extra spaces, concatenate to one string.
        """
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Remove special characters (keep only alphanum and spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Remove extra spaces and concatenate
        text = ' '.join(text.split())
        return text

    def analyze_frame(self, frame_path: str, debug: bool = False) -> dict:
        """
        Analyze a frame and determine if it represents a scene change.
        Uses holistic text comparison, similarity threshold, and fallback to SSIM if OCR fails.
        """
        current_text = self.extract_text(frame_path)
        norm_text = self.holistic_text(current_text)
        frame_name = os.path.basename(frame_path)
        # Dynamic threshold adjustment (example: more text, lower threshold)
        dynamic_threshold = self.similarity_threshold
        if len(norm_text.split()) > 20:
            dynamic_threshold = max(0.8, self.similarity_threshold - 0.1)
        if self.text_cache is None:
            self.text_cache = norm_text
            if debug:
                print(f"First frame: {frame_name}\nText: {current_text[:100]}...")
            return {
                "is_new_scene": True,
                "similarity": 0.0,
                "text": current_text,
                "normalized_text": norm_text,
                "frame_name": frame_name
            }
        similarity = self.compare_text(self.text_cache, current_text)
        is_new_scene = similarity < dynamic_threshold
        if debug:
            print(f"\nFrame: {frame_name}")
            print(f"Similarity: {similarity:.4f} (Threshold: {dynamic_threshold})")
            print(f"Is new scene: {is_new_scene}")
            print(f"Current text: {current_text[:50]}...")
            print(f"Cached text: {self.text_cache[:50]}...")        # Fallback: if OCR text is empty, use image comparison
        if not norm_text.strip() and self.enable_fallback:
            try:
                img1 = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(self.last_frame_cache, cv2.IMREAD_GRAYSCALE) if self.last_frame_cache else None
                if img1 is not None and img2 is not None:
                    if SKIMAGE_AVAILABLE:
                        # Use structural_similarity from skimage if available
                        ssim = structural_similarity(img1, img2)
                        is_new_scene = ssim < 0.95
                        if debug:
                            print(f"[Fallback] SSIM: {ssim:.4f}")
                    else:
                        # Fallback to simple difference if skimage is not available
                        diff = cv2.absdiff(img1, img2)
                        similarity = 1.0 - (cv2.countNonZero(diff) / (img1.shape[0] * img1.shape[1]))
                        is_new_scene = similarity < 0.95
                        if debug:
                            print(f"[Fallback] Image diff similarity: {similarity:.4f}")
            except Exception as e:
                if debug:
                    print(f"[Fallback] Image comparison failed: {e}")
        if is_new_scene:
            self.text_cache = norm_text
            self.last_frame_cache = frame_path
        return {
            "is_new_scene": is_new_scene,
            "similarity": similarity,
            "text": current_text,
            "normalized_text": norm_text,
            "frame_name": frame_name
        }
    
    def analyze_frames(self, frame_paths: List[str], debug: bool = False) -> List[str]:
        """
        Analyze frames, store unique preprocessed texts, return list of them.
        Falls back to image-based similarity when OCR is not available.
        """
        self.text_cache = None
        self.last_frame_cache = None
        stored_texts = []
        
        # Check if we need to use fallback image similarity
        use_image_similarity = not self.use_ocr
        
        for i, frame_path in enumerate(frame_paths):
            if use_image_similarity:
                # Image-based similarity fallback when OCR is not available
                current_img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if current_img is None:
                    continue
                    
                # For the first frame
                if self.last_frame_cache is None:
                    self.last_frame_cache = frame_path
                    frame_text = f"image:{os.path.basename(frame_path)}"
                    stored_texts.append(frame_text)
                    if debug:
                        print(f"First frame: {os.path.basename(frame_path)}")
                    continue
                    
                # Compare with previous frame
                prev_img = cv2.imread(self.last_frame_cache, cv2.IMREAD_GRAYSCALE)
                if prev_img is not None and prev_img.shape == current_img.shape:
                    # Calculate structural similarity
                    try:
                        score = cv2.matchTemplate(current_img, prev_img, cv2.TM_CCOEFF_NORMED)[0][0]
                        similarity = (score + 1) / 2  # Normalize to 0-1 range
                    except Exception:
                        # Fallback to simple difference if template matching fails
                        diff = cv2.absdiff(current_img, prev_img)
                        similarity = 1.0 - (cv2.countNonZero(diff) / (current_img.shape[0] * current_img.shape[1]))
                    
                    if similarity < self.similarity_threshold:
                        # This is a new scene
                        frame_text = f"image:{os.path.basename(frame_path)}"
                        stored_texts.append(frame_text)
                        self.last_frame_cache = frame_path
                        if debug:
                            print(f"New scene at {os.path.basename(frame_path)}, similarity: {similarity:.4f}")
                else:
                    # Images have different shapes, consider as new scene
                    frame_text = f"image:{os.path.basename(frame_path)}"
                    stored_texts.append(frame_text)
                    self.last_frame_cache = frame_path
                    if debug:
                        print(f"New scene at {os.path.basename(frame_path)} (different dimensions)")
            else:
                # OCR-based similarity (original implementation)
                raw_text = self.extract_text(frame_path)
                processed = self.preprocess_text(raw_text)
                
                if self.text_cache is None:
                    self.text_cache = processed
                    stored_texts.append(processed)
                    if debug:
                        print(f"First frame: {os.path.basename(frame_path)}\nText: {processed}")
                    continue
                
                similarity = difflib.SequenceMatcher(None, self.text_cache, processed).ratio()
                if similarity < self.similarity_threshold:
                    stored_texts.append(processed)
                    self.text_cache = processed
                    if debug:
                        print(f"New scene at {os.path.basename(frame_path)}: {processed}, similarity: {similarity:.4f}")
        
        return stored_texts

    def extract_keyframes(self, frame_paths: List[str], output_dir: str) -> List[str]:
        """
        Extract key frames that represent scene changes.
        
        Args:
            frame_paths: List of paths to frame image files
            output_dir: Directory to save extracted key frames
            
        Returns:
            List of paths to the extracted key frames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Reset cache
        self.text_cache = None
        self.normalized_cache = None
        
        keyframe_paths = []
        
        for i, frame_path in enumerate(frame_paths):
            result = self.analyze_frame(frame_path)
            
            if result["is_new_scene"]:
                # This is a key frame representing a scene change
                frame_filename = os.path.basename(frame_path)
                keyframe_path = os.path.join(output_dir, f"keyframe_{i:05d}_{frame_filename}")
                
                # Copy the frame to the output directory
                img = cv2.imread(frame_path)
                cv2.imwrite(keyframe_path, img)
                
                keyframe_paths.append(keyframe_path)
        
        return keyframe_paths
