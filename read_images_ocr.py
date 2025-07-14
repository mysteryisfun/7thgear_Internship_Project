import os
import sys
from paddleocr import PaddleOCR
from PIL import Image

def read_images_and_extract_text(directory):
    """
    Reads all images in the specified directory and extracts text using PaddleOCR.

    Args:
        directory (str): Path to the directory containing images.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Initialize PaddleOCR with angle classifier enabled
    ocr = PaddleOCR(use_angle_cls=True)

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            try:
                print(f"Processing image: {filename}")

                # Open the image
                image = Image.open(file_path)

                # Perform OCR
                result = ocr.ocr(file_path, cls=True)

                # Validate OCR result structure
                if not isinstance(result, list) or not result or not isinstance(result[0], list):
                    print(f"Unexpected OCR result format for file '{filename}': {result}")
                    continue

                # Collect and print all extracted text for the image
                extracted_texts = []
                for line in result[0]:
                    # Skip box-only lines (list of coordinates)
                    if isinstance(line, list) and all(isinstance(x, (int, float)) or (isinstance(x, list) and all(isinstance(y, (int, float)) for y in x)) for x in line):
                        continue
                    # Handle tuple (text, confidence)
                    if isinstance(line, tuple) and len(line) > 0 and isinstance(line[0], str):
                        extracted_texts.append(line[0])
                    # Handle [box, (text, confidence)]
                    elif isinstance(line, list) and len(line) > 1 and isinstance(line[1], tuple) and len(line[1]) > 0 and isinstance(line[1][0], str):
                        extracted_texts.append(line[1][0])
                if extracted_texts:
                    print(f"Extracted text for {filename}: {', '.join(extracted_texts)}")
                else:
                    print(f"No valid text extracted for {filename}.")

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")
        else:
            print(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_images_ocr.py <directory_path>")
        sys.exit(1)

    input_directory = sys.argv[1]
    read_images_and_extract_text(input_directory)
