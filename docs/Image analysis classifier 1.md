# Image Analysis Classifier 1: People/Presentation Binary Classification

## Overview
This document describes the implementation, usage, and evaluation of the image classification module (Classifier 1) for distinguishing between "people" and "presentation" images. The module is designed for integration into the Intelligent Data Extraction System and supports both a custom CNN and an EfficientNetV2B0-based transfer learning model.

## Implemented Models

### 1. Custom CNN Classifier
- **File:** `src/image_processing/classifier1_models/custom_cnn_classifier.py`
- **Architecture:** Sequential CNN with 4 convolutional blocks, batch normalization, dropout, and dense layers.
- **Input Size:** 224x224 RGB images
- **Output:** Softmax over 2 classes (people, presentation)
- **Training:** Uses Keras `ImageDataGenerator` with validation split, early stopping, and learning rate reduction.
- **Model Save Format:** H5 (`custom_cnn_classifier_model.h5`)

### 2. EfficientNetV2B0 Transfer Learning Classifier
- **File:** `src/image_processing/classifier1_models/efficientnet_functional.py`
- **Architecture:** Keras Functional API, EfficientNetV2B0 backbone (frozen), dropout, dense output.
- **Input Size:** 224x224 RGB images
- **Output:** Softmax over 2 classes
- **Training:** Uses Keras `ImageDataGenerator` with EfficientNet preprocessing, early stopping, and learning rate reduction.
- **Model Save Formats:** H5 (`efficientnet_functional_model.h5`) and TensorFlow SavedModel (`efficientnet_savedmodel/`)

## Training & Evaluation Workflow

- **Data Directory:** `data/classifier1/augmented/` (subfolders: `people/`, `presentation/`)
- **Validation Split:** 20% (via `ImageDataGenerator`)
- **Metrics:** Accuracy, Precision, Loss, Prediction Time, Model Size
- **Evaluation Notebook:** `src/image_processing/classifier1_models/test_custom_cnn_classifier_eval.ipynb`
    - Compares both models on the same validation set
    - Reports all metrics and prints a summary table

## Usage

### Training (PowerShell, in `internproj` conda env)
```powershell
conda activate internproj
python src/image_processing/classifier1_models/custom_cnn_classifier.py --train
python src/image_processing/classifier1_models/efficientnet_functional.py --train
```

### Evaluation (Jupyter Notebook)
```powershell
conda activate internproj
jupyter notebook src/image_processing/classifier1_models/test_custom_cnn_classifier_eval.ipynb
```

## Model Comparison Example Output
| Model                   | Loss   | Accuracy | Precision | Time(s) | Size(MB) |
|-------------------------|--------|----------|-----------|---------|----------|
| Custom CNN              |0.2487     0.9825     0.9830       4.38      78
| EfficientNet Functional | 0.0939     0.9864     0.9867      13.90      23

*Values are for illustration; see notebook for actual results.*

## Integration Notes
- Both models can be imported and used for prediction in downstream pipelines.
- EfficientNet model is recommended for higher accuracy; custom CNN is smaller and faster.
- All scripts and notebooks are compatible with the `pygpu` conda environment.

## Files Added/Updated
- `src/image_processing/classifier1_models/custom_cnn_classifier.py`
- `src/image_processing/classifier1_models/efficientnet_functional.py`
- `src/image_processing/classifier1_models/test_custom_cnn_classifier_eval.ipynb`
- `src/image_processing/classifier1_models/custom_cnn_classifier_model.h5`
- `src/image_processing/classifier1_models/efficientnet_functional_model.h5`
- `src/image_processing/classifier1_models/efficientnet_savedmodel/`

## See Also
- `docs/execution_guide.md` for pipeline integration
- `docs/text_extraction&processing.md` for text pipeline details
