"""
Custom CNN Classifier for People/Presentation Binary Classification

- Defines, builds, trains, and saves a CNN model using TensorFlow/Keras.
- Provides a reusable prediction function for later import.
- Uses GPU acceleration if available.
- Model and weights are saved in this directory after training.

Usage (PowerShell, inside pygpu env):
    conda activate pygpu
    python src/image_processing/classifier1_models/custom_cnn_classifier.py --train
    # For prediction, import and use predict_image_class(image_path)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2  # Added import for cv2

# Optional: Enable mixed precision for memory savings (uncomment if needed)
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'custom_cnn_classifier_model.h5')
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Reduced from 32 to prevent GPU OOM errors with 224x224 images
EPOCHS = 20
CLASS_NAMES = ['people', 'presentation']


def build_model():
    """Builds the custom CNN model architecture."""
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128),
        layers.ReLU(),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(data_dir, save_path=MODEL_PATH):
    """
    Trains the custom CNN model on images in data_dir (expects subfolders 'people' and 'presentation').
    Saves the trained model to save_path.
    
    Notes:
        - If you encounter OOM (out-of-memory) errors, reduce BATCH_SIZE further or enable mixed precision (see code).
        - Always run inside the 'pygpu' conda environment for GPU acceleration.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15
    )
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    model = build_model()
    cb = [
        #callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ]
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=cb
    )
    model.save(save_path)
    print(f"Model saved to {save_path}")


def load_model(model_path=MODEL_PATH):
    """Loads the trained model from disk."""
    return tf.keras.models.load_model(model_path)


def predict_image_class(image_path, model_path=MODEL_PATH):
    """
    Predicts the class ('people' or 'presentation') for a single image.
    Returns the predicted class label and probability.
    """
    model = load_model(model_path)
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    return CLASS_NAMES[idx], float(preds[0][idx])


def predict_frame_class(model, frame: np.ndarray):
    """
    Predicts the class ('people' or 'presentation') for a single frame (NumPy array).
    Returns the predicted class label and probability.
    Args:
        model: Loaded Keras model
        frame: NumPy array (BGR or RGB)
    Returns:
        tuple: (class_label, probability)
    """
    img = cv2.resize(frame, IMG_SIZE)
    if img.shape[2] == 4:
        img = img[:, :, :3]  # Remove alpha if present
    img = img[..., ::-1]  # Convert BGR to RGB if needed
    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    return CLASS_NAMES[idx], float(preds[0][idx])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Custom CNN Classifier for People/Presentation")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data-dir', type=str, default='data/classifier1/augmented', help='Directory with training images')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    args = parser.parse_args()

    if args.train:
        train_model(args.data_dir)
    elif args.predict:
        label, prob = predict_image_class(args.predict)
        print(f"Predicted: {label} (probability: {prob:.2f})")
    else:
        parser.print_help()
