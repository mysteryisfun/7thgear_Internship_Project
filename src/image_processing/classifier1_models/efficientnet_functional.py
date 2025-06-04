"""
EfficientNet-V2B0 Transfer Learning Classifier (Standard Functional API Version)

This is a simplified version of the EfficientNet classifier that uses the standard
Keras Functional API instead of a custom model subclass. This approach avoids
serialization issues that can occur with custom model classes.

Usage (PowerShell, inside pygpu env):
    conda activate pygpu
    python src/image_processing/classifier1_models/efficientnet_functional.py --train
    # For prediction, import and use predict_image_class(image_path)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'efficientnet_functional_model.h5')
SAVEDMODEL_PATH = os.path.join(os.path.dirname(__file__), 'efficientnet_functional_savedmodel')
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Adjust if OOM
EPOCHS = 30
CLASS_NAMES = ['people', 'presentation']


def build_model():
    """
    Builds an EfficientNetV2B0 transfer learning model using the standard Keras Functional API.
    This avoids serialization issues that can occur with custom model classes.
    """
    # Create the base model
    base_model = EfficientNetV2B0(include_top=False, weights='imagenet', 
                                 input_shape=(224, 224, 3), pooling='avg')
    base_model.trainable = False  # Freeze base layers
    
    # Build model using Functional API
    inputs = layers.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)  # Apply preprocessing within the model
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    # Create and compile model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(data_dir, save_path=MODEL_PATH, save_savedmodel=True):
    """
    Trains the EfficientNetV2B0 transfer learning model on images in data_dir.
    Saves the trained model to save_path in H5 format and optionally as a SavedModel.
    
    Args:
        data_dir: Directory containing subdirectories for each class ('people', 'presentation')
        save_path: Path to save the trained model in H5 format
        save_savedmodel: Whether to also save the model in SavedModel format
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
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
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=cb
    )
    
    # Save model in H5 format
    model.save(save_path, save_format='h5', include_optimizer=False)
    print(f"Model saved in H5 format to {save_path}")
    
    # Optionally save in SavedModel format
    if save_savedmodel:
        model.save(SAVEDMODEL_PATH)
        print(f"Model also saved in SavedModel format to {SAVEDMODEL_PATH}")
    
    # Save training history
    history_path = save_path.replace('.h5', '_history.json')
    import json
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved to {history_path}")
    
    return model


def predict_image_class(image_path, model_path=MODEL_PATH):
    """
    Predicts the class ('people' or 'presentation') for a single image.
    Returns the predicted class label and probability.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the saved model (H5 or SavedModel directory)
    
    Returns:
        tuple: (class_label, probability)
    """
    # Load the model - works with either H5 or SavedModel format without custom objects
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    
    # Make prediction
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    
    return CLASS_NAMES[idx], float(preds[0][idx])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EfficientNet-V2B0 Transfer Learning Classifier using Functional API")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data-dir', type=str, default='data/classifier1/augmented', help='Directory with training images')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    parser.add_argument('--format', type=str, choices=['h5', 'savedmodel'], default='h5', help='Model format to use for prediction')
    args = parser.parse_args()

    if args.train:
        train_model(args.data_dir)
    elif args.predict:
        model_path = MODEL_PATH if args.format == 'h5' else SAVEDMODEL_PATH
        label, prob = predict_image_class(args.predict, model_path)
        print(f"Predicted: {label} (probability: {prob:.2f})")
    else:
        parser.print_help()
