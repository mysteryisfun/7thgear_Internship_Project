"""
EfficientNet-B0 Transfer Learning Classifier for People/Presentation Binary Classification

- Defines, builds, trains, and saves a transfer learning model using EfficientNet-B0 (TensorFlow/Keras).
- Provides a reusable prediction function for later import.
- Uses GPU acceleration if available.
- Model and weights are saved in this directory after training.

Usage (PowerShell, inside pygpu env):
    conda activate pygpu
    python src/image_processing/classifier1_models/efficientnet_classifier.py --train
    # For prediction, import and use predict_image_class(image_path)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'efficientnet_classifier_model.h5')
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Adjust if OOM
EPOCHS = 15
CLASS_NAMES = ['people', 'presentation']


def build_model():
    """Builds the EfficientNet-B0 transfer learning model architecture."""
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    base_model.trainable = False  # Freeze base
    inputs = layers.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(data_dir, save_path=MODEL_PATH):
    """
    Trains the EfficientNet-B0 transfer learning model on images in data_dir (expects subfolders 'people' and 'presentation').
    Saves the trained model to save_path.
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
        callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ]
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=cb
    )
    # Save model using 'save_format="h5"' to avoid TensorFlow serialization issues
    model.save(save_path, save_format='h5')
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
    x = tf.keras.utils.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    return CLASS_NAMES[idx], float(preds[0][idx])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EfficientNet-B0 Transfer Learning Classifier for People/Presentation")
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
