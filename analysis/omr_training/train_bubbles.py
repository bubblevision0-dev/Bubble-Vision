"""
train_bubbles.py

Train a CNN to classify bubble patches into:
    - empty
    - filled
    - crossed
    - invalid

Assumed folder structure:

omr_training/
  data/
    train/
      empty/
      filled/
      crossed/
      invalid/
    val/
      empty/
      filled/
      crossed/
      invalid/

The trained model will be saved to:
    analysis/omr_engine/models/bubble_cnn.h5

Run from project root:
    (env) D:\item_analysis\item> python omr_training/train_bubbles.py
"""

from pathlib import Path
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------- CONFIG ----------------

# Project root = folder where manage.py lives
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_ROOT = Path(__file__).resolve().parent / "data"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

# Image config
IMG_SIZE = 40           # bubble patch size (40x40), adjust if needed
COLOR_MODE = "grayscale"  # "grayscale" or "rgb"
CHANNELS = 1 if COLOR_MODE == "grayscale" else 3

BATCH_SIZE = 64
EPOCHS = 25

# Class order MUST match your inference CLASS_NAMES
CLASS_NAMES = ["empty", "filled", "crossed", "invalid"]


# ---------------- REPRODUCIBILITY ----------------

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds(42)


# ---------------- DATASET LOADING ----------------

def make_datasets():
    """
    Create tf.data.Dataset objects for train and validation.
    Uses directory structure and ensures class order == CLASS_NAMES.
    """
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"TRAIN_DIR not found: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"VAL_DIR not found: {VAL_DIR}")

    print(f"[INFO] Loading training data from: {TRAIN_DIR}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",   # one-hot
        class_names=CLASS_NAMES,    # enforce order
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
        seed=42,
    )

    print(f"[INFO] Loading validation data from: {VAL_DIR}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
        seed=123,
    )

    # Performance options
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("[INFO] Train class names:", train_ds.class_names)
    return train_ds, val_ds


# ---------------- MODEL DEFINITION ----------------

def build_model():
    """
    Simple but strong CNN for small 40x40 patches.
    """
    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)

    inputs = keras.Input(shape=input_shape)

    # Normalize inputs to [0,1]
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Data augmentation (small, to avoid overfitting)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.02)(x)
    x = layers.RandomZoom(0.05)(x)

    # Convolutional backbone
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="bubble_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ---------------- TRAINING LOOP ----------------

def train():
    train_ds, val_ds = make_datasets()
    model = build_model()

    # Callbacks: early stopping & best-checkpoint saving (in training folder)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(Path(__file__).resolve().parent / "bubble_cnn_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    print("[INFO] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Evaluate on validation
    print("[INFO] Final evaluation on validation set:")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"[RESULT] val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Save final model into your system folder
    models_dir = PROJECT_ROOT / "analysis" / "omr_engine" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    out_path_h5 = models_dir / "bubble_cnn.h5"
    model.save(out_path_h5)
    print(f"[INFO] Saved model to: {out_path_h5}")

    # Optionally also save as .keras in the same folder
    out_path_keras = models_dir / "bubble_cnn.keras"
    model.save(out_path_keras)
    print(f"[INFO] Saved model to: {out_path_keras}")

    # Save CLASS_NAMES so you never forget the order
    class_txt = models_dir / "class_names.txt"
    class_txt.write_text("\n".join(CLASS_NAMES), encoding="utf-8")
    print(f"[INFO] Saved class names to: {class_txt}")


if __name__ == "__main__":
    # Make sure TF doesn’t spam too much (optional)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    train()
