from pathlib import Path
from typing import Dict, Any
import numpy as np
import cv2
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "bubble_classifier.h5"

# Same order as training
CLASS_NAMES = ["empty", "filled", "crossed", "invalid"]

IMG_SIZE = 40  # must match train_bubbles.py

print(f"[INFO] Loading bubble model from: {MODEL_PATH}")
MODEL = tf.keras.models.load_model(MODEL_PATH)


def preprocess_patch(patch: np.ndarray) -> np.ndarray:
    """
    Take a raw grayscale patch (H,W) or (H,W,1) and prepare for model.
    """
    if patch.ndim == 2:
        # (H,W) -> (H,W,1)
        patch = np.expand_dims(patch, axis=-1)

    # Resize
    patch_resized = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))

    # Ensure shape (IMG_SIZE, IMG_SIZE, 1)
    if patch_resized.ndim == 2:
        patch_resized = np.expand_dims(patch_resized, axis=-1)

    # Convert to float32 & scale to [0,1] (same as Rescaling(1./255) in model)
    patch_resized = patch_resized.astype("float32") / 255.0

    # Add batch dim: (1, H, W, C)
    x = np.expand_dims(patch_resized, axis=0)
    return x


def classify_patch(patch: np.ndarray):
    """
    Classify a single bubble patch.
    Returns: (label, confidence, probs_array)
    """
    x = preprocess_patch(patch)
    probs = MODEL.predict(x, verbose=0)[0]   # shape (num_classes,)
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    conf = float(probs[idx])
    return label, conf, probs


def classify_question(option_patches: Dict[str, np.ndarray], conf_thr: float = 0.6) -> Dict[str, Any]:
    """
    option_patches: {"A": patchA, "B": patchB, ...}

    Decide:
      - which options look "marked" (filled or crossed with enough confidence)
      - question status: blank / answered / invalid

    Returns:
      {
        "status": "blank" | "answered" | "invalid",
        "chosen": "A"/"B"/None,
        "marked": [letters],
        "per_option": {
          "A": {"label": ..., "conf": ..., "probs": [...]},
          ...
        }
      }
    """
    per_option: Dict[str, Any] = {}
    marked: list[str] = []

    for letter, patch in option_patches.items():
        label, conf, probs = classify_patch(patch)
        per_option[letter] = {
            "label": label,
            "conf": conf,
            "probs": probs.tolist(),
        }

        # treat both "filled" and "crossed" as "marked"
        if label in ("filled", "crossed") and conf >= conf_thr:
            marked.append(letter)

    if len(marked) == 0:
        status = "blank"
        chosen = None
    elif len(marked) == 1:
        status = "answered"
        chosen = marked[0]
    else:
        status = "invalid"
        chosen = None

    return {
        "status": status,
        "chosen": chosen,
        "marked": marked,
        "per_option": per_option,
    }


# Quick self-test when run directly
if __name__ == "__main__":
    import numpy as np

    dummy = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    label, conf, probs = classify_patch(dummy)
    print(f"Dummy patch classified as: {label} ({conf:.2f})")
    print("probs:", probs)
