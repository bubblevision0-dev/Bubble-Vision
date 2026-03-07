from pathlib import Path

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import joblib  # pip install joblib

from config_omr_knn import ROOT, BUBBLE_DATA_DIR, KNN_MODEL_PATH, KNN_META_PATH

IMG_SIZE = (24, 24)  # size to which each bubble is resized (small = faster)
TEST_SIZE = 0.2      # 80% train / 20% test
K = 5                # number of neighbors

# Fix a stable, semantic class order
EXPECTED_CLASSES = ["empty", "filled", "crossed", "invalid"]


def load_dataset():
    data_dir = Path(BUBBLE_DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    # Discover folders that actually exist
    available = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print("Detected folders in bubbles_data:", available)

    # Build class_names in a stable, meaningful order
    class_names: list[str] = []
    for cname in EXPECTED_CLASSES:
        if cname in available:
            class_names.append(cname)
        else:
            print(f"[WARN] Expected class folder '{cname}' not found.")

    # Any extra folders (e.g. 'invalid', 'other') are added at the end
    extras = sorted(set(available) - set(class_names))
    for cname in extras:
        print(f"[INFO] Extra class folder found: '{cname}' – will be included.")
        class_names.append(cname)

    if not class_names:
        raise RuntimeError(
            "No class folders found in bubbles_data. "
            "Create e.g. 'empty', 'filled', 'crossed', 'invalid'."
        )

    X = []
    y = []

    for label_idx, class_name in enumerate(class_names):
        folder = data_dir / class_name
        print(f"\nLoading class '{class_name}' from {folder} ...")

        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            files.extend(folder.glob(ext))

        if not files:
            print(f"[WARN] No images found for class '{class_name}'.")
            continue

        count_ok = 0
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read image: {p}")
                continue

            # Resize to fixed size
            img_resized = cv2.resize(
                img, IMG_SIZE, interpolation=cv2.INTER_AREA
            )

            # Normalize to [0,1] and flatten
            arr = img_resized.astype("float32") / 255.0
            feat = arr.flatten()

            X.append(feat)
            y.append(label_idx)
            
            # for p in files:
            #     img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            #     if img is None: 
            #         continue

            #     # Apply adaptive threshold to match the scanning pipeline logic
            #     img_thresh = cv2.adaptiveThreshold(
            #         img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            #         cv2.THRESH_BINARY_INV, 11, 7
            #     )

            #     # Resize the thresholded image
            #     img_resized = cv2.resize(img_thresh, IMG_SIZE, interpolation=cv2.INTER_AREA)
                
            #     # Normalize and store
            #     arr = img_resized.astype("float32") / 255.0
            #     X.append(arr.flatten())
            #     y.append(label_idx)
            count_ok += 1

        print(f"  Loaded {count_ok} images for class '{class_name}'.")

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int64")

    if X.shape[0] == 0:
        raise RuntimeError("No training data found in bubbles_data/*")

    print("\n=== DATASET SUMMARY ===")
    print("Total samples:", X.shape[0])
    print("Feature dimension:", X.shape[1])
    for idx, cname in enumerate(class_names):
        n_cls = int((y == idx).sum())
        print(f"  {cname:8s}: {n_cls:4d} samples")

    return X, y, class_names


def main():
    X, y, class_names = load_dataset()

    # Make sure we have enough data per class for a stratified split
    unique, counts = np.unique(y, return_counts=True)
    if np.any(counts < 5):
        print("\n[WARN] Some classes have < 5 samples. "
              "Train/test split and accuracy may be unreliable.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=123,
        stratify=y,
    )

    print(f"\nTrain samples: {X_train.shape[0]}, "
          f"Test samples: {X_test.shape[0]}")

    # ---- Train k-NN ----
    knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
    knn.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== k-NN evaluation on held-out test set ===")
    print("Accuracy:", acc)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(
        classification_report(
            y_test, y_pred, target_names=class_names, digits=4
        )
    )

    # SAVE MODEL + METADATA 
    KNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, KNN_MODEL_PATH)

    np.savez(
        KNN_META_PATH,
        class_names=np.array(class_names),
        img_h=IMG_SIZE[1],
        img_w=IMG_SIZE[0],
    )

    print("\nSaved k-NN model to :", KNN_MODEL_PATH)
    print("Saved metadata to   :", KNN_META_PATH)


if __name__ == "__main__":
    main()
