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
import joblib

# Ensure these paths are correctly defined in your config file
from config_omr_knn import ROOT, BUBBLE_DATA_DIR, KNN_MODEL_PATH, KNN_META_PATH

IMG_SIZE = (24, 24)  
TEST_SIZE = 0.2      
K = 3  # Lower K is often better for binary bubble classification

EXPECTED_CLASSES = ["empty", "filled", "crossed", "invalid"]

def load_dataset():
    data_dir = Path(BUBBLE_DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    available = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print("Detected folders in bubbles_data:", available)

    class_names: list[str] = []
    for cname in EXPECTED_CLASSES:
        if cname in available:
            class_names.append(cname)
        else:
            print(f"[WARN] Expected class folder '{cname}' not found.")

    extras = sorted(set(available) - set(class_names))
    for cname in extras:
        print(f"[INFO] Extra class folder found: '{cname}' – will be included.")
        class_names.append(cname)

    if not class_names:
        raise RuntimeError("No class folders found in bubbles_data.")

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
                continue

            # 1. Resize
            img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

            # 2. Thresholding: Convert to binary (Black & White)
            # THRESH_BINARY_INV makes the marks white (255) and background black (0)
            # This makes the "fill" the most important mathematical feature
            _, img_bin = cv2.threshold(
                img_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # 3. Normalize & Flatten
            feat = img_bin.astype("float32") / 255.0
            X.append(feat.flatten())
            y.append(label_idx)
            count_ok += 1

        print(f"  Loaded {count_ok} images for class '{class_name}'.")

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int64")

    if X.shape[0] == 0:
        raise RuntimeError("No training data found.")

    return X, y, class_names

def main():
    X, y, class_names = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=123, stratify=y
    )

    # Euclidean distance is fine, but thresholding makes it much more effective
    knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("\n=== k-NN Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save
    KNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, KNN_MODEL_PATH)
    np.savez(KNN_META_PATH, class_names=np.array(class_names), img_h=IMG_SIZE[1], img_w=IMG_SIZE[0])

if __name__ == "__main__":
    main()