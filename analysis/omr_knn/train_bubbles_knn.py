from pathlib import Path
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from config_omr_knn import BUBBLE_DATA_DIR, KNN_MODEL_PATH, KNN_META_PATH

IMG_SIZE = (24, 24)
TEST_SIZE = 0.2
K = 3

EXPECTED_CLASSES = ["empty", "filled", "crossed", "invalid"]


# ---------------------------
# SAME PREPROCESS USED BY OMR
# ---------------------------
def _preprocess_bin_and_mask(gray):
    """
    Match OMR:
    - resize
    - OTSU + THRESH_BINARY_INV
    - inner mask (ignore outline)
    """
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # small denoise
    bin_img = cv2.medianBlur(bin_img, 3)

    # inner mask to ignore ring/outline
    h, w = bin_img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.32)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=mask)

    return bin_img


def _fill_ratio(bin_img):
    return float(np.count_nonzero(bin_img)) / float(bin_img.size + 1e-9)


def load_dataset():
    data_dir = Path(BUBBLE_DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    available = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print("Detected folders in bubbles_data:", available)

    class_names = [c for c in EXPECTED_CLASSES if c in available]
    if not class_names:
        raise RuntimeError("No class folders found in bubbles_data.")

    X, y = [], []
    ratios = []     # store fill ratio for calibration
    labels = []     # store label index for calibration

    for label_idx, class_name in enumerate(class_names):
        folder = data_dir / class_name
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            files.extend(folder.glob(ext))

        print(f"\nLoading '{class_name}' ({len(files)} files)...")
        ok = 0
        for p in files:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.size == 0:
                continue

            bin_img = _preprocess_bin_and_mask(gray)
            feat = (bin_img.astype("float32") / 255.0).flatten()

            X.append(feat)
            y.append(label_idx)

            fr = _fill_ratio(bin_img)
            ratios.append(fr)
            labels.append(label_idx)

            ok += 1

        print(f"  Loaded {ok} images.")

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int64")
    ratios = np.array(ratios, dtype="float32")
    labels = np.array(labels, dtype="int64")

    if X.shape[0] == 0:
        raise RuntimeError("No training data found.")

    return X, y, class_names, ratios, labels


def main():
    X, y, class_names, ratios, labels = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=123, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean", weights="distance")
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("\n=== k-NN Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # ---------------------------
    # CALIBRATION (NO NEW DATA)
    # ---------------------------
    filled_min_ratio = 0.10
    empty_max_ratio = 0.08

    if "filled" in class_names:
        filled_idx = class_names.index("filled")
        filled_ratios = ratios[labels == filled_idx]
        if len(filled_ratios) > 0:
            filled_min_ratio = float(np.percentile(filled_ratios, 5))  # conservative low end
            filled_min_ratio = max(filled_min_ratio, 0.08)

    if "empty" in class_names:
        empty_idx = class_names.index("empty")
        empty_ratios = ratios[labels == empty_idx]
        if len(empty_ratios) > 0:
            empty_max_ratio = float(np.percentile(empty_ratios, 95))   # high end of empty
            empty_max_ratio = min(empty_max_ratio, 0.10)

    print(f"\n[CAL] filled_min_ratio={filled_min_ratio:.4f}  empty_max_ratio={empty_max_ratio:.4f}")

    # Save model + meta
    KNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, KNN_MODEL_PATH)

    np.savez(
        KNN_META_PATH,
        class_names=np.array(class_names),
        img_h=IMG_SIZE[1],
        img_w=IMG_SIZE[0],
        filled_min_ratio=filled_min_ratio,
        empty_max_ratio=empty_max_ratio,
        inner_mask_ratio=0.32
    )

    print(f"\nSaved model to: {KNN_MODEL_PATH}")
    print(f"Saved meta  to: {KNN_META_PATH}")


if __name__ == "__main__":
    main()
