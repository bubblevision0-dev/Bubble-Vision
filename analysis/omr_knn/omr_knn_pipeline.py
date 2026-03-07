from pathlib import Path
import cv2
import numpy as np
import joblib
import sys

from .config_omr_knn import CHOICE_LETTERS, TEMPLATES, KNN_MODEL_PATH, KNN_META_PATH

# ============================================================
# 1) AUTO-DETECTION LOGIC
# ============================================================

def auto_detect_layout(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "70"

    # Focus on the largest rectangle
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    scanned_ratio = w / float(h)

    # Define strict expected ratios (Update these based on your config_omr_knn.py)
    # Example: if 50-item is more "square" than 70-item
    best_match = "70"
    min_diff = float("inf")
    
    for layout_id, cfg in TEMPLATES.items():
        template_ratio = cfg["SHEET_W"] / float(cfg["SHEET_H"])
        diff = abs(scanned_ratio - template_ratio)
        
        if diff < min_diff:
            min_diff = diff
            best_match = layout_id

    # ✅ Manual Overrides for 50/60/70 based on observed ratios
    # If it's a 50-item sheet, the ratio is usually significantly different 
    # from a 70-item sheet. Check your printed sheet ratios.
    if 0.73 < scanned_ratio < 0.78:
        return "50"
    elif 0.69 < scanned_ratio <= 0.73:
        return "60"
    elif 0.65 <= scanned_ratio <= 0.69:
        return "70"

    return best_match

# ============================================================
# 2) LOAD MODEL + META (thresholds from training)
# ============================================================

def load_knn_model():
    knn = joblib.load(KNN_MODEL_PATH)
    meta = np.load(KNN_META_PATH, allow_pickle=True)

    class_names = meta["class_names"].tolist()
    img_size = (int(meta["img_w"]), int(meta["img_h"]))

    filled_min_ratio = float(meta.get("filled_min_ratio", 0.10))
    empty_max_ratio = float(meta.get("empty_max_ratio", 0.08))
    inner_mask_ratio = float(meta.get("inner_mask_ratio", 0.32))

    cal = {
        "filled_min_ratio": filled_min_ratio,
        "empty_max_ratio": empty_max_ratio,
        "inner_mask_ratio": inner_mask_ratio
    }
    return knn, class_names, img_size, cal


# ============================================================
# 3) WARP
# ============================================================

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def warp_sheet(img_bgr, target_w, target_h):
    h_orig, w_orig = img_bgr.shape[:2]
    scale_down = 2000.0 / h_orig if h_orig > 2000 else 1.0
    
    small = cv2.resize(img_bgr, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # Increased block size to ignore internal bubble noise and find the OUTER border
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71, 11)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet_cnt = None
    
    # Filter for large, page-sized contours only
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(cnt)
        if area < (small.shape[0] * small.shape[1] * 0.2): # Ignore if less than 20% of image
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            sheet_cnt = (approx / scale_down).astype(np.float32)
            break

    if sheet_cnt is not None:
        src_pts = _order_points(sheet_cnt.reshape(4, 2))
    else:
        # Improved Fallback: Use full image but trim 2% to avoid scanner black edges
        src_pts = np.array([
            [w_orig*0.02, h_orig*0.02], 
            [w_orig*0.98, h_orig*0.02], 
            [w_orig*0.98, h_orig*0.98], 
            [w_orig*0.02, h_orig*0.98]
        ], dtype="float32")

    dst_pts = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img_bgr, M, (target_w, target_h))


# ============================================================
# 4) BUBBLE PREPROCESS (MATCH TRAINING)
# ============================================================

def _preprocess_bin_and_mask(gray, IMG_SIZE, inner_mask_ratio):
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_img = cv2.medianBlur(bin_img, 3)

    h, w = bin_img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * inner_mask_ratio)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=mask)
    return bin_img

def _fill_ratio(bin_img):
    return float(np.count_nonzero(bin_img)) / float(bin_img.size + 1e-9)

def classify_bubble_knn(img_bgr, knn, class_names, IMG_SIZE, cal):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bin_img = _preprocess_bin_and_mask(gray, IMG_SIZE, cal["inner_mask_ratio"])
    fr = _fill_ratio(bin_img)

    feat = (bin_img.astype("float32") / 255.0).flatten().reshape(1, -1)
    pred_name = class_names[knn.predict(feat)[0]]

    # Gate:
    if fr <= cal["empty_max_ratio"]:
        return "empty", fr

    if pred_name == "filled" and fr < cal["filled_min_ratio"]:
        return "empty", fr

    if fr >= cal["filled_min_ratio"]:
        return "filled", fr

    return pred_name, fr


def crop_bubble(warped_bgr, cx, cy, r):
    x1, y1 = int(cx - r), int(cy - r)
    x2, y2 = int(cx + r), int(cy + r)
    return warped_bgr[max(0, y1):y2, max(0, x1):x2]


# ============================================================
# 5) ANALYZE SHEET (student response only)
# ============================================================

def memory_safe_load(image_path, max_dim=2500):
    """Prevents memory crash by scaling before processing."""
    img_raw = cv2.imread(str(image_path))
    if img_raw is None:
        return None
        
    h, w = img_raw.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_raw = cv2.resize(img_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img_raw

def analyze_sheet(image_path, knn, class_names, img_size, cal):
    # Load and immediately resize to prevent RAM crashes
    img_raw = cv2.imread(str(image_path))
    if img_raw is None: return None, {}, "70"
    
    h, w = img_raw.shape[:2]
    if max(h, w) > 2500: # Standardize maximum dimension to 2500px
        scale = 2500.0 / float(max(h, w))
        img_raw = cv2.resize(img_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    layout_id = auto_detect_layout(img_raw)
    cfg = TEMPLATES.get(layout_id, TEMPLATES["70"])
    
    # Warp the stabilized image
    warped = warp_sheet(img_raw, cfg["SHEET_W"], cfg["SHEET_H"])
    r = cfg["BUBBLE_RADIUS"]

    raw_results = {}
    for q_num, centers in cfg["Q_CENTERS"].items():
        filled = []
        ratios = {}
        for idx, (cx, cy) in enumerate(centers):
            letter = CHOICE_LETTERS[idx]
            bubble_crop = crop_bubble(warped, cx, cy, r)
            if bubble_crop.size == 0:
                continue

            label, fr = classify_bubble_knn(bubble_crop, knn, class_names, img_size, cal)
            ratios[letter] = fr
            if label == "filled":
                filled.append(letter)

        if len(filled) == 0:
            raw_results[q_num] = {"ans": None, "type": "blank"}
        elif len(filled) == 1:
            raw_results[q_num] = {"ans": filled[0], "type": "valid"}
        else:
            best = max(filled, key=lambda L: ratios.get(L, 0.0))
            raw_results[q_num] = {"ans": best, "type": "invalid", "multi": filled}

    return warped, raw_results, layout_id


# ============================================================
# 6) SCORING
# ============================================================

def score_sheet(raw_results, answer_map):
    score_raw = 0
    per_item = {}

    for q_no, correct_ans in answer_map.items():
        q_key = int(q_no)
        det = raw_results.get(q_key, {"ans": None, "type": "blank"})
        student_ans = det.get("ans")
        dtype = det.get("type", "blank")

        if dtype == "valid" and student_ans and student_ans.upper() == correct_ans.upper():
            score_raw += 1
            status = "correct"
        elif dtype == "blank":
            status = "blank"
        elif dtype == "invalid":
            status = "invalid"
        else:
            status = "incorrect"

        per_item[q_no] = {
            "correct_answer": correct_ans,
            "student_answer": student_ans,
            "status": status,
            "multi": det.get("multi", [])
        }

    score_pct = (score_raw / len(answer_map) * 100) if answer_map else 0
    return {"score_raw": score_raw, "score_pct": score_pct, "per_item": per_item}


# ============================================================
# 7) ANNOTATION (GREEN + RED only)
# ============================================================

def annotate_sheet(warped, summary, layout_id):
    annotated = warped.copy()
    cfg = TEMPLATES.get(layout_id, TEMPLATES["70"])
    r = cfg["BUBBLE_RADIUS"]

    for q_no, data in summary["per_item"].items():
        centers = cfg["Q_CENTERS"].get(int(q_no))
        if not centers:
            continue

        correct_ans = data["correct_answer"]
        student_ans = data["student_answer"]
        status = data["status"]
        multi = data.get("multi", []) or []

        def draw(letter, color):
            if letter not in CHOICE_LETTERS:
                return
            idx = CHOICE_LETTERS.index(letter)
            cx, cy = centers[idx]
            cv2.circle(annotated, (int(cx), int(cy)), r + 3, color, 2)

        if status == "correct":
            if student_ans:
                draw(student_ans, (0, 255, 0))

        elif status == "incorrect":
            if student_ans:
                draw(student_ans, (0, 0, 255))
            draw(correct_ans, (0, 255, 0))

        elif status == "invalid":
            for ans in set(multi):
                draw(ans, (0, 0, 255))
            draw(correct_ans, (0, 255, 0))

        else:
                # BLANK -> mark the correct answer (GREEN)
            draw(correct_ans, (0, 255, 0))

    return annotated


# ============================================================
# 8) MAIN (PRINT DETECTED RESULTS)
# ============================================================

def run_omr(image_path, key_path):
    knn, class_names, img_size, cal = load_knn_model()

    warped, raw_results, layout_id = analyze_sheet(image_path, knn, class_names, img_size, cal)
    if warped is None:
        print("Image not found.")
        return

    with open(key_path, "r") as f:
        full_key = [line.strip().upper() for line in f if line.strip()]

    cfg = TEMPLATES.get(layout_id, TEMPLATES["70"])
    num_to_score = min(len(full_key), len(cfg["Q_CENTERS"]))
    answer_map = {i: full_key[i - 1] for i in range(1, num_to_score + 1)}

    summary = score_sheet(raw_results, answer_map)
    annotated = annotate_sheet(warped, summary, layout_id)

    # ✅ PRINT DETECTED RESULTS
    print(f"\n--- DETECTED RESULTS ({layout_id}-ITEM) ---")
    for i in range(1, num_to_score + 1):
        det = raw_results.get(i, {"ans": None, "type": "blank"})
        student = det.get("ans")
        dtype = det.get("type")

        if dtype == "blank":
            det_str = "BLANK"
        elif dtype == "valid":
            det_str = f"{student}"
        else:
            det_str = f"MULT ({','.join(det.get('multi', []))}) -> PICKED {student}"

        correct = answer_map[i]
        status = summary["per_item"][i]["status"].upper()
        print(f"Q{i:02d}: Student={det_str:<20} Key={correct:<2} Status={status}")

    print(f"\nFINAL SCORE: {summary['score_raw']}/{num_to_score} ({summary['score_pct']:.2f}%)")

    out_name = f"result_{Path(image_path).stem}.png"
    cv2.imwrite(out_name, annotated)
    print(f"\nAnnotated image saved as '{out_name}'")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python omr_knn_pipeline.py <image> <key.txt>")
    else:
        run_omr(sys.argv[1], sys.argv[2])
