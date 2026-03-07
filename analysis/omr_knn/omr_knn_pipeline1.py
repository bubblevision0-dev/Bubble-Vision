from pathlib import Path
import cv2
import numpy as np
import joblib
import sys

from .config_omr_knn import ROOT, CHOICE_LETTERS, TEMPLATES, KNN_MODEL_PATH, KNN_META_PATH

# ========== 1. AUTO-DETECTION LOGIC ==========

def auto_detect_layout(img_bgr):
    """
    Detects the layout by finding the sheet first, then calculating its aspect ratio.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Use Adaptive Threshold to find the paper outline better in different lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[WARNING] No contours found. Defaulting to 70-item.")
        return "70" 

    # Find the largest rectangular contour (the paper)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    # Get the bounding box of the detected paper
    x, y, w, h = cv2.boundingRect(cnt)
    scanned_ratio = w / float(h)

    best_match = "70"
    min_diff = float("inf")

    # Compare scanned paper ratio to defined Template ratios
    for layout_id, cfg in TEMPLATES.items():
        template_ratio = cfg["SHEET_W"] / float(cfg["SHEET_H"])
        diff = abs(scanned_ratio - template_ratio)
        if diff < min_diff:
            min_diff = diff
            best_match = layout_id

    print(f"[AUTO-DETECT] Scanned Ratio: {scanned_ratio:.2f} -> Best Match: {best_match}-item")
    return best_match

# ========== 2. LOAD k-NN MODEL & META ==========

def load_knn_model():
    knn = joblib.load(KNN_MODEL_PATH)
    meta = np.load(KNN_META_PATH, allow_pickle=True)
    class_names = meta["class_names"].tolist()
    img_size = (int(meta["img_w"]), int(meta["img_h"]))
    return knn, class_names, img_size

# ========== 3. OMR: SHEET WARPING ==========

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def warp_sheet(img_bgr, target_w, target_h):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 11)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet_cnt = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            sheet_cnt = approx
            break

    if sheet_cnt is not None:
        src_pts = _order_points(sheet_cnt.reshape(4, 2).astype("float32"))
    else:
        h, w = img_bgr.shape[:2]
        src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")

    dst_pts = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img_bgr, M, (target_w, target_h))

# ========== 4. BUBBLE CLASSIFICATION ==========

def classify_bubble_knn(img_bgr, knn, class_names, IMG_SIZE):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    feat = (resized.astype("float32") / 255.0).flatten().reshape(1, -1)
    return class_names[knn.predict(feat)[0]]

def crop_bubble(warped_bgr, cx, cy, r):
    x1, y1 = int(cx - r), int(cy - r)
    x2, y2 = int(cx + r), int(cy + r)
    return warped_bgr[max(0, y1):y2, max(0, x1):x2]

# ========== 5. MAIN PROCESS ==========

def run_omr(image_path, key_path):
    knn, class_names, img_size = load_knn_model()
    original = cv2.imread(str(image_path))
    if original is None: return print("Image not found.")
    
    # Automatic Detection happens here
    layout_id = auto_detect_layout(original)
    
    # Select configuration based on detected layout
    cfg = TEMPLATES[layout_id]
    
    warped = warp_sheet(original, cfg["SHEET_W"], cfg["SHEET_H"])
    annotated = warped.copy()
    
    results = {}
    r = cfg["BUBBLE_RADIUS"]
    
    for q_num, centers in cfg["Q_CENTERS"].items():
        filled = [] 
        for idx, (cx, cy) in enumerate(centers):
            letter = CHOICE_LETTERS[idx]
            bubble_crop = crop_bubble(warped, cx, cy, r)
            if bubble_crop.size == 0: continue

            label = classify_bubble_knn(bubble_crop, knn, class_names, img_size)
            if label == "filled":
                filled.append(letter)
        
        if len(filled) == 1: 
            results[q_num] = {"ans": filled[0], "type": "valid"}
        elif len(filled) > 1: 
            results[q_num] = {"ans": "MULT", "type": "invalid"}
        else: 
            results[q_num] = {"ans": "BLANK", "type": "blank"}

    # Load Answer Key
    with open(key_path, "r") as f:
        full_key = [line.strip().upper() for line in f if line.strip()]
    
    score = 0
    print(f"\n--- RESULTS FOR {layout_id}-ITEM SHEET ---")
    
    num_to_score = min(len(full_key), len(cfg["Q_CENTERS"]))
    
    for i in range(1, num_to_score + 1):
        correct_ans = full_key[i-1]
        res = results.get(i, {"ans": "BLANK", "type": "blank"})
        student_ans = res["ans"]
        
        status = "CORRECT" if student_ans == correct_ans else "INCORRECT"
        if status == "CORRECT": score += 1
        
        print(f"Q{i:02d}: Key={correct_ans}, Student={student_ans} -> {status}")

        # Draw Annotations
        if correct_ans in CHOICE_LETTERS:
            k_idx = CHOICE_LETTERS.index(correct_ans)
            k_cx, k_cy = cfg["Q_CENTERS"][i][k_idx]
            cv2.circle(annotated, (int(k_cx), int(k_cy)), r+2, (0, 255, 0), 2)
        
        if student_ans != correct_ans and student_ans in CHOICE_LETTERS:
            s_idx = CHOICE_LETTERS.index(student_ans)
            s_cx, s_cy = cfg["Q_CENTERS"][i][s_idx]
            cv2.circle(annotated, (int(s_cx), int(s_cy)), r+2, (0, 0, 255), 2)
            
    print(f"\nFINAL SCORE: {score}/{num_to_score} ({(score/num_to_score)*100:.2f}%)")
    
    out_name = f"result_{Path(image_path).stem}.png"
    cv2.imwrite(out_name, annotated)
    print(f"Annotated image saved as '{out_name}'")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python omr_knn_pipeline.py <image> <key.txt>")
    else:
        run_omr(sys.argv[1], sys.argv[2])

# Change this in your NEW omr_knn_pipeline (1).py
from pathlib import Path
import cv2
import numpy as np
import joblib
import sys

from .config_omr_knn import ROOT, CHOICE_LETTERS, TEMPLATES, KNN_MODEL_PATH, KNN_META_PATH

# 1. FIXED ANALYZE_SHEET (Synchronized with views.py)
def analyze_sheet(image_path, knn, class_names, img_size):
    """
    Standardized to handle 4 arguments from views.py.
    Forcing the warp to match the specific template dimensions fixes the alignment.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Detect layout to get correct dimensions (913x483 for 50-item)
    layout_id = auto_detect_layout(img)
    cfg = TEMPLATES[layout_id]
    
    # CRITICAL: Warp to the template's designated SHEET_W and SHEET_H
    warped = warp_sheet(img, cfg["SHEET_W"], cfg["SHEET_H"])
    
    results = {}
    r = cfg["BUBBLE_RADIUS"]

    for q_num, centers in cfg["Q_CENTERS"].items():
        choice_data = {}
        filled_letters = []

        for idx, (cx, cy) in enumerate(centers):
            letter = CHOICE_LETTERS[idx]
            bubble_crop = crop_bubble(warped, cx, cy, r)
            
            if bubble_crop.size == 0:
                continue

            # Classify using the KNN components provided by views.py
            label = classify_bubble_knn(bubble_crop, knn, class_names, img_size)
            
            # Metadata for the UI dashboard
            gray = cv2.cvtColor(bubble_crop, cv2.COLOR_BGR2GRAY)
            ink = 1.0 - (np.mean(gray) / 255.0)
            
            choice_data[letter] = {"label": label, "ink": ink}
            if label == "filled":
                filled_letters.append(letter)

        # Logic for final answer determination
        if len(filled_letters) == 1:
            final_type = "valid"
            final_ans = filled_letters[0]
        elif len(filled_letters) > 1:
            final_type = "invalid"
            final_ans = "MULT"
        else:
            final_type = "blank"
            final_ans = "BLANK"

        results[q_num] = {
            "choices": choice_data,
            "final_type": final_type,
            "final_answer": final_ans
        }

    return warped, results

# 2. SCORE_SHEET
def score_sheet(results, answer_map):
    """
    Compares scan results to answer key.
    """
    total_items = len(answer_map)
    correct = 0
    per_item = {}

    for q_num, correct_ans in answer_map.items():
        res = results.get(q_num, {"final_answer": "BLANK", "final_type": "blank"})
        student_ans = res["final_answer"]
        ftype = res["final_type"]

        if ftype == "blank":
            status = "blank"
        elif ftype == "invalid":
            status = "invalid"
        elif student_ans == correct_ans:
            status = "correct"
            correct += 1
        else:
            status = "incorrect"

        per_item[q_num] = {
            "correct_answer": correct_ans,
            "student_answer": student_ans,
            "status": status
        }

    return {
        "total_items": total_items,
        "score_raw": correct,
        "score_pct": (correct / total_items * 100) if total_items > 0 else 0,
        "per_item": per_item
    }

# 3. FIXED ANNOTATE_SHEET (Prevents KeyError: 11)
def annotate_sheet(warped, summary, raw_results):
    """
    Fixed to accept 3 arguments and only draw coordinates that exist in the layout.
    """
    annotated = warped.copy()
    layout_id = auto_detect_layout(warped)
    cfg = TEMPLATES[layout_id]
    q_centers = cfg["Q_CENTERS"]
    r = cfg["BUBBLE_RADIUS"]
    
    per_item = summary.get("per_item", {})

    for i, data in per_item.items():
        # SAFETY CHECK: If the answer key is longer than the sheet capacity, skip to prevent KeyError
        if i not in q_centers:
            continue

        correct_ans = data.get("correct_answer")
        student_ans = data.get("student_answer")
        status = data.get("status")

        # Draw GREEN circle for Correct Answer
        if correct_ans in CHOICE_LETTERS:
            idx = CHOICE_LETTERS.index(correct_ans)
            cx, cy = q_centers[i][idx]
            cv2.circle(annotated, (int(cx), int(cy)), r + 3, (0, 255, 0), 2)
        
        # Draw RED circle for Wrong or Invalid Marks
        if status in ["incorrect", "invalid"] and student_ans in CHOICE_LETTERS:
            idx = CHOICE_LETTERS.index(student_ans)
            # Second safety check for the choice index
            if idx < len(q_centers[i]):
                cx, cy = q_centers[i][idx]
                cv2.circle(annotated, (int(cx), int(cy)), r + 3, (0, 0, 255), 2)
            
    return annotated
# def analyze_sheet(sheet_path: Path, knn, class_names, IMG_SIZE):
#     """
#     Main OMR + k-NN function for a single sheet.

#     INVALID rules:
#       - half-shaded / not fully shaded (weak fill) → invalid
#       - any crossed bubble in the question → invalid
#       - any bubble predicted as 'invalid' by the model → invalid
#     """
#     img = cv2.imread(str(sheet_path))
#     if img is None:
#         raise FileNotFoundError(f"Could not read sheet image: {sheet_path}")

#     warped = warp_sheet(img)
#     results = {}

#     for q_num, centers in Q_CENTERS.items():
#         choice_data = {}
#         strong_fills = []   # fully shaded bubbles
#         weak_fills = []     # half-shaded / light-shaded bubbles
#         cross_marks = []    # crossed bubbles
#         invalid_marks = []  # explicit 'invalid' from the classifier

#         for idx, (cx, cy) in enumerate(centers):
#             letter = CHOICE_LETTERS[idx]
#             bubble = crop_bubble(warped, cx, cy)

#             # --- KNN classification ---
#             label = classify_bubble_knn(bubble, knn, class_names, IMG_SIZE)

#             # --- ink / darkness score ---
#             g = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
#             g = cv2.resize(g, IMG_SIZE, interpolation=cv2.INTER_AREA)
#             g_norm = g.astype("float32") / 255.0
#             ink = 1.0 - float(g_norm.mean())   # 0 (white) .. 1 (very dark)
#             std = float(g_norm.std())

#             # NEW: if bubble is very light overall, force it to "empty"
#             effective_label = label
#             if ink < MIN_INK_FOR_ANY_MARK:
#                 effective_label = "empty"

#             # store effective label + ink (used later by annotate_sheet)
#             choice_data[letter] = {
#                 "label": effective_label,
#                 "ink": ink,
#                 "std": std,
#             }

#             # ---------- collect marks for decision ----------
#             if effective_label == "filled":
#                 # if it looks very noisy & dark → treat as invalid scribble/cross
#                 if std > STD_SCRIBBLE_THRESH and ink >= MIN_INK_FILLED:
#                     invalid_marks.append((letter, ink))
#                 else:
#                     if ink >= MIN_INK_FILLED:
#                         strong_fills.append((letter, ink))
#                     elif ink >= MIN_INK_WEAK_FILL:
#                         weak_fills.append((letter, ink))

#             elif effective_label == "crossed" and ink >= MIN_INK_CROSS:
#                 cross_marks.append((letter, ink))

#             elif effective_label == "invalid" and ink >= MIN_INK_INVALID:
#                 invalid_marks.append((letter, ink))

#         # -------- decide per question --------
#         has_cross   = len(cross_marks) > 0
#         has_weak    = len(weak_fills) > 0
#         has_invalid = len(invalid_marks) > 0

#         if len(strong_fills) == 0 and not has_cross and not has_weak and not has_invalid:
#             # No strong fills, no crosses, no weak/invalid shading → BLANK
#             final_type = "blank"
#             final_answer = None

#         elif has_cross or has_invalid:
#             # Any crossed or explicit invalid bubble → INVALID
#             final_type = "invalid"
#             final_answer = None

#         elif has_weak and len(strong_fills) == 0:
#             # Only half-shaded bubbles → INVALID
#             final_type = "invalid"
#             final_answer = None

#         elif len(strong_fills) == 1 and not has_cross and not has_weak and not has_invalid:
#             # Exactly one fully shaded bubble, clean → VALID
#             final_type = "valid"
#             final_answer = strong_fills[0][0]

#         else:
#             # Multiple strong fills OR mix of strong + weak → INVALID
#             final_type = "invalid"
#             final_answer = None

#         results[q_num] = {
#             "choices": choice_data,
#             "final_type": final_type,
#             "final_answer": final_answer,
#         }

#     return warped, results


# # ======================================================================
# #  SCORING vs ANSWER KEY
# # ======================================================================


# def score_sheet(results, answer_key):
#     """
#     Compare detected answers with the answer key.

#     answer_key: dict like {1: 'C', 2: 'A', ...}
#     results   : from analyze_sheet()

#     Returns:
#         summary dict with:
#             "total_items", "correct", "incorrect", "blank", "invalid",
#             "score_raw", "score_pct", "per_item"
#     """
#     total_items = len(answer_key)
#     correct = incorrect = blank = invalid = 0
#     per_item = {}

#     for q_num, correct_ans in answer_key.items():
#         res = results.get(q_num)

#         if res is None:
#             status = "missing"
#             student_ans = None
#             invalid += 1
#         else:
#             student_ans = res["final_answer"]
#             ftype = res["final_type"]  # 'valid' / 'blank' / 'invalid'

#             if ftype == "blank":
#                 status = "blank"
#                 blank += 1
#             elif ftype == "invalid":
#                 status = "invalid"
#                 invalid += 1
#             else:  # valid
#                 if student_ans == correct_ans:
#                     status = "correct"
#                     correct += 1
#                 else:
#                     status = "incorrect"
#                     incorrect += 1

#         per_item[q_num] = {
#             "correct_answer": correct_ans,
#             "student_answer": student_ans,
#             "status": status,
#         }

#     score_raw = correct
#     score_pct = (correct / total_items) * 100 if total_items else 0.0

#     summary = {
#         "total_items": total_items,
#         "correct": correct,
#         "incorrect": incorrect,
#         "blank": blank,
#         "invalid": invalid,
#         "score_raw": score_raw,
#         "score_pct": score_pct,
#         "per_item": per_item,
#     }
#     return summary


# # ======================================================================
# #  ANNOTATION (RED/GREEN CIRCLES)
# # ======================================================================


# def annotate_sheet(warped: np.ndarray, summary, results):
#     """
#     Draw circles for each question:

#     - GREEN circle on the correct answer (answer key) for every item.
#     - YELLOW circle on the correct answer if:
#         * there are 2 or more marked bubbles in that item AND
#         * one of the marked bubbles is the correct answer
#       (i.e., multiple-answers but one is the key).
#     - RED circle on any bubble that is considered "marked":
#         * crossed with enough ink
#         * filled with enough ink
#         * weak fill with enough ink
#         * label == 'invalid' with enough ink

#       Exception: if the item is CORRECT and the only mark is the correct answer,
#       we draw only GREEN (no red or yellow).
#     """
#     overlay = warped.copy()

#     GREEN  = (0, 255, 0)
#     RED    = (0, 0, 255)
#     YELLOW = (0, 255, 255)

#     for q_num, item in summary["per_item"].items():
#         correct_opt = item["correct_answer"]      # e.g. 'A'
#         status      = item["status"]              # 'correct','incorrect','blank','invalid','missing'

#         res     = results.get(q_num, {})
#         choices = res.get("choices", {})

#         # ---------- compute all "marked" bubbles ----------
#         marked_letters = []
#         for letter, data in choices.items():
#             if letter not in CHOICE_LETTERS:
#                 continue
#             lab = data.get("label", "")
#             ink = float(data.get("ink", 0.0))

#             is_marked = False

#             # ignore very light bubbles entirely
#             if ink < MIN_INK_FOR_ANY_MARK:
#                 is_marked = False
#             else:
#                 if lab == "crossed" and ink >= MIN_INK_CROSS:
#                     is_marked = True
#                 elif lab == "filled":
#                     # strong or weak fill both treated as a visible mark
#                     if ink >= MIN_INK_FILLED:
#                         is_marked = True
#                     elif ink >= MIN_INK_WEAK_FILL:
#                         is_marked = True
#                 elif lab == "invalid" and ink >= MIN_INK_INVALID:
#                     is_marked = True

#             if is_marked:
#                 marked_letters.append(letter)

#         # ---------- GREEN (correct answer from key) ----------
#         if correct_opt in CHOICE_LETTERS:
#             corr_idx = CHOICE_LETTERS.index(correct_opt)
#             corr_cx, corr_cy = Q_CENTERS[q_num][corr_idx]
#             cv2.circle(
#                 overlay,
#                 (int(corr_cx), int(corr_cy)),
#                 DRAW_RADIUS,
#                 GREEN,
#                 2,
#             )

#         # If item is cleanly correct (only one mark and it's the key) → no red/yellow
#         if status == "correct" and len(marked_letters) == 1 and correct_opt in marked_letters:
#             continue

#         # ---------- decide which bubbles get YELLOW vs RED ----------
#         yellow_letters = []
#         red_letters = []

#         # Multi-answer or messy case where correct is among marks
#         if correct_opt in marked_letters and len(marked_letters) >= 2:
#             yellow_letters.append(correct_opt)
#             red_letters = [l for l in marked_letters if l != correct_opt]
#         else:
#             red_letters = list(marked_letters)

#         # ---------- draw YELLOW on correct bubble (if any) ----------
#         for letter in yellow_letters:
#             idx = CHOICE_LETTERS.index(letter)
#             cx, cy = Q_CENTERS[q_num][idx]
#             cv2.circle(
#                 overlay,
#                 (int(cx), int(cy)),
#                 DRAW_RADIUS,
#                 YELLOW,
#                 2,
#             )

#         # ---------- RED on all other marked bubbles ----------
#         for letter in red_letters:
#             idx = CHOICE_LETTERS.index(letter)
#             cx, cy = Q_CENTERS[q_num][idx]
#             cv2.circle(
#                 overlay,
#                 (int(cx), int(cy)),
#                 DRAW_RADIUS,
#                 RED,
#                 2,
#             )

#     return overlay
