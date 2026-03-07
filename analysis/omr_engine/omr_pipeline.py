# omr_pipeline.py
"""
Full OMR pipeline for your Bubble Vision sheet.

Steps:
  1. Load raw sheet image.
  2. Detect 4 corner boxes and warp to fixed size (SHEET_W x SHEET_H).
  3. For each question, crop bubble patches using Q_CENTERS.
  4. Use trained CNN (via bubble_inference) to classify bubbles.
  5. Apply question-level rules -> blank / answered / invalid.
  6. (Optional) Grade against an answer key.

You can call:
  - process_sheet_image(image_path)     -> per-question detections
  - grade_sheet_image(image_path, key) -> graded result dict
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import os
import sys

# --- make imports work BOTH in Django (package) and as a script ---
try:
    # when imported as analysis.omr_engine.omr_pipeline
    from .q_centers import SHEET_W, SHEET_H, Q_CENTERS
    from .bubble_inference import classify_question, classify_patch
except ImportError:
    # when run directly: python omr_pipeline.py
    from q_centers import SHEET_W, SHEET_H, Q_CENTERS
    from bubble_inference import classify_question, classify_patch


# ---------- CONFIG ----------

PATCH_RADIUS = 20          # how big to crop around bubble center
OPTION_LETTERS = "ABCDE"   # 5 options per question


# ---------- IMAGE WARPING (CORNER DETECTION) ----------

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    pts: array of shape (4, 2)
    """
    pts = np.asarray(pts, dtype="float32")
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]

    # sort top two by x
    if top_two[0, 0] < top_two[1, 0]:
        tl, tr = top_two
    else:
        tr, tl = top_two

    # sort bottom two by x
    if bottom_two[0, 0] < bottom_two[1, 0]:
        bl, br = bottom_two
    else:
        br, bl = bottom_two

    return np.array([tl, tr, br, bl], dtype="float32")


def find_corner_centers(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect the 4 black corner boxes and return their centers,
    ordered as (tl, tr, br, bl).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) < 4:
        return None

    # largest 4 contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    centers = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2.0
        cy = y + h / 2.0
        centers.append((cx, cy))

    if len(centers) != 4:
        return None

    return order_points(np.array(centers, dtype="float32"))


def warp_sheet(img_bgr_or_path: str | Path | np.ndarray) -> np.ndarray:
    """
    Load and warp a raw sheet image to fixed size (SHEET_W x SHEET_H).

    Returns:
        aligned_gray: grayscale warped image of shape (SHEET_H, SHEET_W)

    Raises:
        ValueError if sheet cannot be warped.
    """
    if isinstance(img_bgr_or_path, (str, Path)):
        img_bgr = cv2.imread(str(img_bgr_or_path))
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {img_bgr_or_path}")
    else:
        img_bgr = img_bgr_or_path

    src_pts = find_corner_centers(img_bgr)
    if src_pts is None:
        raise ValueError("Could not detect 4 corner boxes; check sheet / lighting.")

    dst_pts = np.float32([
        [0, 0],
        [SHEET_W - 1, 0],
        [SHEET_W - 1, SHEET_H - 1],
        [0, SHEET_H - 1],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_bgr = cv2.warpPerspective(img_bgr, M, (SHEET_W, SHEET_H))

    aligned_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    return aligned_gray


# ---------- BUBBLE CROPPING & QUESTION CLASSIFICATION ----------

def crop_bubble_patch(
    aligned_gray: np.ndarray,
    cx: float,
    cy: float,
    radius: int = PATCH_RADIUS,
) -> np.ndarray:
    """Crop a square patch around a bubble center (cx, cy)."""
    h, w = aligned_gray.shape[:2]
    x1 = int(cx - radius)
    y1 = int(cy - radius)
    x2 = int(cx + radius)
    y2 = int(cy + radius)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    patch = aligned_gray[y1:y2, x1:x2]
    return patch


def classify_all_questions(
    aligned_gray: np.ndarray,
    conf_thr: float = 0.6,
) -> Dict[int, Dict[str, Any]]:
    """
    For an aligned grayscale sheet, classify all questions.

    Returns:
        results: dict mapping question_no -> {
            "status": "blank" | "answered" | "invalid",
            "chosen": "A"/"B"/.../None,
            "marked": [letters],
            "details": {... per bubble ...}
        }
    """
    results: Dict[int, Dict[str, Any]] = {}

    for q_no, centers in Q_CENTERS.items():
        option_patches = {}

        for idx, (cx, cy) in enumerate(centers):
            if idx >= len(OPTION_LETTERS):
                break
            letter = OPTION_LETTERS[idx]
            patch = crop_bubble_patch(aligned_gray, cx, cy, PATCH_RADIUS)
            option_patches[letter] = patch

        q_result = classify_question(option_patches, conf_thr=conf_thr)
        results[q_no] = q_result

    return results


# ---------- DEBUG HELPERS ----------

def debug_draw_centers(aligned_gray: np.ndarray, out_path: str = "debug_centers.png"):
    """
    Draw all Q_CENTERS on top of the aligned sheet and save as image.
    """
    img_color = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR)

    for q_no, centers in Q_CENTERS.items():
        for idx, (cx, cy) in enumerate(centers):
            cx_i, cy_i = int(cx), int(cy)
            cv2.circle(img_color, (cx_i, cy_i), 8, (0, 0, 255), 2)

            if idx < len(OPTION_LETTERS):
                text = f"{q_no}{OPTION_LETTERS[idx]}"
            else:
                text = str(q_no)
            cv2.putText(
                img_color,
                text,
                (cx_i + 5, cy_i - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(out_path, img_color)
    print(f"[DEBUG] saved overlay with centers to {os.path.abspath(out_path)}")


def debug_dump_some_patches(
    aligned_gray: np.ndarray,
    out_dir: str = "debug_patches",
    max_per_class: int = 5,
):
    """
    Crop some bubble patches from the aligned sheet, classify them,
    print predictions, and save patches to a folder.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"crossed": 0, "filled": 0, "empty": 0, "invalid": 0}

    for q_no, centers in Q_CENTERS.items():
        for idx, (cx, cy) in enumerate(centers):
            patch = crop_bubble_patch(aligned_gray, cx, cy, PATCH_RADIUS)
            label, conf, probs = classify_patch(patch)

            if counts.get(label, 0) < max_per_class:
                counts[label] = counts.get(label, 0) + 1
                fname = f"q{q_no:02d}_{OPTION_LETTERS[idx]}_{label}_{counts[label]}.png"
                cv2.imwrite(str(out_dir / fname), patch)
                print(
                    f"Q{q_no:02d}{OPTION_LETTERS[idx]} -> "
                    f"{label} ({conf:.2f}) saved as {fname}"
                )


# ---------- GRADING ----------

def grade_sheet(
    aligned_gray: np.ndarray,
    answer_key: Dict[int, str],
    conf_thr: float = 0.6,
) -> Dict[str, Any]:
    """
    Grade a sheet given an aligned grayscale image and an answer key.

    answer_key: {question_no: "A"/"B"/...}
    """
    q_results = classify_all_questions(aligned_gray, conf_thr=conf_thr)

    stats = {
        "total_items": len(answer_key),
        "n_correct": 0,
        "n_incorrect": 0,
        "n_blank": 0,
        "n_invalid": 0,
    }

    per_item: Dict[int, Dict[str, Any]] = {}

    for q_no, correct_letter in answer_key.items():
        q_res = q_results.get(q_no, None)
        if q_res is None:
            per_item[q_no] = {
                "correct": correct_letter,
                "chosen": None,
                "is_correct": None,
                "status": "missing",
                "details": {},
            }
            continue

        status = q_res["status"]
        chosen = q_res["chosen"]

        if status == "blank":
            stats["n_blank"] += 1
            is_correct = None
        elif status == "invalid":
            stats["n_invalid"] += 1
            is_correct = None
        else:  # "answered"
            if chosen == correct_letter:
                stats["n_correct"] += 1
                is_correct = True
            else:
                stats["n_incorrect"] += 1
                is_correct = False

        per_item[q_no] = {
            "correct": correct_letter,
            "chosen": chosen,
            "is_correct": is_correct,
            "status": status,
            "details": q_res,
        }

    if stats["total_items"] > 0:
        stats["percent_score"] = (
            stats["n_correct"] / stats["total_items"] * 100.0
        )
    else:
        stats["percent_score"] = 0.0

    return {
        "stats": stats,
        "per_item": per_item,
    }


# ---------- CONVENIENCE ----------

def process_sheet_image(
    image_path: str | Path,
    conf_thr: float = 0.6,
) -> Dict[int, Dict[str, Any]]:
    aligned_gray = warp_sheet(image_path)
    return classify_all_questions(aligned_gray, conf_thr=conf_thr)


def grade_sheet_image(
    image_path: str | Path,
    answer_key: Dict[int, str],
    conf_thr: float = 0.6,
) -> Dict[str, Any]:
    aligned_gray = warp_sheet(image_path)
    return grade_sheet(aligned_gray, answer_key, conf_thr=conf_thr)


# optional CLI testing (not needed in Django)
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run OMR on a bubble sheet image.")
    parser.add_argument("image", help="Path to raw sheet image (photo/scan).")
    parser.add_argument("--answer-key", help='JSON file {"1":"C", "2":"A", ...}')
    parser.add_argument("--conf-thr", type=float, default=0.6)
    parser.add_argument("--debug-centers", action="store_true")
    parser.add_argument("--debug-patches", action="store_true")
    args = parser.parse_args()

    img_path = Path(args.image)
    aligned_gray = warp_sheet(img_path)

    if args.debug_centers:
        debug_draw_centers(aligned_gray, out_path="debug_centers.png")
    if args.debug_patches:
        debug_dump_some_patches(aligned_gray, out_dir="debug_patches")

    if args.answer_key:
        with open(args.answer_key, "r", encoding="utf-8") as f:
            raw_key = json.load(f)
        answer_key = {int(k): v for k, v in raw_key.items()}
        result = grade_sheet(aligned_gray, answer_key, conf_thr=args.conf_thr)
        print(result["stats"])
    else:
        res = classify_all_questions(aligned_gray, conf_thr=args.conf_thr)
        for q_no in sorted(res.keys()):
            print(f"Q{q_no:02d}:", res[q_no])
