# analysis/omr_sheet_detection.py
"""
Detect and warp your Bubble Vision sheet.

- Reads the uploaded image with EXIF orientation applied
- Forces it to landscape if needed
- Tries strict 4-corner-box crop + light zoom
- If that fails, does a WIDE center crop (90% of the image)
- Always returns a (OUT_H, OUT_W, 3) BGR image
"""

import cv2
import numpy as np
from PIL import Image, ImageOps  # handle EXIF orientation

# Final OMR size
OUT_W = 913
OUT_H = 483

# When 4 boxes ARE found, how much margin to cut inside them (for zoom)
INNER_MARGIN_RATIO = 0.08   # 8% each side (lighter than before)


# ---------- image loading with correct orientation ----------

def _load_image_with_orientation(path):
    """
    Load image using Pillow so EXIF orientation is respected,
    then convert to OpenCV BGR array and force landscape.
    """
    pil_img = Image.open(path)

    # Apply EXIF orientation
    pil_img = ImageOps.exif_transpose(pil_img)

    pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # If still portrait (height > width), rotate to landscape
    h, w = bgr.shape[:2]
    if h > w:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

    return bgr

def _crop_inside_boxes(img):
    """
    Strict case: detect 4 corner boxes and warp the ENTIRE sheet
    (including the boxes) to OUT_W x OUT_H.

    This will make the result look like your template sheet:
    full border, NAME/SECTION, BUBBLE VISION, and 4 black boxes.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = thresh.shape[:2]
    img_area = float(h * w)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        # keep reasonable size relative to whole image
        if area < img_area * 0.0005 or area > img_area * 0.05:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, ww, hh = cv2.boundingRect(approx)
        aspect = ww / float(hh + 1e-6)
        if not (0.6 <= aspect <= 1.4):
            continue

        M = cv2.moments(approx)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    if len(centers) < 4:
        raise RuntimeError("Not enough square-like contours for 4 corners.")

    pts = np.array(centers, dtype="float32")

    # Split into left/right by x to find TL, BL, TR, BR
    xs = pts[:, 0]
    median_x = np.median(xs)
    left_pts = pts[xs <= median_x]
    right_pts = pts[xs > median_x]

    if len(left_pts) == 0 or len(right_pts) == 0:
        raise RuntimeError("Cannot split corner candidates into left/right.")

    tl = left_pts[np.argmin(left_pts[:, 1])]  # smallest y on left  = top-left
    bl = left_pts[np.argmax(left_pts[:, 1])]  # largest y on left  = bottom-left
    tr = right_pts[np.argmin(right_pts[:, 1])]  # smallest y on right = top-right
    br = right_pts[np.argmax(right_pts[:, 1])]  # largest y on right = bottom-right

    src = np.float32([tl, tr, br, bl])
    dst = np.float32([
        [0, 0],
        [OUT_W - 1, 0],
        [OUT_W - 1, OUT_H - 1],
        [0, OUT_H - 1],
    ])

    # Perspective warp whole sheet (NO inner crop)
    M = cv2.getPerspectiveTransform(src, dst)
    warped_full = cv2.warpPerspective(img, M, (OUT_W, OUT_H))

    # This already matches your template size, so just return it
    return warped_full

# ---------- fallback crop ----------

def _center_crop(img, crop_ratio=0.9):
    """
    Fallback: wide center crop & resize.

    crop_ratio = fraction of the dimension to keep.
    0.9 = keep central 90% of width and height.
    This avoids over-zooming like in your screenshot.
    """
    h, w = img.shape[:2]
    r = crop_ratio

    cw = int(w * r)
    ch = int(h * r)

    x1 = (w - cw) // 2
    y1 = (h - ch) // 2
    x2 = x1 + cw
    y2 = y1 + ch

    cropped = img[y1:y2, x1:x2]
    final = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    return final


# ---------- public API ----------

def detect_and_warp_sheet_from_file(input_path):
    img = _load_image_with_orientation(input_path)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")

    try:
        processed = _crop_inside_boxes(img)
    except Exception:
        processed = _center_crop(img, crop_ratio=0.9)

    # Make the sheet “downward”, not upside-down
    processed = cv2.rotate(processed, cv2.ROTATE_180)

    return processed