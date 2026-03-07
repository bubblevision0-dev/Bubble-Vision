# analysis/omr_utils.py
import cv2
import numpy as np

CHOICES = ["A", "B", "C", "D", "E"]  # adjust if you use fewer/more

def _read_uploaded_image(uploaded_file):
    """
    Convert Django InMemoryUploadedFile into a cv2 image (numpy array).
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def detect_bubble_answers(uploaded_file, num_questions=50, num_choices=5):
    """
    Very basic OMR:
    - Assume the bubble area is roughly a big grid (questions x choices).
    - Count dark pixels in each cell.
    - Pick the darkest bubble per row.
    - If not dark enough, treat the question as blank.

    Returns: a list like ["A","C","B",None,...] with length num_questions.
    """
    img = _read_uploaded_image(uploaded_file)
    if img is None:
        return [None] * num_questions

    # Resize to a standard size to make math easier
    img = cv2.resize(img, (1000, 1400))  # tweak as needed

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary image: bubbles = dark, background = light
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thresh.shape

    # Use an inner region (ignore borders a bit)
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)
    roi = thresh[margin_y:h-margin_y, margin_x:w-margin_x]
    rh, rw = roi.shape

    # Grid size
    cell_h = rh // num_questions
    cell_w = rw // num_choices

    answers = []

    for q in range(num_questions):
        row_start = q * cell_h
        row_end = (q + 1) * cell_h

        bubble_strengths = []

        for c in range(num_choices):
            col_start = c * cell_w
            col_end = (c + 1) * cell_w

            cell = roi[row_start:row_end, col_start:col_end]

            # count dark pixels (because we used THRESH_BINARY_INV)
            filled_pixels = cv2.countNonZero(cell)

            bubble_strengths.append(filled_pixels)

        # Decide which bubble is "marked"
        best_idx = int(np.argmax(bubble_strengths))
        best_value = bubble_strengths[best_idx]

        # Heuristic: require at least some dark pixels to count as filled
        # You can tune the 0.15 value or log the strengths to debug
        total_pixels = cell_h * cell_w
        if best_value > total_pixels * 0.15:
            answers.append(CHOICES[best_idx])
        else:
            answers.append(None)

    return answers
