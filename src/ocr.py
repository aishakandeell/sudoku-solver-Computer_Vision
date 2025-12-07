import os
import cv2
import numpy as np

GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9
DIGIT_SIZE = 28  

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "data", "template")

_digit_templates_cache = None

def normalize_digit_image_gray(img_gray):
    if len(img_gray.shape) == 3:
        gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_gray.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = binary.shape
    margin = int(0.10 * h)

    binary[0:margin, :] = 0
    binary[h - margin:h, :] = 0
    binary[:, 0:margin] = 0
    binary[:, w - margin:w] = 0

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)

        if cw * ch > 0.01 * (h * w):
            digit_roi = blur[y:y + ch, x:x + cw]
        else:
            digit_roi = blur
    else:
        digit_roi = blur

    digit_resized = cv2.resize(digit_roi, (DIGIT_SIZE, DIGIT_SIZE))
    return digit_resized

def load_digit_templates():
    global _digit_templates_cache
    if _digit_templates_cache is not None:
        return _digit_templates_cache

    templates = {d: [] for d in range(1, 10)}

    if not os.path.isdir(TEMPLATES_DIR):
        print(f"[OCR] Warning: templates dir not found: {TEMPLATES_DIR}")
        _digit_templates_cache = templates
        return templates

    for fname in os.listdir(TEMPLATES_DIR):
        lower = fname.lower()
        if not (lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg")):
            continue

        base = fname.split("_")[0].split(".")[0]
        base = base.lstrip("0")  # "01" -> "1"
        if base == "" or not base.isdigit():
            continue

        digit = int(base)
        if digit < 1 or digit > 9:
            continue

        path = os.path.join(TEMPLATES_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[OCR] Warning: could not read template {path}")
            continue

        norm = normalize_digit_image_gray(img)
        templates[digit].append(norm)

    for d in range(1, 10):
        if not templates[d]:
            print(f"[OCR] Warning: no templates found for digit {d}")

    _digit_templates_cache = templates
    print("[OCR] Templates per digit:", {d: len(v) for d, v in templates.items()})
    return templates

def split_into_cells(warped):
    if len(warped.shape) == 3:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped.copy()

    cells = []
    for r in range(9):
        row_cells = []
        for c in range(9):
            y1 = r * CELL_SIZE
            y2 = (r + 1) * CELL_SIZE
            x1 = c * CELL_SIZE
            x2 = (c + 1) * CELL_SIZE
            cell = gray[y1:y2, x1:x2]
            row_cells.append(cell)
        cells.append(row_cells)
    return cells

def match_digit_to_templates(norm_cell, templates):
    best_digit = 0
    best_score = -1.0

    SCORE_THRESHOLD = 0.55

    a = norm_cell.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)

    for d in range(1, 10):
        for tmpl in templates.get(d, []):
            b = tmpl.astype(np.float32)
            b = (b - b.mean()) / (b.std() + 1e-6)
            res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_score:
                best_score = score
                best_digit = d

    if best_score < SCORE_THRESHOLD:
        return 0
    return best_digit

def cell_has_digit(norm_img):
    blur = cv2.GaussianBlur(norm_img, (3, 3), 0)

    _, bin_inv = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = bin_inv.shape
    margin = int(0.15 * h)
    inner = bin_inv[margin:h - margin, margin:w - margin]

    white = cv2.countNonZero(inner)
    return white > 40

def recognize_board(warped):
    templates = load_digit_templates()
    cells = split_into_cells(warped)

    board = []
    for row_cells in cells:
        row_vals = []
        for cell in row_cells:
            blur = cv2.GaussianBlur(cell, (3, 3), 0)
            _, bin_inv = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            white = cv2.countNonZero(bin_inv)

            if white < 18:      
                row_vals.append(0)
                continue
            norm = normalize_digit_image_gray(cell)
            d = match_digit_to_templates(norm, templates)
            row_vals.append(int(d))
        board.append(row_vals)

    return np.array(board, dtype=int)

