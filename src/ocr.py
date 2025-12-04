import cv2
import numpy as np

GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9
DIGIT_SIZE = 28  

_digit_templates_cache = None


def _generate_digit_template(digit: int, size: int = DIGIT_SIZE) -> np.ndarray:
    """
    Create a synthetic template image for the given digit using cv2.putText,
    White digit on black background.
    """
    img = np.zeros((size, size), dtype=np.uint8)  # black background

    text = str(digit)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    # Center the digit in the square
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (size - tw) // 2
    y = (size + th) // 2

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        255,          
        thickness,
        cv2.LINE_AA,
    )
    return img

def load_digit_templates():
    """
    Create templates for digits 1..9 in memory (no files needed),
    cache them, and return as a dict {digit: template_img}.
    """
    global _digit_templates_cache
    if _digit_templates_cache is not None:
        return _digit_templates_cache

    templates = {}
    for d in range(1, 10):
        templates[d] = _generate_digit_template(d)

    _digit_templates_cache = templates
    return templates

def split_into_cells(warped):
    """
    Split the 450x450 warped grid into 81 (9x9) grayscale cells.
    """
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


def is_cell_empty(cell_gray):
    """
    check if cell is empty by checking pixels.
    """
    blur = cv2.GaussianBlur(cell_gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = binary.shape
    margin = int(0.1 * h)
    inner = binary[margin:h - margin, margin:w - margin]

    non_zero = cv2.countNonZero(inner)
    total = inner.size
    if total == 0:
        return True

    fill_ratio = non_zero / total
    return fill_ratio < 0.01


def extract_digit_image(cell_gray):
    """
    From a non-empty cell, isolate the largest connected component (the digit),
    remove the border lines, and resize it to DIGIT_SIZE x DIGIT_SIZE.
    """
    blur = cv2.GaussianBlur(cell_gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    #dilation
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    h, w = binary.shape
    margin = int(0.1 * h)
    binary[0:margin, :] = 0
    binary[h - margin:h, :] = 0
    binary[:, 0:margin] = 0
    binary[:, w - margin:w] = 0

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # Very tiny component → probably noise
    if cw * ch < 0.01 * (h * w):
        return None

    digit = binary[y:y + ch, x:x + cw]
    digit_resized = cv2.resize(digit, (DIGIT_SIZE, DIGIT_SIZE))
    return digit_resized


def match_digit_to_templates(digit_img, templates):
    """
    Compare the digit image to each template and return the best matching digit.
    Uses sum of absolute differences as a simple distance.
    """
    best_digit = 0
    best_score = None

    for d, tmpl in templates.items():
        # ensure same size
        resized = cv2.resize(
            digit_img, (tmpl.shape[1], tmpl.shape[0])
        )
        resized_blur = cv2.GaussianBlur(resized, (3, 3), 0)
        tmpl_blur = cv2.GaussianBlur(tmpl, (3, 3), 0)

        diff = cv2.absdiff(resized_blur, tmpl_blur)

        score = cv2.sumElems(diff)[0]  # smaller = more similar

        if best_score is None or score < best_score:
            best_score = score
            best_digit = d

    return best_digit


def recognize_board(warped):
    """
    from warped 450x450 image, return a 9x9 numpy array
    where digits are 1-9 and empty cells are 0.
    """
    templates = load_digit_templates()
    cells = split_into_cells(warped)

    board = []
    for row_cells in cells:
        row_vals = []
        for cell in row_cells:
            if is_cell_empty(cell):
                row_vals.append(0)
                continue

            digit_img = extract_digit_image(cell)
            if digit_img is None or not templates:
                row_vals.append(0)
            else:
                d = match_digit_to_templates(digit_img, templates)
                row_vals.append(int(d))
        board.append(row_vals)

    return np.array(board, dtype=int)
