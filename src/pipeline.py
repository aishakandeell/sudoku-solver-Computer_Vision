import cv2
import numpy as np
import os

from .utils import sort_corners_clockwise, save_debug_image
from .ocr import recognize_board

OUTPUT_DIR = os.path.join("data", "output")

def run_pipeline(image_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = original.shape[:2]
    max_side = max(w, h)
    scale = 800.0 / max_side

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        original_resized = cv2.resize(original, (new_w, new_h))
    else:
        original_resized = original.copy()

    save_debug_image("01_original.png", original_resized)

    gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)

    save_debug_image("02_gray.png", gray)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    save_debug_image("03_blur.png", blurred)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    binary_inv = cv2.bitwise_not(binary)
    save_debug_image("04_binary_inverted.png", binary_inv)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    save_debug_image("05_morphed.png", morphed)

    preprocessed_vis = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours found – check preprocessing or image quality.")

    largest = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        print(f"Warning: expected 4 points, got {len(approx)}. Using rotated bounding box instead.")
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)

    else:
        corners = approx.reshape(4, 2).astype(np.float32)
    corners = sort_corners_clockwise(corners)

    contour_vis = original_resized.copy()
    cv2.drawContours(contour_vis, [largest], -1, (0, 255, 0), 2)

    for i, (x, y) in enumerate(corners):
        cv2.circle(contour_vis, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(
            contour_vis,
            str(i),
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    save_debug_image("06_contour_corners.png", contour_vis)

    GRID_SIZE = 450  

    dst_pts = np.array([
        [0, 0],
        [GRID_SIZE - 1, 0],
        [GRID_SIZE - 1, GRID_SIZE - 1],
        [0, GRID_SIZE - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(original_resized, M, (GRID_SIZE, GRID_SIZE))

    save_debug_image("07_warped.png", warped)

    board = recognize_board(warped)
    print("[OCR] Recognized board:")
    print(board)

    results = {
        "original": original_resized,
        "preprocessed": preprocessed_vis,
        "contour": contour_vis,
        "warped": warped,
        "board": board,   
    }

    return results
