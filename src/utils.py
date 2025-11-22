import cv2
import numpy as np
import os

OUTPUT_DIR = os.path.join("data", "output")

def save_debug_image(filename: str, image):
    """
    Save intermediate images to the data/output folder.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, image)

def sort_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    pts: array of shape (4, 2)
    Returns points ordered:
        [top-left, top-right, bottom-right, bottom-left]
    """

    # Use sum and diff trick to identify corners
    s = pts.sum(axis=1)          # x + y
    diff = np.diff(pts, axis=1)  # y - x

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return ordered
