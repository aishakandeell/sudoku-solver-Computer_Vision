import cv2
import os

from .pipeline import run_pipeline

def main():
    # 1. Choose input image
    input_dir = os.path.join("data", "input")
    image_name = "sudoku1.jpg"   # make sure this matches your file
    image_path = os.path.join(input_dir, image_name)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # 2. Run the pipeline
    results = run_pipeline(image_path)

    # 3. Simple viewer for steps
    print("Press:")
    print("  1: original image")
    print("  2: preprocessed image")
    print("  3: contour + corners")
    print("  4: warped (straightened) grid")
    print("  q: quit")

    # start by showing warped
    cv2.imshow("Sudoku Step", results["warped"])

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('1'):
            cv2.imshow("Sudoku Step", results["original"])
        elif key == ord('2'):
            cv2.imshow("Sudoku Step", results["preprocessed"])
        elif key == ord('3'):
            cv2.imshow("Sudoku Step", results["contour"])
        elif key == ord('4'):
            cv2.imshow("Sudoku Step", results["warped"])
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
