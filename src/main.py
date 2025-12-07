import os
import cv2
from .pipeline import run_pipeline
from .ocr import recognize_board      
from .solver import solve_sudoku   

def main():
    input_dir = os.path.join("data", "input")
    image_name = "01.jpg"  
    image_path = os.path.join(input_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    print("Using input image:", image_path)
    results = run_pipeline(image_path)
    warped = results["warped"]

    results = run_pipeline(image_path)
    warped = results["warped"]

    # ---------- OCR: read digits ----------
    board = recognize_board(warped)
    print("[OCR] Recognized board:")
    print(board)

    # ---------- Solve the sudoku ----------
    solution = board.copy()
    if solve_sudoku(solution):
        print("[SOLVER] Solved board:")
        print(solution)
    else:
        print("[SOLVER] No valid solution found (OCR might be wrong).")

    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    warped_path = os.path.join(output_dir, "07_warped.png")
    cv2.imwrite(warped_path, warped)
    print("Pipeline finished.")
    print(f"Final warped grid saved to: {warped_path}")
    cv2.imshow("Sudoku - warped grid", warped)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
