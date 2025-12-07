import numpy as np

def find_empty_cell(board: np.ndarray):
    """
    Find the next empty cell (value 0).
    Returns (row, col) or None if the board is full.
    """
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                return r, c
    return None

def is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:
    """
    Check if placing 'num' at (row, col) is valid
    according to Sudoku rules.
    """
    if num in board[row, :]:
        return False

    if num in board[:, col]:
        return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    if num in board[box_row:box_row + 3, box_col:box_col + 3]:
        return False

    return True

def solve_sudoku(board: np.ndarray) -> bool:
    """
    Solve the Sudoku puzzle in-place using backtracking.
    'board' is a 9x9 numpy array with 0 for empty cells.
    Returns True if a solution is found, False otherwise.
    """
    empty = find_empty_cell(board)
    if empty is None:
        return True

    row, col = empty

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row, col] = num

            if solve_sudoku(board):
                return True
            board[row, col] = 0
    return False
