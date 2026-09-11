import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

#Encuentra la siguiente celda vacía
def find_empty_cell(puzzle):
    for row in range(9):
        for col in range(9):
            if puzzle[row][col] == 0:
                return (row, col)
    return None
#Verifica si un número puede colocarse en una celda
def is_valid(puzzle, row, col, num):
    if num in puzzle[row]:
        return False
    
    if num in [puzzle[i][col] for i in range(9)]:
        return False
    
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for i in range(box_row, box_row+3):
        for j in range(box_col, box_col+3):
            if puzzle[i][j] == num:
                return False
    return True

#Resuelve el Sudoku usando backtracking
def solve_sudoku(puzzle):
    empty_cell = find_empty_cell(puzzle)
    if not empty_cell:
        return True
        
    row, col = empty_cell
    for num in range(1, 10):
        if is_valid(puzzle, row, col, num):
            puzzle[row][col] = num
            if solve_sudoku(puzzle):
                return True
            puzzle[row][col] = 0
    return False

def solve_with_timeout(grid, timeout=5):
    """Ejecuta solve_sudoku con un timeout"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(solve_sudoku, grid)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return False