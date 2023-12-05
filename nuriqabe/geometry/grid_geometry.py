__all__ = [
    "direct_path_between_two_cells",
    "is_continuous_path",
    "find_paths_manhattan_distance",
    "count_manhattan_distance_paths",
]


import numpy as np
from scipy.special import binom


def direct_path_between_two_cells(cell_0, cell_1):
    if cell_0[0] == cell_1[0]:
        return [
            (cell_0[0], y)
            for y in range(min(cell_0[1], cell_1[1]), max(cell_0[1], cell_1[1]) + 1)
        ]
    elif cell_0[1] == cell_1[1]:
        return [
            (x, cell_0[1])
            for x in range(min(cell_0[0], cell_1[0]), max(cell_0[0], cell_1[0]) + 1)
        ]
    else:
        return []


def is_continuous_path(array) -> bool:
    rows, cols = array.shape

    # Helper function for DFS
    def depth_first_search(row, col):
        # Check if the indices are within bounds and the value is True
        if 0 <= row < rows and 0 <= col < cols and array[row, col]:
            # Mark the current cell as visited
            array[row, col] = False

            # Recursively explore neighboring cells
            depth_first_search(row + 1, col)
            depth_first_search(row - 1, col)
            depth_first_search(row, col + 1)
            depth_first_search(row, col - 1)

    # Find the first True value to start DFS
    for i in range(rows):
        for j in range(cols):
            if array[i, j]:
                depth_first_search(i, j)  # Start DFS from the first True value
                return not np.any(array)  # Check if all True values have been visited

    return False


def find_paths_manhattan_distance(start: tuple[int, int], end: tuple[int, int]):
    x1, y1 = start
    x2, y2 = end

    distance_x = abs(x2 - x1)
    distance_y = abs(y2 - y1)

    # Initialize a 2D array to store the number of paths
    paths = [[0] * (distance_y + 1) for _ in range(distance_x + 1)]

    # Initialize a 2D array to store the paths
    path_matrix = [[""] * (distance_y + 1) for _ in range(distance_x + 1)]

    # There is only one way to reach any cell in the first row or first column
    for i in range(distance_x + 1):
        paths[i][0] = 1
        path_matrix[i][0] = "R" * i

    for j in range(distance_y + 1):
        paths[0][j] = 1
        path_matrix[0][j] = "D" * j

    # Fill in the DP arrays
    for i in range(1, distance_x + 1):
        for j in range(1, distance_y + 1):
            paths[i][j] = paths[i - 1][j] + paths[i][j - 1]
            path_matrix[i][j] = "R" * i + "D" * j

    # The result is the number of paths to the bottom-right cell
    num_paths = paths[distance_x][distance_y]
    path = path_matrix[distance_x][distance_y]

    return num_paths, path


def count_manhattan_distance_paths(
    cell_0: tuple[int, int],
    cell_1: tuple[int, int],
) -> float:
    # Difference between the 'x'
    # coordinates of the given points
    x1, y1 = cell_0
    x2, y2 = cell_1
    m = abs(x1 - x2)

    # Difference between the 'y'
    # coordinates of the given points
    n = abs(y1 - y2)

    return binom(m + n, n)
