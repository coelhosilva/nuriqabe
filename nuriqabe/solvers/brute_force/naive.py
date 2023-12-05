__all__ = [
    "brute_force_solver_naive",
]

import itertools
import numpy as np
from copy import deepcopy
from nuriqabe.game import NurikabeBoard, CellCategory


def brute_force_solver_naive(board: NurikabeBoard, max_n_steps: int = None):
    for remaining_cell_combination in itertools.product(
        *[[0, 1] for _ in range(np.sum(board.painted_grid == -1))]
    ):
        working_board = deepcopy(board)
        for cell, color in zip(
            np.argwhere(board.painted_grid == -1), remaining_cell_combination
        ):
            working_board.paint_cell(*cell, CellCategory(color))
        working_board.fill_unknown_cells(CellCategory.BLACK)
        if working_board.solved():
            working_board.plot()
            return working_board
