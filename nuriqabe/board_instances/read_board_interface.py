__all__ = [
    "read_board_formulation",
    "read_board_solution",
    "read_board",
    "fetch_all_boards_and_solutions",
    "boards_summary",
]

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal


_base_path = Path(__file__).parent

BOARDS_PATH = _base_path / "nurikabe_boards"


_custom_board_ids = {
    "5x5_easy": 5555,
    "5x5_hard": 5556,
    "7x7_easy": 7777,
    "10x10_easy": 9910,
    "10x10_hard": 9911,
}


class BoardNotFound(Exception):
    pass


def read_board_component(
    board_id: int,
    component: Literal["problem", "solution"],
) -> np.ndarray:
    if component not in ["problem", "solution"]:
        raise ValueError(f"Component must be 'problem' or 'solution', not {component}.")

    if board_id in _custom_board_ids:
        board_id = _custom_board_ids[board_id]

    processed_board_id = str(int(board_id)).zfill(4)

    try:
        return np.loadtxt(
            BOARDS_PATH / f"{processed_board_id}.a.htm_{component}.txt"
        ).astype(np.int64)
    except FileNotFoundError:
        raise BoardNotFound(f"Board {board_id} not found.")


def read_board_formulation(board_id: int) -> np.ndarray:
    return read_board_component(board_id, "problem")


def read_board_solution(board_id: int) -> np.ndarray:
    return read_board_component(board_id, "solution")


def read_board(board_id: int) -> tuple[np.ndarray, np.ndarray]:
    return read_board_formulation(board_id), read_board_solution(board_id)


def fetch_all_boards_and_solutions(sort_boards: bool = True):
    boards = (
        sorted(list(BOARDS_PATH.rglob("*htm_problem.txt")))
        if sort_boards
        else BOARDS_PATH.rglob("*htm_problem.txt")
    )
    for board in boards:
        board_id = int(board.stem[:4])
        yield board_id, *read_board(board_id)


def boards_summary() -> pd.DataFrame:
    ids = []
    shapes = []
    sizes = []
    square = []
    n_islands = []
    for board_id, board, _ in fetch_all_boards_and_solutions():
        ids.append(board_id)
        shapes.append(board.shape)
        sizes.append(board.shape[0] * board.shape[1])
        square.append(board.shape[0] == board.shape[1])
        n_islands.append(np.sum(board != 0))

    return pd.DataFrame(
        {
            "id": ids,
            "shape": shapes,
            "size": sizes,
            "square": square,
            "n_islands": n_islands,
        }
    )
