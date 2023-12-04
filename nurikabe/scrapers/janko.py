import time
import requests
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path

_base_path = Path(__file__).parent.parent
_index_url = "https://www.janko.at/Raetsel/Nurikabe/index-1.htm"
_base_board_url = "https://www.janko.at/Raetsel/Nurikabe"
BOARDS_PATH = _base_path / "board_instances/nurikabe_boards"


def fetch_board_ids() -> list[str]:
    request_index = requests.get(_index_url)
    soup = BeautifulSoup(request_index.text, "html.parser")
    links = soup.find_all("a", class_="sv")

    return [link.get("href") for link in links]


def fetch_board_and_solution(board_id: str) -> tuple[np.ndarray, np.ndarray]:
    board_request = requests.get(f"{_base_board_url}/{board_id}")

    problem_text = board_request.text.split("[problem]")[-1].split("[solution]")[0]
    solution_text = board_request.text.split("[solution]")[-1].split("[moves]")[0]

    problem = np.array(
        [
            [int(char) if char != "-" else 0 for char in line.split()]
            for line in problem_text.strip().split("\n")
        ],
        dtype=np.int64,
    )

    solution = np.array(
        [
            [1 if char == "x" else 0 for char in line.split()]
            for line in solution_text.strip().split("\n")
        ],
        dtype=np.int64,
    )

    return problem, solution


def download_boards(
    continue_on_error: bool = True,
    skip_existing_file: bool = True,
) -> None:
    BOARDS_PATH.mkdir(parents=True, exist_ok=True)

    board_ids = fetch_board_ids()
    counter = 1
    n_boards = len(board_ids)
    for board_id in board_ids:
        if skip_existing_file:
            if (BOARDS_PATH / f"{board_id}_problem.txt").exists():
                print(f"Skipping board {board_id} [{counter}/{n_boards}]...")
                counter += 1
                continue

        print(f"Downloading board {board_id} [{counter}/{n_boards}]...")
        try:
            problem, solution = fetch_board_and_solution(board_id)
            np.savetxt(BOARDS_PATH / f"{board_id}_problem.txt", problem)
            np.savetxt(BOARDS_PATH / f"{board_id}_solution.txt", solution)
        except Exception as e:
            print(f"Error while fetching board {board_id}: {e}")
            if not continue_on_error:
                raise e

        time.sleep(1)
        counter += 1

    print("Finished downloading boards.")
