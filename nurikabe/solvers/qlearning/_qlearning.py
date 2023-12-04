import numpy as np
import itertools
from copy import deepcopy
from nurikabe.game import NurikabeBoard, CellCategory
from nurikabe.solvers.brute_force import brute_force_solver_naive


def is_valid_move(
    board: NurikabeBoard, row: int, col: int, color_to_paint=CellCategory.BLACK
) -> bool:
    if board.paint_and_check(row, col, color_to_paint):
        return True
    else:
        return False


def get_starting_location_island(board: NurikabeBoard, random_state):
    chosen_island = random_state.choice(board.get_incomplete_islands_info())
    return chosen_island["hint"]


def get_reward_qlearning(board: NurikabeBoard, row: int, col: int):
    if is_valid_move(board, row, col, CellCategory.WHITE):
        return 1  # Slightly negative reward for valid moves
    else:
        return -1  # Negative reward for invalid moves


def get_next_location(
    current_row_index: int,
    current_column_index: int,
    action_index: int,
    environment_rows: int,
    environment_columns: int,
):
    actions = ["up", "right", "down", "left"]
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == "up" and current_row_index > 0:
        new_row_index -= 1
    elif (
        actions[action_index] == "right"
        and current_column_index < environment_columns - 1
    ):
        new_column_index += 1
    elif actions[action_index] == "down" and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == "left" and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index


def get_next_action(
    q_table: np.ndarray,
    current_row_index,
    current_column_index,
    epsilon,
    random_state,
) -> int:
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    if random_state.random() < epsilon:
        return np.argmax(q_table[current_row_index, current_column_index])
    else:
        return random_state.randint(4)  # choose a random action


def q_learning(
    board: NurikabeBoard,
    epsilon: float = 0.9,
    discount_factor: float = 0.9,
    learning_rate: float = 0.9,
    max_n_episodes: int = 10,
    max_n_steps: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray]:
    random_state = np.random.RandomState(seed)
    q_table = np.zeros((*board.shape, 4))
    environment_rows = board.shape[0]
    environment_columns = board.shape[1]
    original_board = deepcopy(board)

    if original_board.is_terminal_state():
        return q_table, []

    previously_completed_islands = [
        island["hint"] for island in board.get_completed_islands_info()
    ]
    generated_islands = {
        str(completed_island["hint"]): []
        for completed_island in original_board.get_incomplete_islands_info()
        if completed_island["size"] > 1
    }

    for episode in range(max_n_episodes):
        board = deepcopy(original_board)
        print(f"Training episode: {episode}...")

        row_index, column_index = get_starting_location_island(
            board, random_state
        )  # some island definition
        print(
            f"Starting island: {row_index}, {column_index}. Island size: {board.grid[row_index, column_index]}."
        )

        step = 0
        step_ix = 0
        printing_instances = np.linspace(0, max_n_steps, 10)

        while not board.is_terminal_state():
            if step > printing_instances[step_ix]:
                print("." * int(printing_instances[step_ix] / max_n_steps * 100))
                step_ix += 1

            if step > max_n_steps:
                break

            step += 1

            action_index = get_next_action(
                q_table,
                row_index,
                column_index,
                epsilon,
                random_state=random_state,
            )
            old_row_index, old_column_index = row_index, column_index
            row_index, column_index = get_next_location(
                row_index,
                column_index,
                action_index,
                environment_rows,
                environment_columns,
            )
            reward = get_reward_qlearning(board, row_index, column_index)
            old_q_value = q_table[old_row_index, old_column_index, action_index]
            temporal_difference = (
                reward
                + (discount_factor * np.max(q_table[row_index, column_index]))
                - old_q_value
            )
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_table[old_row_index, old_column_index, action_index] = new_q_value

            for (
                completed_island
            ) in board.get_completed_islands_info():  # append every tried combination.
                if completed_island["size"] > 1:
                    if (
                        completed_island["hint"] not in previously_completed_islands
                        and completed_island["cells"]
                        not in generated_islands[str(completed_island["hint"])]
                    ):
                        generated_islands[str(completed_island["hint"])].append(
                            completed_island["cells"]
                        )

    print("Training completed!")

    return q_table, generated_islands


def solve_board_via_generated_islands(
    board: NurikabeBoard,
    generated_islands,
    brute_force_threshold: int = 10000,
):
    if board.solved():
        return board
    else:
        if len(generated_islands) == 0:
            return None

        for island_group in itertools.product(*list(generated_islands.values())):
            working_board = deepcopy(board)
            for island in island_group:
                for cell in island:
                    working_board.paint_cell(*cell, CellCategory.WHITE)

            if working_board.search_space_size_naive_current() < brute_force_threshold:
                bfs = brute_force_solver_naive(working_board)
                if bfs is not None:
                    return bfs

            working_board.fill_unknown_cells(CellCategory.BLACK)
            if working_board.solved():
                return working_board
