"""


markov decision process
interact -> history -> approximate env models
use policy/value iteration to find optimal policy
Derivde optmial policy directly - model free learning


binary reward system does not account for better moves that fill more cells.
variable reward proportional to cells filled?

if reaches a dead end, backtrack until a different outcome is possible.

might be the case of having a problem grid, a painted grid, and a certainty grid.

draw betas for the training process if exploring TS

todo: build a verifier that loops through islands, check if they are solved, check continuous black see, check no black pool.
todo: check if verifier works with solutions.
todo: sat solver
todo: feed neural network
board.solved() seems correct. I tried changing one of the islands and it correctly returned the unsolved response.
"""

import itertools
import numpy as np


from copy import deepcopy
from nurikabe.game import NurikabeBoard

# from dataclasses import dataclass

from nurikabe.board_instances import read_board, BoardNotFound


# def island_combinations()

"""
Might need islands
detect which islands are in the regions of influence.
"""


# def get_reward(
#     board: NurikabeBoard, row: int, col: int, color_to_paint=CellCategory.BLACK
# ):
#     if is_valid_move(board, row, col, color_to_paint):
#         return 1  # Positive reward for valid moves
#     else:
#         return -1  # Negative reward for invalid moves


# def thompson_sampling(
#     board: NurikabeBoard,
#     max_n_steps: int,
#     allow_revisiting: bool,
#     color_to_paint=CellCategory.BLACK,
# ) -> tuple[np.ndarray]:
#     success_counts = np.zeros(
#         (len(board), len(board)),
#     )
#     failure_counts = np.zeros(
#         (len(board), len(board)),
#     )
#     visited_cells = np.zeros((len(board), len(board)), dtype=bool)

#     step = 0

#     while True:
#         if step > max_n_steps:
#             break

#         # Cell selection via Thompson-Sampling
#         row, col = np.unravel_index(
#             np.argmax(
#                 np.random.beta(1 + success_counts, 1 + failure_counts),
#                 axis=None,
#             ),
#             success_counts.shape,
#         )

#         if not allow_revisiting:
#             if visited_cells[row][col]:
#                 step += 1
#                 continue
#             else:
#                 visited_cells[row][col] = True

#         # Simulating coloring the selected cell
#         reward = get_reward(board, row, col, color_to_paint)

#         # Updating counts and rewards
#         if reward > 0:
#             success_counts[row][col] += 1
#         elif reward < 0:
#             failure_counts[row][col] += 1

#         if board.solved():
#             break

#         if not board.solvable():
#             print("Board solution diverged.")
#             break

#         step += 1

#     return success_counts, failure_counts


# def get_starting_location(board: NurikabeBoard, random_state):
#     environment_rows = board.shape[0]
#     environment_columns = board.shape[1]

#     current_row_index = random_state.randint(board.shape[0])
#     current_column_index = random_state.randint(environment_columns)

#     # continue choosing random row and column indexes until a certain black cell is chosen
#     while not (
#         board.cell_is_known[current_row_index, current_column_index]
#         and board.painted_grid[current_row_index, current_column_index]
#         == CellCategory.WHITE.value
#     ):
#         current_row_index = random_state.randint(environment_rows)
#         current_column_index = random_state.randint(environment_columns)

#     return current_row_index, current_column_index


# environment_rows = board.shape[0]
# environment_columns = board.shape[1]

# current_row_index = random_state.randint(board.shape[0])
# current_column_index = random_state.randint(environment_columns)

# # continue choosing random row and column indexes until a certain black cell is chosen
# while not (
#     board.cell_is_known[current_row_index, current_column_index]
#     and board.painted_grid[current_row_index, current_column_index]
#     == CellCategory.WHITE.value
# ):
#     current_row_index = random_state.randint(environment_rows)
#     current_column_index = random_state.randint(environment_columns)

# return current_row_index, current_column_index


# else:

#     for island in island_group:
#         for cell in island:
#             board.paint_cell(*cell, CellCategory.UNKNOWN)


# def solve_board_via_q_table(
#     board: NurikabeBoard,
#     learned_q_table,
#     max_n_steps: int = 100,
#     seed: int = 0,
# ) -> NurikabeBoard:
#     # return immediately if this is an invalid starting location
#     random_state = np.random.RandomState(seed)

#     if board.is_terminal_state():
#         return board
#     else:
#         environment_rows = board.shape[0]
#         environment_columns = board.shape[1]

#         start_row_index, start_column_index = get_starting_location_island(
#             board,
#             random_state,
#         )  # some cell known to be black

#         current_row_index, current_column_index = start_row_index, start_column_index
#         # continue moving along the path until we reach the goal (i.e., the item packaging location)
#         step = 0

#         while not board.is_terminal_state():
#             if step > max_n_steps:
#                 break

#             step += 1
#             # get the best action to take
#             action_index = get_next_action(
#                 learned_q_table,
#                 current_row_index,
#                 current_column_index,
#                 epsilon=1.0,
#                 random_state=random_state,
#             )
#             # move to the next location on the path, and add the new location to the list
#             current_row_index, current_column_index = get_next_location(
#                 current_row_index,
#                 current_column_index,
#                 action_index,
#                 environment_rows,
#                 environment_columns,
#             )
#             board.paint_and_check(
#                 current_row_index,
#                 current_column_index,
#                 CellCategory.WHITE,
#             )
#         return board


# def brute_force_solver_polyomino(board: NurikabeBoard, max_n_steps: int = None):
#     incomplete_islands = board.get_incomplete_islands_info()
#     i = 0
#     original_board = deepcopy(board)
#     for island_combination in itertools.product(
#         *[
#             generate_polyomino(island["hint"], island["size"])
#             for island in incomplete_islands
#         ]
#     ):
#         working_board = deepcopy(original_board)
#         print(i)
#         i += 1
#         if max_n_steps is not None:
#             if i > max_n_steps:
#                 break
#         # print(island_combination)
#         can_paint_all = [
#             working_board.cell_cannot_be_painted(*cell)
#             for island in island_combination
#             for cell in island
#             if cell not in working_board.hint_cells
#         ]
#         if not any(can_paint_all):
#             # latest_board = deepcopy(working_board)
#             for island in island_combination:
#                 for cell in island:
#                     working_board.paint_cell(*cell, CellCategory.WHITE)
#                     # try:
#                     #     board.paint_cell(*cell, CellCategory.WHITE)
#                     # except IndexError:
#                     #     latest_board = deepcopy(board)
#                     #     continue
#                 # board.plot()
#             working_board.fill_unknown_cells(CellCategory.BLACK)
#             # working_board.plot()
#             if working_board.solved():
#                 return working_board
#                 # else:
#                 # board = deepcopy(latest_board)


# incomplete_islands = board.get_incomplete_islands_info()
# i = 0
# original_board = deepcopy(board)
# for island_combination in itertools.product(
#     *[
#         generate_polyomino(island["hint"], island["size"])
#         for island in incomplete_islands
#     ]
# ):
#     working_board = deepcopy(original_board)
#     print(i)
#     i += 1
#     if max_n_steps is not None:
#         if i > max_n_steps:
#             break
#     # print(island_combination)
#     can_paint_all = [
#         working_board.cell_cannot_be_painted(*cell)
#         for island in island_combination
#         for cell in island
#         if cell not in working_board.hint_cells
#     ]
#     if not any(can_paint_all):
#         # latest_board = deepcopy(working_board)
#         for island in island_combination:
#             for cell in island:
#                 working_board.paint_cell(*cell, CellCategory.WHITE)
#                 # try:
#                 #     board.paint_cell(*cell, CellCategory.WHITE)
#                 # except IndexError:
#                 #     latest_board = deepcopy(board)
#                 #     continue
#             # board.plot()
#         working_board.fill_unknown_cells(CellCategory.BLACK)
#         # working_board.plot()
#         if working_board.solved():
#             return working_board
#             # else:
#             # board = deepcopy(latest_board)


# from typing import Generator


# def generate_polyomino(
#     indexing_point: tuple[int, int],
#     polyomino_size: int,
#     previous_point: tuple[int, int] = None,
# ) -> Generator[tuple[tuple[int, int]], None, None]:
#     if previous_point is None:
#         previous_point = indexing_point

#     if polyomino_size == 1:
#         yield (indexing_point),
#         return

#     for offset_x, offset_y in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
#         if not (offset_x == 0 and offset_y == 0):
#             new_indexing_point = (
#                 indexing_point[0] + offset_x,
#                 indexing_point[1] + offset_y,
#             )
#             if (
#                 new_indexing_point != previous_point
#                 and new_indexing_point[0] >= 0
#                 and new_indexing_point[1] >= 0
#             ):
#                 for sub_poly in generate_polyomino(
#                     new_indexing_point, polyomino_size - 1, indexing_point
#                 ):
#                     yield indexing_point, *sub_poly


# def brute_force_solver_hybrid()

if __name__ == "__main__":
    # pg = generate_polyomino((5, 5), 3)
    # [p for p in pg]
    # pp = next(pg)
    # from tqdm import tqdm

    # for i in tqdm(range(1091)):
    #     try:
    #         grid, solution = read_board(i)
    #         board = NurikabeBoard(grid)
    #         board.painted_grid = solution
    #         if not board.solved():
    #             print(f"Failed to detect solution in board: {i}")
    #             board.plot()
    #     except ValueError:
    #         pass
    #     except BoardNotFound:
    #         pass
    #     except Exception as e:
    #         print(e)
    #         print(f"Failed to process board: {i}")

    # grid, solution = read_board(0)
    # grid, solution = read_board(1073)
    # grid, solution = read_board(2)
    # grid, solution = read_board(130)
    # grid, solution = read_board(122)
    # grid, solution = read_board("5x5_easy")
    # grid, solution = read_board("7x7_easy")
    # grid, solution = read_board("5x5_hard")
    # grid, solution = read_board("10x10_easy")
    # grid, solution = read_board("10x10_hard")
    # grid, solution = read_board(5555)
    grid, solution = read_board(9999)

    board = NurikabeBoard(grid)  # , max_n_steps_deterministic_iterations=100)
    # todo: make it until the point it becomes unsolvable. If unsolvable, backtrack.
    # sb = brute_force_solver_naive(board)
    # sb = brute_force_solver_polyomino(board)
    board.plot()
    # r = brute_force_solver(board)
    # # plot_grid(solution)

    solvable_board = NurikabeBoard(grid)
    # # # # # # q learning
    q_table, generated_islands = q_learning(
        solvable_board,
        max_n_episodes=100,
        max_n_steps=200,
    )
    s = solve_board_via_generated_islands(solvable_board, generated_islands)
    if s:
        print("found solution")
        s.plot()
        print(f"Match recorded solution: {np.all(s.painted_grid == solution)}")

    # agent_response = solve_board_via_q_table(
    #     solvable_board,
    #     q_table,
    #     max_n_steps=1000,
    # )  # testar as combinações. Preencher com preto? o que faltar ou fazer brute force naive.
    # agent_response.plot()

    # # # Thompson sampling
    # todo: try and change to step by step. Or also see all completed islands.
    # solvable_board = NurikabeBoard(grid)
    # thompson_sampling(
    #     solvable_board,
    #     max_n_steps=10000,
    #     allow_revisiting=False,
    #     color_to_paint=CellCategory.WHITE,
    # )
    # solvable_board.plot()


"""
Presentation:
- comparar tamanho novo do problema com o inicial, upper bound sem parede.
- Cada polyomino tem uma 

Ideas:
- incorporate island expansion to agent
- improve the solvable function
- brute force of few cases
- Test other values of epsilon, learning rate and rewards schematics
- statistics of solvability

Maybe keep a q table of the board state an integer resulting of the binary representation
"""
