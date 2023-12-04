import numpy as np
from nurikabe.game import NurikabeBoard
from nurikabe.board_instances import read_board
from nurikabe.solvers.qlearning import q_learning, solve_board_via_generated_islands

if __name__ == "__main__":
    # grid, solution = read_board(0)
    # grid, solution = read_board(1073)
    # grid, solution = read_board(2)
    # grid, solution = read_board(130)
    grid, solution = read_board(122)
    # grid, solution = read_board("5x5_easy")
    # grid, solution = read_board("7x7_easy")
    # grid, solution = read_board("5x5_hard")
    # grid, solution = read_board("10x10_easy")
    # grid, solution = read_board("10x10_hard")
    # grid, solution = read_board(5555)
    # grid, solution = read_board(9999)

    board = NurikabeBoard(grid)  # , max_n_steps_deterministic_iterations=100)
    board.plot()

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
