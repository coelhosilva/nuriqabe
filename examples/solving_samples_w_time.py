from nuriqabe.board_instances import read_board
from nuriqabe.solvers.qlearning import QLearningSolver
from time import perf_counter

if __name__ == "__main__":
    # board_input = 1073  # ! Not solving
    # board_input = 100  # ! Not solving
    # board_input = 0
    # board_input = 2
    # board_input = 130
    # board_input = 122
    # board_input = 9999
    # board_input = "5x5_easy"
    # board_input = "5x5_hard"
    # board_input = "7x7_easy"
    # board_input = "10x10_easy"
    board_input = "10x10_hard"

    grid, solution = read_board(board_input)
    board_solver = QLearningSolver(problem_grid=grid, verbose=False)
    board_solver.board.plot()
    st = perf_counter()
    board_solver.solve(
        max_n_episodes=100,
        max_n_steps=200,
    )
    et = perf_counter()
    print(f"Solved board {board_input} in {et - st} seconds.")
    board_solver.plot_solution()
