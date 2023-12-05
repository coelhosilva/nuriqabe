from nurikabe.board_instances import read_board
from nurikabe.solvers.qlearning import QLearningSolver

if __name__ == "__main__":
    # grid, solution = read_board(1073)  # ! Not solving
    # grid, solution = read_board(0)
    grid, solution = read_board(2)
    # grid, solution = read_board(130)
    # grid, solution = read_board(122)
    # grid, solution = read_board("5x5_easy")
    # grid, solution = read_board("7x7_easy")
    # grid, solution = read_board("5x5_hard")
    # grid, solution = read_board("10x10_easy")
    grid, solution = read_board(9999)

    board_solver = QLearningSolver(problem_grid=grid)
    board_solver.board.plot()
    # board_solver.solve(max_n_episodes=100, max_n_steps=200)
    # board_solver.plot_solution()
