from nuriqabe.board_instances import read_board
from nuriqabe.solvers.qlearning import QLearningSolver

grid, solution = read_board(9999)

board_solver = QLearningSolver(problem_grid=grid)

board_solver.board.plot()  # inspect base board, filled with deterministic rules

board_solver.solve(max_n_episodes=100, max_n_steps=200)

board_solver.plot_solution()
