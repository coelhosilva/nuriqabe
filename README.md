# nuriQabe

`nuriQabe` is a Python package for solving Nurikabe puzzles with Q-Learning.


## Installation

Create a project folder and navigate to it:

1. ```mkdir nurikabe-qlearning-user```

2. ```cd nurikabe-qlearning-user```

From a terminal within the `nurikabe-qlearning-user` project folder, create a new `Python (>=3.10.4)` virtual environment:

3. ```python -m venv venv```

Activate the virtual environment:

4. Shell: `source venv/bin/activate` | Windows CMD: `venv\Scripts\activate`

Install the nuriQabe package:

5. ```pip install git+https://github.com/coelhosilva/nuriqabe.git```

Alternatively, one may clone this repo and perform a local installation according to the following steps:

1. ```git clone https://github.com/coelhosilva/nuriqabe.git``` (HTTP) or ```git clone git@github.com:coelhosilva/nuriqabe.git`` (SSH)

2. ```cd nuriqabe```

3. ```python -m venv venv```

4. Shell: `source venv/bin/activate` | Windows CMD: `venv\Scripts\activate`

5. ```pip install ./nuriqabe```

## Getting started

`nuriQabe` provides a Q-Learning-based solver for the board game Nurikabe. The package is divided into the following modules: `board_instances`, `game`, `geometry`, `graphics`, and `solvers`.

The following code exemplifies how to instantiate a game and solve it using the Q-Learning solver.

```python
from nuriqabe.board_instances import read_board
from nuriqabe.solvers.qlearning import QLearningSolver

grid, solution = read_board(9999)

board_solver = QLearningSolver(problem_grid=grid)

board_solver.board.plot() # inspect base board, filled with deterministic rules

board_solver.solve(max_n_episodes=100, max_n_steps=200)

board_solver.plot_solution()
```

For inspecting the available board instances, please refer to the following code:

```python
from nuriqabe.board_instances import boards_summary

# Pandas DataFrame with id, shape, size, whether it is a square board, and n_islands
df_boards = boards_summary()

print(df_boards)
```
