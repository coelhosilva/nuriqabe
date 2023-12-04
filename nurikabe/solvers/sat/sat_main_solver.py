from pysat.formula import IDPool
from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.card import CardEnc
from .sat_utils import Problem, read_problem_grid, try_find_wall


def _implies(p, q):
    return [-p_i for p_i in p] + [q]


class Encoder:
    def __init__(self, problem_instance: Problem, wall_assumption: (int, int)):
        self.problem = problem_instance
        self.wall_assumption = wall_assumption
        self.pool = IDPool()
        self.v = {}
        self.field_indices = {}
        self.positions_of_indices = {}
        self.field_amount = 0
        self.max_field_size = 0
        self.wall_size = 0

    def _initialize_variables(self):
        xt = self.problem.x
        yt = self.problem.y

        self.wall_size = xt * yt

        for y in range(yt):
            for x in range(xt):
                # there is a wall at pos (x, y)
                self.v["w", x, y] = self.pool.id(("w", x, y))

                # field indices
                if self.problem.fields[x][y] is not None:
                    self.max_field_size = max(
                        self.max_field_size, self.problem.fields[x][y]
                    )
                    self.wall_size -= self.problem.fields[x][y]
                    field_index = len(self.field_indices)
                    self.field_indices[(x, y)] = field_index
                    self.positions_of_indices[field_index] = (x, y)
                    self.field_amount += 1

        # field connectivity
        for y in range(yt):
            for x in range(xt):
                for field_index in range(self.field_amount):
                    for dist in range(self.max_field_size):
                        # pos (x, y) belongs to field_index, and is at distance k from field anchor
                        self.v["f", x, y, field_index, dist] = self.pool.id(
                            ("f", x, y, field_index, dist)
                        )

        # wall connectivity
        for y in range(yt):
            for x in range(xt):
                for dist in range(self.wall_size):
                    # pos (x, y) is at distance dist from wall_assumption
                    self.v["d", x, y, dist] = self.pool.id(("d", x, y, dist))

    def _index_neighbours(self, x, y):
        neighbours = []
        if x > 0:
            neighbours.append((x - 1, y))
        if x < self.problem.x - 1:
            neighbours.append((x + 1, y))
        if y > 0:
            neighbours.append((x, y - 1))
        if y < self.problem.y - 1:
            neighbours.append((x, y + 1))
        return neighbours

    def encode(self):
        self._initialize_variables()
        formula = CNF()
        xt = self.problem.x
        yt = self.problem.y

        # every position is in a field at distance or a wall
        for y in range(yt):
            for x in range(xt):
                literals = [self.v["w", x, y]]
                for field_index in range(self.field_amount):
                    for dist in range(self.max_field_size):
                        literals.append(self.v["f", x, y, field_index, dist])
                clauses = CardEnc.equals(literals, bound=1, vpool=self.pool)
                for clause in clauses:
                    formula.append(clause)

        # numbers belong to appropriate fields at distance 0
        for y in range(yt):
            for x in range(xt):
                if self.problem.fields[x][y] is not None:
                    field_index = self.field_indices[(x, y)]
                    formula.append([self.v["f", x, y, field_index, 0]])
                else:
                    for field_index in range(self.field_amount):
                        formula.append([-self.v["f", x, y, field_index, 0]])

        # a ^ b => c v d
        # !a v !b v c v d

        # ..x.4 <- this has to be prevented
        # if in field at distance d, then I have a neighbour at distance d-1
        for y in range(yt):
            for x in range(xt):
                for field_index in range(self.field_amount):
                    for dist in range(1, self.max_field_size):
                        clause = [-self.v["f", x, y, field_index, dist]]
                        for neigh_x, neigh_y in self._index_neighbours(x, y):
                            clause.append(
                                self.v["f", neigh_x, neigh_y, field_index, dist - 1]
                            )
                        formula.append(clause)

        # if I am in field at distance d, all my non-wall neighbours are in the same field at dist d-1 or d+1
        for y in range(yt):
            for x in range(xt):
                for field_index in range(self.field_amount):
                    for dist in range(0, self.max_field_size):
                        for neigh_x, neigh_y in self._index_neighbours(x, y):
                            clause = [
                                -self.v["f", x, y, field_index, dist],
                                self.v["w", neigh_x, neigh_y],
                            ]
                            if dist > 0:
                                clause.append(
                                    self.v["f", neigh_x, neigh_y, field_index, dist - 1]
                                )
                            if dist < self.max_field_size - 1:
                                clause.append(
                                    self.v["f", neigh_x, neigh_y, field_index, dist + 1]
                                )
                            formula.append(clause)

        # every field has the appropriate size
        for field_index in range(self.field_amount):
            pos_x, pos_y = self.positions_of_indices[field_index]
            value = self.problem.fields[pos_x][pos_y]
            assert value is not None

            literals = []
            for y in range(yt):
                for x in range(xt):
                    for dist in range(self.max_field_size):
                        literals.append(self.v["f", x, y, field_index, dist])
            clauses = CardEnc.equals(literals, bound=value, vpool=self.pool)
            for clause in clauses:
                formula.append(clause)

        # no squares in walls
        for y in range(yt - 1):
            for x in range(xt - 1):
                a, b, c, d = (
                    self.v["w", x, y],
                    self.v["w", x + 1, y],
                    self.v["w", x, y + 1],
                    self.v["w", x + 1, y + 1],
                )
                formula.append([-a, -b, -c, -d])

        # ----- WALL CONNECTIVITY -----

        # this is the anchor
        wall_x, wall_y = self.wall_assumption
        for y in range(yt):
            for x in range(xt):
                if (x, y) == (wall_x, wall_y):
                    formula.append([self.v["d", x, y, 0]])
                else:
                    formula.append([-self.v["d", x, y, 0]])

        # not a wall or there is exactly one distance possible
        for y in range(yt):
            for x in range(xt):
                literals = [-self.v["w", x, y]]
                for dist in range(self.wall_size):
                    literals.append(self.v["d", x, y, dist])
                clauses = CardEnc.equals(literals, bound=1, vpool=self.pool)
                for clause in clauses:
                    formula.append(clause)

        # if wall distance d>0, then I have a neighbour at distance d-1
        for y in range(yt):
            for x in range(xt):
                for dist in range(1, self.wall_size):
                    clause = [-self.v["d", x, y, dist]]
                    for neigh_x, neigh_y in self._index_neighbours(x, y):
                        clause.append(self.v["d", neigh_x, neigh_y, dist - 1])
                    formula.append(clause)

        # if wall distance d, then all wall-neighbours have wall distance d-1 or d+1
        for y in range(yt):
            for x in range(xt):
                for dist in range(1, self.wall_size):
                    for neigh_x, neigh_y in self._index_neighbours(x, y):
                        clause = [
                            -self.v["d", x, y, dist],
                            -self.v["w", neigh_x, neigh_y],
                        ]
                        if dist > 0:
                            clause.append(self.v["d", neigh_x, neigh_y, dist - 1])
                        if dist < self.wall_size - 1:
                            clause.append(self.v["d", neigh_x, neigh_y, dist + 1])
                        formula.append(clause)

        # if wall, then connected
        for y in range(yt):
            for x in range(xt):
                clause = [-self.v["w", x, y]]
                for dist in range(self.wall_size):
                    clause.append(self.v["d", x, y, dist])
                formula.append(clause)

        return formula

    def decode(self, model: list[int]):
        xt = self.problem.x
        yt = self.problem.y

        walls = [[False for _ in range(yt)] for _ in range(xt)]
        for y in range(yt):
            for x in range(xt):
                walls[x][y] = model[self.v["w", x, y] - 1] > 0

        return walls


def try_solve_with_assumed_wall(assumed_wall: (int, int)):
    encoder = Encoder(instance, assumed_wall)
    formula = encoder.encode()
    with Solver(bootstrap_with=formula, name="gluecard4") as solver:
        if solver.solve():
            return encoder.decode(solver.get_model())
    return None


def find_solution(problem_instance: Problem):
    pos = try_find_wall(problem_instance)
    if pos is not None:
        return try_solve_with_assumed_wall(assumed_wall=pos)

    for y in range(problem_instance.x):
        for x in range(problem_instance.y):
            result = try_solve_with_assumed_wall(assumed_wall=(x, y))
            if result is not None:
                return result

    return None


import numpy as np
from board_instances.read_board_interface import read_board


def read_problem_grid_from_array(grid: np.ndarray):
    xt, yt = grid.shape
    hint_cells = [tuple(hint_cell) for hint_cell in np.argwhere(grid != 0).tolist()]

    instance = Problem(xt, yt)
    for hint_cell in hint_cells:
        instance.add_field(*hint_cell, grid[hint_cell])
    return instance


if __name__ == "__main__":
    # from pathlib import Path
    # base_path = Path(__file__).parent
    # path = base_path / "instances/1.in"
    # instance = read_problem_grid(path)
    from board import NurikabeBoard
    from qlearning_dev_islands import plot_grid

    # grid, solution_reference = read_board(9999)
    # board = NurikabeBoard(grid)
    # instance = read_problem_grid_from_array(grid)

    # solution = find_solution(problem_instance=instance)
    # board.painted_grid = np.array(solution).astype(int)

    # if solution is None:
    #     print("Unsolvable")
    # else:
    #     instance.add_solution(solution)
    #     print(instance)

    from board_instances.read_board_interface import fetch_all_boards_and_solutions
    from time import perf_counter

    board_solving_times = {}

    start_time_solving = perf_counter()
    for board_id, grid, solution in fetch_all_boards_and_solutions():
        start_time_solving_board = perf_counter()
        board = NurikabeBoard(grid)
        instance = read_problem_grid_from_array(grid)
        solution = find_solution(problem_instance=instance)
        if solution is None:
            print("Unsolvable")
        else:
            board.painted_grid = np.array(solution).astype(int)
            if not board.solved():
                print("Wrong solution")
            else:
                print(f"Correct solution for board {board_id}")
        end_time_solving_board = perf_counter()
        board_solving_times[board_id] = (
            end_time_solving_board - start_time_solving_board
        )
    end_time_solving = perf_counter()
    print(f"Time solving: {end_time_solving - start_time_solving}")
