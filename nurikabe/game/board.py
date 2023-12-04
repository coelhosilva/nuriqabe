__all__ = [
    "NurikabeBoard",
]

import numpy as np
from functools import reduce, cached_property
from scipy.spatial import distance as scipy_distance
from nurikabe.graphics import plot_grid
from nurikabe.geometry import (
    direct_path_between_two_cells,
    is_continuous_path,
    count_manhattan_distance_paths,
)
from .enums import CellCategory


class NurikabeDeterministicRules:
    def __init__(self):
        self.deterministic_rules = [
            self.surrounded_cells,
            self.black_pool_prevention,
            self.clear_island_expansion,
            self.surrounded_complete_islands,
            self.island_expansion,
            # self.merging_dark_areas,
        ]

    def surrounded_cells(self):
        # ! surrounded cells - if all neighbors are known and share the same color, the cell must be the same color
        for coordinate in self.board_coordinates:
            if not self.cell_is_known[coordinate]:
                neighbors = self.neighbors(*coordinate)
                neighboring_colors = {
                    self.painted_grid[neighbor] for neighbor in neighbors
                }
                neighboring_certainty = all(
                    self.cell_is_known[neighbor] for neighbor in neighbors
                )
                if neighboring_certainty:
                    if len(neighboring_colors) == 1:
                        neighboring_color = neighboring_colors.pop()
                        if neighboring_color in [
                            CellCategory.BLACK.value,
                            CellCategory.WHITE.value,
                        ]:
                            self.paint_cell(
                                *coordinate, CellCategory(neighboring_color)
                            )

    def black_pool_prevention(self):
        # ! black pool prevention - If there are 2x2 regions with 3 certain black cells, the fourth cell must be white.
        for ix, iy in np.ndindex(self.painted_grid.shape):
            pool_coordinates = [
                (ix, iy),
                (ix + 1, iy) if ix < self.grid_size - 1 else None,
                (ix, iy + 1) if iy < self.grid_size - 1 else None,
                (ix + 1, iy + 1)
                if ix < self.grid_size - 1 and iy < self.grid_size - 1
                else None,
            ]
            if None not in pool_coordinates:
                if (
                    sum(
                        map(
                            lambda cell_value: cell_value == CellCategory.BLACK.value,
                            [
                                self.painted_grid[pool_coordinate]
                                for pool_coordinate in pool_coordinates
                            ],
                        )
                    )
                    == 3
                ):
                    for pool_coordinate in pool_coordinates:
                        if (
                            self.painted_grid[pool_coordinate]
                            == CellCategory.UNKNOWN.value
                        ):
                            self.paint_cell(*pool_coordinate, CellCategory.WHITE)

    def clear_island_expansion(self):
        # ! clear expansion
        # For each cell, keep record of the islands that can reach it. If it is white
        # and can only be reached by one and there is only one path to do it, it must
        # be part of the island.
        for cell in self.board_coordinates:
            if (
                self.painted_grid[cell] == CellCategory.WHITE.value
                and cell not in self.hint_cells
            ):
                cell_reachable_by = self.islands_that_reach_here(cell)
                if len(cell_reachable_by) == 1:
                    n_paths_between_cells = count_manhattan_distance_paths(
                        cell, cell_reachable_by[0]
                    )
                    if n_paths_between_cells == 1:
                        path_between_cells = direct_path_between_two_cells(
                            cell, cell_reachable_by[0]
                        )
                        for cell_within_path in path_between_cells:
                            self.paint_cell(*cell_within_path, CellCategory.WHITE)

    def surrounded_complete_islands(self):
        # islands that are complete must be surrounded by black cells
        for complete_island in self.get_completed_islands():
            for cell in complete_island:
                for neighbor in self.neighbors(*cell):
                    if (
                        self.painted_grid[neighbor] == CellCategory.UNKNOWN.value
                        and neighbor not in complete_island
                    ):
                        self.paint_cell(*neighbor, CellCategory.BLACK)

    def merging_dark_areas(self):
        # # merging dark areas. Must be recursive.
        # ! must be improved not to block an island expansion
        for ix, iy in np.ndindex(self.painted_grid.shape):
            if self.painted_grid[ix, iy] == CellCategory.BLACK.value:
                # check if neighbor is unknown and neighbor neighbor is black
                for neighbor in self.neighbors(ix, iy):
                    if self.painted_grid[neighbor] == CellCategory.UNKNOWN.value:
                        for neighbor_neighbor in self.neighbors(*neighbor):
                            if (
                                (neighbor_neighbor != (ix, iy))
                                and (
                                    self.painted_grid[neighbor_neighbor]
                                    == CellCategory.BLACK.value
                                )
                                and count_manhattan_distance_paths(
                                    (ix, iy), neighbor_neighbor
                                )
                                == 1
                            ):
                                self.paint_cell(*neighbor, CellCategory.BLACK)

    def island_expansion(self):
        # island expansion - one direction:
        # self.get_incomplete_islands_sizes
        # change this so it is able to expand partial islands as well.
        # """
        # for island in incomplete_islands:
        # number_of_expandible_directions = 0
        # if number_of_expandible_directions == 1: # must account that if it expands to one side it will join another island.
        #     expand_island()
        #     do it recursively. Stop if it is completed or number_of_expandible_directions>1.
        # ! must be improved to incorporate isolated cells.
        # """
        for island in self.get_incomplete_islands_info():
            target_size = island["size"]
            if len(island["cells"]) == 1:
                current_island_size = 1
                can_expand_directly = True
                reference_cell = island["cells"][0]
                while can_expand_directly and current_island_size < target_size:
                    # isolated cell that is only reachable by it?
                    other_whites = [
                        tuple(c)
                        for c in np.argwhere(
                            self.painted_grid == CellCategory.WHITE.value
                        ).tolist()
                        if tuple(c) not in self.hint_cells
                    ]
                    for w in other_whites:
                        irh = self.islands_that_reach_here(w)
                        if len(irh) == 1:
                            if irh[0] == reference_cell:
                                # neighbors.append(w)
                                new_reference_cell = w
                                self.paint_cell(*w, CellCategory.WHITE)
                                current_island_size += 1
                                reference_cell = w
                                continue
                    neighbors = self.neighbors(*reference_cell)

                    conquerable_cells = [
                        (
                            self.cell_is_known[neighbor] == False
                            and all(
                                self.painted_grid[neighbor_neighbor]
                                != CellCategory.WHITE.value
                                for neighbor_neighbor in self.neighbors(*neighbor)
                                if neighbor_neighbor != reference_cell
                            )
                        )  # and would not be a part of another island and would not create an isolated black cell
                        for neighbor in neighbors
                    ]
                    if conquerable_cells.count(True) == 1:
                        new_reference_cell = conquerable_cells.index(True)
                        self.paint_cell(
                            *neighbors[new_reference_cell], CellCategory.WHITE
                        )
                        current_island_size += 1
                        reference_cell = neighbors[new_reference_cell]
                    else:
                        can_expand_directly = False

    def deterministic_fill_iterative(self):
        # for deterministic_rule in self.deterministic_rules:
        # deterministic_rule()
        for deterministic_rule in self.deterministic_rules:
            previous_grid = self.painted_grid.copy()
            previous_certainty = self.cell_is_known.copy()
            deterministic_rule()
            if not self.solvable():
                self.painted_grid = previous_grid
                self.cell_is_known = previous_certainty


class NurikabeBoard(NurikabeDeterministicRules):
    grid_size: int
    board_size: int
    grid: np.ndarray
    painted_grid: np.ndarray

    def __repr__(self):
        return f"NurikabeBoard(size={self.shape},grid={self.grid})"

    def __init__(
        self,
        grid: np.ndarray | list[list[int]],
        max_n_steps_deterministic_iterations: int = 10,
    ):
        super(NurikabeBoard, self).__init__()
        self.grid = self._init_grid(grid)
        self.painted_grid = np.full_like(
            self.grid,
            CellCategory.UNKNOWN.value,
        )
        self.cell_is_known = np.full_like(
            self.grid,
            False,
            dtype=bool,
        )
        self.grid_size = self.grid.shape[0]
        self.board_size = self.grid.shape[0] * self.grid.shape[1]
        self.max_n_steps_deterministic_iterations = max_n_steps_deterministic_iterations

        self.deterministic_fill_one_timer()
        self.evolve_deterministic_fills_iterative()

    def evolve_deterministic_fills_iterative(self):
        step = 0
        while step < self.max_n_steps_deterministic_iterations:
            previous_grid = self.painted_grid.copy()
            previous_certainty = self.cell_is_known.copy()

            self.deterministic_fill_iterative()
            if not self.solvable():
                self.painted_grid = previous_grid
                self.cell_is_known = previous_certainty
                break

            if np.array_equal(previous_grid, self.painted_grid):
                break
            step += 1

    @property
    def search_space_size_naive(self) -> int:
        return 2 ** (self.grid.shape[0] * self.grid.shape[1])

    @property
    def search_space_size(self):
        return reduce(
            lambda x, y: x * y,
            map(
                lambda l: (l - 1) ** 2 + l**2 - 1 if l > 1 else 1,
                [self.grid[i[0], i[1]] for i in np.argwhere(self.grid != 0)],
            ),
        )

    def search_space_size_naive_current(self):
        return 2 ** (np.sum(self.painted_grid.flatten() == -1))

    def search_space_size_current(self):
        return reduce(
            lambda x, y: x * y,
            map(
                lambda l: (l - 1) ** 2 + l**2 - 1 if l > 1 else 1,
                self.get_incomplete_islands_sizes(),
            ),
        )

    def __len__(self):
        return len(self.grid)

    def _init_grid(self, grid: np.ndarray | list[list[int]]):
        if isinstance(grid, list):
            grid = np.array(grid)

        if not self.check_game_validity(grid):
            raise ValueError("Invalid board")

        return grid

    def check_game_validity(self, grid):
        # add more checks that indicates the game is properly formatted and solvable
        return grid.shape[0] == grid.shape[1]

    def cell_cannot_be_painted(self, row, col) -> bool:
        return (
            row < 0
            or row >= self.grid_size
            or col < 0
            or col >= self.grid_size
            or self.grid[row][col] != 0
            or self.cell_is_known[row, col]
        )

    def fill_unknown_cells(self, category: CellCategory):
        for iy, ix in np.ndindex(self.grid.shape):
            if self.painted_grid[iy, ix] == CellCategory.UNKNOWN.value:
                self.paint_cell(iy, ix, category)

    def paint_cell(self, row, col, category: CellCategory):
        if not self.cell_is_known[row, col]:
            self.painted_grid[row, col] = category.value
            self.cell_is_known[row, col] = True

    def paint_and_check(self, row, col, category: CellCategory) -> bool:
        if self.cell_cannot_be_painted(row, col):
            return False

        previous_grid = self.painted_grid.copy()
        previous_certainty = self.cell_is_known.copy()
        self.painted_grid[row][col] = category.value
        self.cell_is_known[row, col] = True

        if self.solvable():
            previous_grid = self.painted_grid.copy()
            previous_certainty = self.cell_is_known.copy()

            self.in_between_deterministic_fills()

            if self.solvable():
                return True
            else:
                self.painted_grid = previous_grid
                self.cell_is_known = previous_certainty
                return True
        else:
            self.painted_grid = previous_grid
            self.cell_is_known = previous_certainty
            return False

        # self.in_between_deterministic_fills()

        # if self.solvable():
        #     return True
        # else:
        #     self.painted_grid = previous_grid
        #     self.cell_is_known = previous_certainty
        #     return False

    def in_between_deterministic_fills(self):
        self.deterministic_fill_iterative()

    @property
    def shape(self) -> tuple[int]:
        return self.grid.shape

    @property
    def board_state_as_int(self) -> int:
        return int(
            "".join(np.abs(self.painted_grid.flatten()).astype(str)),
            base=2,
        )

    @staticmethod
    def diagonals(grid: np.ndarray, row: int, col: int) -> bool:
        return list(
            filter(
                lambda x: x[-1] is not None,
                [
                    ((row - 1, col - 1), grid[row - 1][col - 1])
                    if row > 1 and col > 1
                    else (None, None),
                    ((row - 1, col + 1), grid[row - 1][col + 1])
                    if row > 1 and col < grid.shape[0] - 1
                    else (None, None),
                    ((row + 1, col - 1), grid[row + 1][col - 1])
                    if row < grid.shape[0] - 1 and col > 1
                    else (None, None),
                    ((row + 1, col + 1), grid[row + 1][col + 1])
                    if row < grid.shape[0] - 1 and col < grid.shape[0] - 1
                    else (None, None),
                ],
            )
        )

    def complement_diagonals(self, cell_0: tuple[int, int], cell_1: tuple[int, int]):
        """Scenarios
        [', 1]
        [0, ']

        [', 0]
        [1, ']

        [1, ']
        [', 0]

        [0, ']
        [', 1]
        """
        cell_0_row, cell_0_column = cell_0
        cell_1_row, cell_1_column = cell_1
        if cell_0_row + 1 == cell_1_row and cell_0_column + 1 == cell_1_column:
            # scenario:
            # [c0, ']
            # [', c1]
            return (cell_0_row, cell_1_column), (cell_1_row, cell_0_column)
        elif cell_0_row - 1 == cell_1_row and cell_0_column - 1 == cell_1_column:
            # ! can be wrong. Check
            # scenario:
            # [', 0]
            # [1, ']
            return (cell_0_row, cell_1_column), (cell_1_row, cell_0_column)
            # return (cell_0_row, cell_0_column - 1), (cell_0_row - 1, cell_0_column)
        elif cell_1_row + 1 == cell_0_row and cell_1_column - 1 == cell_0_column:
            # scenario:
            # [', c1]
            # [c0, ']
            return (cell_0_row, cell_1_column), (cell_1_row, cell_0_column)
        elif cell_0_row + 1 == cell_1_row and cell_0_column - 1 == cell_1_column:
            # scenario
            # [', c0]
            # [c1, ']
            return (cell_0_row, cell_1_column), (cell_1_row, cell_0_column)
        else:
            return ()

    @cached_property
    def board_coordinates(self) -> list[tuple[int, int]]:
        return [
            (i, j) for i in range(self.grid.shape[0]) for j in range(self.grid.shape[1])
        ]

    @cached_property
    def manhattan_distances_within_board(self):
        return scipy_distance.squareform(
            scipy_distance.pdist(
                self.board_coordinates,
                metric="cityblock",
            )
        )

    def cityblock_distance_between_two_cells(
        self, cell_0: tuple[int, int], cell_1: tuple[int, int]
    ):
        return self.manhattan_distances_within_board[
            self.board_coordinates.index(cell_0), self.board_coordinates.index(cell_1)
        ]

    def islands_that_reach_here(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        # does not consider black cells that block a pathway
        return [
            hint_cell
            for hint_cell in self.hint_cells
            if self.cityblock_distance_between_two_cells(hint_cell, cell) + 1
            <= self.grid[hint_cell]
        ]

    @cached_property
    def hint_cells(self) -> list[tuple[int, int]]:
        return [tuple(hint_cell) for hint_cell in np.argwhere(self.grid != 0).tolist()]

    def neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        return list(
            filter(
                lambda x: x is not None,
                [
                    (row - 1, col) if row > 0 else None,
                    (row + 1, col) if row < self.grid_size - 1 else None,
                    (row, col - 1) if col > 0 else None,
                    (row, col + 1) if col < self.grid_size - 1 else None,
                ],
            )
        )

    def deterministic_fill_one_timer(self):
        self.painted_grid[self.grid != 0] = CellCategory.WHITE.value
        self.cell_is_known[self.grid != 0] = True

        # ! one islands
        for one_island in np.argwhere(self.grid == 1):
            try:
                coord = one_island[0], one_island[1] + 1
                if not any(map(lambda x: x == -1, coord)):
                    self.paint_cell(*coord, CellCategory.BLACK)
            except IndexError:
                pass
            try:
                coord = one_island[0] + 1, one_island[1]
                if not any(map(lambda x: x == -1, coord)):
                    self.paint_cell(*coord, CellCategory.BLACK)
            except IndexError:
                pass
            try:
                coord = one_island[0], one_island[1] - 1
                if not any(map(lambda x: x == -1, coord)):
                    self.paint_cell(*coord, CellCategory.BLACK)
            except IndexError:
                pass
            try:
                coord = one_island[0] - 1, one_island[1]
                if not any(map(lambda x: x == -1, coord)):
                    self.paint_cell(*coord, CellCategory.BLACK)
            except IndexError:
                pass

        # ! fields of contact - guaranteeing island isolation
        for ix, iy in np.ndindex(self.grid.shape):
            if (
                (ix < self.grid_size - 2)
                and self.grid[ix, iy] > 0
                and self.grid[ix + 2, iy] > 0
            ):
                self.paint_cell(ix + 1, iy, CellCategory.BLACK)
            if (
                (iy < self.grid_size - 2)
                and self.grid[ix, iy] > 0
                and self.grid[ix, iy + 2] > 0
            ):
                self.paint_cell(ix, iy + 1, CellCategory.BLACK)

        # ! adjacent hints
        for hint_cell in self.hint_cells:
            for diag_coord, diag_value in self.diagonals(
                self.grid, hint_cell[0], hint_cell[1]
            ):
                if diag_value != 0:
                    for complement_cell_diagonal in self.complement_diagonals(
                        hint_cell, diag_coord
                    ):
                        self.paint_cell(*complement_cell_diagonal, CellCategory.BLACK)

        # ! unreachable cells due to island size
        manhattan_distances = scipy_distance.squareform(
            scipy_distance.pdist(
                self.board_coordinates,
                metric="cityblock",
            )
        )
        reachable = np.full_like(manhattan_distances, False, dtype=bool)
        for hint_cell in self.hint_cells:
            island_size = self.grid[hint_cell[0], hint_cell[1]]
            island_cell_index = self.board_coordinates.index(hint_cell)
            reachable[island_cell_index, :] = (
                manhattan_distances[island_cell_index, :] < island_size
            )
        reachable_subset = reachable[
            [self.board_coordinates.index(hint_cell) for hint_cell in self.hint_cells],
            :,
        ]
        reachable_by_any = reachable_subset.any(axis=0)
        for coordinate, reachable in zip(self.board_coordinates, reachable_by_any):
            if not reachable:
                self.paint_cell(*coordinate, CellCategory.BLACK)

    def black_pool_exists(self) -> bool:
        for ix, iy in np.ndindex(self.painted_grid.shape):
            if all(
                map(
                    lambda cell_value: cell_value == CellCategory.BLACK.value,
                    [
                        self.painted_grid[ix, iy],
                        self.painted_grid[ix + 1, iy]
                        if ix < self.grid_size - 1
                        else None,
                        self.painted_grid[ix, iy + 1]
                        if iy < self.grid_size - 1
                        else None,
                        self.painted_grid[ix + 1, iy + 1]
                        if ix < self.grid_size - 1 and iy < self.grid_size - 1
                        else None,
                    ],
                )
            ):
                return True
        return False

    def get_islands(self) -> list:
        islands = []
        for hint_cell in self.hint_cells:
            island = [hint_cell]
            neighborhood = self.neighbors(*hint_cell)
            for neighbor in neighborhood:
                if (
                    self.painted_grid[neighbor] == CellCategory.WHITE.value
                    and neighbor not in island
                ):
                    island.append(neighbor)
                    neighborhood.extend(self.neighbors(*neighbor))  # list concat
            # print(island)
            islands.append(island)

        return islands

    # def get_islands_with_size(self) -> dict[int, list]:
    #     islands = {}
    #     for hint_cell in self.hint_cells:
    #         islands[self.grid[hint_cell]] = [hint_cell]
    #         neighborhood = self.neighbors(*hint_cell)
    #         for neighbor in neighborhood:
    #             if (
    #                 self.painted_grid[neighbor] == CellCategory.WHITE.value
    #                 and neighbor not in islands[self.grid[hint_cell]]
    #             ):
    #                 islands[self.grid[hint_cell]].append(neighbor)
    #                 neighborhood.extend(self.neighbors(*neighbor))  # list concat

    #     return islands

    def get_islands_with_size_alternative(self) -> dict[int, list]:
        islands = {}
        for hint_cell in self.hint_cells:
            islands[hint_cell] = {"hint": hint_cell}
            islands[hint_cell]["size"] = self.grid[hint_cell]
            islands[hint_cell]["cells"] = [hint_cell]
            neighborhood = self.neighbors(*hint_cell)
            for neighbor in neighborhood:
                if (
                    self.painted_grid[neighbor] == CellCategory.WHITE.value
                    and neighbor not in islands[hint_cell]["cells"]
                ):
                    islands[hint_cell]["cells"].append(neighbor)
                    neighborhood.extend(self.neighbors(*neighbor))  # list concat

        return islands

    def get_completed_islands(self) -> list:
        return [
            island_info["cells"]
            for _, island_info in self.get_islands_with_size_alternative().items()
            if len(island_info["cells"]) == island_info["size"]
        ]

    def get_completed_islands_info(self) -> list:
        return [
            island_info
            for _, island_info in self.get_islands_with_size_alternative().items()
            if len(island_info["cells"]) == island_info["size"]
        ]

    def get_incomplete_islands_info(self) -> list:
        return [
            island_info
            for _, island_info in self.get_islands_with_size_alternative().items()
            if len(island_info["cells"]) != island_info["size"]
        ]

    def get_incomplete_islands_sizes(self) -> list:
        return [
            island_info["size"]
            for _, island_info in self.get_islands_with_size_alternative().items()
            if len(island_info["cells"]) != island_info["size"]
        ]

    def solved(self) -> bool:
        # ! right-sized islands. Also covers no island intersection
        for isle in self.get_islands():
            if len(isle) != self.grid[isle[0][0], isle[0][1]]:
                return False

        # ! continuous black sea
        if not is_continuous_path(self.painted_grid == CellCategory.BLACK.value):
            return False

        # ! no pools
        if self.black_pool_exists():
            return False

        return True

    def constructable(
        self,
        island: list[tuple[int, int]],
        other_islands: list[list[tuple[int, int]]],
    ) -> bool:
        # if neighbors belong to other islands or black sea, island is not constructable
        current_island_size = len(island)
        island_hint_cell = island[0]
        island_size_requirement = self.grid[island_hint_cell]

        if len(island) == island_size_requirement:
            return True

        region_of_influence = [island_hint_cell]
        neighborhood = self.neighbors(*island_hint_cell)
        discarded_neighbors = []
        n_steps = 0
        for neighbor in neighborhood:
            n_steps += 1

            if (
                (
                    self.painted_grid[neighbor] == CellCategory.WHITE.value
                    or self.painted_grid[neighbor] == CellCategory.UNKNOWN.value
                )
                and neighbor not in region_of_influence
                and neighbor not in discarded_neighbors
                and all(neighbor not in other_island for other_island in other_islands)
            ):
                region_of_influence.append(neighbor)
                neighborhood.extend(self.neighbors(*neighbor))  # list concat
            else:
                discarded_neighbors.append(neighbor)

            if len(region_of_influence) + len(discarded_neighbors) == self.board_size:
                return (
                    current_island_size + len(region_of_influence) - 1
                    >= island_size_requirement
                )

            if n_steps > 1000:
                return (
                    current_island_size + len(region_of_influence) - 1
                    >= island_size_requirement
                )

        return (
            current_island_size + len(region_of_influence) - 1
            >= island_size_requirement
        )

    def solvable(self) -> bool:
        # remaining_cells = np.where(self.painted_grid == -1)
        # if len(np.where(self.painted_grid.flatten() == -1)[0]) == 2:
        #     color_combinations = list(
        #         itertools.product(
        #             [CellCategory.WHITE.value, CellCategory.BLACK.value],
        #             [CellCategory.WHITE.value, CellCategory.BLACK.value],
        #         )
        #     )
        #     for color_combination in color_combinations:
        #         previous_grid = self.painted_grid.copy()
        #         previous_certainty = self.cell_is_known.copy()
        #         self.painted_grid[remaining_cells[0]] = color_combination[0]
        #         self.cell_is_known[remaining_cells[0]] = True
        #         self.painted_grid[remaining_cells[1]] = color_combination[1]
        #         self.cell_is_known[remaining_cells[0]] = True
        #         if self.solved():
        #             return True
        #         else:
        #             self.painted_grid = previous_grid
        #             self.cell_is_known = previous_certainty

        # if len(np.where(self.painted_grid.flatten() == -1)[0]) == 1:
        #     rem_cell = (remaining_cells[0][0], remaining_cells[1][0])
        #     for color in [CellCategory.WHITE.value, CellCategory.BLACK.value]:
        #         previous_grid = self.painted_grid.copy()
        #         previous_certainty = self.cell_is_known.copy()
        #         self.painted_grid[rem_cell] = color
        #         self.cell_is_known[rem_cell] = True
        #         if self.solved():
        #             return True
        #         else:
        #             self.painted_grid = previous_grid
        #             self.cell_is_known = previous_certainty
        #     return False

        # ! islands solved or can be constructed
        archipelago = self.get_islands()
        for isle_number, isle in enumerate(archipelago):
            if len(isle) > self.grid[isle[0][0], isle[0][1]]:
                return False
            if not self.constructable(
                isle,
                [archipelago[i] for i in range(len(archipelago)) if i != isle_number],
            ):
                return False

        # ! no pools
        if self.black_pool_exists():
            return False

        # ! black sea would be continuous if all remaining cells were black
        adapted_grid = self.painted_grid.copy()
        adapted_grid[
            adapted_grid == CellCategory.UNKNOWN.value
        ] = CellCategory.BLACK.value
        if not is_continuous_path(adapted_grid == CellCategory.BLACK.value):
            return False

        return True

    def is_terminal_state(self):
        return self.solved() or not self.solvable()

    def plot(self):
        ax = plot_grid(self.painted_grid)
        for hint_cell in self.hint_cells:
            ax.text(
                hint_cell[1],
                hint_cell[0],
                self.grid[
                    hint_cell[0],
                    hint_cell[1],
                ],
                ha="center",
                va="center",
            )
