import os


class Problem:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.fields = [[None for _ in range(y)] for _ in range(x)]
        self.solution = None

    def add_field(self, x, y, value):
        self.fields[x][y] = value

    def add_solution(self, solution: list[list[bool]]):
        self.solution = solution

    def __str__(self):
        lines = []
        for y in range(self.y):
            line = ""
            for x in range(self.x):
                elem = self.fields[x][y]
                if elem is not None:
                    line += str(elem)
                elif self.solution is not None:
                    if self.solution[x][y] and self.fields[x][y] is not None:
                        raise RuntimeError("Algorithm error: wall at a number!")
                    line += "x" if self.solution[x][y] else "."
                else:
                    line += "."
            lines.append(line)
        return "\n".join(lines)


def DFS(visited, x, y, moves_left):
    if (x, y) in visited:
        return

    if moves_left == 0:
        return

    visited.add((x, y))

    # ignore boundaries once again
    for new_x, new_y in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        DFS(visited, new_x, new_y, moves_left - 1)


def try_find_wall(problem_instance: Problem):
    # neighbour numbers
    tiles_with_neighbours = set()
    for y in range(problem_instance.y):
        for x in range(problem_instance.x):
            if problem_instance.fields[x][y] is None:
                continue
            # we can ignore tiles outside boundary
            for pos in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if pos in tiles_with_neighbours:
                    return pos
                tiles_with_neighbours.add(pos)

    # unreachable tiles
    visited = set()
    for y in range(problem_instance.y):
        for x in range(problem_instance.x):
            if problem_instance.fields[x][y] is None:
                continue
            DFS(visited, x, y, problem_instance.fields[x][y])
    for y in range(problem_instance.y):
        for x in range(problem_instance.x):
            if (x, y) not in visited:
                return x, y

    return None


def read_problem_grid(path):
    with open(path, "r") as f:
        data = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            data.append(line)

        xt = len(data[0])
        yt = len(data)

        instance = Problem(xt, yt)
        for y, line in enumerate(data):
            for x, char in enumerate(line):
                if "0" <= char <= "9":
                    instance.add_field(x, y, int(char))
        return instance
