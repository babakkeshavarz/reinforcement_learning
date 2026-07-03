import random


class ComplexRandomCliffEnv:
    """
    Randomized cliff maze environment used for hyperparameter experiments.

    Legend:
        S = start
        G = goal
        X = wall
        C = cliff / danger
        . = safe cell

    Rewards:
        normal move     = -1
        wall/boundary   = -5 and stay in place
        goal            = +20
        cliff           = -100 and reset to start
    """

    def __init__(self, rows=9, cols=12, maze_seed=7, cliff_probability=0.22):
        self.rows = rows
        self.cols = cols
        self.maze_seed = maze_seed
        self.cliff_probability = cliff_probability
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)

        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }

        self.action_names = {
            0: "Up",
            1: "Down",
            2: "Left",
            3: "Right",
        }

        self.grid = self._generate_grid()
        self.agent_position = self.start
        self.fell_off_cliff = False

    def _generate_grid(self):
        rng = random.Random(self.maze_seed)
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]

        safe_path = set()

        # Guaranteed safe path from start to goal.
        for r in range(self.rows - 1, 1, -1):
            safe_path.add((r, 0))

        for c in range(self.cols):
            safe_path.add((2, c))

        for r in range(2, self.rows):
            safe_path.add((r, self.cols - 1))

        # Extra corridor for alternative paths.
        for c in range(self.cols):
            if c not in [3, 7]:
                safe_path.add((5, c))

        for r in range(2, 6):
            safe_path.add((r, 4))

        fixed_walls = {
            (1, 2), (1, 3), (1, 7), (1, 8),
            (3, 2), (3, 3), (3, 7), (3, 8),
            (4, 6),
            (6, 2), (6, 3), (6, 8), (6, 9),
            (7, 5), (7, 6),
        }

        for pos in fixed_walls:
            if pos not in safe_path and pos not in [self.start, self.goal]:
                r, c = pos
                grid[r][c] = "X"

        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)

                if pos in safe_path or pos in [self.start, self.goal]:
                    continue

                if grid[r][c] == "X":
                    continue

                local_probability = self.cliff_probability

                if r >= self.rows - 3:
                    local_probability += 0.10

                if rng.random() < local_probability:
                    grid[r][c] = "C"

        sr, sc = self.start
        gr, gc = self.goal
        grid[sr][sc] = "S"
        grid[gr][gc] = "G"

        return grid

    def reset(self):
        self.agent_position = self.start
        self.fell_off_cliff = False
        return self.agent_position

    def is_inside(self, position):
        r, c = position
        return 0 <= r < self.rows and 0 <= c < self.cols

    def cell_type(self, position):
        r, c = position
        return self.grid[r][c]

    def step(self, action):
        self.fell_off_cliff = False

        r, c = self.agent_position
        dr, dc = self.actions[action]
        next_position = (r + dr, c + dc)

        if not self.is_inside(next_position):
            return self.agent_position, -5, False

        cell = self.cell_type(next_position)

        if cell == "X":
            return self.agent_position, -5, False

        if cell == "C":
            self.agent_position = self.start
            self.fell_off_cliff = True
            return self.start, -100, False

        if next_position == self.goal:
            self.agent_position = next_position
            return next_position, 20, True

        self.agent_position = next_position
        return next_position, -1, False
