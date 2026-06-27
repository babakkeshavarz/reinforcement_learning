class MazeEnv:
    """
    The environment is the world the agent interacts with.

    This file contains only the maze logic:
    - maze layout
    - start and goal positions
    - valid and invalid moves
    - rewards
    - terminal display

    It does not contain Q-learning.
    """

    def __init__(self):
        self.maze = [
            ["S", ".", ".", ".", "."],
            [".", "X", "X", ".", "."],
            [".", ".", ".", "X", "."],
            [".", "X", ".", ".", "."],
            [".", ".", ".", ".", "G"],
        ]

        self.rows = len(self.maze)
        self.cols = len(self.maze[0])

        self.start = (0, 0)
        self.goal = (4, 4)

        # Action indexes:
        # 0 = up, 1 = down, 2 = left, 3 = right
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

        self.agent_position = self.start

    def reset(self):
        """
        Start a new episode and return the starting state.
        """
        self.agent_position = self.start
        return self.agent_position

    def is_valid_position(self, position):
        """
        A position is valid if it is inside the maze and not a wall.
        """
        r, c = position

        inside_maze = 0 <= r < self.rows and 0 <= c < self.cols
        if not inside_maze:
            return False

        if self.maze[r][c] == "X":
            return False

        return True

    def step(self, action):
        """
        Move the agent one step.

        Returns:
            next_state, reward, done

        next_state:
            the new agent location

        reward:
            -5 for hitting a wall or boundary
            -1 for a normal move
            10 for reaching the goal

        done:
            True if the goal is reached
        """
        r, c = self.agent_position
        dr, dc = self.actions[action]

        next_position = (r + dr, c + dc)

        if not self.is_valid_position(next_position):
            reward = -5
            done = False
            next_position = self.agent_position

        elif next_position == self.goal:
            reward = 10
            done = True
            self.agent_position = next_position

        else:
            reward = -1
            done = False
            self.agent_position = next_position

        return next_position, reward, done

    def render(self, visited=None):
        """
        Print the maze in the terminal.

        A = agent
        S = start
        G = goal
        X = wall
        * = already visited in this episode
        """
        if visited is None:
            visited = []

        for r in range(self.rows):
            line = ""

            for c in range(self.cols):
                position = (r, c)

                if position == self.agent_position:
                    line += "A "
                elif position == self.start:
                    line += "S "
                elif position == self.goal:
                    line += "G "
                elif self.maze[r][c] == "X":
                    line += "X "
                elif position in visited:
                    line += "* "
                else:
                    line += ". "

            print(line)

        print()
