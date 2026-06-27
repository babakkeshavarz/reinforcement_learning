import numpy as np
import random


class QLearningAgent:
    """
    The agent is the learner.

    This file contains only the learning logic:
    - Q-table
    - action selection
    - Q-value updates
    - Q-table display

    The agent does not know the maze layout directly.
    It only receives state, action, reward and next_state.
    """

    def __init__(
        self,
        rows,
        cols,
        number_of_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
    ):
        self.rows = rows
        self.cols = cols
        self.number_of_actions = number_of_actions

        # Learning rate: how strongly new information changes old Q-values.
        self.alpha = alpha

        # Discount factor: how much the agent cares about future rewards.
        self.gamma = gamma

        # Exploration rate: chance of taking a random action.
        self.epsilon = epsilon

        # Q-table shape:
        # rows x cols x actions
        #
        # For a 5 x 5 maze with 4 actions:
        # 5 x 5 x 4
        #
        # Each cell stores:
        # [Up, Down, Left, Right]
        self.q_table = np.zeros((rows, cols, number_of_actions))

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy selection.

        Sometimes the agent explores randomly.
        Otherwise, it chooses the action with the highest Q-value.
        """
        r, c = state

        if random.random() < self.epsilon:
            return random.randint(0, self.number_of_actions - 1)

        return int(np.argmax(self.q_table[r, c]))

    def update(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning formula.

        new_q = old_q + alpha * (reward + gamma * future_best_q - old_q)
        """
        r, c = state
        nr, nc = next_state

        old_q = self.q_table[r, c, action]
        future_best_q = np.max(self.q_table[nr, nc])

        new_q = old_q + self.alpha * (
            reward + self.gamma * future_best_q - old_q
        )

        self.q_table[r, c, action] = new_q

        return old_q, new_q, future_best_q

    def get_best_action(self, state):
        """
        Return the best learned action for one state.
        Used after training to show the final route.
        """
        r, c = state
        return int(np.argmax(self.q_table[r, c]))

    def print_current_q_values(self, state):
        """
        Print all four Q-values for the current state.
        """
        r, c = state
        values = self.q_table[r, c]

        print("Q-values for current position:")
        print(f"Position: {state}")
        print(f"Up:    {values[0]:7.2f}")
        print(f"Down:  {values[1]:7.2f}")
        print(f"Left:  {values[2]:7.2f}")
        print(f"Right: {values[3]:7.2f}")
        print()

    def print_best_q_matrix(self, maze, goal):
        """
        Print a simplified Q-table.

        The actual Q-table has four values per cell.
        This display shows only the best action and its value.
        """
        arrows = {
            0: "^",
            1: "v",
            2: "<",
            3: ">",
        }

        print("Best Q-value matrix:")
        print("Each open cell shows best action and best Q-value.")
        print()

        for r in range(self.rows):
            line = ""

            for c in range(self.cols):
                if maze[r][c] == "X":
                    line += "   X       "
                elif (r, c) == goal:
                    line += "   G       "
                else:
                    best_action = int(np.argmax(self.q_table[r, c]))
                    best_value = np.max(self.q_table[r, c])
                    line += f"{arrows[best_action]} {best_value:6.2f}  "

            print(line)

        print()

    def print_full_q_table(self, maze, goal):
        """
        Print the actual final Q-table.

        Each open cell shows:
        [Up, Down, Left, Right]
        """
        print("Full Q-table:")
        print("Each open cell shows: [Up, Down, Left, Right]")
        print()

        for r in range(self.rows):
            for c in range(self.cols):
                if maze[r][c] == "X":
                    print(f"Cell ({r},{c}) = WALL")
                elif (r, c) == goal:
                    print(f"Cell ({r},{c}) = GOAL")
                else:
                    values = np.round(self.q_table[r, c], 2)
                    print(f"Cell ({r},{c}) = {values}")
            print()
