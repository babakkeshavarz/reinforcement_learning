import random
import numpy as np


class QLearningAgent:
    """
    Q-learning is off-policy.

    It updates using the best possible next action:
        max(Q[next_state])
    """

    def __init__(self, rows, cols, number_of_actions, alpha, gamma, epsilon):
        self.rows = rows
        self.cols = cols
        self.number_of_actions = number_of_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((rows, cols, number_of_actions))

    def choose_action(self, state):
        r, c = state

        if random.random() < self.epsilon:
            return random.randint(0, self.number_of_actions - 1)

        return int(np.argmax(self.q_table[r, c]))

    def update(self, state, action, reward, next_state):
        r, c = state
        nr, nc = next_state

        old_q = self.q_table[r, c, action]
        future_best_q = np.max(self.q_table[nr, nc])

        new_q = old_q + self.alpha * (
            reward + self.gamma * future_best_q - old_q
        )

        self.q_table[r, c, action] = new_q


class SarsaAgent:
    """
    SARSA is on-policy.

    It updates using the next action the agent actually selected:
        Q[next_state, next_action]
    """

    def __init__(self, rows, cols, number_of_actions, alpha, gamma, epsilon):
        self.rows = rows
        self.cols = cols
        self.number_of_actions = number_of_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((rows, cols, number_of_actions))

    def choose_action(self, state):
        r, c = state

        if random.random() < self.epsilon:
            return random.randint(0, self.number_of_actions - 1)

        return int(np.argmax(self.q_table[r, c]))

    def update(self, state, action, reward, next_state, next_action, done):
        r, c = state
        nr, nc = next_state

        old_q = self.q_table[r, c, action]
        future_q = 0 if done else self.q_table[nr, nc, next_action]

        new_q = old_q + self.alpha * (
            reward + self.gamma * future_q - old_q
        )

        self.q_table[r, c, action] = new_q
