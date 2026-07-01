import random
import numpy as np


class SarsaAgent:
    """
    SARSA agent.

    SARSA is on-policy.

    It updates using the next action the agent actually selected:

        Q[next_state, next_action]

    In risky environments, this can make SARSA more cautious because it learns
    from its real exploratory behaviour.
    """

    def __init__(self, rows, cols, number_of_actions, alpha=0.1, gamma=0.95, epsilon=0.15):
        self.rows = rows
        self.cols = cols
        self.number_of_actions = number_of_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((rows, cols, number_of_actions))

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        r, c = state

        if random.random() < self.epsilon:
            return random.randint(0, self.number_of_actions - 1)

        return int(np.argmax(self.q_table[r, c]))

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update.

        Q(s,a) = Q(s,a) + alpha * [reward + gamma * Q(s',a') - Q(s,a)]
        """
        r, c = state
        nr, nc = next_state

        old_q = self.q_table[r, c, action]

        if done:
            future_q = 0
        else:
            future_q = self.q_table[nr, nc, next_action]

        new_q = old_q + self.alpha * (
            reward + self.gamma * future_q - old_q
        )

        self.q_table[r, c, action] = new_q

    def get_best_action(self, state):
        """
        Choose the best learned action after training.
        """
        r, c = state
        return int(np.argmax(self.q_table[r, c]))
