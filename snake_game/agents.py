import random
import pickle


class QLearningAgent:
    """
    Tabular Q-learning agent.

    Off-policy:
        learns from max Q-value in the next state.
    """

    def __init__(self, actions=3, alpha=0.1, gamma=0.95, epsilon=0.3):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = {}

    def get_values(self, state):
        if state not in self.q:
            self.q[state] = [0.0 for _ in range(self.actions)]
        return self.q[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)

        values = self.get_values(state)
        return max(range(self.actions), key=lambda action: values[action])

    def update(self, state, action, reward, next_state, done):
        values = self.get_values(state)
        old_q = values[action]

        if done:
            future_best_q = 0.0
        else:
            future_best_q = max(self.get_values(next_state))

        values[action] = old_q + self.alpha * (
            reward + self.gamma * future_best_q - old_q
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            agent = pickle.load(f)
        agent.epsilon = 0
        return agent


class SarsaAgent:
    """
    Tabular SARSA agent.

    On-policy:
        learns from the next action it actually selected.
    """

    def __init__(self, actions=3, alpha=0.1, gamma=0.95, epsilon=0.3):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = {}

    def get_values(self, state):
        if state not in self.q:
            self.q[state] = [0.0 for _ in range(self.actions)]
        return self.q[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)

        values = self.get_values(state)
        return max(range(self.actions), key=lambda action: values[action])

    def update(self, state, action, reward, next_state, next_action, done):
        values = self.get_values(state)
        old_q = values[action]

        if done:
            future_q = 0.0
        else:
            future_q = self.get_values(next_state)[next_action]

        values[action] = old_q + self.alpha * (
            reward + self.gamma * future_q - old_q
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            agent = pickle.load(f)
        agent.epsilon = 0
        return agent
