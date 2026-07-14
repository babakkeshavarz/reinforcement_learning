import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    DQN agent.

    DQN learns:
        state -> Q-values for all actions

    It uses:
        - neural network
        - replay memory
        - target network
        - epsilon decay
    """

    def __init__(
        self,
        state_size,
        action_size=3,
        gamma=0.95,
        epsilon=0.5,
        epsilon_min=0.02,
        epsilon_decay=0.997,
        learning_rate=0.001,
        memory_size=20_000,
        batch_size=64,
        target_update_every=250,
        train_every=4,
        seed=1,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.train_every = train_every
        self.training_steps = 0
        self.environment_steps = 0

        self.memory = deque(maxlen=memory_size)

        self.policy_net = DQNNetwork(state_size, action_size)
        self.target_net = DQNNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def _state_tensor(self, state):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            q_values = self.policy_net(self._state_tensor(state))

        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        self.environment_steps += 1

        if len(self.memory) < self.batch_size:
            return 0.0

        if self.environment_steps % self.train_every != 0:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)

        states = torch.tensor(np.array([item[0] for item in batch]), dtype=torch.float32)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([item[3] for item in batch]), dtype=torch.float32)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(
            {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        agent = cls(
            state_size=checkpoint["state_size"],
            action_size=checkpoint["action_size"],
        )

        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.epsilon = 0

        return agent
