import random
from pathlib import Path

import pandas as pd
import torch

from snake_env import SnakeEnv
from dqn_agent import DQNAgent


def train_dqn(episodes=3000, seed=1):
    """
    Train DQN.

    DQN needs more time than tabular agents.
    For better results, use 5000 to 10000 episodes.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    env = SnakeEnv(width=10, height=10, seed=seed)
    state = env.reset()

    agent = DQNAgent(
        state_size=len(state),
        action_size=3,
        gamma=0.95,
        epsilon=0.6,
        epsilon_min=0.02,
        epsilon_decay=0.997,
        learning_rate=0.001,
        batch_size=64,
        train_every=4,
        seed=seed,
    )

    results = []
    best_score = -1

    assets = Path("assets")
    assets.mkdir(exist_ok=True)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        losses = []

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()

            if loss:
                losses.append(loss)

            total_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()

        score = info["score"]
        avg_loss = sum(losses) / len(losses) if losses else 0

        if score > best_score:
            best_score = score
            agent.save(assets / "dqn_best_agent.pt")

        results.append(
            {
                "episode": episode,
                "score": score,
                "reward": total_reward,
                "epsilon": agent.epsilon,
                "loss": avg_loss,
                "best_score_so_far": best_score,
            }
        )

        if (episode + 1) % 500 == 0:
            recent = results[-100:]
            avg_score = sum(row["score"] for row in recent) / len(recent)
            print(
                f"DQN episode {episode + 1}, "
                f"avg score last 100: {avg_score:.2f}, "
                f"best score: {best_score}"
            )

    agent.save(assets / "dqn_last_agent.pt")
    pd.DataFrame(results).to_csv(assets / "dqn_training.csv", index=False)

    return agent, results


if __name__ == "__main__":
    train_dqn()
