import random
from pathlib import Path

import pandas as pd

from snake_env import SnakeEnv
from agents import QLearningAgent


def train_q_learning(episodes=5000, seed=1):
    """
    Train Q-learning.

    For quick tests, reduce episodes.
    For better results, use 5000+.
    """
    random.seed(seed)

    env = SnakeEnv(width=10, height=10, seed=seed)
    agent = QLearningAgent(alpha=0.12, gamma=0.95, epsilon=0.5)

    min_epsilon = 0.02
    epsilon_decay = 0.997

    results = []
    best_score = -1

    assets = Path("assets")
    assets.mkdir(exist_ok=True)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        score = info["score"]

        if score > best_score:
            best_score = score
            agent.save(assets / "q_learning_best_agent.pkl")

        results.append(
            {
                "episode": episode,
                "score": score,
                "reward": total_reward,
                "epsilon": agent.epsilon,
                "best_score_so_far": best_score,
            }
        )

        if (episode + 1) % 500 == 0:
            recent = results[-100:]
            avg_score = sum(row["score"] for row in recent) / len(recent)
            print(
                f"Q-learning episode {episode + 1}, "
                f"avg score last 100: {avg_score:.2f}, "
                f"best score: {best_score}"
            )

    agent.save(assets / "q_learning_last_agent.pkl")
    pd.DataFrame(results).to_csv(assets / "q_learning_training.csv", index=False)

    return agent, results


if __name__ == "__main__":
    train_q_learning()
