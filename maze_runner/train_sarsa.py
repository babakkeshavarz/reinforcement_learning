import random
import numpy as np

from complex_random_cliff_env import ComplexRandomCliffEnv
from sarsa_agent import SarsaAgent
from visualization import get_final_path, save_path_gif, save_path_png, save_maze_png


def train_sarsa(episodes=10000, max_steps_per_episode=500, seed=1, maze_seed=7):
    """
    Train SARSA on the randomized cliff maze.

    This version uses epsilon decay:
        high exploration at the beginning
        low exploration near the end

    SARSA is on-policy, so its learning update includes the effect of
    exploration. This can make it more cautious around cliffs.
    """
    np.random.seed(seed)
    random.seed(seed)

    env = ComplexRandomCliffEnv(maze_seed=maze_seed)
    agent = SarsaAgent(
        rows=env.rows,
        cols=env.cols,
        number_of_actions=len(env.actions),
        alpha=0.1,
        gamma=0.95,
        epsilon=0.30,
    )

    min_epsilon = 0.02
    epsilon_decay = 0.9995

    rewards = []
    steps = []
    cliff_falls = []

    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)

        total_reward = 0
        falls = 0
        done = False

        for step in range(max_steps_per_episode):
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                done=done,
            )

            if env.fell_off_cliff:
                falls += 1

            total_reward += reward
            state = next_state
            action = next_action

            if done:
                break

        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        rewards.append(total_reward)
        steps.append(step + 1)
        cliff_falls.append(falls)

    return env, agent, rewards, steps, cliff_falls


if __name__ == "__main__":
    env, agent, rewards, steps, cliff_falls = train_sarsa()
    path, path_falls = get_final_path(env, agent)

    save_maze_png(env, "generated_random_maze.png")
    gif_path = save_path_gif(env, path, "sarsa_random_cliff.gif", "SARSA final path")
    png_path = save_path_png(env, path, "sarsa_random_cliff.png", "SARSA final path")

    print("SARSA complete.")
    print(f"Final path: {path}")
    print(f"Greedy playback cliff falls: {path_falls}")
    print(f"GIF saved to: {gif_path}")
    print(f"PNG saved to: {png_path}")
