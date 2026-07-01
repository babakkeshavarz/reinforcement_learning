import random
import numpy as np

from complex_random_cliff_env import ComplexRandomCliffEnv
from q_learning_agent import QLearningAgent
from visualization import get_final_path, save_path_gif, save_path_png, save_maze_png


def train_q_learning(episodes=10000, max_steps_per_episode=500, seed=1, maze_seed=7):
    """
    Train Q-learning on the randomized cliff maze.

    This version uses epsilon decay:
        high exploration at the beginning
        low exploration near the end
    """
    np.random.seed(seed)
    random.seed(seed)

    env = ComplexRandomCliffEnv(maze_seed=maze_seed)
    agent = QLearningAgent(
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
        total_reward = 0
        falls = 0
        done = False

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            if env.fell_off_cliff:
                falls += 1

            total_reward += reward
            state = next_state

            if done:
                break

        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        rewards.append(total_reward)
        steps.append(step + 1)
        cliff_falls.append(falls)

    return env, agent, rewards, steps, cliff_falls


if __name__ == "__main__":
    env, agent, rewards, steps, cliff_falls = train_q_learning()
    path, path_falls = get_final_path(env, agent)

    save_maze_png(env, "generated_random_maze.png")
    gif_path = save_path_gif(env, path, "q_learning_random_cliff.gif", "Q-learning final path")
    png_path = save_path_png(env, path, "q_learning_random_cliff.png", "Q-learning final path")

    print("Q-learning complete.")
    print(f"Final path: {path}")
    print(f"Greedy playback cliff falls: {path_falls}")
    print(f"GIF saved to: {gif_path}")
    print(f"PNG saved to: {png_path}")
