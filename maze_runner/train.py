import os
import random
import time

import numpy as np

from maze_env import MazeEnv
from agent import QLearningAgent


# Seeds make the random parts reproducible.
np.random.seed(1)
random.seed(1)


def clear_screen():
    """
    Clear the terminal so the animation looks cleaner.
    """
    os.system("cls" if os.name == "nt" else "clear")


def show_training_step(
    env,
    agent,
    episode,
    step_number,
    state,
    action,
    reward,
    total_reward,
    next_state,
    visited,
    old_q,
    new_q,
    future_best_q,
    animation_speed,
):
    """
    Display one training step.

    This function is only for visualization.
    The actual learning happens in agent.update().
    """
    clear_screen()

    print("Q-learning maze training")
    print()
    print(f"Episode: {episode}")
    print(f"Step: {step_number}")
    print(f"Current state: {state}")
    print(f"Action: {env.action_names[action]}")
    print(f"Next state: {next_state}")
    print(f"Reward from this move: {reward}")
    print(f"Total reward in this episode: {total_reward}")
    print()

    env.render(visited=visited)

    print("Q-value update for this move:")
    print(f"Old Q-value:       {old_q:.2f}")
    print(f"Reward:            {reward:.2f}")
    print(f"Future best value: {future_best_q:.2f}")
    print(f"New Q-value:       {new_q:.2f}")
    print()

    agent.print_current_q_values(state)
    agent.print_best_q_matrix(env.maze, env.goal)

    print("Legend:")
    print("S = start")
    print("G = goal")
    print("X = wall")
    print("A = agent")
    print("* = places visited in this episode")
    print("^ = best action is Up")
    print("v = best action is Down")
    print("< = best action is Left")
    print("> = best action is Right")
    print()

    time.sleep(animation_speed)


def train_agent():
    """
    Train the Q-learning agent.

    Main RL cycle:

    1. Start from a state.
    2. Agent chooses an action.
    3. Environment returns next_state, reward, done.
    4. Agent updates its Q-table.
    5. Continue until goal is reached or max steps are used.
    """
    env = MazeEnv()

    agent = QLearningAgent(
        rows=env.rows,
        cols=env.cols,
        number_of_actions=len(env.actions),
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
    )

    episodes = 5000
    max_steps_per_episode = 100
    animation_speed = 0.4

    # Only these episodes are animated step by step.
    # Animating every episode would be very slow.
    show_episodes = {1, 2, 3, 10, 50, 100, 500, 1000, 5000}

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        visited = [state]

        while not done and steps < max_steps_per_episode:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            old_q, new_q, future_best_q = agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
            )

            total_reward += reward
            steps += 1
            visited.append(next_state)

            if episode in show_episodes:
                show_training_step(
                    env=env,
                    agent=agent,
                    episode=episode,
                    step_number=steps,
                    state=state,
                    action=action,
                    reward=reward,
                    total_reward=total_reward,
                    next_state=next_state,
                    visited=visited,
                    old_q=old_q,
                    new_q=new_q,
                    future_best_q=future_best_q,
                    animation_speed=animation_speed,
                )

            state = next_state

    return env, agent


def show_final_path(env, agent):
    """
    After training, follow the best learned action from each cell.
    This shows the final learned route.
    """
    clear_screen()
    print("Training complete.")
    print("Now showing the final learned path.")
    time.sleep(1)

    state = env.reset()
    visited = [state]

    for step_number in range(1, 31):
        action = agent.get_best_action(state)
        next_state, reward, done = env.step(action)
        visited.append(next_state)

        clear_screen()
        print("Final learned path")
        print()
        print(f"Step: {step_number}")
        print(f"Current state: {state}")
        print(f"Action: {env.action_names[action]}")
        print(f"Next state: {next_state}")
        print(f"Reward: {reward}")
        print()

        env.render(visited=visited)
        agent.print_best_q_matrix(env.maze, env.goal)

        time.sleep(0.4)

        state = next_state

        if done:
            print("Goal reached!")
            break


if __name__ == "__main__":
    env, agent = train_agent()
    show_final_path(env, agent)

    print()
    print("Final actual Q-table:")
    agent.print_full_q_table(env.maze, env.goal)
