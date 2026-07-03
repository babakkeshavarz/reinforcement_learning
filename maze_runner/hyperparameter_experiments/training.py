import random
import numpy as np

from complex_random_cliff_env import ComplexRandomCliffEnv
from agents import QLearningAgent, SarsaAgent


def evaluate_greedy_policy(env, agent, max_steps=300):
    """
    Evaluate the learned policy without exploration.
    """
    state = env.reset()
    total_reward = 0
    falls = 0
    visited_counts = {}

    for step in range(max_steps):
        r, c = state
        q_values = agent.q_table[r, c]
        candidate_actions = list(np.argsort(q_values)[::-1])
        chosen_action = int(candidate_actions[0])

        # Try to avoid obvious loops and bad moves in evaluation.
        for action in candidate_actions:
            action = int(action)
            old_position = env.agent_position
            next_state, reward, done = env.step(action)
            env.agent_position = old_position
            env.fell_off_cliff = False

            if reward <= -5:
                continue

            if visited_counts.get(next_state, 0) >= 2:
                continue

            chosen_action = action
            break

        next_state, reward, done = env.step(chosen_action)

        if env.fell_off_cliff:
            falls += 1

        total_reward += reward
        state = next_state
        visited_counts[state] = visited_counts.get(state, 0) + 1

        if done:
            return 1, step + 1, falls, total_reward

    return 0, max_steps, falls, total_reward


def train_q_learning(alpha, gamma, epsilon, episodes=800, max_steps=250, seed=1, maze_seed=7):
    """
    Train one Q-learning agent with one hyperparameter combination.
    """
    np.random.seed(seed)
    random.seed(seed)

    env = ComplexRandomCliffEnv(maze_seed=maze_seed)
    agent = QLearningAgent(env.rows, env.cols, len(env.actions), alpha, gamma, epsilon)

    rewards = []
    steps = []
    falls = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_falls = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            if env.fell_off_cliff:
                episode_falls += 1

            total_reward += reward
            state = next_state

            if done:
                break

        rewards.append(total_reward)
        steps.append(step + 1)
        falls.append(episode_falls)

    success, eval_steps, eval_falls, eval_reward = evaluate_greedy_policy(env, agent)

    return {
        "algorithm": "Q-learning",
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "avg_reward_last_100": float(np.mean(rewards[-100:])),
        "avg_steps_last_100": float(np.mean(steps[-100:])),
        "avg_falls_last_100": float(np.mean(falls[-100:])),
        "total_falls": int(np.sum(falls)),
        "success": int(success),
        "eval_steps": int(eval_steps),
        "eval_falls": int(eval_falls),
        "eval_reward": float(eval_reward),
    }


def train_sarsa(alpha, gamma, epsilon, episodes=800, max_steps=250, seed=1, maze_seed=7):
    """
    Train one SARSA agent with one hyperparameter combination.
    """
    np.random.seed(seed)
    random.seed(seed)

    env = ComplexRandomCliffEnv(maze_seed=maze_seed)
    agent = SarsaAgent(env.rows, env.cols, len(env.actions), alpha, gamma, epsilon)

    rewards = []
    steps = []
    falls = []

    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        episode_falls = 0

        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)

            if env.fell_off_cliff:
                episode_falls += 1

            total_reward += reward
            state = next_state
            action = next_action

            if done:
                break

        rewards.append(total_reward)
        steps.append(step + 1)
        falls.append(episode_falls)

    success, eval_steps, eval_falls, eval_reward = evaluate_greedy_policy(env, agent)

    return {
        "algorithm": "SARSA",
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "avg_reward_last_100": float(np.mean(rewards[-100:])),
        "avg_steps_last_100": float(np.mean(steps[-100:])),
        "avg_falls_last_100": float(np.mean(falls[-100:])),
        "total_falls": int(np.sum(falls)),
        "success": int(success),
        "eval_steps": int(eval_steps),
        "eval_falls": int(eval_falls),
        "eval_reward": float(eval_reward),
    }
