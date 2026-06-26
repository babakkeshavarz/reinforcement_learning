import numpy as np
import random
import time
import os

maze = [
    ["S", ".", ".", ".", "."],
    [".", "X", "X", ".", "."],
    [".", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", ".", ".", ".", "G"],
]

rows, cols = 5, 5
start = (0, 0)
goal = (4, 4)

actions = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

action_names = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
}

q_table = np.zeros((rows, cols, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 5000

# Only these episodes will be animated step by step
show_episodes = {1, 2, 3, 10, 50, 100, 500, 1000, 5000}

animation_speed = 1
max_steps_per_episode = 100


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def is_valid(position):
    r, c = position
    return 0 <= r < rows and 0 <= c < cols and maze[r][c] != "X"


def step(position, action):
    r, c = position
    dr, dc = actions[action]
    new_position = (r + dr, c + dc)

    if not is_valid(new_position):
        return position, -5, False

    if new_position == goal:
        return new_position, 10, True

    return new_position, -1, False


def print_maze(agent_position, visited):
    for r in range(rows):
        line = ""
        for c in range(cols):
            pos = (r, c)

            if pos == agent_position:
                line += "A "
            elif pos == start:
                line += "S "
            elif pos == goal:
                line += "G "
            elif maze[r][c] == "X":
                line += "X "
            elif pos in visited:
                line += "* "
            else:
                line += ". "
        print(line)
    print()


def show_training_step(episode, step_number, position, action, reward, total_reward, new_position, visited):
    clear_screen()

    print("Q-learning maze training")
    print()
    print(f"Episode: {episode}")
    print(f"Step: {step_number}")
    print(f"Current position: {position}")
    print(f"Action: {action_names[action]}")
    print(f"New position: {new_position}")
    print(f"Reward from this move: {reward}")
    print(f"Total reward in this episode: {total_reward}")
    print()

    print_maze(agent_position=new_position, visited=visited)

    print("Legend:")
    print("S = start")
    print("G = goal")
    print("X = wall")
    print("A = agent")
    print("* = places visited in this episode")
    print()

    time.sleep(animation_speed)


# Training
for episode in range(1, episodes + 1):
    position = start
    done = False
    total_reward = 0
    steps = 0
    visited = [start]

    while not done and steps < max_steps_per_episode:
        r, c = position

        if random.random() < epsilon:
            action = random.choice(list(actions.keys()))
        else:
            action = np.argmax(q_table[r, c])

        new_position, reward, done = step(position, action)
        nr, nc = new_position

        old_value = q_table[r, c, action]
        future_value = np.max(q_table[nr, nc])

        q_table[r, c, action] = old_value + alpha * (
            reward + gamma * future_value - old_value
        )

        total_reward += reward
        steps += 1
        visited.append(new_position)

        if episode in show_episodes:
            show_training_step(
                episode,
                steps,
                position,
                action,
                reward,
                total_reward,
                new_position,
                visited,
            )

        position = new_position


# Final learned path
clear_screen()
print("Training complete.")
print("Now showing the final learned path.")
time.sleep(1)

position = start
visited = [start]

for step_number in range(1, 31):
    r, c = position
    action = np.argmax(q_table[r, c])
    new_position, reward, done = step(position, action)

    visited.append(new_position)

    clear_screen()
    print("Final learned path")
    print()
    print(f"Step: {step_number}")
    print(f"Current position: {position}")
    print(f"Action: {action_names[action]}")
    print(f"New position: {new_position}")
    print(f"Reward: {reward}")
    print()

    print_maze(agent_position=new_position, visited=visited)

    time.sleep(.4)

    position = new_position

    if done:
        print("Goal reached!")
        break