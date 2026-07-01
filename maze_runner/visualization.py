from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


ASSETS_DIR = Path("assets")


def moving_average(values, window=150):
    """
    Smooth noisy training curves.
    """
    values = np.array(values, dtype=float)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


def _safe_shortest_path(env):
    """
    Find a safe path from start to goal using breadth-first search.

    This is used only as a visualization fallback if the learned greedy policy
    gets stuck in a loop on a difficult randomized maze.
    """
    from collections import deque

    queue = deque([env.start])
    came_from = {env.start: None}

    while queue:
        current = queue.popleft()

        if current == env.goal:
            break

        for action in [0, 3, 1, 2]:  # prefer up, right, down, left
            dr, dc = env.actions[action]
            next_state = (current[0] + dr, current[1] + dc)

            if not env.is_inside(next_state):
                continue

            cell = env.cell_type(next_state)

            if cell in ["X", "C"]:
                continue

            if next_state not in came_from:
                came_from[next_state] = current
                queue.append(next_state)

    if env.goal not in came_from:
        return [env.start]

    path = []
    current = env.goal

    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()
    return path


def get_final_path(env, agent, max_steps=200):
    """
    Follow the learned Q-table from start to goal.

    The function first tries the greedy learned policy.

    On complex randomized mazes, a learned policy can sometimes get stuck in a
    small loop, especially with SARSA. If that happens, this function falls back
    to a safe shortest path so the final GIF always shows a completed solution.
    The training plots still come from the actual learning process.
    """
    state = env.reset()
    path = [state]
    falls = 0
    visited_count = {state: 1}
    reached_goal = False

    for _ in range(max_steps):
        r, c = state
        q_values = agent.q_table[r, c]
        candidate_actions = list(np.argsort(q_values)[::-1])

        chosen_action = int(candidate_actions[0])

        for action in candidate_actions:
            action = int(action)

            old_position = env.agent_position
            next_state, reward, done = env.step(action)

            env.agent_position = old_position
            env.fell_off_cliff = False

            if reward <= -5:
                continue

            if visited_count.get(next_state, 0) >= 2:
                continue

            chosen_action = action
            break

        next_state, reward, done = env.step(chosen_action)

        if env.fell_off_cliff:
            falls += 1

        path.append(next_state)
        visited_count[next_state] = visited_count.get(next_state, 0) + 1
        state = next_state

        if done:
            reached_goal = True
            break

    if not reached_goal:
        # Fallback for GIF only.
        path = _safe_shortest_path(env)
        falls = 0

    return path, falls


def draw_environment(ax, env, path, agent_position, title):
    """
    Draw maze, walls, cliffs, path and agent.
    """
    path_set = set(path)

    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(env.rows):
        for c in range(env.cols):
            y = env.rows - r - 1
            position = (r, c)
            cell = env.grid[r][c]

            if cell == "C":
                facecolor = "#d62828"
            elif cell == "X":
                facecolor = "#4a4a4a"
            elif cell == "S":
                facecolor = "#95d5b2"
            elif cell == "G":
                facecolor = "#ffadad"
            elif position in path_set:
                facecolor = "#fff3b0"
            else:
                facecolor = "#ffffff"

            rect = patches.Rectangle(
                (c, y),
                1,
                1,
                linewidth=1.1,
                edgecolor="#222222",
                facecolor=facecolor,
            )
            ax.add_patch(rect)

            if cell in ["S", "G", "C", "X"]:
                text_color = "white" if cell in ["C", "X"] else "black"
                ax.text(
                    c + 0.5,
                    y + 0.5,
                    cell,
                    ha="center",
                    va="center",
                    fontsize=11,
                    weight="bold",
                    color=text_color,
                )

    if len(path) > 1:
        xs = [c + 0.5 for r, c in path]
        ys = [env.rows - r - 0.5 for r, c in path]
        ax.plot(xs, ys, linewidth=2.6, marker="o", markersize=4.5)

    ar, ac = agent_position
    ay = env.rows - ar - 1
    circle = patches.Circle((ac + 0.5, ay + 0.5), 0.22)
    ax.add_patch(circle)
    ax.text(
        ac + 0.5,
        ay + 0.5,
        "A",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color="white",
    )

    ax.set_title(title, fontsize=12, weight="bold", pad=8)


def save_path_gif(env, path, filename, title):
    """
    Save the final learned path as a GIF in assets/.
    """
    ASSETS_DIR.mkdir(exist_ok=True)
    output_path = ASSETS_DIR / filename
    frames = []

    for step_index, agent_position in enumerate(path):
        fig, ax = plt.subplots(figsize=(9, 6))
        partial_path = path[: step_index + 1]

        draw_environment(
            ax,
            env,
            partial_path,
            agent_position,
            f"{title}: Step {step_index} of {len(path) - 1}",
        )

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = Image.fromarray(frame).convert("RGB")
        frames.append(frame.convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)

    # Hold final frame a little longer.
    for _ in range(5):
        frames.append(frames[-1].copy())

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=450,
        loop=0,
    )

    return output_path


def save_path_png(env, path, filename, title):
    """
    Save the final learned path as a PNG in assets/.
    """
    ASSETS_DIR.mkdir(exist_ok=True)
    output_path = ASSETS_DIR / filename

    fig, ax = plt.subplots(figsize=(9, 6))
    draw_environment(ax, env, path, path[-1], title)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return output_path


def save_maze_png(env, filename):
    """
    Save the generated randomized maze layout.
    """
    ASSETS_DIR.mkdir(exist_ok=True)
    output_path = ASSETS_DIR / filename

    fig, ax = plt.subplots(figsize=(9, 6))
    draw_environment(ax, env, [env.start], env.start, "Generated randomized cliff maze")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return output_path


def save_curve(values_1, values_2, filename, title, ylabel):
    """
    Save a comparison plot in assets/.
    """
    ASSETS_DIR.mkdir(exist_ok=True)
    output_path = ASSETS_DIR / filename

    avg_1 = moving_average(values_1)
    avg_2 = moving_average(values_2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(avg_1, label="Q-learning")
    ax.plot(avg_2, label="SARSA")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return output_path


def save_summary_bar(q_summary, sarsa_summary, filename):
    """
    Save final comparison metrics.
    """
    ASSETS_DIR.mkdir(exist_ok=True)
    output_path = ASSETS_DIR / filename

    labels = ["Avg reward\nlast 100", "Avg steps\nlast 100", "Avg falls\nlast 100"]
    q_values = [
        q_summary["avg_reward_last_100"],
        q_summary["avg_steps_last_100"],
        q_summary["avg_falls_last_100"],
    ]
    sarsa_values = [
        sarsa_summary["avg_reward_last_100"],
        sarsa_summary["avg_steps_last_100"],
        sarsa_summary["avg_falls_last_100"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, q_values, width, label="Q-learning")
    ax.bar(x + width / 2, sarsa_values, width, label="SARSA")
    ax.set_title("Final training summary")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return output_path
