import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(values, window=100):
    return values.rolling(window=window, min_periods=1).mean()


def patch_episode_count(file_path, function_name, episodes):
    """
    Temporarily patch the default episode count in a training file.
    This keeps command-line usage simple for beginner learning.
    """
    path = Path(file_path)
    text = path.read_text()

    import re

    updated = re.sub(
        rf"def {function_name}\(episodes=\d+, seed=1\):",
        f"def {function_name}(episodes={episodes}, seed=1):",
        text,
    )

    path.write_text(updated)


def plot_training_curves():
    assets = Path("assets")

    q = pd.read_csv(assets / "q_learning_training.csv")
    s = pd.read_csv(assets / "sarsa_training.csv")
    d = pd.read_csv(assets / "dqn_training.csv")

    plt.figure(figsize=(9, 5))
    plt.plot(q["episode"], moving_average(q["score"]), label="Q-learning")
    plt.plot(s["episode"], moving_average(s["score"]), label="SARSA")
    plt.plot(d["episode"], moving_average(d["score"]), label="DQN")
    plt.title("Snake training score comparison")
    plt.xlabel("Episode")
    plt.ylabel("Moving average score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(assets / "score_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(q["episode"], moving_average(q["reward"]), label="Q-learning")
    plt.plot(s["episode"], moving_average(s["reward"]), label="SARSA")
    plt.plot(d["episode"], moving_average(d["reward"]), label="DQN")
    plt.title("Snake training reward comparison")
    plt.xlabel("Episode")
    plt.ylabel("Moving average reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(assets / "reward_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(q["episode"], q["best_score_so_far"], label="Q-learning")
    plt.plot(s["episode"], s["best_score_so_far"], label="SARSA")
    plt.plot(d["episode"], d["best_score_so_far"], label="DQN")
    plt.title("Best score reached during training")
    plt.xlabel("Episode")
    plt.ylabel("Best score so far")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(assets / "best_score_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(d["episode"], moving_average(d["loss"]), label="DQN loss")
    plt.title("DQN training loss")
    plt.xlabel("Episode")
    plt.ylabel("Moving average loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(assets / "dqn_loss.png", bbox_inches="tight", dpi=150)
    plt.close()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "Q-learning",
                "avg_score_last_100": q["score"].tail(100).mean(),
                "best_score": q["score"].max(),
                "avg_reward_last_100": q["reward"].tail(100).mean(),
            },
            {
                "algorithm": "SARSA",
                "avg_score_last_100": s["score"].tail(100).mean(),
                "best_score": s["score"].max(),
                "avg_reward_last_100": s["reward"].tail(100).mean(),
            },
            {
                "algorithm": "DQN",
                "avg_score_last_100": d["score"].tail(100).mean(),
                "best_score": d["score"].max(),
                "avg_reward_last_100": d["reward"].tail(100).mean(),
            },
        ]
    )

    summary.to_csv(assets / "summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(summary["algorithm"], summary["avg_score_last_100"])
    plt.title("Average score over last 100 episodes")
    plt.ylabel("Score")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(assets / "summary_score_bar.png", bbox_inches="tight", dpi=150)
    plt.close()


def run_script(script_name):
    subprocess.run([sys.executable, script_name], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-episodes", type=int, default=None)
    parser.add_argument("--sarsa-episodes", type=int, default=None)
    parser.add_argument("--dqn-episodes", type=int, default=None)
    args = parser.parse_args()

    Path("assets").mkdir(exist_ok=True)

    if args.q_episodes is not None:
        patch_episode_count("train_q_learning.py", "train_q_learning", args.q_episodes)

    if args.sarsa_episodes is not None:
        patch_episode_count("train_sarsa.py", "train_sarsa", args.sarsa_episodes)

    if args.dqn_episodes is not None:
        patch_episode_count("train_dqn.py", "train_dqn", args.dqn_episodes)

    print("Training Q-learning...")
    run_script("train_q_learning.py")

    print("Training SARSA...")
    run_script("train_sarsa.py")

    print("Training DQN...")
    run_script("train_dqn.py")

    print("Plotting comparison charts...")
    plot_training_curves()

    print("Creating gameplay GIFs and MP4 videos from the best saved agents...")
    run_script("evaluate_and_video.py")

    print("Done. Outputs saved to assets/.")


if __name__ == "__main__":
    main()
