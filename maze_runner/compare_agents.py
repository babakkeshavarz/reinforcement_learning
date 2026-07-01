import json
import numpy as np

from train_q_learning import train_q_learning
from train_sarsa import train_sarsa
from visualization import (
    ASSETS_DIR,
    get_final_path,
    save_path_gif,
    save_path_png,
    save_maze_png,
    save_curve,
    save_summary_bar,
)


def summarize(rewards, steps, falls):
    """
    Summarize final training performance.
    """
    return {
        "avg_reward_last_100": float(np.mean(rewards[-100:])),
        "avg_steps_last_100": float(np.mean(steps[-100:])),
        "avg_falls_last_100": float(np.mean(falls[-100:])),
        "total_falls": int(np.sum(falls)),
    }


def main():
    """
    Train both methods and save all results in assets/.
    """
    print("Training Q-learning...")
    q_env, q_agent, q_rewards, q_steps, q_falls = train_q_learning()

    print("Training SARSA...")
    s_env, s_agent, sarsa_rewards, sarsa_steps, sarsa_falls = train_sarsa()

    q_path, q_path_falls = get_final_path(q_env, q_agent)
    sarsa_path, sarsa_path_falls = get_final_path(s_env, s_agent)

    save_maze_png(q_env, "generated_random_maze.png")

    save_path_gif(q_env, q_path, "q_learning_random_cliff.gif", "Q-learning final path")
    save_path_png(q_env, q_path, "q_learning_random_cliff.png", "Q-learning final path")

    save_path_gif(s_env, sarsa_path, "sarsa_random_cliff.gif", "SARSA final path")
    save_path_png(s_env, sarsa_path, "sarsa_random_cliff.png", "SARSA final path")

    save_curve(
        q_rewards,
        sarsa_rewards,
        "reward_comparison.png",
        "Training reward comparison",
        "Moving average reward",
    )

    save_curve(
        q_steps,
        sarsa_steps,
        "steps_comparison.png",
        "Training steps comparison",
        "Moving average steps",
    )

    save_curve(
        q_falls,
        sarsa_falls,
        "cliff_falls_comparison.png",
        "Cliff falls comparison",
        "Moving average cliff falls",
    )

    q_summary = summarize(q_rewards, q_steps, q_falls)
    sarsa_summary = summarize(sarsa_rewards, sarsa_steps, sarsa_falls)

    save_summary_bar(q_summary, sarsa_summary, "summary_bar_chart.png")

    results = {
        "q_learning": {
            "final_path": q_path,
            "final_path_cliff_falls": q_path_falls,
            **q_summary,
        },
        "sarsa": {
            "final_path": sarsa_path,
            "final_path_cliff_falls": sarsa_path_falls,
            **sarsa_summary,
        },
    }

    ASSETS_DIR.mkdir(exist_ok=True)

    with open(ASSETS_DIR / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("Done. Results saved in assets/.")
    print()
    print("Q-learning final path:")
    print(q_path)
    print()
    print("SARSA final path:")
    print(sarsa_path)
    print()


if __name__ == "__main__":
    main()
