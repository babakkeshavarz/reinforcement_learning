import itertools
from pathlib import Path

import pandas as pd

from training import train_q_learning, train_sarsa


def main():
    """
    Run hyperparameter experiments.

    Tested values:

        epsilon: 0.05, 0.10, 0.20, 0.30
        alpha:   0.05, 0.10, 0.30, 0.50
        gamma:   0.70, 0.90, 0.95, 0.99

    The default episode count is kept moderate so the project runs quickly.
    Increase episodes if you want smoother, more reliable results.
    """

    epsilon_values = [0.05, 0.10, 0.20, 0.30]
    alpha_values = [0.05, 0.10, 0.30, 0.50]
    gamma_values = [0.70, 0.90, 0.95, 0.99]

    episodes = 800
    max_steps = 250

    results = []
    combinations = list(itertools.product(alpha_values, gamma_values, epsilon_values))

    total_runs = len(combinations) * 2
    run_number = 1

    for alpha, gamma, epsilon in combinations:
        print(f"[{run_number}/{total_runs}] Q-learning: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
        results.append(
            train_q_learning(
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                episodes=episodes,
                max_steps=max_steps,
                seed=1,
                maze_seed=7,
            )
        )
        run_number += 1

        print(f"[{run_number}/{total_runs}] SARSA: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
        results.append(
            train_sarsa(
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                episodes=episodes,
                max_steps=max_steps,
                seed=1,
                maze_seed=7,
            )
        )
        run_number += 1

    df = pd.DataFrame(results)

    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    output_path = assets_dir / "hyperparameter_results.csv"
    df.to_csv(output_path, index=False)

    print()
    print("Experiment complete.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
