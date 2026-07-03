from pathlib import Path

import pandas as pd


def main():
    """
    Print a simple text analysis of the hyperparameter results.
    """
    path = Path("assets/hyperparameter_results.csv")

    if not path.exists():
        raise FileNotFoundError(
            "assets/hyperparameter_results.csv not found. Run python run_experiments.py first."
        )

    df = pd.read_csv(path)

    print()
    print("Best settings by reward:")
    print(
        df.sort_values("avg_reward_last_100", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Safest settings by lowest cliff falls:")
    print(
        df.sort_values("avg_falls_last_100", ascending=True)
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Average performance by algorithm:")
    print(
        df.groupby("algorithm")[
            [
                "avg_reward_last_100",
                "avg_steps_last_100",
                "avg_falls_last_100",
                "success",
                "eval_reward",
            ]
        ]
        .mean()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
