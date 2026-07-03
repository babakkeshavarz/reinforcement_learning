from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ASSETS_DIR = Path("assets")


def save_bar_chart(df, metric, filename, title, ylabel, top_n=12):
    """
    Save a horizontal bar chart for the best hyperparameter combinations.

    For reward and success, higher is better.
    For steps and falls, lower is better.
    """
    if metric in ["avg_reward_last_100", "success", "eval_reward"]:
        sorted_df = df.sort_values(metric, ascending=False).head(top_n)
    else:
        sorted_df = df.sort_values(metric, ascending=True).head(top_n)

    labels = [
        f"{row.algorithm}\na={row.alpha}, g={row.gamma}, e={row.epsilon}"
        for row in sorted_df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, sorted_df[metric])
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    output_path = ASSETS_DIR / filename
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_parameter_effect_plot(df, parameter, metric, filename, title, ylabel):
    """
    Show the average effect of one hyperparameter.
    """
    grouped = (
        df.groupby(["algorithm", parameter])[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for algorithm in grouped["algorithm"].unique():
        subset = grouped[grouped["algorithm"] == algorithm]
        ax.plot(subset[parameter], subset[metric], marker="o", label=algorithm)

    ax.set_title(title)
    ax.set_xlabel(parameter)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path = ASSETS_DIR / filename
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_heatmap(df, algorithm, x_param, y_param, metric, filename, title):
    """
    Save a simple heatmap.
    """
    subset = df[df["algorithm"] == algorithm]

    pivot = subset.pivot_table(
        index=y_param,
        columns=x_param,
        values=metric,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot.values, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label=metric)

    output_path = ASSETS_DIR / filename
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    """
    Create plots from assets/hyperparameter_results.csv.
    """
    ASSETS_DIR.mkdir(exist_ok=True)

    results_path = ASSETS_DIR / "hyperparameter_results.csv"

    if not results_path.exists():
        raise FileNotFoundError(
            "assets/hyperparameter_results.csv not found. Run python run_experiments.py first."
        )

    df = pd.read_csv(results_path)

    save_bar_chart(
        df,
        metric="avg_reward_last_100",
        filename="top_reward_settings.png",
        title="Top hyperparameter settings by final reward",
        ylabel="Average reward over last 100 episodes",
    )

    save_bar_chart(
        df,
        metric="avg_falls_last_100",
        filename="lowest_cliff_fall_settings.png",
        title="Safest hyperparameter settings",
        ylabel="Average cliff falls over last 100 episodes",
    )

    save_parameter_effect_plot(
        df,
        parameter="epsilon",
        metric="avg_reward_last_100",
        filename="epsilon_vs_reward.png",
        title="Effect of epsilon on reward",
        ylabel="Average reward over last 100 episodes",
    )

    save_parameter_effect_plot(
        df,
        parameter="alpha",
        metric="avg_reward_last_100",
        filename="alpha_vs_reward.png",
        title="Effect of alpha on reward",
        ylabel="Average reward over last 100 episodes",
    )

    save_parameter_effect_plot(
        df,
        parameter="gamma",
        metric="avg_reward_last_100",
        filename="gamma_vs_reward.png",
        title="Effect of gamma on reward",
        ylabel="Average reward over last 100 episodes",
    )

    save_parameter_effect_plot(
        df,
        parameter="epsilon",
        metric="avg_falls_last_100",
        filename="epsilon_vs_cliff_falls.png",
        title="Effect of epsilon on cliff falls",
        ylabel="Average cliff falls over last 100 episodes",
    )

    save_heatmap(
        df,
        algorithm="Q-learning",
        x_param="epsilon",
        y_param="alpha",
        metric="avg_reward_last_100",
        filename="q_learning_alpha_epsilon_heatmap.png",
        title="Q-learning reward heatmap: alpha vs epsilon",
    )

    save_heatmap(
        df,
        algorithm="SARSA",
        x_param="epsilon",
        y_param="alpha",
        metric="avg_reward_last_100",
        filename="sarsa_alpha_epsilon_heatmap.png",
        title="SARSA reward heatmap: alpha vs epsilon",
    )

    print("Plots saved in assets/.")


if __name__ == "__main__":
    main()
