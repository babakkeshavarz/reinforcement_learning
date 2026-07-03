# Reinforcement Learning Projects


## Repository structure

```text
reinforcement_learning/
├── README.md
├── maze_runner/
│   ├── complex_random_cliff_env.py
│   ├── q_learning_agent.py
│   ├── sarsa_agent.py
│   ├── train_q_learning.py
│   ├── train_sarsa.py
│   ├── compare_agents.py
│   ├── visualization.py
│   ├── requirements.txt
│   └── assets/
└── hyperparameter_experiments/
    ├── complex_random_cliff_env.py
    ├── agents.py
    ├── training.py
    ├── run_experiments.py
    ├── plot_results.py
    ├── analyze_results.py
    ├── requirements.txt
    └── assets/
```

## Maze Runner: Randomized Cliff Environment

This project compares **Q-learning** and **SARSA** in a larger randomized cliff maze.


### Run

```bash
cd maze_runner
python compare_agents.py
```

The code saves all outputs to:

```text
maze_runner/assets/
```

### Install dependencies

```bash
cd maze_runner
pip install -r requirements.txt
```

or:

```bash
pip install numpy matplotlib pillow
```

### Generated randomized maze

The maze is randomized but reproducible using a seed.

![Generated randomized maze](maze_runner/assets/generated_random_maze.png)

Legend:

```text
S = start
G = goal
C = cliff / danger
X = wall
. = safe cell
```

### Final learned paths

### Q-learning

Q-learning is **off-policy**. It updates using the best possible next action.

![Q-learning final path](maze_runner/assets/q_learning_random_cliff.gif)

### SARSA

SARSA is **on-policy**. It updates using the next action the agent actually selected.

![SARSA final path](maze_runner/assets/sarsa_random_cliff.gif)

### Training comparison

### Cliff falls

![Cliff falls comparison](maze_runner/assets/cliff_falls_comparison.png)

### Reward comparison

![Reward comparison](maze_runner/assets/reward_comparison.png)

### Steps comparison

![Steps comparison](maze_runner/assets/steps_comparison.png)

### Final summary

![Summary bar chart](maze_runner/assets/summary_bar_chart.png)

### Files

```text
maze_runner/
├── complex_random_cliff_env.py
├── q_learning_agent.py
├── sarsa_agent.py
├── train_q_learning.py
├── train_sarsa.py
├── compare_agents.py
├── visualization.py
├── requirements.txt
├── README.md
└── assets/
```

### Change the random maze

Open `train_q_learning.py`, `train_sarsa.py` or `compare_agents.py`.

Change:

```python
maze_seed=7
```

to another number.

For example:

```python
maze_seed=15
```

Then rerun:

```bash
python compare_agents.py
```


### Training detail

This version uses **epsilon decay**.

That means the agent explores more at the beginning and becomes more confident over time.

```text
early training: more random exploration
late training: mostly learned behaviour
```

The final GIFs are created from the learned Q-table and saved to `assets/`.


### Visualization note

The training plots come from the actual Q-learning and SARSA training.

For the final GIFs, the visualizer first tries the learned greedy policy. If the
policy gets stuck in a loop on a difficult randomized maze, it falls back to a
safe valid route so the GIF still reaches the goal.



## Hyperparameter Experiments

The `hyperparameter_experiments/` project compares Q-learning and SARSA across different values of:

```text
alpha
gamma
epsilon
```

Run it:

```bash
cd hyperparameter_experiments
pip install -r requirements.txt
python run_experiments.py
python plot_results.py
python analyze_results.py
```


## What each hyperparameter does

### Epsilon

`epsilon` controls exploration.

```text
low epsilon  = mostly uses what it already knows
high epsilon = tries more random actions
```

In the cliff maze, higher epsilon can increase cliff falls because the agent takes more random actions near danger cells.

### Alpha

`alpha` is the learning rate.

```text
low alpha  = slower but more stable learning
high alpha = faster but more aggressive learning
```

A very high alpha can make learning unstable because the agent may overreact to recent rewards or penalties.

### Gamma

`gamma` is the discount factor.

```text
low gamma  = focuses more on immediate reward
high gamma = cares more about future reward
```

In this project, gamma is especially interesting because SARSA and Q-learning can respond differently.

## Observation from the current experiment

In the current experiment results:

```text
SARSA: higher gamma improves reward
Q-learning: higher gamma decreases reward
```

This is a useful result.

SARSA uses the action the agent actually takes:

```python
future_q = Q[next_state, next_action]
```

So SARSA learns from its real behaviour, including exploration risk. With higher gamma, SARSA can value safer long-term routes that avoid cliffs.

Q-learning uses the best possible next action:

```python
future_best_q = max(Q[next_state])
```

So Q-learning can become more optimistic. With higher gamma, it may overvalue risky states because it strongly values the future goal reward while assuming future actions will be optimal.

In plain language:

```text
SARSA says: I care about the future, but I know I may still make exploratory mistakes.
Q-learning says: I care about the future and assume I will act perfectly later.
```

That is why high gamma can help SARSA but hurt Q-learning in this specific cliff environment.


## Example outputs

### Top reward settings

![Top reward settings](hyperparameter_experiments/assets/top_reward_settings.png)

### Epsilon effect

![Epsilon vs reward](hyperparameter_experiments/assets/epsilon_vs_reward.png)

### Q-learning heatmap

![Q-learning alpha epsilon heatmap](hyperparameter_experiments/assets/q_learning_alpha_epsilon_heatmap.png)

### SARSA heatmap

![SARSA alpha epsilon heatmap](hyperparameter_experiments/assets/sarsa_alpha_epsilon_heatmap.png)


### Alpha effect

![Alpha vs reward](hyperparameter_experiments/assets/alpha_vs_reward.png)

### Gamma effect

![Gamma vs reward](hyperparameter_experiments/assets/gamma_vs_reward.png)

### Epsilon and cliff falls

![Epsilon vs cliff falls](hyperparameter_experiments/assets/epsilon_vs_cliff_falls.png)

