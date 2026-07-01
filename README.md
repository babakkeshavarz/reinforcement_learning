# Reinforcement Learning Projects

This repository contains small reinforcement learning projects built from the ground up in Python.

## Maze Runner

The first project is a randomized cliff maze comparing:

```text
Q-learning = off-policy
SARSA      = on-policy
```

Go into the project folder and run:

```bash
cd maze_runner
python compare_agents.py
```

## Generated randomized maze

![Generated randomized maze](maze_runner/assets/generated_random_maze.png)

## Final learned paths

### Q-learning

![Q-learning final path](maze_runner/assets/q_learning_random_cliff.gif)

### SARSA

![SARSA final path](maze_runner/assets/sarsa_random_cliff.gif)

## Training comparison

### Cliff falls

![Cliff falls comparison](maze_runner/assets/cliff_falls_comparison.png)

### Reward comparison

![Reward comparison](maze_runner/assets/reward_comparison.png)

### Steps comparison

![Steps comparison](maze_runner/assets/steps_comparison.png)

### Final summary

![Summary bar chart](maze_runner/assets/summary_bar_chart.png)

## Install dependencies

```bash
cd maze_runner
pip install -r requirements.txt
```


## Visualization note

The training plots are generated from the actual training results.

The final GIF playback first tries the learned greedy policy. If a policy loops
on a complex randomized maze, the visualizer falls back to a safe valid route so
the GIF still reaches the goal.
