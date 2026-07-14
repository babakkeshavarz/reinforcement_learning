import shutil
from pathlib import Path

from agents import QLearningAgent, SarsaAgent
from dqn_agent import DQNAgent
from rendering import draw_frame, frames_to_gif, frames_to_video
from snake_env import SnakeEnv


def play_and_record(agent, algorithm_name, seed=100, max_steps=600, use_small_exploration=False):
    """
    Play one evaluation game and create GIF/MP4.

    By default, epsilon is 0 for a pure greedy policy.
    Set use_small_exploration=True if the early-trained agent gets stuck.
    """
    assets = Path("assets")
    frames_dir = assets / f"{algorithm_name.lower().replace('-', '_')}_frames"

    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)

    if use_small_exploration and hasattr(agent, "epsilon"):
        agent.epsilon = 0.02

    env = SnakeEnv(width=10, height=10, seed=seed)
    state = env.reset()
    info = {"score": 0}
    total_reward = 0
    frame_paths = []

    for step in range(max_steps):
        frame_path = frames_dir / f"frame_{step:04d}.png"
        draw_frame(env.clone_for_render(), algorithm_name, frame_path)
        frame_paths.append(frame_path)

        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state

        if done:
            final_frame = frames_dir / f"frame_{step + 1:04d}.png"
            draw_frame(env.clone_for_render(), f"{algorithm_name} final", final_frame)
            frame_paths.append(final_frame)
            break

    stem = algorithm_name.lower().replace("-", "_").replace(" ", "_")

    gif_path = assets / f"{stem}_game.gif"
    video_path = assets / f"{stem}_game.mp4"

    frames_to_gif(frame_paths, gif_path)
    frames_to_video(frame_paths, video_path)

    return {
        "algorithm": algorithm_name,
        "score": info["score"],
        "steps": len(frame_paths),
        "reward": total_reward,
        "gif": str(gif_path),
        "video": str(video_path),
    }


def main():
    assets = Path("assets")

    q_agent = QLearningAgent.load(assets / "q_learning_best_agent.pkl")
    sarsa_agent = SarsaAgent.load(assets / "sarsa_best_agent.pkl")
    dqn_agent = DQNAgent.load(assets / "dqn_best_agent.pt")

    results = [
        play_and_record(q_agent, "Q-learning", seed=100),
        play_and_record(sarsa_agent, "SARSA", seed=100),
        play_and_record(dqn_agent, "DQN", seed=100),
    ]

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
