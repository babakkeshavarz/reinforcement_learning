from pathlib import Path

import imageio.v2 as imageio


FRAMES_DIR = Path("frames")
ASSETS_DIR = Path("assets")


def frames_to_video(frames_dir=FRAMES_DIR, output_path=ASSETS_DIR / "snake_demo.mp4", fps=12):
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    if not frame_paths:
        raise FileNotFoundError("No frames found. Run python generate_snake_frames.py first.")

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    frames_to_video()
