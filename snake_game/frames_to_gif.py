from pathlib import Path

from PIL import Image


FRAMES_DIR = Path("frames")
ASSETS_DIR = Path("assets")


def frames_to_gif(frames_dir=FRAMES_DIR, output_path=ASSETS_DIR / "snake_demo.gif", duration=80):
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    if not frame_paths:
        raise FileNotFoundError("No frames found. Run python generate_snake_frames.py first.")

    frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)

    print(f"GIF saved to: {output_path}")


if __name__ == "__main__":
    frames_to_gif()
