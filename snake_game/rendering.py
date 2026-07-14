from pathlib import Path

from PIL import Image, ImageDraw
import imageio.v2 as imageio


CELL_SIZE = 30


def draw_frame(render_data, title, output_path):
    width = render_data["width"]
    height = render_data["height"]
    snake = render_data["snake"]
    food = render_data["food"]
    score = render_data["score"]
    steps = render_data.get("steps", 0)

    image_width = width * CELL_SIZE
    image_height = height * CELL_SIZE + 48

    image = Image.new("RGB", (image_width, image_height), (18, 18, 18))
    draw = ImageDraw.Draw(image)

    draw.text((10, 8), f"{title} | Score: {score} | Steps: {steps}", fill=(240, 240, 240))

    y_offset = 48

    for x in range(width):
        for y in range(height):
            rect = [
                x * CELL_SIZE,
                y_offset + y * CELL_SIZE,
                (x + 1) * CELL_SIZE,
                y_offset + (y + 1) * CELL_SIZE,
            ]
            draw.rectangle(rect, outline=(35, 35, 35))

    if food is not None:
        fx, fy = food
        rect = [
            fx * CELL_SIZE + 3,
            y_offset + fy * CELL_SIZE + 3,
            (fx + 1) * CELL_SIZE - 3,
            y_offset + (fy + 1) * CELL_SIZE - 3,
        ]
        draw.rectangle(rect, fill=(220, 60, 60))

    for index, (x, y) in enumerate(snake):
        color = (70, 210, 90) if index == 0 else (50, 150, 70)
        rect = [
            x * CELL_SIZE + 3,
            y_offset + y * CELL_SIZE + 3,
            (x + 1) * CELL_SIZE - 3,
            y_offset + (y + 1) * CELL_SIZE - 3,
        ]
        draw.rectangle(rect, fill=color)

    image.save(output_path)


def frames_to_gif(frame_paths, output_path, duration=90):
    frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def frames_to_video(frame_paths, output_path, fps=12):
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
