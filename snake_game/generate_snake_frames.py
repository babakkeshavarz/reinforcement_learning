from pathlib import Path
import random

from PIL import Image, ImageDraw


CELL_SIZE = 25
GRID_WIDTH = 24
GRID_HEIGHT = 18
IMAGE_WIDTH = CELL_SIZE * GRID_WIDTH
IMAGE_HEIGHT = CELL_SIZE * GRID_HEIGHT

FRAMES_DIR = Path("frames")
ASSETS_DIR = Path("assets")


def draw_frame(snake, food, score, frame_number):
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (18, 18, 18))
    draw = ImageDraw.Draw(image)

    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            rect = [
                x * CELL_SIZE,
                y * CELL_SIZE,
                (x + 1) * CELL_SIZE,
                (y + 1) * CELL_SIZE,
            ]
            draw.rectangle(rect, outline=(35, 35, 35))

    fx, fy = food
    draw.rectangle(
        [
            fx * CELL_SIZE + 2,
            fy * CELL_SIZE + 2,
            (fx + 1) * CELL_SIZE - 2,
            (fy + 1) * CELL_SIZE - 2,
        ],
        fill=(220, 60, 60),
    )

    for i, (x, y) in enumerate(snake):
        color = (70, 200, 90) if i == 0 else (50, 150, 70)
        draw.rectangle(
            [
                x * CELL_SIZE + 2,
                y * CELL_SIZE + 2,
                (x + 1) * CELL_SIZE - 2,
                (y + 1) * CELL_SIZE - 2,
            ],
            fill=color,
        )

    draw.text((10, 8), f"Score: {score}", fill=(240, 240, 240))
    draw.text((10, 30), f"Frame: {frame_number}", fill=(180, 180, 180))

    image.save(FRAMES_DIR / f"frame_{frame_number:04d}.png")


def spawn_food(snake):
    while True:
        food = (
            random.randint(0, GRID_WIDTH - 1),
            random.randint(0, GRID_HEIGHT - 1),
        )
        if food not in snake:
            return food


def choose_direction(head, food, current_direction):
    hx, hy = head
    fx, fy = food

    if fx > hx and current_direction != (-1, 0):
        return (1, 0)
    if fx < hx and current_direction != (1, 0):
        return (-1, 0)
    if fy > hy and current_direction != (0, -1):
        return (0, 1)
    if fy < hy and current_direction != (0, 1):
        return (0, -1)

    return current_direction


def generate_frames(number_of_frames=180, seed=1):
    random.seed(seed)
    FRAMES_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)

    for old_frame in FRAMES_DIR.glob("frame_*.png"):
        old_frame.unlink()

    snake = [
        (GRID_WIDTH // 2, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2),
    ]
    direction = (1, 0)
    food = spawn_food(snake)
    score = 0

    for frame_number in range(number_of_frames):
        draw_frame(snake, food, score, frame_number)

        direction = choose_direction(snake[0], food, direction)
        hx, hy = snake[0]
        dx, dy = direction
        new_head = (hx + dx, hy + dy)

        hit_wall = (
            new_head[0] < 0
            or new_head[0] >= GRID_WIDTH
            or new_head[1] < 0
            or new_head[1] >= GRID_HEIGHT
        )
        hit_self = new_head in snake

        if hit_wall or hit_self:
            snake = [
                (GRID_WIDTH // 2, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2),
            ]
            direction = (1, 0)
            food = spawn_food(snake)
            continue

        snake.insert(0, new_head)

        if new_head == food:
            score += 1
            food = spawn_food(snake)
        else:
            snake.pop()


if __name__ == "__main__":
    generate_frames()
    print("Frames saved to frames/")
