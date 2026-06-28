import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = BASE_DIR / "data" / "fonts"

# -----------------------------
# Config
# -----------------------------

IMAGE_SIZE = 28
FONT_SIZE = 22

# -----------------------------
# Load Fonts
# -----------------------------

font_paths = list(FONT_DIR.glob("*.ttf")) + list(FONT_DIR.glob("*.ttc"))

if not font_paths:
    raise ValueError("No fonts found in data/fonts")


def apply_rotation(image):
    angle = random.uniform(-8, 8)
    return image.rotate(angle, fillcolor=255)


def apply_noise(image):
    array = np.array(image).astype(np.float32)
    noise = np.random.normal(loc=0, scale=5, size=array.shape)
    array += noise
    array = np.clip(array, 0, 255)
    return Image.fromarray(array.astype(np.uint8))


def apply_blur(image):
    radius = random.uniform(0, 1.2)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


CHARACTERS = ["<", ">"] + [str(digit) for digit in range(10)] + [chr(code) for code in range(ord("A"), ord("Z") + 1)]

def generate_character_image(character):
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)
    draw = ImageDraw.Draw(image)

    font_path = random.choice(font_paths)
    font = ImageFont.truetype(str(font_path), FONT_SIZE)

    x = random.randint(4, 10)
    y = random.randint(0, 6)

    draw.text((x, y), str(character), font=font, fill=0)

    image = apply_rotation(image)
    image = apply_blur(image)
    image = apply_noise(image)

    return image


# -----------------------------
# Visualization
# -----------------------------

if __name__ == "__main__":
    fig, axes = plt.subplots(4, 10, figsize=(10, 5))

    for character, ax in zip(CHARACTERS, axes.flatten()):
        image = generate_character_image(character)
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Char {character}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# poetry run python -m generator.render_digits