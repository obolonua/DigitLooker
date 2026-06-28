import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = BASE_DIR / "data" / "fonts"
OUTPUT_DIR = BASE_DIR / "data" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Image and dataset settings used for synthetic character generation.
IMAGE_SIZE = 28
FONT_SIZE = 22
SAMPLES_PER_CHAR = 2500
CHARACTERS = ["<", ">"] + [str(digit) for digit in range(10)] + [chr(code) for code in range(ord("A"), ord("Z") + 1)]

# Collect every available font file so each sample can be rendered with small style variations.
font_paths = list(FONT_DIR.glob("*.ttf")) + list(FONT_DIR.glob("*.ttc"))

if not font_paths:
    raise ValueError("No fonts found in data/fonts")


def apply_rotation(image):
    # Small rotations make the synthetic digits closer to real handwritten or scanned input.
    angle = random.uniform(-8, 8)
    return image.rotate(angle, fillcolor=255)


def apply_noise(image):
    # Add pixel-level noise to simulate imperfect scan quality and image compression artifacts.
    array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 12, array.shape)
    array += noise
    array = np.clip(array, 0, 255)
    return Image.fromarray(array.astype(np.uint8))


def apply_blur(image):
    # Slight blur helps the model handle soft or out-of-focus receipt text.
    radius = random.uniform(0, 1.2)
    return image.filter(ImageFilter.GaussianBlur(radius))


def generate_character_image(character):
    # Draw one character on a clean canvas, then apply the augmentations above.
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


def create_dataset():
    X = []
    y = []

    # Generate the same number of samples for each character class.
    for label, character in enumerate(CHARACTERS):
        print(f"Generating character {character}...")
        for _ in range(SAMPLES_PER_CHAR):
            image = generate_character_image(character)
            # Normalize to [0, 1] and flatten into a vector for later model training.
            array = np.array(image) / 255.0
            vector = array.flatten()
            X.append(vector)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


if __name__ == "__main__":
    X, y = create_dataset()
    print("\nDataset created:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Split the generated samples into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)

    print("\nDatasets saved in data/generated/")

# poetry run python -m generator.create_dataset
