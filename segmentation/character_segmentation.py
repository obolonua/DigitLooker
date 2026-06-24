from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageSequence


IMAGE_SIZE = 28


def _load_image(path):
    # Load the first frame of an image file as a grayscale numpy array.
    image_path = Path(path)
    with Image.open(image_path) as image:
        frame = next(ImageSequence.Iterator(image))
        return np.array(frame.convert("L"))


def _box_blur(image):
    # Apply a light blur to reduce noise before thresholding.
    pil_image = Image.fromarray(image)
    return np.array(pil_image.filter(ImageFilter.GaussianBlur(radius=1.0)))


def _binarize(image):
    # Convert a grayscale image into a binary foreground mask.
    blurred = _box_blur(image)
    threshold = float(np.mean(blurred))
    foreground = blurred < threshold
    return foreground.astype(np.uint8)


def _connected_components(mask, min_area=20):
    # Find bounding boxes for connected foreground regions.
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or visited[y, x]:
                continue

            queue = deque([(y, x)])
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0

            while queue:
                cy, cx = queue.popleft()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue
                        if visited[ny, nx] or mask[ny, nx] == 0:
                            continue
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if area >= min_area:
                boxes.append((min_x, min_y, max_x + 1, max_y + 1))

    boxes.sort(key=lambda box: (box[0], box[1]))
    return boxes


def _resize_with_padding(image, size=IMAGE_SIZE):
    # Resize a crop to fit inside a square canvas while preserving aspect ratio.
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Cannot resize an empty image crop")

    scale = min((size - 4) / width, (size - 4) / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    pil_image = Image.fromarray(image)
    resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_array = np.array(resized, dtype=np.uint8)

    canvas = np.full((size, size), 255, dtype=np.uint8)
    y_offset = (size - new_height) // 2
    x_offset = (size - new_width) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized_array
    return canvas


def segment_characters(image_path, size=IMAGE_SIZE):
    # Segment characters from a file path and return 28x28 grayscale crops.
    grayscale = _load_image(image_path)
    mask = _binarize(grayscale)
    boxes = _connected_components(mask)

    characters: list[np.ndarray] = []
    for x0, y0, x1, y1 in boxes:
        crop = grayscale[y0:y1, x0:x1]
        characters.append(_resize_with_padding(crop, size=size))

    return characters


def segment_character_arrays(images, size=IMAGE_SIZE):
    # Segment characters from already-loaded image arrays.
    segmented = []
    for image in images:
        if image.ndim == 3:
            image = image.mean(axis=2)
        image = image.astype(np.uint8)
        mask = _binarize(image)
        boxes = _connected_components(mask)
        chars = [_resize_with_padding(image[y0:y1, x0:x1], size=size) for x0, y0, x1, y1 in boxes]
        segmented.append(chars)
    return segmented


def show_segmented_characters(characters, cols=8):
    # Display segmented characters in a grid for quick inspection.
    if not characters:
        print("No characters to display.")
        return

    rows = (len(characters) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)

    for index, ax in enumerate(axes):
        ax.axis("off")
        if index < len(characters):
            ax.imshow(characters[index], cmap="gray")
            ax.set_title(str(index), fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 -m segmentation.character_segmentation <image.tif>")

    chars = segment_characters(sys.argv[1])
    # show_segmented_characters(chars)
    print(f"Segmented {len(chars)} character(s)")

# poetry run python -m segmentation.character_segmentation codes/_85534873_schengenvisanewafp.jpg-0.tif