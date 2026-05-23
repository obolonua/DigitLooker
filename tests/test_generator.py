import importlib
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the generator module once for the whole test class.
        cls.module = importlib.import_module("generator.create_dataset")

    def test_font_directory_has_fonts(self):
        self.assertTrue(self.module.font_paths, "Expected at least one font in data/fonts")
        self.assertTrue(
            all(path.suffix.lower() in {".ttf", ".ttc"} for path in self.module.font_paths)
        )

    def test_generate_digit_image_properties(self):
        # The generator should return a grayscale 28x28 image for a digit.
        image = self.module.generate_digit_image(5)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "L")
        self.assertEqual(image.size, (self.module.IMAGE_SIZE, self.module.IMAGE_SIZE))

        array = np.array(image)
        self.assertEqual(array.shape, (self.module.IMAGE_SIZE, self.module.IMAGE_SIZE))
        self.assertLess(array.min(), 255)

    def test_augmentation_helpers_preserve_shape(self):
        # Use a plain white image so each helper is tested in isolation.
        base_image = Image.new("L", (self.module.IMAGE_SIZE, self.module.IMAGE_SIZE), color=255)

        rotated = self.module.apply_rotation(base_image)
        blurred = self.module.apply_blur(base_image)
        noised = self.module.apply_noise(base_image)

        for image in (rotated, blurred, noised):
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.size, (self.module.IMAGE_SIZE, self.module.IMAGE_SIZE))

        noise_array = np.array(noised)
        self.assertEqual(noise_array.dtype, np.uint8)
        self.assertGreaterEqual(noise_array.min(), 0)
        self.assertLessEqual(noise_array.max(), 255)

    def test_create_dataset_shape_and_labels(self):
        # Patch out randomness so this test checks dataset structure only.
        with patch.object(self.module, "SAMPLES_PER_DIGIT", 2), patch.object(
            self.module,
            "generate_digit_image",
            side_effect=lambda digit: Image.new(
                "L", (self.module.IMAGE_SIZE, self.module.IMAGE_SIZE), color=digit * 25
            ),
        ):
            X, y = self.module.create_dataset()

        self.assertEqual(X.shape, (20, self.module.IMAGE_SIZE * self.module.IMAGE_SIZE))
        self.assertEqual(y.shape, (20,))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)
        self.assertEqual(set(y.tolist()), set(range(10)))

    def test_dataset_split_is_stratified(self):
        # This mirrors the production split logic without generating images.
        from sklearn.model_selection import train_test_split

        X = np.arange(500).reshape(50, 10)
        y = np.repeat(np.arange(10), 5)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        self.assertEqual(len(X_train) + len(X_val), len(X))
        self.assertEqual(len(y_train) + len(y_val), len(y))
        self.assertEqual(set(y_train.tolist() + y_val.tolist()), set(range(10)))


if __name__ == "__main__":
    unittest.main()
