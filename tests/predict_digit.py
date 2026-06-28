from pathlib import Path

import numpy as np
import random

from generator.render_digits import CHARACTERS, generate_character_image
from mlp.activations import ReLU, Softmax
from mlp.layers import Layer


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR.parent / "models" / "digit_mlp_weights.npz"


def load_model():
    weights = np.load(MODEL_FILE)

    layer1 = Layer(784, 128)
    layer2 = Layer(128, len(CHARACTERS))

    layer1.weights = weights["layer1_weights"]
    layer1.biases = weights["layer1_biases"]
    layer2.weights = weights["layer2_weights"]
    layer2.biases = weights["layer2_biases"]

    return layer1, layer2

def predict(image_array):
    layer1, layer2 = load_model()
    relu = ReLU()
    softmax = Softmax()

    x = image_array.reshape(1, -1).astype(np.float32) / 255.0

    layer1.forward(x)
    relu.forward(layer1.output)
    layer2.forward(relu.output)
    probabilities = softmax.forward(layer2.output)

    return int(np.argmax(probabilities, axis=1)[0]), probabilities[0]

if __name__ == "__main__":
    # Generate a fresh synthetic character image the model has not seen before.
    correct = 0
    n = 100
    for i in range(n):

        target_digit = random.randrange(len(CHARACTERS))
        image = generate_character_image(CHARACTERS[target_digit])

        predicted_digit, probabilities = predict(np.array(image))
        if target_digit == predicted_digit:
            correct += 1

    print("accuracy:", correct/n)

# poetry run python -m tests.predict_digit
