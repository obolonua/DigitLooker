from pathlib import Path

import numpy as np

from data.mnist import load_mnist
from mlp.activations import ReLU, Softmax
from mlp.layers import Layer


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR.parent / "models" / "digit_mlp_weights.npz"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing model weights at {MODEL_FILE}. Train the model first with `poetry run python train.py`."
        )

    weights = np.load(MODEL_FILE)

    layer1 = Layer(784, 128)
    layer2 = Layer(128, 10)

    layer1.weights = weights["layer1_weights"]
    layer1.biases = weights["layer1_biases"]
    layer2.weights = weights["layer2_weights"]
    layer2.biases = weights["layer2_biases"]

    return layer1, layer2


def predict(image_array):
    layer1, layer2 = load_model()
    relu = ReLU()
    softmax = Softmax()

    x = image_array.reshape(1, -1).astype(np.float32)

    layer1.forward(x)
    relu.forward(layer1.output)
    layer2.forward(relu.output)
    probabilities = softmax.forward(layer2.output)[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return predicted_digit, confidence, probabilities


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    sample_count = 1000
    sample_indices = np.random.choice(len(X_test), size=sample_count, replace=False)

    correct = 0
    for index in sample_indices:
        image = (X_test[index].reshape(28, 28) * 255).astype(np.uint8)
        predicted_digit, confidence, _ = predict(image)
        correct += int(predicted_digit == int(y_test[index]))
        # print(
        #     f"target={int(y_test[index])} predicted={predicted_digit} "
        #     f"confidence={confidence:.3f}"
        # )

    print(f"accuracy: {correct / sample_count:.4f}")

# poetry run python -m tests.predict_digit
