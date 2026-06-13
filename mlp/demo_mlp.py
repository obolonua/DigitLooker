from pathlib import Path
import sys

import numpy as np


# Allow running this file directly from the repo root without packaging the folder.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mlp.layers import Layer
from mlp.activations import ReLU, Softmax
from mlp.losses import CategoricalCrossEntropy, SoftmaxCrossEntropy


# Simple MLP walkthrough
# ======================
#
# This script shows one full forward pass and one full backward pass through a
# small multilayer perceptron.
#
# What it demonstrates:
# - input tensor shape and values
# - layer weights and biases
# - linear output after each dense layer
# - ReLU activation output
# - softmax probabilities
# - loss and accuracy
# - gradients for each layer
# - a single parameter update step
#
# Use this file when you want to understand how data moves through the network
# and how gradients flow back through it.


def print_array(name, array):
    print(f"\n{name}:")
    print(array)
    print(f"shape={array.shape}, dtype={array.dtype}")


def main():
    np.set_printoptions(precision=4, suppress=True)

    # Small, readable example batch.
    X = np.array(
        [
            [1.0, 0.5, -1.2, 0.3],
            [-0.7, 0.8, 0.2, 1.1],
        ],
        dtype=np.float32,
    )
    y = np.array([1, 0], dtype=np.int64)

    print_array("Input X", X)
    print_array("Labels y", y)

    layer1 = Layer(4, 5)
    activation1 = ReLU()
    layer2 = Layer(5, 3)
    softmax = Softmax()
    loss_function = CategoricalCrossEntropy()
    loss_backward = SoftmaxCrossEntropy()

    print_array("Layer 1 weights", layer1.weights)
    print_array("Layer 1 biases", layer1.biases)
    print_array("Layer 2 weights", layer2.weights)
    print_array("Layer 2 biases", layer2.biases)

    # Forward pass
    print("\n=== Forward pass ===")
    z1 = layer1.forward(X)
    print_array("Z1 = X @ W1 + b1", z1)

    a1 = activation1.forward(z1)
    print_array("A1 = ReLU(Z1)", a1)

    z2 = layer2.forward(a1)
    print_array("Z2 = A1 @ W2 + b2", z2)

    probs = softmax.forward(z2)
    print_array("P = Softmax(Z2)", probs)

    loss = loss_function.forward(probs, y)
    print(f"\nLoss: {loss:.6f}")

    predictions = np.argmax(probs, axis=1)
    print_array("Predictions", predictions)
    print(f"Accuracy: {np.mean(predictions == y):.4f}")

    # Backward pass
    print("\n=== Backward pass ===")
    loss_backward.backward(probs, y)
    print_array("dL/dZ2 from SoftmaxCrossEntropy", loss_backward.dinputs)

    layer2.backward(loss_backward.dinputs)
    print_array("dW2", layer2.dweights)
    print_array("db2", layer2.dbiases)
    print_array("dA1", layer2.dinputs)

    activation1.backward(layer2.dinputs)
    print_array("dZ1 after ReLU backward", activation1.dinputs)

    layer1.backward(activation1.dinputs)
    print_array("dW1", layer1.dweights)
    print_array("db1", layer1.dbiases)
    print_array("dX", layer1.dinputs)

    learning_rate = 0.1
    print("\n=== One parameter update ===")
    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases

    print_array("Updated Layer 1 weights", layer1.weights)
    print_array("Updated Layer 1 biases", layer1.biases)
    print_array("Updated Layer 2 weights", layer2.weights)
    print_array("Updated Layer 2 biases", layer2.biases)


if __name__ == "__main__":
    main()
