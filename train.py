from pathlib import Path

import numpy as np

from mlp.activations import ReLU, Softmax
from mlp.layers import Layer
from mlp.losses import CategoricalCrossEntropy, SoftmaxCrossEntropy
from data.mnist import load_mnist


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "generated"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_train, y_train, X_val, y_val = load_mnist()

# Use a smaller training subset to keep experimentation fast while iterating.
X_train = X_train[:6000]
y_train = y_train[:6000]
X_val = X_val[:1000]
y_val = y_val[:1000]

print("labels:", np.unique(y_train))
print("train:", X_train.shape, y_train.shape)

# Simple two-layer MLP for 28x28 digit classification.
layer1 = Layer(784, 128)
relu = ReLU()
layer2 = Layer(128, 10)
softmax = Softmax()
loss_function = CategoricalCrossEntropy()
loss_backward = SoftmaxCrossEntropy()

# Learning rate for gradient descent updates.
learning_rate = 0.016

for epoch in range(20000):
    # Forward pass: compute logits and class probabilities.
    layer1.forward(X_train)
    relu.forward(layer1.output)
    layer2.forward(relu.output)
    softmax.forward(layer2.output)

    # Measure training loss and accuracy on the current batch.
    loss = loss_function.forward(softmax.output, y_train)
    predictions = np.argmax(softmax.output, axis=1)
    accuracy = np.mean(predictions == y_train)

    # Backward pass: propagate gradients from the loss through the network.
    loss_backward.backward(softmax.output, y_train)
    layer2.backward(loss_backward.dinputs)
    relu.backward(layer2.dinputs)
    layer1.backward(relu.dinputs)

    # Update weights and biases with vanilla gradient descent.
    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases

    # Run the same forward path on the validation set to track generalization.
    layer1.forward(X_val)
    relu.forward(layer1.output)
    layer2.forward(relu.output)
    softmax.forward(layer2.output)

    val_predictions = np.argmax(softmax.output, axis=1)
    val_acc = np.mean(val_predictions == y_val)

    if epoch % 100 == 0 or epoch == 19999:
        print(
            f"epoch {epoch:5d} | loss {loss:.4f} | "
            f"train_acc {accuracy:.4f} | val_acc {val_acc:.4f}"
        )

# Save the learned weights so they can be reused without retraining.
np.savez(
    MODEL_DIR / "digit_mlp_weights.npz",
    layer1_weights=layer1.weights,
    layer1_biases=layer1.biases,
    layer2_weights=layer2.weights,
    layer2_biases=layer2.biases,
)

print(f"saved weights to {MODEL_DIR / 'digit_mlp_weights.npz'}")

# Usage note: run with `poetry run python train.py`.
