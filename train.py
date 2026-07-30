from pathlib import Path

import numpy as np

from data.mnist import load_mnist
from mlp.activations import ReLU, Softmax
from mlp.layers import Layer
from mlp.losses import CategoricalCrossEntropy, SoftmaxCrossEntropy


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.008
VALIDATION_SIZE = 5000

np.random.seed(RANDOM_SEED)


def make_batches(images, labels, batch_size):
    # Shuffle the indexes so the batches are different in every epoch.
    indexes = np.arange(len(images))
    np.random.shuffle(indexes)

    batches = []
    for start in range(0, len(images), batch_size):
        batch_indexes = indexes[start:start + batch_size]
        image_batch = images[batch_indexes]
        label_batch = labels[batch_indexes]
        batches.append((image_batch, label_batch))

    return batches


def calculate_accuracy(probabilities, correct_labels):
    predictions = np.argmax(probabilities, axis=1)
    return np.mean(predictions == correct_labels)


# Load the MNIST images and labels.
X_train, y_train, X_test, y_test = load_mnist()

# Use the last 5000 training images for validation.
X_val = X_train[-VALIDATION_SIZE:]
y_val = y_train[-VALIDATION_SIZE:]
X_train = X_train[:-VALIDATION_SIZE]
y_train = y_train[:-VALIDATION_SIZE]

print("train:", X_train.shape, y_train.shape)
print("val:", X_val.shape, y_val.shape)
print("test:", X_test.shape, y_test.shape)
print("labels:", np.unique(y_train))

# Create the neural network.
# The sizes are 784 -> 256 -> 128 -> 10.
layer1 = Layer(784, 256)
relu1 = ReLU()
layer2 = Layer(256, 128)
relu2 = ReLU()
layer3 = Layer(128, 10)
softmax = Softmax()
loss_function = CategoricalCrossEntropy()
loss_backward = SoftmaxCrossEntropy()

for epoch in range(EPOCHS):
    total_loss = 0
    total_accuracy = 0
    batches = make_batches(X_train, y_train, BATCH_SIZE)

    for X_batch, y_batch in batches:
        # Forward pass: move the images through all the layers.
        layer1.forward(X_batch)
        relu1.forward(layer1.output)
        layer2.forward(relu1.output)
        relu2.forward(layer2.output)
        layer3.forward(relu2.output)
        probabilities = softmax.forward(layer3.output)

        loss = loss_function.forward(probabilities, y_batch)
        accuracy = calculate_accuracy(probabilities, y_batch)
        total_loss += loss
        total_accuracy += accuracy

        # Backward pass: calculate how the weights should change.
        loss_backward.backward(probabilities, y_batch)
        layer3.backward(loss_backward.dinputs)
        relu2.backward(layer3.dinputs)
        layer2.backward(relu2.dinputs)
        relu1.backward(layer2.dinputs)
        layer1.backward(relu1.dinputs)

        # Update the weights and biases using gradient descent.
        layer1.weights -= LEARNING_RATE * layer1.dweights
        layer1.biases -= LEARNING_RATE * layer1.dbiases
        layer2.weights -= LEARNING_RATE * layer2.dweights
        layer2.biases -= LEARNING_RATE * layer2.dbiases
        layer3.weights -= LEARNING_RATE * layer3.dweights
        layer3.biases -= LEARNING_RATE * layer3.dbiases

    # Check the model with validation data after each epoch.
    layer1.forward(X_val)
    relu1.forward(layer1.output)
    layer2.forward(relu1.output)
    relu2.forward(layer2.output)
    layer3.forward(relu2.output)
    val_probabilities = softmax.forward(layer3.output)

    val_loss = loss_function.forward(val_probabilities, y_val)
    val_accuracy = calculate_accuracy(val_probabilities, y_val)

    average_loss = total_loss / len(batches)
    average_accuracy = total_accuracy / len(batches)

    print(
        f"epoch {epoch + 1:02d}/{EPOCHS} | "
        f"loss {average_loss:.4f} | "
        f"acc {average_accuracy:.4f} | "
        f"val_loss {val_loss:.4f} | "
        f"val_acc {val_accuracy:.4f}"
    )

# Test the finished model.
layer1.forward(X_test)
relu1.forward(layer1.output)
layer2.forward(relu1.output)
relu2.forward(layer2.output)
layer3.forward(relu2.output)
test_probabilities = softmax.forward(layer3.output)

test_accuracy = calculate_accuracy(test_probabilities, y_test)
test_loss = loss_function.forward(test_probabilities, y_test)

# Save the trained weights so app.py can use them.
np.savez(
    MODEL_DIR / "digit_mlp_weights.npz",
    layer1_weights=layer1.weights,
    layer1_biases=layer1.biases,
    layer2_weights=layer2.weights,
    layer2_biases=layer2.biases,
    layer3_weights=layer3.weights,
    layer3_biases=layer3.biases,
)

print(f"test_loss {test_loss:.4f} | test_acc {test_accuracy:.4f}")
print(f"saved weights to {MODEL_DIR / 'digit_mlp_weights.npz'}")

# Usage note: run with `poetry run python train.py`.
