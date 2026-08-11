"""End-to-end learning test for the hand-written MLP implementation."""

import unittest

import numpy as np
from sklearn.datasets import load_digits

from mlp.activations import ReLU, Softmax
from mlp.layers import Layer
from mlp.losses import CategoricalCrossEntropy, SoftmaxCrossEntropy


class NetworkOverfittingTest(unittest.TestCase):
    def test_network_overfits_tiny_multiclass_dataset(self):
        """Gradients and parameter updates can memorize a few real digit images."""
        digits = load_digits()
        selected = np.flatnonzero(np.isin(digits.target, (0, 1)))[:6]
        images = digits.data[selected].astype(np.float64) / 16.0
        labels = digits.target[selected].astype(np.int64)

        # Guard against accidentally weakening the test to a single class.
        self.assertEqual(set(labels), {0, 1})

        np.random.seed(7)
        hidden_layer = Layer(images.shape[1], 8)
        relu = ReLU()
        output_layer = Layer(8, 2)
        softmax = Softmax()
        loss_function = CategoricalCrossEntropy()
        loss_backward = SoftmaxCrossEntropy()

        learning_rate = 0.2
        target_loss = 1e-2
        probabilities = None

        # The complete subset is one minibatch, repeated until it is memorized.
        for _ in range(100):
            hidden_layer.forward(images)
            relu.forward(hidden_layer.output)
            output_layer.forward(relu.output)
            probabilities = softmax.forward(output_layer.output)

            if loss_function.forward(probabilities, labels) < target_loss:
                break

            loss_backward.backward(probabilities, labels)
            output_layer.backward(loss_backward.dinputs)
            relu.backward(output_layer.dinputs)
            hidden_layer.backward(relu.dinputs)

            hidden_layer.weights -= learning_rate * hidden_layer.dweights
            hidden_layer.biases -= learning_rate * hidden_layer.dbiases
            output_layer.weights -= learning_rate * output_layer.dweights
            output_layer.biases -= learning_rate * output_layer.dbiases

        final_loss = loss_function.forward(probabilities, labels)
        accuracy = np.mean(np.argmax(probabilities, axis=1) == labels)

        self.assertLess(final_loss, target_loss)
        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()


# poetry run python -m unittest tests.network_overfitting -v