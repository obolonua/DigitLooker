"""Training-signal test for the hand-written MLP implementation."""

import unittest

import numpy as np
from sklearn.datasets import load_digits

from mlp.activations import ReLU, Softmax
from mlp.layers import Layer
from mlp.losses import CategoricalCrossEntropy, SoftmaxCrossEntropy


class LossDecreasesTest(unittest.TestCase):
    def test_gradients_are_non_zero_and_loss_decreases(self):
        """Several minibatch updates propagate gradients and improve the loss."""
        digits = load_digits()
        images = digits.data[:256].astype(np.float64) / 16.0
        labels = digits.target[:256].astype(np.int64)

        np.random.seed(13)
        hidden_layer = Layer(images.shape[1], 32)
        relu = ReLU()
        output_layer = Layer(32, 10)
        softmax = Softmax()
        loss_function = CategoricalCrossEntropy()
        loss_backward = SoftmaxCrossEntropy()

        def forward(batch):
            hidden_layer.forward(batch)
            relu.forward(hidden_layer.output)
            output_layer.forward(relu.output)
            return softmax.forward(output_layer.output)

        initial_loss = loss_function.forward(forward(images), labels)
        learning_rate = 0.1
        batch_size = 32

        for _ in range(20):
            for start in range(0, len(images), batch_size):
                image_batch = images[start:start + batch_size]
                label_batch = labels[start:start + batch_size]
                probabilities = forward(image_batch)

                loss_backward.backward(probabilities, label_batch)
                output_layer.backward(loss_backward.dinputs)
                relu.backward(output_layer.dinputs)
                hidden_layer.backward(relu.dinputs)

                gradients = (
                    hidden_layer.dweights,
                    hidden_layer.dbiases,
                    output_layer.dweights,
                    output_layer.dbiases,
                )
                
                for gradient in gradients:
                    self.assertTrue(np.all(np.isfinite(gradient)))
                    self.assertGreater(np.linalg.norm(gradient), 0.0)

                hidden_layer.weights -= learning_rate * hidden_layer.dweights
                hidden_layer.biases -= learning_rate * hidden_layer.dbiases
                output_layer.weights -= learning_rate * output_layer.dweights
                output_layer.biases -= learning_rate * output_layer.dbiases

        final_loss = loss_function.forward(forward(images), labels)
        print(initial_loss, final_loss)
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()


# poetry run python -m tests.loss_decreases