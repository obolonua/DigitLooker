"""Verify that every trainable layer participates in optimization."""

import unittest

import numpy as np
from sklearn.datasets import load_digits

from mlp.activations import ReLU, Softmax
from mlp.layers import Layer
from mlp.losses import SoftmaxCrossEntropy


class AllLayersChangeTest(unittest.TestCase):
    def test_all_layer_weights_change_after_each_optimizer_step(self):
        """Each minibatch update changes every dense layer's weights."""
        digits = load_digits()
        images = digits.data[:128].astype(np.float64) / 16.0
        labels = digits.target[:128].astype(np.int64)

        np.random.seed(19)
        layers = (
            Layer(images.shape[1], 32),
            Layer(32, 16),
            Layer(16, 10),
        )
        relu1 = ReLU()
        relu2 = ReLU()
        softmax = Softmax()
        loss_backward = SoftmaxCrossEntropy()

        learning_rate = 0.05
        batch_size = 32

        for batch_number, start in enumerate(range(0, len(images), batch_size), 1):
            image_batch = images[start:start + batch_size]
            label_batch = labels[start:start + batch_size]

            layers[0].forward(image_batch)
            relu1.forward(layers[0].output)
            layers[1].forward(relu1.output)
            relu2.forward(layers[1].output)
            layers[2].forward(relu2.output)
            probabilities = softmax.forward(layers[2].output)

            loss_backward.backward(probabilities, label_batch)
            layers[2].backward(loss_backward.dinputs)
            relu2.backward(layers[2].dinputs)
            layers[1].backward(relu2.dinputs)
            relu1.backward(layers[1].dinputs)
            layers[0].backward(relu1.dinputs)

            weights_before_step = [layer.weights.copy() for layer in layers]
            for layer in layers:
                layer.weights -= learning_rate * layer.dweights
                layer.biases -= learning_rate * layer.dbiases

            for layer_number, (before, layer) in enumerate(
                zip(weights_before_step, layers), 1
            ):
                self.assertFalse(
                    np.array_equal(before, layer.weights),
                    msg=(
                        f"layer {layer_number} weights did not change after "
                        f"optimizer step {batch_number}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
