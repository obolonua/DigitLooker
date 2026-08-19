"""Deterministic unit tests for the hand-written MLP building blocks."""

import unittest

import numpy as np

from mlp.activations import ReLU, Softmax
from mlp.layers import Layer
from mlp.losses import CategoricalCrossEntropy, SoftmaxCrossEntropy


class LayerTest(unittest.TestCase):
    def test_forward_and_backward_compute_expected_values(self):
        layer = Layer(2, 2)
        layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.biases = np.array([[0.5, -0.5]])
        inputs = np.array([[1.0, 2.0], [-1.0, 3.0]])

        output = layer.forward(inputs)
        np.testing.assert_allclose(output, [[7.5, 9.5], [8.5, 9.5]])

        gradients = np.array([[1.0, -1.0], [2.0, 0.5]])
        layer.backward(gradients)

        np.testing.assert_allclose(layer.dweights, [[-1.0, -1.5], [8.0, -0.5]])
        np.testing.assert_allclose(layer.dbiases, [[3.0, -0.5]])
        np.testing.assert_allclose(layer.dinputs, [[-1.0, -1.0], [3.0, 8.0]])


class ActivationTest(unittest.TestCase):
    def test_relu_forward_and_backward_zero_non_positive_values(self):
        relu = ReLU()
        output = relu.forward(np.array([[-2.0, 0.0, 3.0]]))
        np.testing.assert_array_equal(output, [[0.0, 0.0, 3.0]])

        relu.backward(np.array([[4.0, 5.0, 6.0]]))
        np.testing.assert_array_equal(relu.dinputs, [[0.0, 0.0, 6.0]])

    def test_softmax_is_stable_and_normalizes_each_sample(self):
        probabilities = Softmax().forward(
            np.array([[1000.0, 1000.0], [1.0, 2.0]])
        )

        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
        np.testing.assert_allclose(probabilities[0], [0.5, 0.5])
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertGreater(probabilities[1, 1], probabilities[1, 0])


class LossTest(unittest.TestCase):
    def test_cross_entropy_accepts_class_indices(self):
        loss = CategoricalCrossEntropy().forward(
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            np.array([1, 0]),
        )
        self.assertAlmostEqual(loss, np.mean(-np.log([0.9, 0.8])))

    def test_cross_entropy_accepts_one_hot_labels(self):
        loss = CategoricalCrossEntropy().forward(
            np.array([[0.25, 0.75], [0.6, 0.4]]),
            np.array([[0, 1], [1, 0]]),
        )
        self.assertAlmostEqual(loss, np.mean(-np.log([0.75, 0.6])))

    def test_cross_entropy_clips_zero_probability(self):
        loss = CategoricalCrossEntropy().forward(
            np.array([[1.0, 0.0]]),
            np.array([1]),
        )
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(loss, -np.log(1e-7))

    def test_softmax_cross_entropy_backward_computes_batch_gradient(self):
        backward = SoftmaxCrossEntropy()
        predictions = np.array([[0.2, 0.8], [0.6, 0.4]])
        original = predictions.copy()

        backward.backward(predictions, np.array([1, 0]))

        np.testing.assert_allclose(backward.dinputs, [[0.1, -0.1], [-0.2, 0.2]])
        np.testing.assert_array_equal(predictions, original)


if __name__ == "__main__":
    unittest.main()
