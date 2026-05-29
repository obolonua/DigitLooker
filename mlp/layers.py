# mlp/layers.py

import numpy as np


# Layer is a fully connected layer that applies a learned linear transform.
class Layer:
    # Initialize weights with He initialization and start biases at zero.
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

    # Store the input for later use and compute the linear output.
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases
