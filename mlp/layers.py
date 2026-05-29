# mlp/layers.py

import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases