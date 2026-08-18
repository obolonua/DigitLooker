# mlp/activations.py

import numpy as np


# ReLU keeps positive values unchanged and clips negative values to zero.
class ReLU:
    # Save the input for future gradient calculations and apply the ReLU transform.
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0

# Softmax turns raw scores into probabilities that sum to 1 across each row.
class Softmax:
    # Shift values for numerical stability, then normalize each row into probabilities.
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return self.output