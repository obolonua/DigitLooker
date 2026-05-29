# mlp/activations.py

import numpy as np


# ReLU keeps positive values unchanged and clips negative values to zero.
class ReLU:
    # Save the input for future gradient calculations and apply the ReLU transform.
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

# Softmax turns raw scores into probabilities that sum to 1 across each row.
class Softmax:
    # Shift values for numerical stability, then normalize each row into probabilities.
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return probabilities

if __name__ == "__main__":
    sample_input = np.array([[-2.0, 0.0, 3.0], [1.0, -4.0, 2.0]])

    print("input:")
    print(sample_input)

    print("\nReLU:")
    print(ReLU().forward(sample_input))

    print("\nSoftmax:")
    print(Softmax().forward(sample_input))
