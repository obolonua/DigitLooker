
import numpy as np

from layers import Layer
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy


# Sample batch of 5 flattened images with 784 input features each.
X = np.random.randn(5, 784)

# First hidden layer: 784 inputs -> 128 features.
layer1 = Layer(784, 128)
activation1 = ReLU()

# Second hidden layer: 128 features -> 64 features.
layer2 = Layer(128, 64)
activation2 = ReLU()

# Output layer: 64 features -> 10 class scores.
layer3 = Layer(64, 10)
softmax = Softmax()

# Loss function used to compare predicted probabilities with labels.
loss_function = CategoricalCrossEntropy()


# Forward pass
# Run the batch through the network one layer at a time.
output1 = layer1.forward(X)
output2 = activation1.forward(output1)

output3 = layer2.forward(output2)
output4 = activation2.forward(output3)

output5 = layer3.forward(output4)

predictions = softmax.forward(output5)

# Print the predicted class probabilities and their shape.
print(predictions)
print(predictions.shape)








# ==========================================================
# Simple MLP Forward Pass Example
# ==========================================================
#
# Network architecture:
#
# Input Layer (784 features)
#          ↓
# Dense Layer (784 → 128)
#          ↓
# ReLU Activation
#          ↓
# Dense Layer (128 → 64)
#          ↓
# ReLU Activation
#          ↓
# Dense Layer (64 → 10)
#          ↓
# Softmax Activation
#          ↓
# Class Probabilities
#
# Explanation:
#
# 1. X contains 5 samples, each with 784 input features.
#    (Equivalent to flattened 28x28 images.)
#
# 2. Layer 1:
#    Computes:
#        output = X · W + b
#    producing 128 features for each sample.
#
# 3. ReLU:
#    Replaces negative values with 0 while keeping
#    positive values unchanged.
#
# 4. Layer 2:
#    Transforms the 128 features into 64 features.
#
# 5. ReLU:
#    Introduces non-linearity again.
#
# 6. Layer 3:
#    Produces 10 output values (one for each class).
#    These values are called logits.
#
# 7. Softmax:
#    Converts logits into probabilities that sum to 1.
#
# Example output:
#    [0.02, 0.01, 0.65, ..., 0.05]
#
# This means:
#    Class 2 has a 65% predicted probability.
#
# Final shape:
#    (5, 10)
#
# Meaning:
#    5 samples
#    10 class probabilities per sample
#
# At this stage the network only performs prediction.
# No learning occurs yet because:
#    - Loss is not calculated
#    - Gradients are not computed
#    - Weights are not updated
#
# These steps will be added during backpropagation and training.
# ==========================================================