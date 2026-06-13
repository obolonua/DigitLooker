import numpy as np

class CategoricalCrossEntropy:
    # Categorical cross-entropy loss for multi-class classification.
    # This loss measures how well predicted class probabilities match the true
    # labels. Lower values indicate that the model assigns more probability to
    # the correct class.

    def forward(self, y_pred, y_true):
        # Compute the average categorical cross-entropy loss.
        # Parameters:
        #     y_pred: Predicted class probabilities with shape
        #         (batch_size, num_classes).
        #     y_true: True labels as class indices or one-hot encoded vectors.
        # Returns:
        #     The mean negative log likelihood over the batch.
        samples = len(y_pred)

        y_pred_clipped = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        else:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(
            correct_confidences
        )

        return np.mean(
            negative_log_likelihoods
        )


class SoftmaxCrossEntropy:
    # Backward pass helper for softmax + cross-entropy.
    # This computes the gradient of the combined softmax activation and
    # categorical cross-entropy loss with respect to the softmax outputs.

    def backward(self, y_pred, y_true):
        # Compute gradients for the softmax-cross-entropy combination.
        # Parameters:
        #     y_pred: Softmax probabilities with shape (batch_size, num_classes).
        #     y_true: True labels as class indices.
        samples = len(y_pred)
        self.dinputs = y_pred.copy()
        self.dinputs[
            range(samples),
            y_true
        ] -= 1

        self.dinputs /= samples


if __name__ == "__main__":
    y_pred = np.array([
        [0.05, 0.10, 0.85],
        [0.70, 0.20, 0.10],
        [0.02, 0.15, 0.83],
    ])
    y_true = np.array([2, 0, 2])

    loss = CategoricalCrossEntropy().forward(y_pred, y_true)

    print("predictions:")
    print(y_pred)
    print("\nlabels:")
    print(y_true)
    print("\nloss:")
    print(loss)


# Categorical Cross-Entropy Loss
# ==============================
#
# Purpose
# -------
# Measures how well the predicted class probabilities match the true labels.
#
# The lower the loss:
#     - the better the predictions
#     - the more confident the network is about the correct classes
#
# The higher the loss:
#     - the more incorrect the predictions are
#     - especially when the model is confidently wrong
#
# Example
# -------
#
# Predicted probabilities:
#
#     [
#         [0.05, 0.10, 0.85],
#         [0.70, 0.20, 0.10],
#         [0.02, 0.15, 0.83]
#     ]
#
# True labels:
#
#     [2, 0, 2]
#
# Step 1: Select the probability assigned to each correct class
#
#     Sample 1 -> class 2 -> 0.85
#     Sample 2 -> class 0 -> 0.70
#     Sample 3 -> class 2 -> 0.83
#
# Result:
#
#     [0.85, 0.70, 0.83]
#
# Step 2: Compute negative log likelihood
#
#     -log(0.85) ≈ 0.163
#     -log(0.70) ≈ 0.357
#     -log(0.83) ≈ 0.186
#
# Result:
#
#     [0.163, 0.357, 0.186]
#
# Step 3: Compute average loss
#
#     (0.163 + 0.357 + 0.186) / 3
#
#     ≈ 0.235
#
#
# Softmax + Cross-Entropy
# =======================
#
# Purpose
# -------
# Used during backpropagation to efficiently compute the gradient of the
# combined softmax activation and categorical cross-entropy loss.
#
# Why combine them:
#     - Softmax converts raw scores into probabilities
#     - Cross-entropy measures how far those probabilities are from the truth
#     - The combined gradient is simpler and more numerically stable than
#       handling the two operations separately
#
# What the backward pass does:
#     - Copies the predicted probabilities
#     - Subtracts 1 from the probability assigned to the correct class
#     - Divides by the number of samples to get the batch average gradient
#
# Example
# -------
#
# Predicted probabilities:
#
#     [
#         [0.05, 0.10, 0.85],
#         [0.70, 0.20, 0.10],
#         [0.02, 0.15, 0.83]
#     ]
#
# True labels:
#
#     [2, 0, 2]
#
# Gradient before averaging:
#
#     [
#         [0.05, 0.10, -0.15],
#         [-0.30, 0.20, 0.10],
#         [0.02, 0.15, -0.17]
#     ]
#
# After dividing by 3 samples:
#
#     [
#         [0.0167, 0.0333, -0.0500],
#         [-0.1000, 0.0667, 0.0333],
#         [0.0067, 0.0500, -0.0567]
#     ]
