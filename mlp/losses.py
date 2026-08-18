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