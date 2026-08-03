# Week 4

During the fourth week, I focused on testing whether the hand-written neural network actually learns from digit data. I added two end-to-end tests using real digit images from scikit-learn's digits dataset.

The first test checks that the network can overfit a very small dataset containing images of zeros and ones. It trains a network with one hidden layer until it reaches a low loss and verifies that every image is classified correctly. This confirms that the forward pass, backpropagation, and parameter updates work together well enough for the model to memorize a simple multiclass dataset.

The second test checks the training signal on a larger subset containing all ten digit classes. It verifies that the weight and bias gradients are finite and non-zero during training, and that repeated minibatch updates reduce the loss from its initial value. Fixed random seeds are used in both tests so that their results remain reproducible.

The main challenge was designing tests that exercise the complete learning process without being too slow or unreliable. The datasets, network sizes, learning rates, and stopping conditions had to be small enough for automated testing while still clearly demonstrating that the implementation learns.

Next week, I plan to evaluate the model more broadly, measure its classification accuracy on unseen digit images, and investigate any weaknesses found during testing.

## Time Tracking

| Date | Time Spent | Description |
| ----- | ---------- | ----------- |
| 31.7. | 1.5 h | Creating an end-to-end test that verifies the network can overfit a tiny digit dataset |
| 3.8. | 1.5 h | Testing gradients and confirming that minibatch training decreases the loss |
| **Total** | **3 h** | |
