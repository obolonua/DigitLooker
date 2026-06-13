import numpy as np


class Layer:
    # Fully connected dense layer.
    # The layer computes:
    #     output = inputs @ weights + biases
    # during the forward pass, and stores the values needed to compute gradients
    # during the backward pass.

    def __init__(self, input_size, output_size):
        # Create a dense layer with He initialization.
        # Parameters:
        #     input_size: Number of features in each input sample.
        #     output_size: Number of neurons in the layer.
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        # Run the forward pass for a batch of inputs.
        # Parameters:
        #     inputs: Array of shape (batch_size, input_size).
        # Returns:
        #     The linear output of shape (batch_size, output_size).
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        # Backpropagate gradients through the dense layer.
        # Parameters:
        #     dvalues: Gradient of the loss with respect to the layer output.
        # Produces:
        #     dweights: Gradient with respect to the weights.
        #     dbiases: Gradient with respect to the biases.
        #     dinputs: Gradient with respect to the inputs.
        self.dweights = np.dot(
            self.inputs.T,
            dvalues
        )

        self.dbiases = np.sum(
            dvalues,
            axis=0,
            keepdims=True
        )

        self.dinputs = np.dot(
            dvalues,
            self.weights.T
        )

if __name__ == "__main__":
    layer = Layer(4, 3)
    X = np.random.randn(5, 4)
    layer.forward(X)
    fake_gradient = np.random.randn(5, 3)
    layer.backward(fake_gradient)
    print(layer.dweights.shape)
    print(layer.dbiases.shape)
    print(layer.dinputs.shape)
