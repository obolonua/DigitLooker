# Week 3

During the third week, I improved the neural network and training process to achieve better accuracy on MNIST. I expanded the model from two layers to three layers, with an architecture of 784, 256, 128, and 10 neurons. I also added a second ReLU activation between the hidden layers.

I changed the training process to use shuffled mini-batches instead of training repeatedly on one fixed subset. The model now uses most of the MNIST training data, keeps 5,000 images for validation, and reports the average training loss and accuracy together with validation results after each epoch. After training, it also measures loss and accuracy on the separate test set.

The web application was updated to load all three trained layers and use the same architecture for predictions. The saved model file now includes the weights and biases of the additional layer, so the training script and web interface remain compatible.

The main challenge was making the larger network work consistently throughout the project. Adding a layer required changes to the forward pass, backward pass, parameter updates, saved model data, and prediction code. The training and application architectures must match exactly for the saved weights to load and produce valid predictions.

Durign next weeks a plan to focus mainly on testing

## Time Tracking

| Date | Time Spent | Description |
| ----- | ---------- | ----------- |
| 28.7. | 4 h | Expanding the neural network with an additional hidden layer |
| 29.7. | 3 h | Adding shuffled mini-batch training and validation and test evaluation |
| 30.7. | 2 h | Updating the web application and saved model format for the new architecture |
| **Total** | **9 h** | |
