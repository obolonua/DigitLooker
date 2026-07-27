# Week 2

During the second week, I replaced the project's synthetic character dataset with the MNIST handwritten digit dataset. I added a loader that downloads MNIST through scikit-learn, converts the labels to integers, normalizes the image values to the range `[0, 1]`, and divides the data into training and test sets.

I updated the neural network and training pipeline for the new dataset. The model now has 10 output classes instead of the previous 38 character classes, and training uses subsets of the MNIST training and test data to keep experiments reasonably fast. I also adjusted the learning rate and number of training epochs.

The web interface was changed to select and display a random MNIST image. It now shows the target digit, the model's prediction, its confidence, and the probabilities for all 10 digit classes. The old font-based dataset generator, its font files, and the related instructions were removed because they are no longer needed.

The main challenge was changing every part of the application consistently from synthetic character recognition to MNIST digit recognition. The data format, model output size, training process, prediction view, dependencies, and documentation all had to agree with the new 10-class problem.

Next week, I plan to train and test the updated model, measure its prediction accuracy, and fix any issues revealed by evaluating it with MNIST samples.

## Time Tracking

| Date | Time Spent | Description |
| ----- | ---------- | ----------- |
| 23.7. | 1 h | Adding the MNIST loader and preparing normalized training and test data |
| 23.7. | 1 h | Updating the network architecture and training pipeline for 10 digit classes |
| 23.7. | 1 h | Updating the web interface, dependencies, and documentation and removing the synthetic generator |
| **Total** | **3 h** | |
