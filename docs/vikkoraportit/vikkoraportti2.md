# Week 2

During the second week, I focused more on the practical implementation of the project. I started building the synthetic dataset generator and worked on creating training data that can be used later for the OCR model.

Most of the work went into preparing the image generation pipeline. I used Pillow, OpenCV, and NumPy to create digit images with different fonts and small variations in appearance. The goal was to make the dataset more realistic by adding rotation, blur, noise, and other distortions that can happen in receipt images.

I also spent time testing different preprocessing steps and checking how the generated images look before saving them into the dataset. This helped me understand which kinds of augmentations are useful and which ones make the images too hard to read. In addition, I continued organizing the project structure so the data generation process is easier to reuse later.

A challenge this week was balancing realism and simplicity. The generated images need enough variation to help the model generalize, but they still need to remain clear enough for training. Another challenge was making the generator stable so it can produce a large number of samples without errors.

Next week, I want to continue improving the dataset generation process and start moving toward the actual OCR model. The libraries we are using so far should be fine for the dataset and preprocessing work, but for the MLP itself I will implement it from scratch and use only NumPy for matrix operations. If the dataset pipeline is stable enough, I will start implementing the MLP. The first step is to teach the model to understand digits and letters, and later we will start working with receipts and parsing data from them. After that, I can begin experimenting with training and evaluating a first simple prototype.

## Time Tracking

| Date | Time Spent | Description |
| ----- | ----------- | ----------- |
| 23.5. | 3 h | Implementing the synthetic digit generator and testing image creation |
| 23.5. | 3 h | Adding data augmentation such as rotation, blur, and noise |
| 23.5. | 1 h | Reviewing generated samples and adjusting the project structure |
| **Total** | **5 h** | |
