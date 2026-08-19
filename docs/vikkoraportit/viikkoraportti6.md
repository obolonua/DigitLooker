# Week 6

During the sixth week, I completed the testing work by adding deterministic
unit tests for the individual building blocks of the hand-written neural
network. The tests verify the forward and backward calculations of a dense
layer, ReLU behavior, Softmax normalization and numerical stability, and the
categorical cross-entropy loss with both class indices and one-hot labels. I
also tested probability clipping and the gradient produced by the combined
Softmax and cross-entropy backward pass.

I changed the coverage configuration so that fully covered files are also
shown in the terminal report. The complete test suite now contains 10 tests,
and all of them pass. The current statement and branch coverage for the `mlp`
package is 100%, including the activation, layer, and loss modules.

I also cleaned up the project by removing unused example code and excessive
comments from the MLP modules. The obsolete character segmentation file was
deleted because the project now works with individual MNIST digit images and
no longer needs the earlier character-segmentation approach. Finally, I fixed
the misspelled filename of the third weekly report.

The main challenge was choosing small, deterministic inputs with expected
outputs that could be calculated by hand. This was especially important for
testing backpropagation, because the tests need to detect incorrect gradients
without depending on random training behavior.

The project now has both end-to-end learning tests and focused unit tests for
the core neural-network components. The testing goals are complete, and the
codebase has been cleaned of obsolete material.

## Time Tracking

| Date | Time Spent | Description |
| ----- | ---------- | ----------- |
| 14.8. | 1 h | Removing obsolete character-segmentation code and unnecessary example comments |
| 16.8. | 7 h | Adding deterministic unit tests for layers, activations, and loss functions |
| 17.8. | 1 h | Updating the coverage output and correcting the weekly-report filename |
| **Total** | **9 h** | |
