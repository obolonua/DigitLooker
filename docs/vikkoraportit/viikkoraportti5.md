# Week 5

During the fifth week, I continued improving the automated testing of the
hand-written neural network. I added a third test that checks whether the
weights of every trainable layer change after each optimizer step. This helps
detect errors where gradients are calculated but one of the layers is
accidentally left out of the parameter update.

I also added the Coverage.py tool to the development dependencies and enabled
statement and branch coverage measurement for the `mlp` package. Instructions
for running the tests, printing the terminal report, and generating an HTML
coverage report were added to the README. All three automated tests currently
pass, and the latest measured coverage of the `mlp` package is 68%.

I created a separate testing document describing the testing approach, the
purpose of each automated test, the commands needed to run them, the current
coverage results, and the parts of the application that are not yet tested.

The main challenge was understanding the difference between running the
tests and collecting useful coverage information. The coverage report also
showed that passing end-to-end tests do not execute every alternative branch in
the individual MLP components.

Next week i plan to finish with testing.

## Time Tracking

| Date | Time Spent | Description |
| ----- | ---------- | ----------- |
| 10.8. | 4 h | Adding a test that verifies updates to every trainable layer |
| 11.8. | 2 h | Configuring Coverage.py and measuring statement and branch coverage |
| 11.8. | 1 h | Writing the testing document and test instructions |
| **Total** | **7 h** | |
