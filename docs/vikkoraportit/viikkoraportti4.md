# Week 4

During the fourth week, I focused on the Flask application and made it possible to test the digit recognition model through a simple web interface. The main goal was to connect the trained network to the browser so the application can generate a digit image and show the prediction result to the user.

Most of the work went into the route logic and model loading in `app.py`. I added code that loads the saved MLP weights from disk, creates fresh layer instances, and runs a generated digit through the network to produce a prediction and confidence value. I also added helpers for converting the generated image into a data URI so it can be embedded directly in the HTML page without saving temporary files.

The main challenge this week was keeping the UI logic simple while still making the prediction flow reliable. I had to make sure the image data was converted correctly before passing it into the model, and I also had to verify that the browser could display the generated image cleanly after encoding it. Another challenge was organizing the code so the model-loading and prediction steps stayed readable inside a small Flask application.

Next week, I want to continue improving the user interface and start working with real receipts. The plan is to segment the receipt into text lines, pass those lines into the network, and use the extracted predictions to recover useful information from the receipt.

## Time Tracking

| Date | Time Spent | Description |
| ----- | ----------- | ----------- |
| 6.6. | 4 h | Building the Flask route and model loading logic |
| 7.6. | 3 h | Connecting predictions to the browser UI |
| 8.6. | 2 h | Adding image encoding and testing the response flow |
| **Total** | **9 h** | |
