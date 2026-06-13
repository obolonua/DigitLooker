# Flask UI for generating synthetic digits and viewing model predictions.

from base64 import b64encode
from io import BytesIO
from pathlib import Path
import random

from flask import Flask, render_template, request
import numpy as np

from generator.render_digits import generate_digit_image
from mlp.activations import ReLU, Softmax
from mlp.layers import Layer


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "digit_mlp_weights.npz"

app = Flask(__name__)


def load_model():
    # Load the trained MLP weights from disk into fresh layer instances.
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing model weights at {MODEL_FILE}. Train the model first with `poetry run python train.py`."
        )

    weights = np.load(MODEL_FILE)

    layer1 = Layer(784, 128)
    layer2 = Layer(128, 10)

    layer1.weights = weights["layer1_weights"]
    layer1.biases = weights["layer1_biases"]
    layer2.weights = weights["layer2_weights"]
    layer2.biases = weights["layer2_biases"]

    return layer1, layer2


def predict(image_array):
    # Run a single digit image through the MLP and return prediction details.
    layer1, layer2 = load_model()
    relu = ReLU()
    softmax = Softmax()

    x = image_array.reshape(1, -1).astype(np.float32) / 255.0

    layer1.forward(x)
    relu.forward(layer1.output)
    layer2.forward(relu.output)
    probabilities = softmax.forward(layer2.output)[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return predicted_digit, confidence, probabilities


def image_to_data_uri(image):
    # Convert a PIL image into a data URI so it can be embedded in HTML.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/", methods=["GET", "POST"])
def index():
    # Render the landing page and optionally generate a fresh digit sample.
    result = None

    if request.method == "POST":
        target_digit = random.randint(0, 9)
        image = generate_digit_image(target_digit)
        predicted_digit, confidence, probabilities = predict(np.array(image))

        result = {
            "target_digit": target_digit,
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "is_correct": predicted_digit == target_digit,
            "image_data": image_to_data_uri(image),
            # "probabilities": [
            #     {"digit": digit, "value": float(probabilities[digit])}
            #     for digit in range(10)
            # ],
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
