# Flask UI for generating synthetic characters and viewing model predictions.

from base64 import b64encode
from io import BytesIO
from pathlib import Path
import random

from flask import Flask, render_template, request
import numpy as np

from generator.render_digits import CHARACTERS, generate_character_image
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
    layer2 = Layer(128, 38)

    layer1.weights = weights["layer1_weights"]
    layer1.biases = weights["layer1_biases"]
    layer2.weights = weights["layer2_weights"]
    layer2.biases = weights["layer2_biases"]

    return layer1, layer2


def predict(image_array):
    # Run a single character image through the MLP and return prediction details.
    layer1, layer2 = load_model()
    relu = ReLU()
    softmax = Softmax()

    x = image_array.reshape(1, -1).astype(np.float32) / 255.0

    layer1.forward(x)
    relu.forward(layer1.output)
    layer2.forward(relu.output)
    probabilities = softmax.forward(layer2.output)[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return predicted_class, confidence, probabilities


def image_to_data_uri(image):
    # Convert a PIL image into a data URI so it can be embedded in HTML.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/", methods=["GET", "POST"])
def index():
    # Render the landing page and optionally generate a fresh character sample.
    result = None

    if request.method == "POST":
        target_class = random.randrange(len(CHARACTERS))
        target_character = CHARACTERS[target_class]
        image = generate_character_image(target_character)
        predicted_class, confidence, probabilities = predict(np.array(image))
        predicted_character = CHARACTERS[predicted_class]

        result = {
            "target_character": target_character,
            "predicted_character": predicted_character,
            "confidence": confidence,
            "is_correct": predicted_class == target_class,
            "image_data": image_to_data_uri(image),
            # "probabilities": [
            #     {"class": idx, "value": float(probabilities[idx])}
            #     for idx in range(len(CHARACTERS))
            # ],
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)

# poetry run python -m app
