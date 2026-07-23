# DigitLooker

This is an assignment for the Algorithms and Artificial Intelligence course in the Computer Science bachelor’s program at the University of Helsinki.

## Usage

### Requirements

- Python 3.11

### Install dependencies

This project uses Poetry.

```bash
poetry install
```

### MNIST data

The training and demo code now load MNIST directly instead of the synthetic character dataset.


### Run the Flask UI

After training the model, start the web app with:

```bash
poetry run python -m app
```

Open `http://127.0.0.1:5000/` in your browser and click **Generate character** to create a new sample, see the image, and inspect the model prediction.

## Project Notes

- MNIST images are loaded as flattened 28x28 grayscale vectors normalized to `[0, 1]`.

## Dokumentaatio
