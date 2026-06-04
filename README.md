# ReceiptLooker

This is an assignment for the Algorithms and Artificial Intelligence course in the Computer Science bachelor’s program at the University of Helsinki.

## Usage

### Requirements

- Python 3.11
- Font files in `data/fonts/` with `.ttf` or `.ttc` extensions

### Install dependencies

This project uses Poetry.

```bash
poetry install
```

### Preview synthetic digits

Generate and display a sample grid of synthetic digit images:

```bash
poetry run python generator/render_digits.py
```

### Generate the dataset

Run the dataset generator to create the NumPy arrays in `data/generated/`:

```bash
poetry run python generator/create_dataset.py
```

The script writes:

- `data/generated/X_train.npy`
- `data/generated/X_val.npy`
- `data/generated/y_train.npy`
- `data/generated/y_val.npy`

### Run tests

Run the unit test suite with Python's built-in test runner:

```bash
poetry run python -m unittest discover -s tests
```

### Test model accuracy

After training and saving the model weights, you can estimate accuracy on fresh synthetic digits with:

```bash
poetry run python -m tests.predict_digit
```

The script expects the trained weights at `models/digit_mlp_weights.npz`, which is created by `poetry run python train.py`.

### Run the Flask UI

After training the model, start the web app with:

```bash
poetry run python app.py
```

Open `http://127.0.0.1:5000/` in your browser and click **Generate number** to create a new digit, see the generated image, and inspect the model prediction.

## Project Notes

- The generators expect at least one font file in `data/fonts/`.
- The dataset generator creates 28x28 grayscale digit images and splits them into training and validation sets.

## Dokumentaatio
