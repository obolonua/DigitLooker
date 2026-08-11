# Testing Document

## Testing approach

The neural network is tested with automated `unittest` tests. The tests use
small samples from scikit-learn's digits dataset and fixed NumPy random seeds.
This keeps the tests fast and makes their results reproducible.

The test suite contains three tests:

- `network_overfitting.py` verifies that the network can learn and completely
  memorize a small dataset containing examples of digits zero and one. The test
  checks that the final loss is below `0.01` and the training accuracy is 100%.
- `loss_decreases.py` verifies that gradients are finite and non-zero and that
  repeated minibatch updates reduce the loss. In the latest test run, the loss
  decreased from approximately `2.516` to `0.193`.
- `all_layers_change.py` verifies that the weights of every trainable layer
  change after each optimizer step.

Together, these tests exercise the forward pass, backpropagation, loss
calculation, and parameter updates of the hand-written MLP implementation.

## Running the tests

Install the dependencies and run the tests with coverage measurement:

```bash
poetry install
poetry run coverage run -m unittest discover -s tests -p "*.py"
```

The first command installs the project dependencies. The second command finds
the tests in the `tests` directory, runs them, and saves the collected coverage
data in the `.coverage` file.

Print the coverage report in the terminal with:

```bash
poetry run coverage report
```

An HTML report can also be generated with:

```bash
poetry run coverage html
```

The generated report can be opened from `htmlcov/index.html`.

## Test coverage

The latest coverage report was generated on 11 August 2026. All three tests
passed.

| File | Statements | Missing | Branches | Partially covered branches | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlp/activations.py` | 16 | 1 | 2 | 1 | 89% |
| `mlp/layers.py` | 22 | 8 | 2 | 1 | 62% |
| `mlp/losses.py` | 26 | 10 | 4 | 2 | 60% |
| **Total** | **64** | **19** | **8** | **4** | **68%** |

Branch coverage is enabled. The report measures the `mlp` package configured in
`pyproject.toml`.