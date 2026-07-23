
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def load_mnist():
    """
    Load MNIST as flattened float32 images in [0, 1].
    """
    from sklearn.datasets import fetch_openml

    mnist_data = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist_data.data.astype(np.float32)
    y = mnist_data.target.astype(np.int64)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    if X_train.ndim == 3:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return X_train, y_train, X_test, y_test
