"""Model: learns from data.csv, remembers the result in boundary.json and recognizes circle or square.

The model is a logistic regression trained in small steps (mini-batch gradient descent), written in
plain numpy. Every feature gets a weight: z = sum(weight * standardized_feature) + bias.
z > 0 -> square, z <= 0 -> circle. The same training run is recorded in "history" and replayed
in the animation, so the simulation shows real learning, not a smooth move "from point A to B".

Command line usage: python model.py
"""
import json
import math
import os

import numpy as np

import data_handler
from feature_extractor import FEATURE_NAMES

FOLDER = data_handler.FOLDER
BOUNDARY_PATH = os.path.join(FOLDER, "boundary.json")
NEGATIVE_CLASS, POSITIVE_CLASS = "circle", "square"

MIN_SAMPLES = 20      # minimal amount of data needed to train
MIN_PER_CLASS = 5     # and minimal number of examples of each shape
TEST_FRACTION = 0.2   # share of data held out for an honest test (the model never sees it)
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.05
REGULARIZATION = 1e-3

BOUNDARY_KEYS = ("weights", "bias", "mean", "std", "negative_class", "positive_class")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train(X, y, epochs=EPOCHS, seed=42):
    """Trains the model in small steps. Returns a dict:
      boundary         - ready to be saved as JSON (weights, bias, feature means and stds, class names)
      history          - list of (weights, bias, seen, test_accuracy) after every training step
      X_train, y_train - training data in the order the model saw it
      X_test, y_test   - data held out for testing
      accuracy         - accuracy of the final model on the test data
    """
    rng = np.random.default_rng(seed)
    target = (np.asarray(y) == POSITIVE_CLASS).astype(float)

    # train/test split, done separately in each class so that proportions match
    is_test = np.zeros(len(target), dtype=bool)
    for cls in (0.0, 1.0):
        idx = rng.permutation(np.flatnonzero(target == cls))
        is_test[idx[:max(1, int(len(idx) * TEST_FRACTION))]] = True
    train_idx = rng.permutation(np.flatnonzero(~is_test))
    test_idx = np.flatnonzero(is_test)

    X_train, target_train = X[train_idx], target[train_idx]
    X_test, target_test = X[test_idx], target[test_idx]
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    Z_train = (X_train - mean) / std
    Z_test = (X_test - mean) / std

    weights = np.zeros(Z_train.shape[1])
    bias = 0.0
    history = [(weights.copy(), bias, 0, float(np.mean((Z_test @ weights + bias > 0) == (target_test > 0.5))))]
    seen = 0
    for epoch in range(epochs):
        order = np.arange(len(target_train)) if epoch == 0 else rng.permutation(len(target_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            error = _sigmoid(Z_train[batch] @ weights + bias) - target_train[batch]
            weights = weights - LEARNING_RATE * (Z_train[batch].T @ error / len(batch) + REGULARIZATION * weights)
            bias = bias - LEARNING_RATE * error.mean()
            seen += len(batch)
            history.append((weights.copy(), float(bias), seen,
                            float(np.mean((Z_test @ weights + bias > 0) == (target_test > 0.5)))))

    boundary = {"features": list(FEATURE_NAMES), "weights": [float(x) for x in weights], "bias": float(bias),
                "mean": [float(x) for x in mean], "std": [float(x) for x in std],
                "negative_class": NEGATIVE_CLASS, "positive_class": POSITIVE_CLASS}
    return {"boundary": boundary, "history": history,
            "X_train": X_train, "y_train": np.asarray(y)[train_idx],
            "X_test": X_test, "y_test": np.asarray(y)[test_idx],
            "accuracy": history[-1][3]}


def score(boundary, X):
    """z for one row of features (or an array of rows)."""
    Z = (np.asarray(X, dtype=float) - boundary["mean"]) / boundary["std"]
    return Z @ np.array(boundary["weights"]) + boundary["bias"]


def predict(boundary, X):
    return np.where(score(boundary, X) > 0, boundary["positive_class"], boundary["negative_class"])


def accuracy(boundary, X, y):
    return float(np.mean(predict(boundary, X) == np.asarray(y)))


def classify(boundary, features):
    """Returns (label, confidence, z). Confidence = sigmoid of z (the farther from the boundary, the surer)."""
    z = float(score(boundary, features))
    p = 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))
    if z > 0:
        return boundary["positive_class"], p, z
    return boundary["negative_class"], 1 - p, z


def save_boundary(boundary, path=BOUNDARY_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(boundary, f, indent=2, ensure_ascii=False)


def load_boundary(path=BOUNDARY_PATH):
    """Returns the boundary from the file, or None if the file is missing, corrupted
    or was trained on a different set of features."""
    try:
        with open(path, encoding="utf-8") as f:
            boundary = json.load(f)
    except (OSError, ValueError):
        return None
    if not all(key in boundary for key in BOUNDARY_KEYS) or boundary.get("features") != FEATURE_NAMES:
        return None
    return boundary


def enough_data(y):
    y = np.asarray(y)
    return len(y) >= MIN_SAMPLES and min((y == NEGATIVE_CLASS).sum(), (y == POSITIVE_CLASS).sum()) >= MIN_PER_CLASS


def train_and_save(data_path=data_handler.DATA_PATH, boundary_path=BOUNDARY_PATH):
    """Trains on the whole CSV and saves the boundary. Returns the result of train() with an added
    'samples' key, or None when there is not enough data."""
    X, y, _ = data_handler.load_data(data_path)
    if not enough_data(y):
        return None
    result = train(X, y)
    save_boundary(result["boundary"], boundary_path)
    result["samples"] = len(y)
    return result


if __name__ == "__main__":
    result = train_and_save()
    if result is None:
        raise SystemExit(f"Not enough data in {data_handler.DATA_PATH} "
                         f"(need at least {MIN_SAMPLES} samples, {MIN_PER_CLASS} of each shape). "
                         f"Run: python data_generator.py")
    print(f"Samples: {result['samples']}, accuracy on test data: {result['accuracy']:.1%}")
    print(f"Boundary saved to: {BOUNDARY_PATH}")
