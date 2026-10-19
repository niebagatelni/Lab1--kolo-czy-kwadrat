"""Model: learns from data.csv and recognizes circle or square.

The model is a logistic regression on two features (circularity and number of corners), learned
one figure at a time (stochastic gradient descent). There is one training that never starts over:
every new learning continues from the weights the previous one ended with. Two files hold the result:
  weights.csv   - the whole history: after every single step the 3 numbers (2 weights and a bias)
                  are appended; simulate.py replays it
  boundary.json - the latest weights only, overwritten after each learning; recognize.py uses it

    z = w_circularity * (circularity - mean) / scale + w_corners * (corners - mean) / scale + bias

z > 0 -> square, z <= 0 -> circle. The means and scales are fixed constants (below), not taken
from the data, so a later row cannot change what an earlier boundary meant.

Command line usage: python model.py   (one pass over the whole data.csv, same as the button in train.py)
"""
import csv
import json
import math
import os

import numpy as np

import data_handler
from feature_extractor import MODEL_FEATURES

FOLDER = data_handler.FOLDER
BOUNDARY_PATH = os.path.join(FOLDER, "boundary.json")
WEIGHTS_PATH = os.path.join(FOLDER, "weights.csv")
WEIGHTS_HEADER = [f"w_{name}" for name in MODEL_FEATURES] + ["bias"]
NEGATIVE_CLASS, POSITIVE_CLASS = "circle", "square"

MIN_SAMPLES = 20      # minimal amount of data needed to train
MIN_PER_CLASS = 5     # and minimal number of examples of each shape
# (mean, scale) of each model feature: typical values of circles and squares, fixed on purpose
FEATURE_SCALE = {"circularity": (0.90, 0.05), "corners": (6.0, 2.0)}
LEARNING_RATE = 0.1
REGULARIZATION = 1e-3

BOUNDARY_KEYS = ("features", "weights", "bias", "negative_class", "positive_class")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _standardize(X):
    mean = np.array([FEATURE_SCALE[name][0] for name in MODEL_FEATURES])
    scale = np.array([FEATURE_SCALE[name][1] for name in MODEL_FEATURES])
    return (np.asarray(X, dtype=float) - mean) / scale


def train(X, y, start=None):
    """Learns from the rows one by one. X has the 2 model features per row.
    start: (weights, bias) to continue from; None = start from zeros. Returns a dict:
      boundary - ready to be saved as JSON
      history  - array (rows, 3): weights and bias after each row
      accuracy - share of the last 200 rows that the boundary from before that row got right
                 (an honest test: every row is judged by a model that has not seen it yet)
    """
    Z = _standardize(X)
    target = (np.asarray(y) == POSITIVE_CLASS).astype(float)
    weights = np.zeros(Z.shape[1]) if start is None else np.array(start[0], dtype=float)
    bias = 0.0 if start is None else float(start[1])
    history = np.empty((len(target), Z.shape[1] + 1))
    correct = np.empty(len(target), dtype=bool)
    for i in range(len(target)):
        z = Z[i] @ weights + bias
        correct[i] = (z > 0) == (target[i] > 0.5)
        error = _sigmoid(z) - target[i]
        weights = weights - LEARNING_RATE * (error * Z[i] + REGULARIZATION * weights)
        bias = bias - LEARNING_RATE * error
        history[i] = (*weights, bias)
    boundary = {"features": list(MODEL_FEATURES), "weights": [float(w) for w in weights], "bias": float(bias),
                "negative_class": NEGATIVE_CLASS, "positive_class": POSITIVE_CLASS}
    return {"boundary": boundary, "history": history,
            "accuracy": float(correct[-200:].mean()) if len(correct) else 0.0}


def score(boundary, X):
    """z for one row of features (or an array of rows); X has the 2 model features per row."""
    return _standardize(X) @ np.array(boundary["weights"]) + boundary["bias"]


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


def line_y(weights, bias, x):
    """Number of corners at which z = 0, for the given circularity values x (NaN if the line is vertical)."""
    x = np.asarray(x, dtype=float)
    if abs(weights[1]) < 1e-9:
        return np.full(x.shape, np.nan)
    (circ_mean, circ_scale), (corn_mean, corn_scale) = (FEATURE_SCALE[name] for name in MODEL_FEATURES)
    z_circ = (x - circ_mean) / circ_scale
    return corn_mean + corn_scale * (-(bias + weights[0] * z_circ) / weights[1])


def circularity_on_line(weights, bias):
    """Circularity at which z = 0 when the line is vertical (the corners weight is 0); None otherwise."""
    if abs(weights[0]) < 1e-9:
        return None
    (circ_mean, circ_scale), _ = (FEATURE_SCALE[name] for name in MODEL_FEATURES)
    return circ_mean + circ_scale * (-bias / weights[0])


def z_grid(weights, bias, xx, yy):
    """z over a grid of (circularity, corners) points."""
    (circ_mean, circ_scale), (corn_mean, corn_scale) = (FEATURE_SCALE[name] for name in MODEL_FEATURES)
    return (weights[0] * (xx - circ_mean) / circ_scale + weights[1] * (yy - corn_mean) / corn_scale + bias)


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
    if not all(key in boundary for key in BOUNDARY_KEYS) or boundary.get("features") != MODEL_FEATURES:
        return None
    return boundary


def enough_data(y):
    y = np.asarray(y)
    return len(y) >= MIN_SAMPLES and min((y == NEGATIVE_CLASS).sum(), (y == POSITIVE_CLASS).sum()) >= MIN_PER_CLASS


def append_weights(history, path=WEIGHTS_PATH):
    """Appends the weights after each step (array (steps, 3)) to the end of weights.csv."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(WEIGHTS_HEADER)
        writer.writerows([f"{x:.6f}" for x in row] for row in history)


def load_weights(path=WEIGHTS_PATH):
    """The whole history of the weights, array (steps, 3); empty if there is no file yet."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.empty((0, len(WEIGHTS_HEADER)))
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if header != WEIGHTS_HEADER:
        raise ValueError(f"{path} has columns {header}, expected {WEIGHTS_HEADER}")
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def learn(X, y, boundary_path=BOUNDARY_PATH, weights_path=WEIGHTS_PATH):
    """Learns from the given rows, continuing from the weights saved in boundary.json (zeros if there
    is none yet). Appends the weights of every step to weights.csv and overwrites boundary.json with
    the latest ones. Returns the result of train() with an added 'samples' key."""
    previous = load_boundary(boundary_path)
    result = train(X, y, None if previous is None else (previous["weights"], previous["bias"]))
    append_weights(result["history"], weights_path)
    save_boundary(result["boundary"], boundary_path)
    result["samples"] = len(y)
    return result


if __name__ == "__main__":
    X, y, _ = data_handler.load_data()
    if load_boundary() is None and not enough_data(y):
        raise SystemExit(f"Not enough data in {data_handler.DATA_PATH} "
                         f"(need at least {MIN_SAMPLES} samples, {MIN_PER_CLASS} of each shape). "
                         f"Run: python data_generator.py")
    result = learn(X, y)
    print(f"Samples: {result['samples']}, accuracy on the last 200 steps: {result['accuracy']:.1%}")
    print(f"Weights appended to: {WEIGHTS_PATH}; latest saved to: {BOUNDARY_PATH}")
