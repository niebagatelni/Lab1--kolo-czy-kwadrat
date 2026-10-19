"""Data handling: appending rows to data.csv and loading them back.

data.csv: one row = one figure (2 features, label, source). Data is only ever appended.
The weights of the model are not kept here (see weights.csv and boundary.json in model.py).
"""
import csv
import os

import numpy as np

from feature_extractor import MODEL_FEATURES

FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FOLDER, "data.csv")
HEADER = MODEL_FEATURES + ["label", "source"]


class IncompatibleData(ValueError):
    """The CSV file was made with a different set of columns than the current one."""


def _check_header(path):
    """Raises IncompatibleData if the file exists, is not empty and has a different header."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header != HEADER:
        raise IncompatibleData(f"{path} has columns {header}, expected {HEADER}")


def append_data(rows, path=DATA_PATH):
    """rows: list of tuples (feature1, feature2, label, source). The header is written only to a new file."""
    _check_header(path)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(HEADER)
        for *features, label, source in rows:
            writer.writerow([f"{x:.5f}" for x in features] + [label, source])


def load_data(path=DATA_PATH):
    """Returns (X, y, sources); empty arrays if the file does not exist."""
    if not os.path.exists(path):
        return np.empty((0, len(MODEL_FEATURES))), np.empty(0, dtype=str), np.empty(0, dtype=str)
    _check_header(path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[name]) for name in MODEL_FEATURES] for r in rows]).reshape(-1, len(MODEL_FEATURES))
    y = np.array([r["label"] for r in rows], dtype=str)
    sources = np.array([r["source"] for r in rows], dtype=str)
    return X, y, sources
