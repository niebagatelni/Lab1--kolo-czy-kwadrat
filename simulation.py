"""Training simulation: replays, step by step, how the model learned on the whole data set.

Four charts (one per feature) show the training points as they arrive (circles at the bottom,
squares at the top) and the line where the model separates the shapes in that feature.
The fifth chart shows the feature weights. The line on a feature chart is the place where z = 0
when the other features sit at the mean value of the points seen so far. Weights are computed on
standardized features, so they can be compared with each other.

The module does not depend on the window: it draws on the matplotlib Figure it is given,
and the app calls the frames.
"""
import math

import numpy as np
from matplotlib.figure import Figure

import model
from feature_extractor import FEATURE_NAMES

COLORS = {"circle": "tab:blue", "square": "tab:red"}
FRAMES = 150

# Texts shown to the user (front end, Polish)
FEATURE_LABELS = {
    "norm_radius_cv": "Falowanie promienia (CV)",
    "norm_radius_spread": "Rozpiętość promienia",
    "circularity": "Kolistość",
    "corners": "Liczba rogów",
}
CLASS_LABELS = {"circle": "koło", "square": "kwadrat"}


def new_figure():
    return Figure(figsize=(11.5, 6.6), constrained_layout=True)


def _frame_indices(step_count, frames):
    """Which training steps to show. Denser at the beginning, where the model changes fastest."""
    t = np.linspace(0, 1, min(frames, step_count))
    return np.unique(np.round((step_count - 1) * t ** 1.7).astype(int))


class Simulation:
    def __init__(self, fig, result, frames=FRAMES):
        self.fig = fig
        self.history = result["history"]
        self.boundary = result["boundary"]
        self.X = result["X_train"]
        self.y = result["y_train"]
        self.n = len(self.y)
        self.frame_steps = _frame_indices(len(self.history), frames)
        self.frame_count = len(self.frame_steps)
        self.epochs = model.EPOCHS
        self.steps_per_epoch = math.ceil(self.n / model.BATCH_SIZE)

        mean = np.array(self.boundary["mean"])
        std = np.array(self.boundary["std"])
        self.mean, self.std = mean, std
        Z = (self.X - mean) / std
        self.cumulative_z = np.vstack([np.zeros(Z.shape[1]), np.cumsum(Z, axis=0)])  # partial sums

        # points of each class in the order the model saw them + fixed vertical scatter
        rng = np.random.default_rng(1)
        self.classes = {}
        for label, (low, high) in (("circle", (0.04, 0.46)), ("square", (0.54, 0.96))):
            positions = np.flatnonzero(self.y == label)
            self.classes[label] = {"positions": positions, "X": self.X[positions],
                                   "vertical": rng.uniform(low, high, len(positions))}

        self._build()

    def _build(self):
        fig = self.fig
        fig.clear()
        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.85])
        self.axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                     fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        self.weights_axis = fig.add_subplot(grid[:, 2])

        self.points, self.lines = [], []
        for i, (axis, name) in enumerate(zip(self.axes, FEATURE_NAMES)):
            low, high = self.X[:, i].min(), self.X[:, i].max()
            margin = (high - low) * 0.06 or 0.5
            axis.set_xlim(low - margin, high + margin)
            axis.set_ylim(0, 1)
            axis.set_yticks([0.25, 0.75])
            axis.set_yticklabels([CLASS_LABELS["circle"], CLASS_LABELS["square"]])
            axis.set_xlabel(FEATURE_LABELS[name])
            axis.grid(alpha=0.3)
            scatters = {label: axis.scatter([], [], s=7, c=COLORS[label], alpha=0.45, linewidths=0)
                        for label in COLORS}
            self.points.append(scatters)
            self.lines.append(axis.axvline(np.nan, color="black", linestyle="--", linewidth=1.8))

        max_weight = max(np.abs(h[0]).max() for h in self.history) or 1.0
        self.weights_axis.set_xlim(-max_weight * 1.6, max_weight * 1.6)
        self.weights_axis.set_ylim(-0.6, len(FEATURE_NAMES) - 0.4)
        self.weights_axis.invert_yaxis()
        self.weights_axis.set_yticks(range(len(FEATURE_NAMES)))
        self.weights_axis.set_yticklabels([FEATURE_LABELS[name] for name in FEATURE_NAMES])
        self.weights_axis.axvline(0, color="black", linewidth=0.8)
        self.weights_axis.set_xlabel("← koło        waga        kwadrat →")
        self.weights_axis.set_title("Wagi cech", fontsize=11)
        self.weights_axis.grid(axis="x", alpha=0.3)
        self.bars = self.weights_axis.barh(range(len(FEATURE_NAMES)), np.zeros(len(FEATURE_NAMES)), height=0.55)
        # weight values in a fixed column at the right edge (x in axes units, y in data units)
        self.weight_texts = [self.weights_axis.text(0.98, i, "", va="center", ha="right", fontsize=9,
                                                    transform=self.weights_axis.get_yaxis_transform())
                             for i in range(len(FEATURE_NAMES))]
        fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.965))  # room for the footnote
        self.title = fig.suptitle("", fontsize=12)
        fig.text(0.01, 0.005, "Linia przerywana: granica w danej cesze przy średnich wartościach "
                 "pozostałych cech.", fontsize=8, color="0.35")

    def draw_frame(self, k):
        """Draws frame k (0 .. frame_count - 1)."""
        step = int(self.frame_steps[min(max(k, 0), self.frame_count - 1)])
        weights, bias, seen, accuracy = self.history[step]
        visible = min(seen, self.n)

        for label, data in self.classes.items():
            count = int(np.searchsorted(data["positions"], visible))
            for i in range(len(FEATURE_NAMES)):
                xy = np.c_[data["X"][:count, i], data["vertical"][:count]] if count else np.empty((0, 2))
                self.points[i][label].set_offsets(xy)

        z_mean = self.cumulative_z[visible] / visible if visible else np.zeros(len(FEATURE_NAMES))
        for i in range(len(FEATURE_NAMES)):
            if abs(weights[i]) > 1e-6:
                rest = bias + float(np.dot(weights, z_mean) - weights[i] * z_mean[i])
                x = self.mean[i] + self.std[i] * (-rest / weights[i])
                self.lines[i].set_xdata([x, x])
            else:
                self.lines[i].set_xdata([np.nan, np.nan])
            self.axes[i].set_title(f"waga {weights[i]:+.2f}", fontsize=10)

        for i, (bar, text) in enumerate(zip(self.bars, self.weight_texts)):
            bar.set_width(weights[i])
            bar.set_color(COLORS["square"] if weights[i] > 0 else COLORS["circle"])
            text.set_text(f"{weights[i]:+.2f}")

        epoch = min(self.epochs, (step - 1) // self.steps_per_epoch + 1) if step else 0
        self.title.set_text(f"Krok {step}/{len(self.history) - 1}  ·  zobaczone punkty: {visible}/{self.n}"
                            f"  ·  epoka {epoch}/{self.epochs}  ·  dokładność na teście: {accuracy:.1%}")
