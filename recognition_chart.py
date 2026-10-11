"""Charts for the recognition tab: where the drawn figure lands among the training data.

Four charts (one per feature) show the training points in the background, the line where the model
separates the shapes in that feature (the other features at their training mean) and the drawn figure
as a star. The fifth chart shows how much each feature pushed the decision towards a circle or a square.
The sum of the pushes plus the bias is z; z > 0 means a square.

The module does not depend on the window: it draws on the matplotlib Figure it is given.
"""
import numpy as np
from matplotlib.figure import Figure

from feature_extractor import FEATURE_NAMES
from simulation import CLASS_LABELS, COLORS, FEATURE_LABELS

MAX_BACKGROUND_POINTS = 1500


def new_figure():
    return Figure(figsize=(8.4, 5.6), constrained_layout=True)


class RecognitionChart:
    def __init__(self, fig):
        self.fig = fig
        self.clear()

    def clear(self):
        self.fig.clear()
        self.fig.text(0.5, 0.5, "Narysuj figurę i kliknij „Rozpoznaj”.\nTutaj zobaczysz, gdzie wypadła "
                      "wśród danych uczących.", ha="center", va="center", fontsize=12, color="0.4")

    def show(self, boundary, features, X, y):
        """boundary: model boundary, features: 4 numbers of the drawn figure,
        X, y: training data used as the background."""
        fig = self.fig
        fig.clear()
        weights = np.array(boundary["weights"])
        mean, std = np.array(boundary["mean"]), np.array(boundary["std"])
        bias = boundary["bias"]
        values = np.asarray(features, dtype=float)
        pushes = weights * (values - mean) / std
        z = float(pushes.sum() + bias)

        rng = np.random.default_rng(1)
        if len(y) > MAX_BACKGROUND_POINTS:
            keep = rng.choice(len(y), MAX_BACKGROUND_POINTS, replace=False)
            X, y = X[keep], y[keep]
        bands = {"circle": (0.04, 0.46), "square": (0.54, 0.96)}
        vertical = {label: rng.uniform(*bands[label], int((y == label).sum())) for label in bands}

        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.95])
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        for i, (axis, name) in enumerate(zip(axes, FEATURE_NAMES)):
            low = min(X[:, i].min(), values[i]) if len(y) else values[i] - 1
            high = max(X[:, i].max(), values[i]) if len(y) else values[i] + 1
            margin = (high - low) * 0.06 or 0.5
            axis.set_xlim(low - margin, high + margin)
            axis.set_ylim(0, 1)
            axis.set_yticks([0.25, 0.75])
            axis.set_yticklabels([CLASS_LABELS["circle"], CLASS_LABELS["square"]])
            axis.set_xlabel(FEATURE_LABELS[name])
            axis.grid(alpha=0.3)
            for label in bands:
                chosen = X[y == label, i]
                axis.scatter(chosen, vertical[label][:len(chosen)], s=6, c=COLORS[label],
                             alpha=0.2, linewidths=0)
            if abs(weights[i]) > 1e-6:
                axis.axvline(mean[i] + std[i] * (-bias / weights[i]), color="black",
                             linestyle="--", linewidth=1.6)
            axis.scatter([values[i]], [0.5], marker="*", s=320, c="gold", edgecolors="black",
                         linewidths=1.2, zorder=5)
            vote = "kwadrat" if pushes[i] > 0 else "koło"
            axis.set_title(f"głos: {vote} ({pushes[i]:+.2f})", fontsize=10,
                           color=COLORS["square"] if pushes[i] > 0 else COLORS["circle"])
        fig.suptitle("★ = Twój rysunek", fontsize=10, color="0.25")

        axis = fig.add_subplot(grid[:, 2])
        names = [FEATURE_LABELS[n] for n in FEATURE_NAMES] + ["Stała", "Razem (z)"]
        amounts = list(pushes) + [bias, z]
        positions = list(range(len(names)))
        colors = [COLORS["square"] if a > 0 else COLORS["circle"] for a in amounts]
        axis.barh(positions, amounts, color=colors, height=0.55)
        reach = max(1.0, max(abs(a) for a in amounts)) * 1.5
        axis.set_xlim(-reach, reach)
        axis.set_ylim(-0.6, len(names) - 0.4)
        axis.invert_yaxis()
        axis.set_yticks(positions)
        axis.set_yticklabels(names)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.axhline(len(names) - 1.5, color="0.6", linewidth=0.8)
        axis.grid(axis="x", alpha=0.3)
        axis.set_xlabel("← koło        wpływ na decyzję        kwadrat →")
        for position, amount in zip(positions, amounts):
            axis.text(0.98, position, f"{amount:+.2f}", va="center", ha="right", fontsize=9,
                      transform=axis.get_yaxis_transform())
        verdict = "KWADRAT" if z > 0 else "KOŁO"
        axis.set_title(f"z = {z:+.2f}\n→ {verdict}", fontsize=11, fontweight="bold",
                       color=COLORS["square"] if z > 0 else COLORS["circle"])
