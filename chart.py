"""The 2D chart shared by recognize.py and simulate.py.

X axis = circularity, Y axis = number of corners. Circles are blue, squares are red. The black dashed
line is the decision boundary (z = 0) and the faint colored background shows on which side of it
the model says "circle" and on which "square". The right column holds the legend and the numbers.

The module does not depend on a window: it draws on the matplotlib Figure it is given.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

import model

COLORS = {"circle": "tab:blue", "square": "tab:red"}
X_COLUMN, Y_COLUMN = 0, 1   # columns of X: circularity, corners (the order of MODEL_FEATURES)
JITTER = 0.25   # corners are whole numbers, so points get a small vertical shift to not hide each other
AREA_ALPHA = 0.13   # strength of the colored background

# Texts shown to the user (front end, Polish)
FEATURE_LABELS = {"circularity": "Kolistość", "corners": "Liczba rogów"}
X_LABEL = "Kolistość  (1,0 = idealne koło, kwadrat ok. 0,8)"
Y_LABEL = "Liczba rogów  (kwadrat ok. 4, koło ok. 8)"
LEGEND_TITLE = "Legenda"


def jitter(count, seed=1):
    """Fixed vertical shifts, one per data row."""
    return np.random.default_rng(seed).uniform(-JITTER, JITTER, count)


def setup_axes(fig, X, extra_points=()):
    """Main chart on the left, a column for the legend and numbers on the right.
    X: all rows (2 features); extra_points: (circularity, corners) pairs that must fit as well.
    Returns (axes, side_axes)."""
    fig.clear()
    grid = fig.add_gridspec(1, 2, width_ratios=[4, 1.4])
    ax, side = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    side.axis("off")
    xs = list(X[:, X_COLUMN]) + [p[0] for p in extra_points] or [0.75, 1.0]
    ys = list(X[:, Y_COLUMN]) + [p[1] for p in extra_points] or [3, 9]
    x_margin = (max(xs) - min(xs)) * 0.05 or 0.02
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_autoscale_on(False)
    ax.set_xlabel(X_LABEL, fontsize=11)
    ax.set_ylabel(Y_LABEL, fontsize=11)
    ax.grid(alpha=0.3, zorder=0)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    return ax, side


def _clip(corners, z, sign):
    """The part of the polygon `corners` where sign * z > 0 (z is given at the corners)."""
    clipped = []
    for i, p in enumerate(corners):
        q = corners[(i + 1) % len(corners)]
        sp, sq = sign * z[i], sign * z[(i + 1) % len(corners)]
        if sp > 0:
            clipped.append(p)
        if (sp > 0) != (sq > 0):
            t = sp / (sp - sq)
            clipped.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return np.array(clipped) if clipped else np.full((1, 2), np.nan)


class Boundary:
    """The decision boundary: the dashed line and the colored background behind the points
    (one polygon for the side of circles, one for the side of squares)."""

    def __init__(self, ax):
        self.ax = ax
        self.circle_area = ax.fill([], [], color=COLORS["circle"], alpha=AREA_ALPHA, linewidth=0, zorder=0)[0]
        self.square_area = ax.fill([], [], color=COLORS["square"], alpha=AREA_ALPHA, linewidth=0, zorder=0)[0]
        (self.line,) = ax.plot([], [], color="black", linestyle="--", linewidth=2, zorder=4)
        self.artists = [self.circle_area, self.square_area, self.line]

    def line_points(self, weights, bias):
        """Two (x, y) arrays of the line z = 0 within the chart."""
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        vertical = model.circularity_on_line(weights, bias) if abs(weights[1]) < 1e-9 else None
        if vertical is not None:
            return np.array([vertical, vertical]), np.array(ylim)
        xs = np.array(xlim)
        return xs, model.line_y(weights, bias, xs)

    def update(self, weights, bias):
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        z = model.z_grid(weights, bias, np.array([c[0] for c in corners]), np.array([c[1] for c in corners]))
        self.square_area.set_xy(_clip(corners, z, 1))
        self.circle_area.set_xy(_clip(corners, z, -1))
        self.line.set_data(*self.line_points(weights, bias))


def legend_handles(extra=()):
    handles = [
        Line2D([], [], marker="o", linestyle="", color=COLORS["circle"], markersize=7, label="koło (dane)"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["square"], markersize=7, label="kwadrat (dane)"),
        Line2D([], [], color="black", linestyle="--", linewidth=2, label="granica (z = 0)"),
        Patch(facecolor=COLORS["circle"], alpha=0.25, label="tu model mówi: koło"),
        Patch(facecolor=COLORS["square"], alpha=0.25, label="tu model mówi: kwadrat"),
    ]
    return handles + list(extra)


def show_legend(side, extra=()):
    side.legend(handles=legend_handles(extra), loc="upper left", title=LEGEND_TITLE, fontsize=9,
                title_fontsize=10, frameon=True)


def weights_text(weights, bias):
    return (f"Wagi granicy:\n"
            f"  {FEATURE_LABELS['circularity'].lower()}: {weights[0]:+.2f}\n"
            f"  {FEATURE_LABELS['corners'].lower()}: {weights[1]:+.2f}\n"
            f"  stała: {bias:+.2f}\n"
            f"z > 0 → kwadrat, z ≤ 0 → koło")
