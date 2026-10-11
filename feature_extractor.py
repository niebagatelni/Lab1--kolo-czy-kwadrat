"""Feature extraction: a drawing (outline) -> 4 numbers describing the figure.

All features are independent of the figure's size and rotation:
  norm_radius_cv      - how much the distance from the outline pixels to the center varies, measured
                        after "straightening the ellipse" (see _normalized_radius), so an oval circle
                        is not mistaken for a square
  norm_radius_spread  - (p95 - p5) / p95 of those distances (same idea, robust to single stray pixels)
  circularity         - 4*pi*area / perimeter^2 of the convex hull (circle ~1.0, square ~0.78)
  corners             - number of vertices of the polygon the hull simplifies to (square ~4, circle more)
"""
import numpy as np

LINE_WIDTH = 3           # outline thickness (px), used when drawing and in the generator
MIN_SIDE = 15            # minimal figure size (px), smaller drawings are rejected
CORNER_TOLERANCE = 0.02  # hull simplification tolerance, as a fraction of its perimeter
FEATURE_NAMES = ["norm_radius_cv", "norm_radius_spread", "circularity", "corners"]


class EmptyDrawing(ValueError):
    """Nothing was drawn."""


class FigureTooSmall(ValueError):
    """The figure is smaller than MIN_SIDE."""


class DegenerateFigure(ValueError):
    """The figure has zero area or zero perimeter."""


def _normalized_radius(xs, ys):
    """Distances of the outline pixels from their center after "straightening the ellipse":
    the points are rescaled along their principal axes so that their spread is equal in every
    direction. An ellipse (an oval circle) becomes a circle, while a square stays a square,
    so oval hand-drawn circles no longer look like squares."""
    points = np.c_[xs - xs.mean(), ys - ys.mean()].astype(float)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    straight = points @ vectors / np.sqrt(np.maximum(values, 1e-9))
    return np.hypot(straight[:, 0], straight[:, 1])


def _convex_hull(xs, ys):
    """Convex hull (monotone chain) of the points (xs, ys). Returns an (n, 2) array."""
    order = np.lexsort((xs, ys))
    ys_sorted, xs_sorted = ys[order], xs[order]
    first = np.r_[0, np.flatnonzero(np.diff(ys_sorted)) + 1]
    last = np.r_[first[1:], len(ys_sorted)] - 1
    # only the outermost pixels of each row can be hull vertices
    candidates = np.concatenate([np.c_[xs_sorted[first], ys_sorted[first]],
                                 np.c_[xs_sorted[last], ys_sorted[last]]])
    points = np.unique(candidates, axis=0)
    if len(points) < 3:
        return points

    def half_hull(pts):
        hull = []
        for q in pts:
            while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (q[1] - hull[-2][1])
                                      - (hull[-1][1] - hull[-2][1]) * (q[0] - hull[-2][0])) <= 0:
                hull.pop()
            hull.append(q)
        return hull

    lower, upper = half_hull(points), half_hull(points[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _simplify_polyline(points, tolerance):
    """Ramer-Douglas-Peucker simplification of an open polyline, points is (n, 2)."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    direction = end - start
    length = np.hypot(*direction)
    if length == 0:
        distance = np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
    else:
        distance = np.abs(direction[0] * (points[:, 1] - start[1])
                          - direction[1] * (points[:, 0] - start[0])) / length
    k = int(distance.argmax())
    if distance[k] > tolerance:
        return np.vstack([_simplify_polyline(points[:k + 1], tolerance)[:-1],
                          _simplify_polyline(points[k:], tolerance)])
    return np.vstack([start, end])


def _count_corners(hull, tolerance):
    """Number of vertices of the closed polygon `hull` after simplification with `tolerance`."""
    if len(hull) <= 3:
        return len(hull)
    d = np.hypot(hull[:, None, 0] - hull[None, :, 0], hull[:, None, 1] - hull[None, :, 1])
    i, j = np.unravel_index(d.argmax(), d.shape)  # cut at the two farthest points
    i, j = min(i, j), max(i, j)
    first = _simplify_polyline(hull[i:j + 1], tolerance)
    second = _simplify_polyline(np.vstack([hull[j:], hull[:i + 1]]), tolerance)
    return len(first) + len(second) - 2


def image_to_features(outline):
    """outline: 2D boolean array, True = pixel of the (already closed) outline.
    Returns a tuple of 4 numbers in the order of FEATURE_NAMES."""
    ys, xs = np.nonzero(outline)
    if len(ys) == 0:
        raise EmptyDrawing("Empty drawing.")
    if ys.max() - ys.min() + 1 < MIN_SIDE or xs.max() - xs.min() + 1 < MIN_SIDE:
        raise FigureTooSmall("Figure is too small.")

    radius = _normalized_radius(xs, ys)
    norm_radius_cv = radius.std() / radius.mean()
    p5, p95 = np.percentile(radius, [5, 95])
    norm_radius_spread = (p95 - p5) / p95

    hull = _convex_hull(xs, ys)
    next_vertex = np.roll(hull, -1, axis=0)
    area = abs(np.sum(hull[:, 0] * next_vertex[:, 1] - next_vertex[:, 0] * hull[:, 1])) / 2
    perimeter = np.hypot(*(next_vertex - hull).T).sum()
    if area == 0 or perimeter == 0:
        raise DegenerateFigure("Degenerate figure.")
    circularity = 4 * np.pi * area / perimeter ** 2
    corners = _count_corners(hull, CORNER_TOLERANCE * perimeter)

    return float(norm_radius_cv), float(norm_radius_spread), float(circularity), float(corners)
