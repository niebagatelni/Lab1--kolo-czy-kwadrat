"""The computer draws circles and squares by itself (in memory, no image files are saved),
computes their features and appends the rows to data.csv. Nothing is ever deleted.

The shapes imitate drawing with a mouse. Circles are ovals drawn with a shaky, uneven line;
squares have fairly straight sides and corners that may be rounded.

Command line usage: python data_generator.py [count_per_shape]   (default 1000)
"""
import math
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

import data_handler
from feature_extractor import LINE_WIDTH, image_to_features

CANVAS_SIZE = 300  # same as the drawing canvas in the app
CIRCLE_OVALNESS = (0.62, 1.0)  # short axis / long axis of the hand-drawn "circle" (1.0 = round)
SLOPPY_SHARE = 0.3             # share of circles drawn sloppily (strongly wobbling line)
SLOPPY_FACTOR = (1.8, 2.6)     # how much stronger the wobble of a sloppy circle is
SQUARE_ASPECT = (0.82, 1.0)    # short side / long side of the hand-drawn "square"
SQUARE_MAX_ROUNDING = 0.12     # how much the corners may be rounded (share of the side length)


def _jitter(point, strength):
    return (point[0] + random.uniform(-strength, strength), point[1] + random.uniform(-strength, strength))


def _stroke_ends(points):
    """A hand-drawn stroke rarely ends exactly where it started: it either stops short (gap)
    or overshoots (overlap). The app then closes it with a straight line."""
    factor = random.uniform(0.93, 1.07)
    return (points * 2)[:max(3, int(len(points) * factor))]


def circle_points():
    """A hand-drawn circle: an oval (ellipse) whose line is uneven and asymmetric, as if the hand
    were shaking: slow distortions, a local bump, uneven drawing speed, trembling radius and
    an outline that does not close exactly."""
    rx = random.uniform(20, 140)
    ry = rx * random.uniform(*CIRCLE_OVALNESS)  # ellipse; the axis direction is random (see tilt)
    cx = random.uniform(rx * 1.4 + 2, CANVAS_SIZE - rx * 1.4 - 2)
    cy = random.uniform(rx * 1.4 + 2, CANVAS_SIZE - rx * 1.4 - 2)
    n = random.randint(50, 120)
    start = random.uniform(0, 2 * math.pi)
    turn = 2 * math.pi + random.uniform(-0.5, 0.5)  # too short = gap, too long = overlap
    drift = random.uniform(-0.08, 0.08)             # spiral: the end does not meet the start
    tilt = random.uniform(0, math.pi)               # rotation of the oval's axes
    # some circles are drawn sloppily: the line wobbles a lot, yet the shape stays roughly round
    sloppiness = random.uniform(*SLOPPY_FACTOR) if random.random() < SLOPPY_SHARE else 1.0
    harmonics = [(k, random.uniform(0, 2 * math.pi), random.uniform(0.0, amplitude * sloppiness))
                 for k, amplitude in ((1, 0.06), (2, 0.07), (3, 0.07), (4, 0.04), (5, 0.03))]
    bump_center = random.uniform(0, 2 * math.pi)    # one flatter or pointier place on the line
    bump_width = random.uniform(0.4, 0.9)
    bump_height = random.uniform(-0.12, 0.12) * sloppiness
    speed_warp = random.uniform(0.0, 0.8)           # the hand speeds up and slows down
    speed_phase = random.uniform(0, 2 * math.pi)
    tremble_strength = random.uniform(0.0, 0.012) * sloppiness  # correlated trembling of the radius
    tremble = 0.0
    points = []
    for i in range(n):
        t = i / n
        t_uneven = t + speed_warp * math.sin(2 * math.pi * t + speed_phase) / (2 * math.pi)
        angle = start + turn * t_uneven
        tremble = 0.85 * tremble + random.gauss(0, tremble_strength)
        distance = (angle - bump_center + math.pi) % (2 * math.pi) - math.pi
        scale = (1 + drift * t + tremble
                 + sum(a * math.sin(k * angle + phase) for k, phase, a in harmonics)
                 + bump_height * math.exp(-distance ** 2 / (2 * bump_width ** 2)))
        x, y = rx * scale * math.cos(angle), ry * scale * math.sin(angle)
        points.append((cx + x * math.cos(tilt) - y * math.sin(tilt),
                       cy + x * math.sin(tilt) + y * math.cos(tilt)))
    return [_jitter(p, random.uniform(0.3, 1.5)) for p in points]


def square_points():
    """A hand-drawn square: fairly straight sides, slightly shifted corners that may be rounded,
    a bit of hand tremble and an outline that does not close exactly."""
    a = random.uniform(40, 220)
    b = a * random.uniform(*SQUARE_ASPECT)
    angle = random.uniform(-0.9, 0.9)
    corners = [(-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2)]
    shift = 0.07 * min(a, b)
    corners = [(x + random.uniform(-shift, shift), y + random.uniform(-shift, shift)) for x, y in corners]
    reach = max(abs(c) for p in corners for c in p) * 1.5
    cx = random.uniform(reach + 2, CANVAS_SIZE - reach - 2)
    cy = random.uniform(reach + 2, CANVAS_SIZE - reach - 2)
    corners = [(cx + x * math.cos(angle) - y * math.sin(angle),
                cy + x * math.sin(angle) + y * math.cos(angle)) for x, y in corners]

    # rounding: every corner becomes two points closer to the middles of the neighboring sides
    rounding = SQUARE_MAX_ROUNDING * random.random() ** 2  # mostly slight, sometimes clearly rounded
    outline = []
    for i in range(4):
        p, previous, following = corners[i], corners[i - 1], corners[(i + 1) % 4]
        outline.append((p[0] + (previous[0] - p[0]) * rounding, p[1] + (previous[1] - p[1]) * rounding))
        outline.append((p[0] + (following[0] - p[0]) * rounding, p[1] + (following[1] - p[1]) * rounding))

    strength = random.uniform(0.4, 1.8)
    bow = [random.uniform(-0.015, 0.015) for _ in range(len(outline))]  # sides stay almost straight
    points = []
    for i in range(len(outline)):
        p, q = outline[i], outline[(i + 1) % len(outline)]
        length = math.hypot(q[0] - p[0], q[1] - p[1]) or 1.0
        normal = (-(q[1] - p[1]) / length, (q[0] - p[0]) / length)
        for t in np.linspace(0, 1, 14, endpoint=False):
            bend = bow[i] * length * math.sin(math.pi * t)
            points.append(_jitter((p[0] + (q[0] - p[0]) * t + normal[0] * bend,
                                   p[1] + (q[1] - p[1]) * t + normal[1] * bend), strength))
    first = random.randrange(len(points))  # the stroke starts at a random place
    return _stroke_ends(points[first:] + points[:first])


def draw_outline(points):
    """Draws a closed polyline (like in the app: the end is joined to the start), returns the pixel mask."""
    image = Image.new("1", (CANVAS_SIZE, CANVAS_SIZE), 0)
    ImageDraw.Draw(image).line(points + [points[0]], fill=1, width=LINE_WIDTH)
    return np.array(image, dtype=bool)


def _generate_one(make_points):
    """Draws figures until one fits on the canvas and its features can be computed."""
    while True:
        points = make_points()
        if any(not (2 <= x <= CANVAS_SIZE - 3 and 2 <= y <= CANVAS_SIZE - 3) for x, y in points):
            continue
        try:
            return image_to_features(draw_outline(points))
        except ValueError:
            continue


def generate(n, progress=None):
    """n circles and n squares, alternating. progress(i, n) is called after every pair.
    Returns a list of rows (2 features, label, source)."""
    rows = []
    for i in range(n):
        for label, make_points in (("circle", circle_points), ("square", square_points)):
            rows.append((*_generate_one(make_points), label, "generated"))
        if progress is not None:
            progress(i + 1, n)
    return rows


def generate_and_append(n, progress=None, path=data_handler.DATA_PATH):
    """Generates n circles and n squares and appends them to the CSV. Returns the appended rows."""
    rows = generate(n, progress)
    data_handler.append_data(rows, path)
    return rows


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    generate_and_append(count, progress=lambda i, total: print(f"\r{i}/{total}", end="")
                        if i % 50 == 0 or i == total else None)
    print(f"\nAppended {2 * count} samples to {data_handler.DATA_PATH}")
