"""A canvas for drawing a figure in a single stroke, used by train.py and recognize.py."""
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import data_generator
from feature_extractor import LINE_WIDTH, DegenerateFigure, EmptyDrawing, FigureTooSmall

CANVAS_SIZE = data_generator.CANVAS_SIZE

# Texts shown to the user (front end, Polish)
ERROR_TEXTS = {
    EmptyDrawing: "Pusty rysunek.",
    FigureTooSmall: "Figura jest za mała.",
    DegenerateFigure: "Figura jest zdegenerowana.",
}


def error_text(error):
    return ERROR_TEXTS.get(type(error), "Nieprawidłowy rysunek.")


INK_THRESHOLD = 128  # a pixel darker than this belongs to the figure (on a light background)


class DrawingCanvas(tk.Frame):
    """A canvas for drawing in a single stroke; the outline closes itself when the mouse is released.
    A figure can also be loaded from an image file (load_image)."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white",
                                highlightthickness=1, highlightbackground="black")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("1", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.points = []
        self.closed = False

    def _draw_line(self, a, b):
        self.canvas.create_line(*a, *b, width=LINE_WIDTH, capstyle=tk.ROUND)
        self.pen.line([a, b], fill=1, width=LINE_WIDTH)

    def _on_press(self, event):
        if self.closed:  # a new stroke after a finished figure clears the canvas
            self.clear()
        self.points = [(event.x, event.y)]

    def _on_drag(self, event):
        point = (event.x, event.y)
        self._draw_line(self.points[-1], point)
        self.points.append(point)

    def _on_release(self, _):
        if len(self.points) > 1:
            self._draw_line(self.points[-1], self.points[0])  # close the outline
            self.closed = True

    def outline_mask(self):
        return np.array(self.image, dtype=bool)

    def load_image(self, path):
        """Replaces the drawing with the figure from an image file (a dark figure on a light background,
        or the opposite; an outline or a filled shape, both give the same result). The figure is scaled
        to fit the canvas. Raises OSError if the file cannot be read as an image."""
        picture = Image.open(path)
        if picture.mode in ("RGBA", "LA", "P"):  # transparent parts count as white
            picture = picture.convert("RGBA")
            background = Image.new("RGBA", picture.size, "white")
            picture = Image.alpha_composite(background, picture)
        gray = np.array(picture.convert("L"))
        ink = gray < INK_THRESHOLD
        if ink.mean() > 0.5:  # light figure on a dark background
            ink = ~ink
        scale = CANVAS_SIZE / max(ink.shape)
        size = (max(1, round(ink.shape[1] * scale)), max(1, round(ink.shape[0] * scale)))
        # BOX averaging, and any ink in a block counts, so thin lines survive the shrinking
        fitted = Image.fromarray(ink.astype(np.uint8) * 255).resize(size, Image.BOX)
        fitted = fitted.point(lambda value: 255 if value > 0 else 0).convert("1")

        self.clear()
        offset = ((CANVAS_SIZE - size[0]) // 2, (CANVAS_SIZE - size[1]) // 2)
        self.image.paste(fitted, offset)
        self.points = [(0, 0), (0, 0)]
        self.closed = True
        shown = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        shown.paste(fitted.point(lambda value: 0 if value else 255).convert("L"), offset)
        self.photo = ImageTk.PhotoImage(shown)  # kept, otherwise tkinter drops the image
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
