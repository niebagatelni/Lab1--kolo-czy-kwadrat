"""Program 2 of 3: recognizing a figure (circle or square).

Draw a figure in a single stroke (the outline closes itself) or load it from an image file. The program uses the model taught by
train.py (boundary.json), says circle or square and shows on the 2D chart where the figure landed
among the data, relative to the decision boundary.

All texts shown in the window are in Polish; everything else is in English.

Run: python recognize.py
"""
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import chart
import data_handler
import model
from data_handler import IncompatibleData
from drawing_canvas import DrawingCanvas, error_text
from feature_extractor import MODEL_FEATURES, image_to_features

MAX_BACKGROUND_POINTS = 1500

# Texts shown to the user (front end, Polish)
RESULT_NAMES = {"circle": "KOŁO", "square": "KWADRAT"}
HINT = "Narysuj figurę jednym pociągnięciem."
NO_MODEL_TEXT = "Brak wyuczonego modelu."
NO_MODEL_HINT = "Uruchom najpierw train.py, wygeneruj dane i naucz model."
CHART_TITLE = "Gdzie wypadł Twój rysunek: kolistość i liczba rogów"
STAR_LABEL = "Twój rysunek"
FILE_DIALOG_TITLE = "Wybierz obraz z figurą"
FILE_TYPES = [("Obrazy", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Wszystkie pliki", "*.*")]


class RecognizeApp:
    def __init__(self, root):
        self.root = root
        root.title("Koło czy kwadrat — rozpoznawanie")

        left = tk.Frame(root)
        left.pack(side="left", fill="y", padx=(10, 0))
        self.canvas = DrawingCanvas(left)
        self.canvas.pack(pady=(10, 5))

        panel = tk.Frame(left)
        panel.pack(pady=(0, 5))
        tk.Button(panel, text="Wyczyść", command=self.clear).grid(row=0, column=0, padx=3)
        tk.Button(panel, text="Rozpoznaj", command=self.recognize).grid(row=0, column=1, padx=3)
        tk.Button(panel, text="Wczytaj z pliku", command=self.load_file).grid(row=0, column=2, padx=3)

        self.result = tk.Label(left, text=HINT, font=("Segoe UI", 14, "bold"), wraplength=320, justify="center")
        self.result.pack(pady=(10, 0))
        self.details = tk.Label(left, fg="#444", wraplength=320, justify="center")
        self.details.pack(pady=(2, 10))

        self.figure = Figure(figsize=(9.6, 6.0), constrained_layout=True)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.figure_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.show_chart(None, None)

    def clear(self):
        self.canvas.clear()
        self.result.config(text=HINT)
        self.details.config(text="")
        self.show_chart(None, None)

    def _load_background(self):
        try:
            X, y, _ = data_handler.load_data()
        except IncompatibleData:
            return np.empty((0, len(MODEL_FEATURES))), np.empty(0, dtype=str)
        return X, y

    def show_chart(self, features, verdict):
        """Data and boundary; when `features` are given, also the drawn figure as a star."""
        boundary = model.load_boundary()  # read every time, so no restart is needed after training
        X, y = self._load_background()
        extra = [(features[chart.X_COLUMN], features[chart.Y_COLUMN])] if features is not None else []
        ax, side = chart.setup_axes(self.figure, X, extra)
        ax.set_title(CHART_TITLE, fontsize=12, fontweight="bold")
        if boundary is None:
            ax.text(0.5, 0.5, NO_MODEL_HINT, transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="0.4")
            self.figure_canvas.draw_idle()
            return

        offsets = chart.jitter(len(y))
        if len(y) > MAX_BACKGROUND_POINTS:
            keep = np.sort(np.random.default_rng(1).choice(len(y), MAX_BACKGROUND_POINTS, replace=False))
        else:
            keep = np.arange(len(y))
        for label in ("circle", "square"):
            chosen = keep[y[keep] == label]
            ax.scatter(X[chosen, chart.X_COLUMN], X[chosen, chart.Y_COLUMN] + offsets[chosen], s=9,
                       c=chart.COLORS[label], alpha=0.35, linewidths=0, zorder=2)
        region = chart.Boundary(ax)
        region.update(boundary["weights"], boundary["bias"])

        extra_handles = []
        text = chart.weights_text(boundary["weights"], boundary["bias"])
        if features is not None:
            ax.scatter([features[chart.X_COLUMN]], [features[chart.Y_COLUMN]], marker="*", s=420,
                       c="gold", edgecolors="black", linewidths=1.3, zorder=6)
            extra_handles.append(Line2D([], [], marker="*", linestyle="", markerfacecolor="gold",
                                        markeredgecolor="black", markersize=14, label=STAR_LABEL))
            text = (f"{STAR_LABEL}:\n"
                    f"  kolistość: {features[chart.X_COLUMN]:.3f}\n"
                    f"  liczba rogów: {features[chart.Y_COLUMN]:.0f}\n"
                    f"  z = {float(model.score(boundary, features)):+.2f}  →  {verdict}\n\n") + text
        chart.show_legend(side, extra_handles)
        side.text(0.0, 0.42, text, transform=side.transAxes, va="top", fontsize=9)
        self.figure_canvas.draw_idle()

    def load_file(self):
        path = filedialog.askopenfilename(title=FILE_DIALOG_TITLE, filetypes=FILE_TYPES)
        if not path:
            return
        try:
            self.canvas.load_image(path)
        except (OSError, ValueError) as error:
            self.canvas.clear()
            self.result.config(text="Nie udało się wczytać pliku.")
            self.details.config(text=str(error))
            return
        self.recognize()

    def recognize(self):
        boundary = model.load_boundary()
        if boundary is None:
            self.result.config(text=NO_MODEL_TEXT)
            self.details.config(text=NO_MODEL_HINT)
            self.show_chart(None, None)
            return
        if not self.canvas.closed:
            self.result.config(text="Najpierw narysuj figurę.")
            self.details.config(text="")
            return
        try:
            features = image_to_features(self.canvas.outline_mask())
        except ValueError as error:
            self.result.config(text=f"Błąd: {error_text(error)}")
            self.details.config(text="")
            return
        label, confidence, z = model.classify(boundary, features)
        verdict = RESULT_NAMES.get(label, label)
        self.result.config(text=f"To jest {verdict}  (pewność {confidence:.0%})")
        self.details.config(text=(f"kolistość {features[chart.X_COLUMN]:.3f} · "
                                  f"liczba rogów {features[chart.Y_COLUMN]:.0f} · z {z:+.2f}"))
        self.show_chart(features, verdict)


if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(1100, 640)
    RecognizeApp(root)
    root.mainloop()
