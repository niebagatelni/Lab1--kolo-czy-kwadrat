"""Circle or square: one window, three tabs.

  Ucz        (train)      - the computer draws thousands of figures by itself ("Generuj" appends to
                            data.csv) and trains the model; a figure can also be drawn and labeled by hand
  Rozpoznaj  (recognize)  - draw a figure in a single stroke; the program says circle or square and shows
                            where the figure landed on the feature charts
  Symulacja  (simulation) - an animation of how the model learned from data.csv, step by step

All texts shown in the window are in Polish; everything else is in English.

Run: python app.py
"""
import os
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw

import data_generator
import data_handler
import model
import recognition_chart
import simulation
from data_handler import IncompatibleData
from feature_extractor import (FEATURE_NAMES, LINE_WIDTH, DegenerateFigure, EmptyDrawing,
                               FigureTooSmall, image_to_features)

CANVAS_SIZE = data_generator.CANVAS_SIZE
DEFAULT_COUNT = 1000
MAX_COUNT = 20000
FRAME_INTERVAL_MS = 30
SIMULATION_PNG = os.path.join(data_handler.FOLDER, "simulation.png")

# Texts shown to the user (front end, Polish)
RESULT_NAMES = {"circle": "KOŁO", "square": "KWADRAT"}
LABEL_NAMES = {"circle": "koło", "square": "kwadrat"}
ERROR_TEXTS = {
    EmptyDrawing: "Pusty rysunek.",
    FigureTooSmall: "Figura jest za mała.",
    DegenerateFigure: "Figura jest zdegenerowana.",
}
INCOMPATIBLE_TEXT = ("Plik data.csv pochodzi ze starej wersji cech i nie pasuje do obecnej. "
                     "Przenieś go w inne miejsce, a program utworzy nowy.")
RECOGNIZE_HINT = "Narysuj figurę jednym pociągnięciem."


def error_text(error):
    return ERROR_TEXTS.get(type(error), "Nieprawidłowy rysunek.")


def format_weights(boundary):
    """Weights as a short text for the status line."""
    parts = [f"{simulation.FEATURE_LABELS.get(name, name)} {weight:+.2f}"
             for name, weight in zip(boundary["features"], boundary["weights"])]
    return "Wagi: " + ", ".join(parts) + f", stała {boundary['bias']:+.2f}"


class DrawingCanvas(tk.Frame):
    """A canvas for drawing in a single stroke; the outline closes itself when the mouse is released."""

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


class App:
    def __init__(self, root):
        self.root = root
        root.title("Koło czy kwadrat")
        self.frame_job = None
        self.simulation = None
        self.frame = 0
        self.progress = (0, 1)
        self.generation_result = None

        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True)
        self._build_train_tab(tabs)
        self._build_recognize_tab(tabs)
        self._build_simulation_tab(tabs)
        self._refresh_counter()

    # ---------- tab: train ----------
    def _build_train_tab(self, tabs):
        frame = tk.Frame(tabs)
        tabs.add(frame, text="  Ucz  ")

        self.train_canvas = DrawingCanvas(frame)
        self.train_canvas.pack(padx=10, pady=(10, 5))

        panel = tk.Frame(frame)
        panel.pack(pady=(0, 5))
        self.draw_buttons = [
            tk.Button(panel, text="Wyczyść", command=self.train_canvas.clear),
            tk.Button(panel, text="To koło", command=lambda: self.save_drawn("circle")),
            tk.Button(panel, text="To kwadrat", command=lambda: self.save_drawn("square")),
        ]
        for i, button in enumerate(self.draw_buttons):
            button.grid(row=0, column=i, padx=3)

        generate_panel = tk.Frame(frame)
        generate_panel.pack(pady=(10, 5))
        tk.Label(generate_panel, text="Komputer narysuje po").grid(row=0, column=0)
        self.count_var = tk.StringVar(value=str(DEFAULT_COUNT))
        self.count_entry = tk.Entry(generate_panel, textvariable=self.count_var, width=7, justify="center")
        self.count_entry.grid(row=0, column=1, padx=4)
        tk.Label(generate_panel, text="kół i kwadratów").grid(row=0, column=2)
        self.generate_button = tk.Button(generate_panel, text="Generuj", command=self.generate)
        self.generate_button.grid(row=0, column=3, padx=8)

        self.counter = tk.Label(frame, fg="#444", wraplength=520)
        self.counter.pack()
        self.train_status = tk.Label(frame, text="Wygeneruj dane albo narysuj figurę i oznacz ją.",
                                     width=60, wraplength=520, justify="center")
        self.train_status.pack(pady=(5, 10))

    def _refresh_counter(self):
        try:
            _, y, sources = data_handler.load_data()
        except IncompatibleData:
            self.counter.config(text=INCOMPATIBLE_TEXT)
            return
        self.counter.config(text=f"W data.csv: {len(y)} próbek "
                                 f"({int((sources == 'drawn').sum())} narysowanych ręcznie)")

    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        for widget in self.draw_buttons + [self.generate_button, self.count_entry]:
            widget.config(state=state)

    def _train(self):
        """Trains on the whole data.csv and saves boundary.json. Returns text for the status line."""
        result = model.train_and_save()
        if result is None:
            _, y, _ = data_handler.load_data()
            return (f"Próbek: {len(y)}. Za mało do uczenia (min. {model.MIN_SAMPLES}, "
                    f"po {model.MIN_PER_CLASS} każdej figury).")
        return (f"Nauczone na {result['samples']} próbkach. "
                f"Dokładność na danych testowych: {result['accuracy']:.1%}\n{format_weights(result['boundary'])}")

    def save_drawn(self, label):
        canvas = self.train_canvas
        if not canvas.closed:
            self.train_status.config(text="Najpierw narysuj figurę.")
            return
        try:
            features = image_to_features(canvas.outline_mask())
        except ValueError as error:
            self.train_status.config(text=f"Błąd: {error_text(error)}")
            return
        try:
            data_handler.append_data([(*features, label, "drawn")])
            canvas.clear()
            self._refresh_counter()
            self.train_status.config(text=f"Zapisano: {LABEL_NAMES[label]}\n{self._train()}")
        except IncompatibleData:
            self.train_status.config(text=INCOMPATIBLE_TEXT)

    def generate(self):
        try:
            count = int(self.count_var.get())
            if not 1 <= count <= MAX_COUNT:
                raise ValueError
        except ValueError:
            self.train_status.config(text=f"Podaj liczbę całkowitą od 1 do {MAX_COUNT}.")
            return
        self._set_busy(True)
        self.progress = (0, count)
        self.generation_result = None

        def work():
            try:
                rows = data_generator.generate(
                    count, lambda i, total: setattr(self, "progress", (i, total)))
                self.generation_result = ("ok", rows)
            except Exception as error:  # shown in the window instead of silently freezing the button
                self.generation_result = ("error", error)

        threading.Thread(target=work, daemon=True).start()
        self.root.after(100, self._poll_generation)

    def _poll_generation(self):
        if self.generation_result is None:
            done, total = self.progress
            self.train_status.config(text=f"Komputer rysuje figury: {done}/{total} kół i kwadratów...")
            self.root.after(100, self._poll_generation)
            return
        kind, value = self.generation_result
        if kind == "error":
            self.train_status.config(text=f"Błąd generowania: {value}")
            self._set_busy(False)
            return
        rows = value
        try:
            previous = model.load_boundary()  # honest test: the old boundary on a new, unseen batch
            before = ""
            if previous is not None:
                X = np.array([row[:len(FEATURE_NAMES)] for row in rows], dtype=float)
                y = np.array([row[len(FEATURE_NAMES)] for row in rows])
                before = f"Stara granica na nowych figurach: {model.accuracy(previous, X, y):.1%}\n"
            data_handler.append_data(rows)
            self._refresh_counter()
            self.train_status.config(text=f"Dopisano {len(rows)} próbek.\n{before}{self._train()}")
        except IncompatibleData:
            self.train_status.config(text=INCOMPATIBLE_TEXT)
        self._set_busy(False)

    # ---------- tab: recognize ----------
    def _build_recognize_tab(self, tabs):
        frame = tk.Frame(tabs)
        tabs.add(frame, text="  Rozpoznaj  ")

        left = tk.Frame(frame)
        left.pack(side="left", fill="y", padx=(10, 0))
        self.recognize_canvas = DrawingCanvas(left)
        self.recognize_canvas.pack(pady=(10, 5))

        panel = tk.Frame(left)
        panel.pack(pady=(0, 5))
        tk.Button(panel, text="Wyczyść", command=self._clear_recognize).grid(row=0, column=0, padx=3)
        tk.Button(panel, text="Rozpoznaj", command=self.recognize).grid(row=0, column=1, padx=3)

        self.result = tk.Label(left, text=RECOGNIZE_HINT, font=("Segoe UI", 14, "bold"),
                               wraplength=320, justify="center")
        self.result.pack(pady=(10, 0))
        self.details = tk.Label(left, fg="#444", wraplength=320, justify="center")
        self.details.pack(pady=(2, 10))

        self.chart = recognition_chart.RecognitionChart(recognition_chart.new_figure())
        self.chart_canvas = FigureCanvasTkAgg(self.chart.fig, master=frame)
        self.chart_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    def _clear_recognize(self):
        self.recognize_canvas.clear()
        self.result.config(text=RECOGNIZE_HINT)
        self.details.config(text="")
        self.chart.clear()
        self.chart_canvas.draw_idle()

    def recognize(self):
        boundary = model.load_boundary()  # read every time, so no restart is needed after training
        if boundary is None:
            self.result.config(text="Brak wyuczonego modelu.")
            self.details.config(text="Przejdź do zakładki „Ucz”, wygeneruj dane i naucz model.")
            return
        if not self.recognize_canvas.closed:
            self.result.config(text="Najpierw narysuj figurę.")
            self.details.config(text="")
            return
        try:
            features = image_to_features(self.recognize_canvas.outline_mask())
        except ValueError as error:
            self.result.config(text=f"Błąd: {error_text(error)}")
            self.details.config(text="")
            return
        label, confidence, z = model.classify(boundary, features)
        self.result.config(text=f"To jest {RESULT_NAMES.get(label, label)}  (pewność {confidence:.0%})")
        norm_radius_cv, norm_radius_spread, circularity, corners = features
        self.details.config(text=(f"kolistość {circularity:.3f} · liczba rogów {corners:.0f} · "
                                  f"falowanie promienia {norm_radius_cv:.3f} · "
                                  f"rozpiętość {norm_radius_spread:.2f} · z {z:+.2f}"))
        try:
            X, y, _ = data_handler.load_data()
        except IncompatibleData:
            X, y = np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=str)
        self.chart.show(boundary, features, X, y)
        self.chart_canvas.draw_idle()

    # ---------- tab: simulation ----------
    def _build_simulation_tab(self, tabs):
        frame = tk.Frame(tabs)
        tabs.add(frame, text="  Symulacja  ")

        panel = tk.Frame(frame)
        panel.pack(pady=6)
        tk.Button(panel, text="Start (od początku)", command=self.start_simulation).grid(row=0, column=0, padx=3)
        tk.Button(panel, text="Stop", command=self.stop_simulation).grid(row=0, column=1, padx=3)
        self.simulation_status = tk.Label(
            panel, text="Naciśnij Start, żeby zobaczyć, jak model uczył się na danych.")
        self.simulation_status.grid(row=0, column=2, padx=10)

        self.figure = simulation.new_figure()
        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    def stop_simulation(self):
        if self.frame_job is not None:
            self.root.after_cancel(self.frame_job)
            self.frame_job = None

    def start_simulation(self):
        self.stop_simulation()
        try:
            X, y, _ = data_handler.load_data()
        except IncompatibleData:
            self.simulation_status.config(text=INCOMPATIBLE_TEXT)
            return
        if not model.enough_data(y):
            self.simulation_status.config(text=f"Za mało danych w data.csv (min. {model.MIN_SAMPLES}). "
                                               f"Wygeneruj je w zakładce „Ucz”.")
            return
        self.simulation = simulation.Simulation(self.figure, model.train(X, y))
        self.frame = 0
        self.simulation_status.config(text=f"Uczenie na {len(y)} próbkach...")
        self._next_frame()

    def _next_frame(self):
        self.simulation.draw_frame(self.frame)
        self.figure_canvas.draw_idle()
        self.frame += 1
        if self.frame < self.simulation.frame_count:
            self.frame_job = self.root.after(FRAME_INTERVAL_MS, self._next_frame)
            return
        self.frame_job = None
        self.figure.savefig(SIMULATION_PNG, dpi=110)
        self.simulation_status.config(
            text=f"Koniec uczenia. Ostatnia klatka zapisana w {os.path.basename(SIMULATION_PNG)}.")


if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(1180, 700)
    App(root)
    root.mainloop()
