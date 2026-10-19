"""Program 1 of 3: teaching the model (circle or square).

The computer draws thousands of figures by itself ("Generuj" appends them to data.csv), or a figure
can be drawn by hand and labeled. After every addition the model learns from the new figures,
continuing from the weights it already has. The weights after every step are appended to weights.csv
(replayed by simulate.py) and the latest ones overwrite boundary.json (used by recognize.py).
"Ucz ponownie" makes one more pass over the whole data.csv.

All texts shown in the window are in Polish; everything else is in English.

Run: python train.py
"""
import threading
import tkinter as tk

import numpy as np

import data_generator
import data_handler
import model
from data_handler import IncompatibleData
from drawing_canvas import DrawingCanvas, error_text
from feature_extractor import MODEL_FEATURES, image_to_features

DEFAULT_COUNT = 1000
MAX_COUNT = 20000

# Texts shown to the user (front end, Polish)
LABEL_NAMES = {"circle": "koło", "square": "kwadrat"}
INCOMPATIBLE_TEXT = ("Plik data.csv pochodzi ze starej wersji cech i nie pasuje do obecnej. "
                     "Przenieś go w inne miejsce, a program utworzy nowy.")
FILE_ERROR_TEXT = "Nie udało się zapisać pliku (czy jest otwarty w innym programie?): {}"


class TrainApp:
    def __init__(self, root):
        self.root = root
        root.title("Koło czy kwadrat — uczenie")
        self.progress = (0, 1)
        self.generation_result = None

        self.canvas = DrawingCanvas(root)
        self.canvas.pack(padx=10, pady=(10, 5))

        panel = tk.Frame(root)
        panel.pack(pady=(0, 5))
        self.draw_buttons = [
            tk.Button(panel, text="Wyczyść", command=self.canvas.clear),
            tk.Button(panel, text="To koło", command=lambda: self.save_drawn("circle")),
            tk.Button(panel, text="To kwadrat", command=lambda: self.save_drawn("square")),
        ]
        for i, button in enumerate(self.draw_buttons):
            button.grid(row=0, column=i, padx=3)

        generate_panel = tk.Frame(root)
        generate_panel.pack(pady=(10, 5))
        tk.Label(generate_panel, text="Komputer narysuje po").grid(row=0, column=0)
        self.count_var = tk.StringVar(value=str(DEFAULT_COUNT))
        self.count_entry = tk.Entry(generate_panel, textvariable=self.count_var, width=7, justify="center")
        self.count_entry.grid(row=0, column=1, padx=4)
        tk.Label(generate_panel, text="kół i kwadratów").grid(row=0, column=2)
        self.generate_button = tk.Button(generate_panel, text="Generuj", command=self.generate)
        self.generate_button.grid(row=0, column=3, padx=8)

        self.retrain_button = tk.Button(root, text="Ucz ponownie na całym data.csv", command=self.retrain)
        self.retrain_button.pack(pady=(5, 0))

        self.counter = tk.Label(root, fg="#444", wraplength=520)
        self.counter.pack(pady=(10, 0))
        self.status = tk.Label(root, text="Wygeneruj dane albo narysuj figurę i oznacz ją.",
                               width=60, wraplength=520, justify="center")
        self.status.pack(pady=(5, 10))
        self._refresh_counter()

    def _refresh_counter(self):
        try:
            _, y, sources = data_handler.load_data()
        except IncompatibleData:
            self.counter.config(text=INCOMPATIBLE_TEXT)
            return
        known = "model nauczony" if model.load_boundary() else "model jeszcze nienauczony"
        self.counter.config(text=f"W data.csv: {len(y)} próbek "
                                 f"({int((sources == 'drawn').sum())} narysowanych ręcznie); {known}.")

    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        for widget in self.draw_buttons + [self.generate_button, self.count_entry, self.retrain_button]:
            widget.config(state=state)

    def _train(self, new_count=None):
        """Learns and saves. new_count: only the last new_count figures of data.csv are new, learn from
        them; None = one pass over the whole data.csv. The very first learning always uses all the data.
        Returns text for the status line."""
        X, y, _ = data_handler.load_data()
        first = model.load_boundary() is None
        if len(y) == 0 or (first and not model.enough_data(y)):
            return (f"Próbek: {len(y)}. Za mało do uczenia (min. {model.MIN_SAMPLES}, "
                    f"po {model.MIN_PER_CLASS} każdej figury).")
        rows = slice(None) if first or new_count is None else slice(-new_count, None)
        result = model.learn(X[rows], y[rows])
        weights, bias = result["boundary"]["weights"], result["boundary"]["bias"]
        return (f"Liczba figur w tym uczeniu: {result['samples']}. "
                f"Trafność na ostatnich 200 krokach (przed nauką na nich): {result['accuracy']:.1%}\n"
                f"Wagi: kolistość {weights[0]:+.2f}, liczba rogów {weights[1]:+.2f}, stała {bias:+.2f}\n"
                f"Wagi z każdego kroku dopisane do weights.csv.")

    def retrain(self):
        try:
            self.status.config(text=self._train())
        except IncompatibleData:
            self.status.config(text=INCOMPATIBLE_TEXT)
        except OSError as error:
            self.status.config(text=FILE_ERROR_TEXT.format(error))
        self._refresh_counter()

    def save_drawn(self, label):
        canvas = self.canvas
        if not canvas.closed:
            self.status.config(text="Najpierw narysuj figurę.")
            return
        try:
            features = image_to_features(canvas.outline_mask())
        except ValueError as error:
            self.status.config(text=f"Błąd: {error_text(error)}")
            return
        try:
            data_handler.append_data([(*features, label, "drawn")])
            canvas.clear()
            self.status.config(text=f"Zapisano: {LABEL_NAMES[label]}\n{self._train(1)}")
        except IncompatibleData:
            self.status.config(text=INCOMPATIBLE_TEXT)
        except OSError as error:
            self.status.config(text=FILE_ERROR_TEXT.format(error))
        self._refresh_counter()

    def generate(self):
        try:
            count = int(self.count_var.get())
            if not 1 <= count <= MAX_COUNT:
                raise ValueError
        except ValueError:
            self.status.config(text=f"Podaj liczbę całkowitą od 1 do {MAX_COUNT}.")
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
            self.status.config(text=f"Komputer rysuje figury: {done}/{total} kół i kwadratów...")
            self.root.after(100, self._poll_generation)
            return
        kind, value = self.generation_result
        if kind == "error":
            self.status.config(text=f"Błąd generowania: {value}")
            self._set_busy(False)
            return
        rows = value
        try:
            previous = model.load_boundary()  # honest test: the old boundary on a new, unseen batch
            before = ""
            if previous is not None:
                X = np.array([row[:len(MODEL_FEATURES)] for row in rows], dtype=float)
                y = np.array([row[len(MODEL_FEATURES)] for row in rows])
                before = f"Stara granica na nowych figurach: {model.accuracy(previous, X, y):.1%}\n"
            data_handler.append_data(rows)
            self.status.config(text=f"Dopisano {len(rows)} próbek.\n{before}{self._train(len(rows))}")
        except IncompatibleData:
            self.status.config(text=INCOMPATIBLE_TEXT)
        except OSError as error:
            self.status.config(text=FILE_ERROR_TEXT.format(error))
        self._refresh_counter()
        self._set_busy(False)


if __name__ == "__main__":
    root = tk.Tk()
    TrainApp(root)
    root.mainloop()
