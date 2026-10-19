"""Program 3 of 3: simulation of the learning (replay of what train.py did).

Reads data.csv (the figures: X = circularity, Y = number of corners) and weights.csv (the weights after
every step of the learning). Step after step the points appear on the chart and the boundary moves;
faint gray lines are its earlier positions. Nothing is learned here and nothing is changed: the program
only shows the numbers saved by train.py.

The animation only ever plays forward. Controls: the speed slider (0 = standing), the position slider
and the step number field (jump to any step, also back; the animation continues from there).

All texts shown in the window are in Polish; everything else is in English.

Run: python simulate.py
"""
import math
import os
import time
import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import chart
import data_handler
import model

FRAME_INTERVAL_MS = 30
TRAIL_LENGTH = 25         # how many earlier positions of the boundary are shown
TRAIL_PARTS = 300         # the trail lines are (steps / TRAIL_PARTS) steps apart
MAX_FRAME_SECONDS = 0.25  # a slow frame must not make the animation leap ahead
SPEED_MIN = 1.0           # steps per second at the first notch of the speed slider
SPEED_NOTCHES = 100       # the speed slider goes 0 .. SPEED_NOTCHES, 0 = standing, the rest is logarithmic
DEFAULT_SECONDS = 10      # the whole history takes about that long at the starting speed
SIMULATION_PNG = os.path.join(data_handler.FOLDER, "simulation.png")

# Texts shown to the user (front end, Polish)
CHART_TITLE = "Uczenie granicy koło / kwadrat: kolistość i liczba rogów"
INCOMPATIBLE_TEXT = ("Plik data.csv lub weights.csv pochodzi ze starej wersji i nie pasuje do obecnej. "
                     "Przenieś go w inne miejsce, a program utworzy nowy.")
NOT_TRAINED_TEXT = "Brak wag w weights.csv. Uruchom train.py i naucz model."
NO_DATA_TEXT = "Brak danych w data.csv. Uruchom train.py i wygeneruj je."
FOOTNOTE = ("Szare linie: wcześniejsze położenia granicy. Trafność: jak często granica z tego kroku "
            "rozpoznaje poprawnie figury widoczne na wykresie.")


class Blitter:
    """Redraws only the moving parts of the chart. The rest (axes, labels, legend) is drawn once and kept
    as a picture, which makes a frame many times cheaper than redrawing the whole figure."""

    def __init__(self, canvas, artists):
        self.canvas, self.artists, self.background = canvas, artists, None
        self.saving = False
        for artist in artists:
            artist.set_animated(True)
        self.connection = canvas.mpl_connect("draw_event", self._on_draw)

    def close(self):
        self.canvas.mpl_disconnect(self.connection)

    def _on_draw(self, event):  # after every full draw (also after resizing the window)
        if self.saving:  # saving to a file draws the whole figure by itself
            return
        self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_artists()

    def _draw_artists(self):
        for artist in self.artists:
            self.canvas.figure.draw_artist(artist)

    def update(self):
        if self.background is None:
            self.canvas.draw_idle()  # the first full draw, which also takes the picture of the background
            return
        self.canvas.restore_region(self.background)
        self._draw_artists()
        self.canvas.blit(self.canvas.figure.bbox)

    def save(self, path, dpi):
        """Saves the whole figure (animated artists are skipped by a normal drawing)."""
        self.saving = True
        for artist in self.artists:
            artist.set_animated(False)
        self.canvas.figure.savefig(path, dpi=dpi)
        for artist in self.artists:
            artist.set_animated(True)
        self.saving = False


class Simulation:
    """The chart. draw_step shows any step, forward or back: it depends only on the step number."""

    def __init__(self, fig, X, y, history):
        self.fig, self.X, self.y = fig, X, y
        self.history = history
        self.count = len(y)
        self.trail_stride = max(1, len(history) // TRAIL_PARTS)
        self.shift = chart.jitter(self.count)
        self.positions = {label: np.flatnonzero(y == label) for label in chart.COLORS}

        ax, side = chart.setup_axes(fig, X)
        self.region = chart.Boundary(ax)
        self.points = {label: ax.scatter([], [], s=10, c=chart.COLORS[label], alpha=0.5, linewidths=0, zorder=3)
                       for label in chart.COLORS}
        self.trail = [ax.plot([], [], color="0.6", linewidth=1, alpha=0.6, zorder=1)[0]
                      for _ in range(TRAIL_LENGTH)]
        ax.set_title(CHART_TITLE, fontsize=12, fontweight="bold")
        trail_handle = Line2D([], [], color="0.6", linewidth=1, label="wcześniejsze granice")
        chart.show_legend(side, [trail_handle])
        self.info = side.text(0.0, 0.40, "", transform=side.transAxes, va="top", fontsize=9)
        self.animated = self.region.artists + self.trail + list(self.points.values()) + [self.info]
        fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.965))  # room for the footnote
        fig.text(0.01, 0.005, FOOTNOTE, fontsize=8, color="0.35")

    def draw_step(self, row):
        """Draws the state after step `row` (0-based)."""
        row = min(max(row, 0), len(self.history) - 1)
        visible = min(row + 1, self.count)  # after the first pass over the data all figures are shown
        weights, bias = self.history[row, :2], self.history[row, 2]

        self.region.update(weights, bias)
        for j, line in enumerate(self.trail, start=1):
            earlier = row - j * self.trail_stride
            if earlier < 0:
                line.set_data([], [])
            else:
                line.set_data(*self.region.line_points(self.history[earlier, :2], self.history[earlier, 2]))

        for label, positions in self.positions.items():
            shown = positions[positions < visible]
            self.points[label].set_offsets(
                np.c_[self.X[shown, chart.X_COLUMN], self.X[shown, chart.Y_COLUMN] + self.shift[shown]])

        z = model.z_grid(weights, bias, self.X[:visible, chart.X_COLUMN], self.X[:visible, chart.Y_COLUMN])
        right = (z > 0) == (self.y[:visible] == model.POSITIVE_CLASS)
        self.info.set_text(f"Krok {row + 1} z {len(self.history)}\n"
                           f"Figur na wykresie: {visible} z {self.count}\n"
                           f"Trafność: {right.mean():.1%}\n\n" + chart.weights_text(weights, bias))


class SimulateApp:
    def __init__(self, root):
        self.root = root
        root.title("Koło czy kwadrat — symulacja uczenia")
        self.simulation = None
        self.steps = 0              # number of steps in weights.csv
        self.position = 1.0         # the step being shown (fractional while playing)
        self.shown = None           # the step currently drawn
        self.playing = False
        self.frame_job = None
        self.last_tick = 0.0
        self.max_speed = SPEED_MIN * 2
        self.dragging = False       # the mouse holds the position slider
        self.blitter = None

        top = tk.Frame(root)
        top.pack(pady=(6, 0))
        tk.Button(top, text="Start (od początku)", command=self.start).grid(row=0, column=0, padx=3)
        tk.Button(top, text="Stop", command=self.stop).grid(row=0, column=1, padx=3)
        self.status = tk.Label(top, text="Naciśnij Start, żeby zobaczyć, jak model uczył się na danych.")
        self.status.grid(row=0, column=2, padx=10)

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=10, pady=4)
        tk.Label(controls, text="Krok:").grid(row=0, column=0)
        self.step_var = tk.StringVar()
        self.step_entry = tk.Entry(controls, textvariable=self.step_var, width=9, justify="center",
                                   state="disabled")
        self.step_entry.grid(row=0, column=1, padx=4)
        self.step_entry.bind("<Return>", lambda _: self.go())
        self.step_total = tk.Label(controls, text="")
        self.step_total.grid(row=0, column=2)
        self.go_button = tk.Button(controls, text="Idź", command=self.go, state="disabled")
        self.go_button.grid(row=0, column=3, padx=6)
        self.position_slider = tk.Scale(controls, from_=1, to=1, orient="horizontal", showvalue=False,
                                        state="disabled")
        # the slider reacts only to the user (mouse, keys), never to the program moving it
        self.position_slider.bind("<ButtonPress-1>", self._slider_pressed)
        self.position_slider.bind("<B1-Motion>", self._slider_used)
        self.position_slider.bind("<ButtonRelease-1>", self._slider_released)
        self.position_slider.bind("<KeyRelease>", self._slider_used)
        self.position_slider.grid(row=0, column=4, sticky="ew", padx=6)
        controls.columnconfigure(4, weight=1)

        tk.Label(controls, text="Prędkość:").grid(row=1, column=0, pady=(2, 0))
        self.speed_slider = tk.Scale(controls, from_=0, to=SPEED_NOTCHES, orient="horizontal", showvalue=False,
                                     length=260, command=self._on_speed_slider)
        self.speed_slider.grid(row=1, column=1, columnspan=3, pady=(2, 0), sticky="w")
        self.speed_label = tk.Label(controls, text="")
        self.speed_label.grid(row=1, column=4, sticky="w", padx=6)

        self.figure = Figure(figsize=(11, 6.0), constrained_layout=True)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------- speed ----------
    def _speed(self):
        """Steps per second from the speed slider (0 = standing, logarithmic above)."""
        notch = self.speed_slider.get()
        if notch <= 0:
            return 0.0
        return SPEED_MIN * (self.max_speed / SPEED_MIN) ** ((notch - 1) / (SPEED_NOTCHES - 1))

    def _notch_for(self, speed):
        speed = min(max(speed, SPEED_MIN), self.max_speed)
        return 1 + (SPEED_NOTCHES - 1) * math.log(speed / SPEED_MIN) / math.log(self.max_speed / SPEED_MIN)

    def _on_speed_slider(self, _=None):
        speed = self._speed()
        self.speed_label.config(text="stoi" if speed == 0 else f"{speed:,.0f} kroków/s".replace(",", " "))

    # ---------- position ----------
    def _slider_pressed(self, _):
        self.dragging = True
        self._slider_used()

    def _slider_released(self, _):
        self.dragging = False
        self._slider_used()

    def _slider_used(self, _=None):
        self.root.after_idle(self._apply_slider)  # after Tk has moved the slider

    def _apply_slider(self):
        step = int(self.position_slider.get())
        if self.simulation is not None and step != self.shown:
            self.jump(step)

    def go(self):
        """Jumps to the step typed in the field."""
        if self.simulation is None:
            return
        try:
            step = int(self.step_var.get().strip())
        except ValueError:
            self.step_var.set(str(self.shown))  # not a number: back to the current step
            return
        self.figure_canvas.get_tk_widget().focus_set()
        self.jump(step)

    def jump(self, step):
        """Shows the given step (clamped to 1 .. last) and goes on playing forward from it."""
        step = min(max(step, 1), self.steps)
        self.position = float(step)
        self.shown = None
        self._show(step)
        self.playing = True
        self.last_tick = time.perf_counter()
        if self.frame_job is None:
            self.frame_job = self.root.after(FRAME_INTERVAL_MS, self._tick)
        self.status.config(text="Uczenie...")

    def _show(self, step):
        if step == self.shown:
            return
        self.shown = step
        self.simulation.draw_step(step - 1)
        self.blitter.update()
        if not self.dragging:
            self.position_slider.set(step)
        if self.root.focus_get() is not self.step_entry:  # do not overwrite what is being typed
            self.step_var.set(str(step))

    # ---------- playing ----------
    def stop(self):
        self.playing = False
        if self.frame_job is not None:
            self.root.after_cancel(self.frame_job)
            self.frame_job = None
        if self.simulation is not None:
            self.status.config(text="Zatrzymano. Przesuń suwak kroku albo wpisz krok, żeby ruszyć dalej.")

    def start(self):
        self.stop()
        try:
            X, y, _ = data_handler.load_data()
            history = model.load_weights()
        except ValueError:  # includes IncompatibleData
            self.status.config(text=INCOMPATIBLE_TEXT)
            return
        if len(y) == 0:
            self.status.config(text=NO_DATA_TEXT)
            return
        if len(history) == 0:
            self.status.config(text=NOT_TRAINED_TEXT)
            return
        if self.blitter is not None:
            self.blitter.close()
        self.simulation = Simulation(self.figure, X, y, history)
        self.blitter = Blitter(self.figure_canvas, self.simulation.animated)
        self.steps = len(history)
        self.max_speed = max(SPEED_MIN * 2, self.steps / 2)  # the whole history in 2 seconds
        for widget in (self.step_entry, self.go_button, self.position_slider):
            widget.config(state="normal")
        self.position_slider.config(to=self.steps)
        self.step_total.config(text=f"z {self.steps}")
        self.speed_slider.set(round(self._notch_for(self.steps / DEFAULT_SECONDS)))
        self._on_speed_slider()
        self.jump(1)

    def _tick(self):
        self.frame_job = None
        if not self.playing:
            return
        now = time.perf_counter()
        elapsed = min(now - self.last_tick, MAX_FRAME_SECONDS)
        self.position = min(float(self.steps), self.position + self._speed() * elapsed)
        self.last_tick = now
        self._show(int(self.position))
        if self.position >= self.steps:
            self.playing = False
            self.blitter.save(SIMULATION_PNG, dpi=110)
            self.status.config(text=f"Koniec uczenia. Ostatnia klatka zapisana w "
                                    f"{os.path.basename(SIMULATION_PNG)}.")
            return
        self.frame_job = self.root.after(FRAME_INTERVAL_MS, self._tick)


if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(1000, 680)
    SimulateApp(root)
    root.mainloop()
