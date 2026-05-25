"""Interactive matplotlib demo for the CBC simulator.

Run from the project root:

    python apps/interactive_mpl.py

Controls (right-hand panel):
    Radio buttons : feedback algorithm (Off, SPGD, LOCSET, NF sensor)
    Phase sigma   : RMS of the initial Gaussian phase noise per channel
    Pol. jitter   : RMS polarization rotation per channel (degrees)
    Tilt jitter   : RMS per-channel pointing tilt (microradians)
    Linewidth     : Lorentzian FWHM (Hz), drives Wiener phase drift
    PIB R         : radius of the focal-plane bucket (micrometers)
    Reset         : re-randomize disturbances with the current slider values
    Pause / Resume: stop the animation
"""
from collections import deque
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation

from cbc.simulator import Simulator
from cbc.source import LaserSource
from cbc.apertures import GaussianAperture
from cbc.controllers import OpenLoop, SPGD, LOCSET, NearFieldPhaseSensor


SUBSTEPS_PER_FRAME = 20
CONV_BUFFER = 500


class CBCApp:
    def __init__(self, N: int = 19, geometry: str = "hexagonal"):
        self.sim = Simulator(
            N=N, geometry_kind=geometry,
            source=LaserSource(linewidth=0.0),
            sub_aperture=GaussianAperture(w0=2.5e-3),
            controller=OpenLoop(),
            rng=np.random.default_rng(42),
        )
        self.sim.randomize_disturbances(phase_sigma=np.pi)
        self.paused = False
        self.conv = deque(maxlen=CONV_BUFFER)
        self._build_figure()
        self._build_widgets()
        self.anim = FuncAnimation(self.fig, self._animate, interval=60,
                                   blit=False, cache_frame_data=False)

    # ---------------------------------------------------------- figure setup
    def _build_figure(self):
        self.fig = plt.figure(figsize=(12.5, 7.5))
        gs = self.fig.add_gridspec(
            3, 4, height_ratios=[2.4, 0.9, 0.4],
            left=0.05, right=0.78, top=0.94, bottom=0.06,
            wspace=0.35, hspace=0.45,
        )
        self.ax_near = self.fig.add_subplot(gs[0, 0:2])
        self.ax_far = self.fig.add_subplot(gs[0, 2:4])
        self.ax_conv = self.fig.add_subplot(gs[1, :])
        self.ax_metrics = self.fig.add_subplot(gs[2, :])
        self.ax_metrics.axis("off")

        pos_mm = self.sim.positions * 1e3
        self.near = self.ax_near.scatter(
            pos_mm[:, 0], pos_mm[:, 1],
            c=self.sim.channels.residual_phase % (2 * np.pi),
            cmap="hsv", vmin=0, vmax=2 * np.pi, s=250,
            edgecolors="white", linewidths=0.6,
        )
        self.ax_near.set_aspect("equal")
        self.ax_near.set_xlabel("x (mm)")
        self.ax_near.set_ylabel("y (mm)")
        self.ax_near.set_title("Near field — color = residual phase")
        self.fig.colorbar(self.near, ax=self.ax_near, label="phase (rad)")

        I = self.sim.focal_intensity()
        ext_um = [self.sim.x_grid[0] * 1e6, self.sim.x_grid[-1] * 1e6,
                  self.sim.y_grid[0] * 1e6, self.sim.y_grid[-1] * 1e6]
        self.far = self.ax_far.imshow(
            I, cmap="hot", origin="lower", extent=ext_um,
            vmin=0, vmax=self.sim.N ** 2,
        )
        self.ax_far.set_xlabel("x (μm)")
        self.ax_far.set_ylabel("y (μm)")
        self.ax_far.set_title("Far field — combined intensity")
        self.bucket = plt.Circle((0, 0), 100, fill=False, edgecolor="cyan",
                                  linestyle="--", linewidth=1.6)
        self.ax_far.add_patch(self.bucket)
        self.fig.colorbar(self.far, ax=self.ax_far, label="intensity")

        (self.conv_line,) = self.ax_conv.plot([], [], lw=1.5, color="C0")
        self.ax_conv.set_xlim(0, CONV_BUFFER)
        self.ax_conv.set_ylim(0, 1.05)
        self.ax_conv.set_xlabel("frame")
        self.ax_conv.set_ylabel("Strehl")
        self.ax_conv.grid(alpha=0.3)

        self.metrics_text = self.ax_metrics.text(
            0.5, 0.5, "", ha="center", va="center",
            fontsize=11, family="monospace",
            transform=self.ax_metrics.transAxes,
        )

    # ---------------------------------------------------------- widgets
    def _build_widgets(self):
        cl, cw = 0.81, 0.16

        ax_algo = self.fig.add_axes([cl, 0.76, cw, 0.17])
        self.algo_radio = RadioButtons(ax_algo, ("Off", "SPGD", "LOCSET", "NF sensor"))
        self.algo_radio.on_clicked(self._on_algo)

        def slider(y, label, lo, hi, init, fmt=None, callback=None):
            ax = self.fig.add_axes([cl, y, cw, 0.028])
            s = Slider(ax, label, lo, hi, valinit=init, valfmt=fmt)
            if callback:
                s.on_changed(callback)
            return s

        self.s_phase = slider(0.68, "Phase σ", 0, np.pi, np.pi, fmt="%.2f")
        self.s_pol = slider(0.62, "Pol °", 0, 30, 0, fmt="%.0f")
        self.s_tilt = slider(0.56, "Tilt μrad", 0, 200, 0, fmt="%.0f")
        self.s_lw = slider(0.50, "Δν Hz", 0, 10_000, 0, fmt="%.0f",
                            callback=self._on_lw)
        self.s_pib = slider(0.44, "PIB μm", 10, 500, 100, fmt="%.0f",
                             callback=self._on_pib)

        ax_reset = self.fig.add_axes([cl, 0.34, cw / 2 - 0.005, 0.05])
        self.btn_reset = Button(ax_reset, "Reset noise")
        self.btn_reset.on_clicked(self._on_reset)

        ax_pause = self.fig.add_axes([cl + cw / 2 + 0.005, 0.34,
                                       cw / 2 - 0.005, 0.05])
        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_pause.on_clicked(self._on_pause)

    # ---------------------------------------------------------- callbacks
    def _on_algo(self, label):
        rng = self.sim.rng
        if label == "Off":
            self.sim.controller = OpenLoop()
        elif label == "SPGD":
            self.sim.controller = SPGD(rng=rng)
        elif label == "LOCSET":
            self.sim.controller = LOCSET()
        elif label == "NF sensor":
            self.sim.controller = NearFieldPhaseSensor(rng=rng)
        self.sim.channels.correction[:] = 0
        self.sim.iter = 0
        self.conv.clear()

    def _on_reset(self, event=None):
        self.sim.randomize_disturbances(
            phase_sigma=self.s_phase.val,
            pol_jitter_deg=self.s_pol.val,
            tilt_jitter_urad=self.s_tilt.val,
        )
        self.conv.clear()

    def _on_pause(self, event=None):
        self.paused = not self.paused
        self.btn_pause.label.set_text("Resume" if self.paused else "Pause")

    def _on_lw(self, val):
        self.sim.source.linewidth = val

    def _on_pib(self, val):
        self.bucket.set_radius(val)

    # ---------------------------------------------------------- animation
    def _animate(self, frame):
        if not self.paused:
            for _ in range(SUBSTEPS_PER_FRAME):
                self.sim.step()
            self.near.set_array(self.sim.channels.residual_phase % (2 * np.pi))
            self.far.set_data(self.sim.focal_intensity())
            self.conv.append(self.sim.strehl())
            self.conv_line.set_data(range(len(self.conv)), list(self.conv))
            pib = self.sim.power_in_bucket(self.s_pib.val * 1e-6)
            self.metrics_text.set_text(
                f"Strehl: {self.sim.strehl():.3f}    "
                f"PIB: {pib * 100:.1f}%    "
                f"Residual RMS: {self.sim.residual_rms():.2f} rad    "
                f"Iter: {self.sim.iter}"
            )
        return [self.near, self.far, self.conv_line, self.metrics_text]


def main():
    app = CBCApp(N=19, geometry="hexagonal")
    plt.show()


if __name__ == "__main__":
    main()
