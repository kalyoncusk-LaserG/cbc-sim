"""Basic non-interactive demo.

Runs an identical disturbance realization through open-loop and three
different feedback controllers, then saves a comparison figure.

    python examples/basic_demo.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from cbc.simulator import Simulator
from cbc.source import LaserSource
from cbc.apertures import GaussianAperture
from cbc.controllers import OpenLoop, SPGD, LOCSET, NearFieldPhaseSensor


def _fresh_sim(seed: int = 42, N: int = 19) -> Simulator:
    rng = np.random.default_rng(seed)
    sim = Simulator(
        N=N,
        geometry_kind="hexagonal",
        source=LaserSource(linewidth=0.0),
        sub_aperture=GaussianAperture(w0=2.5e-3),
        rng=rng,
    )
    sim.randomize_disturbances(phase_sigma=np.pi, tilt_jitter_urad=20.0)
    return sim


def _run(label: str, ctrl_factory, n_iters: int):
    sim = _fresh_sim()
    sim.controller = ctrl_factory(sim.rng)
    sim.controller.reset()
    strehl, pib = [], []
    for _ in range(n_iters):
        sim.step()
        if sim.iter % 25 == 0:
            strehl.append(sim.strehl())
            pib.append(sim.power_in_bucket(100e-6))
    return label, sim, np.array(strehl), np.array(pib)


def main():
    n_iters = 3000
    results = [
        _run("Open loop", lambda r: OpenLoop(), n_iters),
        _run("SPGD", lambda r: SPGD(rng=r), n_iters),
        _run("LOCSET", lambda r: LOCSET(window=64, gain=0.02), n_iters),
        _run("NF sensor", lambda r: NearFieldPhaseSensor(noise_rad=0.05, gain=0.4, rng=r), n_iters),
    ]

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])

    for col, (label, sim, hist_s, hist_p) in enumerate(results):
        ax_far = fig.add_subplot(gs[0, col])
        ext_um = [sim.x_grid[0] * 1e6, sim.x_grid[-1] * 1e6,
                  sim.y_grid[0] * 1e6, sim.y_grid[-1] * 1e6]
        ax_far.imshow(sim.focal_intensity(), cmap="hot", origin="lower",
                       extent=ext_um, vmin=0, vmax=sim.N ** 2)
        ax_far.set_title(f"{label}\nStrehl={sim.strehl():.3f}")
        ax_far.set_xlabel("x (μm)")
        if col == 0:
            ax_far.set_ylabel("y (μm)")

    ax_conv = fig.add_subplot(gs[1, :])
    x = np.arange(len(results[0][2])) * 25
    for label, _, hist_s, _ in results:
        ax_conv.plot(x, hist_s, label=label, lw=1.4)
    ax_conv.set_xlabel("iteration")
    ax_conv.set_ylabel("Strehl ratio")
    ax_conv.set_ylim(0, 1.05)
    ax_conv.set_title("Convergence (identical disturbance realization)")
    ax_conv.grid(alpha=0.3)
    ax_conv.legend(loc="lower right")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "basic_demo.png")
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
