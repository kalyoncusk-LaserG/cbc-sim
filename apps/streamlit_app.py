"""Streamlit web UI for the CBC simulator.

Install streamlit first:
    pip install streamlit

Then from the project root:
    streamlit run apps/streamlit_app.py

All inputs auto-trigger a rerun (no Run button). The simulator is cached on
its physical inputs, so changes that only affect display (e.g. the PIB
bucket radius) do not re-run the simulator.

The linewidth has both a log-scale slider for quick browsing across decades
and a numeric input for exact values; the two stay in sync.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from cbc.simulator import Simulator
from cbc.source import LaserSource
from cbc.apertures import GaussianAperture, HardCircularAperture
from cbc.controllers import OpenLoop, SPGD, LOCSET, NearFieldPhaseSensor
from cbc.geometry import fill_factor
from cbc.metrics import power_in_bucket


st.set_page_config(page_title="CBC Simulator", layout="wide")
st.title("Coherent Beam Combining Simulator")
st.caption(
    "Tiled-aperture, N-channel coherent beam combining with SPGD, LOCSET "
    "and near-field interferogram phase-sensor feedback. Sliders update live."
)


def fmt_linewidth(v: float) -> str:
    if v < 0.5:
        return "0 (ideal CW)"
    if v < 1e3:
        return f"{v:.1f} Hz"
    if v < 1e6:
        return f"{v/1e3:.2f} kHz"
    if v < 1e9:
        return f"{v/1e6:.2f} MHz"
    return f"{v/1e9:.2f} GHz"


# --- linewidth state: linked log-slider + numeric-input -------------------
if "lw_log" not in st.session_state:
    st.session_state.lw_log = 0.0
if "lw_exact" not in st.session_state:
    st.session_state.lw_exact = 0.0


def _on_log_change():
    """Slider moved — derive an exact Hz value, sync the numeric input."""
    val = 10 ** st.session_state.lw_log if st.session_state.lw_log > 0.05 else 0.0
    st.session_state.lw_exact = float(val)


def _on_exact_change():
    """Numeric input edited — back-fill the slider's log position."""
    val = max(0.0, float(st.session_state.lw_exact))
    if val < 1.0:
        st.session_state.lw_log = 0.0
    else:
        st.session_state.lw_log = float(np.log10(val))


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Array")
    N = st.slider("Channels N", 1, 49, 19)
    geom = st.selectbox("Geometry", ["hexagonal", "square", "linear"])
    aper_type = st.selectbox("Sub-aperture", ["Gaussian", "Hard circular"])

    st.header("Disturbances")
    phase_sigma = st.slider("Initial phase σ (rad)", 0.0, float(np.pi),
                             float(np.pi))
    pol_jitter = st.slider("Polarization jitter (°)", 0.0, 30.0, 0.0)
    tilt_jitter = st.slider("Tilt jitter (μrad)", 0.0, 200.0, 0.0)

    st.markdown("**Linewidth Δν**")
    st.slider(
        "log-scale slider (1 Hz to 100 GHz)",
        min_value=0.0, max_value=11.0, step=0.05,
        key="lw_log",
        on_change=_on_log_change,
        format="10^%.2f",
    )
    st.number_input(
        "or type exact value in Hz (e.g. 5e3, 1.5e6)",
        min_value=0.0, max_value=1e11,
        step=100.0,
        key="lw_exact",
        on_change=_on_exact_change,
        format="%g",
    )
    linewidth = float(st.session_state.lw_exact)
    st.caption(f"**Δν = {fmt_linewidth(linewidth)}**")

    st.header("Feedback")
    algo_name = st.selectbox("Algorithm",
                              ["Off", "SPGD", "LOCSET", "NF sensor"])
    n_iters = st.slider("Substeps to run", 100, 5000, 1000, step=100)

    st.header("Analysis")
    pib_radius_um = st.slider("PIB bucket radius (μm)", 10, 500, 100)
    seed = st.number_input("Random seed", 0, 999_999, 42)


# ----------------------------------------------------- cached simulation
@st.cache_data(show_spinner="Simulating…")
def run_sim(N, geom, aper_type, phase_sigma, pol_jitter, tilt_jitter, linewidth,
            algo_name, n_iters, seed):
    """Run the simulator and return a picklable snapshot dict.

    pib_radius is intentionally NOT an argument — it only affects display, so
    leaving it out of the cache key avoids redundant simulator runs when the
    user only changes the bucket size.
    """
    rng = np.random.default_rng(int(seed))
    src = LaserSource(linewidth=float(linewidth))
    if aper_type == "Gaussian":
        ap = GaussianAperture(w0=2.5e-3)
    else:
        ap = HardCircularAperture(radius=2.5e-3)
    sim = Simulator(N=N, geometry_kind=geom, source=src,
                    sub_aperture=ap, rng=rng)
    sim.randomize_disturbances(phase_sigma=phase_sigma,
                                pol_jitter_deg=pol_jitter,
                                tilt_jitter_urad=tilt_jitter)
    ctrl_map = {
        "Off": OpenLoop(),
        "SPGD": SPGD(rng=rng),
        "LOCSET": LOCSET(),
        "NF sensor": NearFieldPhaseSensor(rng=rng),
    }
    sim.controller = ctrl_map[algo_name]
    sim.controller.reset()

    strehl_hist = []
    for i in range(n_iters):
        sim.step()
        if i % 10 == 0:
            strehl_hist.append(sim.strehl())

    return {
        "positions": sim.positions.copy(),
        "residual_phase": sim.channels.residual_phase.copy(),
        "x_grid": sim.x_grid.copy(),
        "y_grid": sim.y_grid.copy(),
        "intensity": sim.focal_intensity(),
        "N": sim.N,
        "strehl": float(sim.strehl()),
        "residual_rms": float(sim.residual_rms()),
        "iter": int(sim.iter),
        "strehl_hist": np.array(strehl_hist),
    }


# ----------------------------------------------------- run & display
r = run_sim(N, geom, aper_type, phase_sigma, pol_jitter, tilt_jitter,
            linewidth, algo_name, n_iters, seed)

# PIB depends only on the cached intensity grid and the live bucket radius
pib = power_in_bucket(r["intensity"], r["x_grid"], r["y_grid"],
                       pib_radius_um * 1e-6)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Near field")
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    pos_mm = r["positions"] * 1e3
    sc = ax1.scatter(pos_mm[:, 0], pos_mm[:, 1],
                     c=r["residual_phase"] % (2 * np.pi),
                     cmap="hsv", vmin=0, vmax=2 * np.pi, s=300,
                     edgecolors="white", linewidths=0.6)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    plt.colorbar(sc, ax=ax1, label="residual phase (rad)")
    st.pyplot(fig1)

with col2:
    st.subheader("Far field intensity")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ext_um = [r["x_grid"][0] * 1e6, r["x_grid"][-1] * 1e6,
              r["y_grid"][0] * 1e6, r["y_grid"][-1] * 1e6]
    im = ax2.imshow(r["intensity"], cmap="hot", origin="lower",
                     extent=ext_um, vmin=0, vmax=r["N"] ** 2)
    bucket = plt.Circle((0, 0), pib_radius_um, fill=False, edgecolor="cyan",
                         linestyle="--", linewidth=1.5)
    ax2.add_patch(bucket)
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax2, label="intensity (norm.)")
    st.pyplot(fig2)

st.subheader("Convergence (Strehl)")
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(np.arange(len(r["strehl_hist"])) * 10, r["strehl_hist"], lw=1.5)
ax3.set_xlabel("iteration")
ax3.set_ylabel("Strehl ratio")
ax3.set_ylim(0, 1.05)
ax3.grid(alpha=0.3)
st.pyplot(fig3)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strehl ratio", f"{r['strehl']:.3f}")
c2.metric("Power-in-bucket", f"{pib*100:.1f}%")
c3.metric("Residual RMS", f"{r['residual_rms']:.2f} rad")
c4.metric("Iterations", f"{r['iter']}")

if aper_type == "Hard circular" and geom in ("hexagonal", "square"):
    ff = fill_factor(pitch=5.5e-3, aperture_radius=2.5e-3, geometry=geom)
    st.caption(f"Geometric fill factor (hard apertures, {geom}): {ff*100:.1f}%")

with st.expander("About the controllers and the linewidth model"):
    st.markdown(
        "- **Off** — open loop; residual phase is whatever the disturbance "
        "model produces.\n"
        "- **SPGD** — Stochastic Parallel Gradient Descent. Random ±δ dithers "
        "on all channels; the on-axis intensity difference J⁺−J⁻ updates the "
        "phase corrections. Blind (no phase sensor), converges slowly with N.\n"
        "- **LOCSET** — Locking of Optical Coherence by Single-detector "
        "Electronic-frequency Tagging. Each channel dithered at a unique "
        "frequency; the detector signal is lock-in demodulated to extract "
        "per-channel gradients in parallel.\n"
        "- **NF sensor** — idealised near-field interferogram phase sensor: "
        "direct, noisy residual-phase measurement driving a first-order servo. "
        "Reaches noise-limited Strehl quickly, independent of N.\n\n"
        "**Linewidth Δν** drives a Wiener random walk on each channel's phase: "
        "variance 2π·Δν·dt per substep with dt = 1 μs. Order-of-magnitude "
        "guidance:\n"
        "- ≤ 1 kHz — all loops track well\n"
        "- 10 kHz – 100 kHz — only the NF sensor stays locked; SPGD/LOCSET lag\n"
        "- ≥ 1 MHz — feedback can no longer keep up; Strehl drops toward the "
        "open-loop floor\n"
        "- ≥ 100 MHz — system is effectively incoherent regardless of feedback"
    )
