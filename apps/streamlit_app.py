"""Streamlit web UI for the CBC simulator.

Install streamlit first:
    pip install streamlit

Then from the project root:
    streamlit run apps/streamlit_app.py
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


st.set_page_config(page_title="CBC Simulator", layout="wide")
st.title("Coherent Beam Combining Simulator")
st.caption(
    "Tiled-aperture, N-channel coherent beam combining with SPGD, LOCSET "
    "and near-field interferogram phase-sensor feedback."
)

# -------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Array")
    N = st.slider("Channels N", 1, 49, 19)
    geom = st.selectbox("Geometry", ["hexagonal", "square", "linear"])
    aper_type = st.selectbox("Sub-aperture", ["Gaussian", "Hard circular"])

    st.header("Disturbances")
    phase_sigma = st.slider("Initial phase σ (rad)", 0.0, float(np.pi), float(np.pi))
    pol_jitter = st.slider("Polarization jitter (°)", 0.0, 30.0, 0.0)
    tilt_jitter = st.slider("Tilt jitter (μrad)", 0.0, 200.0, 0.0)
    linewidth = st.slider("Linewidth Δν (Hz)", 0.0, 10_000.0, 0.0, step=100.0)

    st.header("Feedback")
    algo_name = st.selectbox("Algorithm",
                              ["Off", "SPGD", "LOCSET", "NF sensor"])
    n_iters = st.slider("Substeps to run", 0, 5000, 1000, step=100)

    st.header("Analysis")
    pib_radius_um = st.slider("PIB bucket radius (μm)", 10, 500, 100)
    seed = st.number_input("Random seed", 0, 999_999, 42)
    run_btn = st.button("Run simulation", use_container_width=True)


# -------------------------------------------------------------- sim cache
@st.cache_data(show_spinner=False)
def run_sim(N, geom, aper_type, phase_sigma, pol_jitter, tilt_jitter, linewidth,
            algo_name, n_iters, seed, pib_radius_um):
    rng = np.random.default_rng(int(seed))
    src = LaserSource(linewidth=linewidth)
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

    strehl_hist, pib_hist = [], []
    for i in range(n_iters):
        sim.step()
        if i % 10 == 0:
            strehl_hist.append(sim.strehl())
            pib_hist.append(sim.power_in_bucket(pib_radius_um * 1e-6))
    return sim, strehl_hist, pib_hist


if run_btn or "sim_result" not in st.session_state:
    with st.spinner("Simulating…"):
        sim, hist_s, hist_p = run_sim(
            N, geom, aper_type, phase_sigma, pol_jitter, tilt_jitter,
            linewidth, algo_name, n_iters, seed, pib_radius_um,
        )
    st.session_state["sim_result"] = (sim, hist_s, hist_p)

sim, hist_s, hist_p = st.session_state["sim_result"]

# -------------------------------------------------------------- plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Near field")
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    pos_mm = sim.positions * 1e3
    sc = ax1.scatter(pos_mm[:, 0], pos_mm[:, 1],
                     c=sim.channels.residual_phase % (2 * np.pi),
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
    I = sim.focal_intensity()
    ext_um = [sim.x_grid[0] * 1e6, sim.x_grid[-1] * 1e6,
              sim.y_grid[0] * 1e6, sim.y_grid[-1] * 1e6]
    im = ax2.imshow(I, cmap="hot", origin="lower", extent=ext_um,
                     vmin=0, vmax=N * N)
    bucket = plt.Circle((0, 0), pib_radius_um, fill=False, edgecolor="cyan",
                         linestyle="--", linewidth=1.5)
    ax2.add_patch(bucket)
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax2, label="intensity (norm.)")
    st.pyplot(fig2)

st.subheader("Convergence")
fig3, ax3 = plt.subplots(figsize=(10, 3))
x_axis = np.arange(len(hist_s)) * 10
ax3.plot(x_axis, hist_s, label="Strehl", lw=1.5)
ax3.plot(x_axis, hist_p, label="Power-in-bucket", lw=1.5, linestyle="--")
ax3.set_xlabel("iteration")
ax3.set_ylabel("figure of merit")
ax3.set_ylim(0, 1.05)
ax3.grid(alpha=0.3)
ax3.legend(loc="best")
st.pyplot(fig3)

# -------------------------------------------------------------- metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Strehl ratio", f"{sim.strehl():.3f}")
c2.metric("Power-in-bucket", f"{sim.power_in_bucket(pib_radius_um*1e-6)*100:.1f}%")
c3.metric("Residual RMS", f"{sim.residual_rms():.2f} rad")
c4.metric("Iterations", f"{sim.iter}")

if aper_type == "Hard circular" and geom in ("hexagonal", "square"):
    ff = fill_factor(pitch=5.5e-3, aperture_radius=2.5e-3, geometry=geom)
    st.caption(f"Geometric fill factor (hard apertures, {geom}): {ff*100:.1f}%")

with st.expander("About the controllers"):
    st.markdown(
        "- **Off** — open loop; the residual phase is whatever the disturbance "
        "model produces.\n"
        "- **SPGD** — Stochastic Parallel Gradient Descent. Random ±δ dithers "
        "on all channels, the on-axis intensity difference J⁺−J⁻ updates the "
        "phase corrections. Blind (no phase sensor) but converges slowly with N.\n"
        "- **LOCSET** — Locking of Optical Coherence by Single-detector "
        "Electronic-frequency Tagging. Each channel dithered at a unique "
        "frequency; the detector signal is lock-in demodulated to extract per-channel "
        "gradients in parallel.\n"
        "- **NF sensor** — idealised near-field interferogram phase sensor: "
        "direct, noisy residual-phase measurement driving a first-order servo. "
        "Reaches noise-limited Strehl quickly, independent of N."
    )
