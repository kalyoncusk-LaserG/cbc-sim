"""Tests for the feedback controllers."""
import numpy as np

from cbc.simulator import Simulator
from cbc.controllers import OpenLoop, SPGD, LOCSET, NearFieldPhaseSensor


def test_nf_sensor_converges_to_unity():
    """Noiseless NF sensor with full gain reaches Strehl ≈ 1."""
    rng = np.random.default_rng(0)
    sim = Simulator(N=7, rng=rng,
                    controller=NearFieldPhaseSensor(noise_rad=0.0, gain=1.0, rng=rng))
    sim.randomize_disturbances(phase_sigma=np.pi)
    for _ in range(50):
        sim.step()
    assert sim.strehl() > 0.999


def test_nf_sensor_noise_limited():
    """Sensor with finite noise gives a known noise-limited Strehl."""
    rng = np.random.default_rng(1)
    sigma = 0.1
    gain = 0.3
    sim = Simulator(N=7, rng=rng,
                    controller=NearFieldPhaseSensor(noise_rad=sigma, gain=gain, rng=rng))
    sim.randomize_disturbances(phase_sigma=np.pi)
    for _ in range(2000):
        sim.step()
    # Steady-state residual variance ≈ σ² · g / (2 − g)
    # Strehl ≈ exp(-var). Expect ≥ 0.9 here.
    assert sim.strehl() > 0.9


def test_spgd_converges():
    """SPGD reaches high Strehl over a few thousand iterations."""
    rng = np.random.default_rng(2)
    sim = Simulator(N=7, rng=rng, controller=SPGD(gain=0.05, rng=rng))
    sim.randomize_disturbances(phase_sigma=np.pi)
    s0 = sim.strehl()
    for _ in range(3000):
        sim.step()
    s1 = sim.strehl()
    assert s1 > 0.9, f"SPGD should converge close to 1, got {s1:.3f}"
    assert s1 > s0, f"SPGD should improve Strehl"


def test_locset_converges():
    """LOCSET reaches high Strehl over a few thousand iterations."""
    rng = np.random.default_rng(3)
    sim = Simulator(N=7, rng=rng, controller=LOCSET(window=32, gain=0.05))
    sim.randomize_disturbances(phase_sigma=np.pi)
    s0 = sim.strehl()
    for _ in range(3000):
        sim.step()
    s1 = sim.strehl()
    assert s1 > 0.9, f"LOCSET should converge close to 1, got {s1:.3f}"
    assert s1 > s0, f"LOCSET should improve Strehl"


def test_openloop_no_change():
    """Open-loop should not change corrections."""
    rng = np.random.default_rng(4)
    sim = Simulator(N=7, rng=rng, controller=OpenLoop())
    sim.randomize_disturbances(phase_sigma=np.pi)
    initial_correction = sim.channels.correction.copy()
    for _ in range(100):
        sim.step()
    assert np.allclose(sim.channels.correction, initial_correction)


def test_simulator_iteration_count():
    rng = np.random.default_rng(5)
    sim = Simulator(N=7, rng=rng)
    sim.randomize_disturbances(phase_sigma=0.5)
    assert sim.iter == 0
    for _ in range(123):
        sim.step()
    assert sim.iter == 123


def test_strehl_bounded():
    """Strehl is between 0 and 1 at all times."""
    rng = np.random.default_rng(6)
    sim = Simulator(N=12, rng=rng, controller=SPGD(rng=rng))
    sim.randomize_disturbances(phase_sigma=np.pi, pol_jitter_deg=10,
                                tilt_jitter_urad=30)
    for _ in range(500):
        sim.step()
        s = sim.strehl()
        assert 0 <= s <= 1.0 + 1e-9
