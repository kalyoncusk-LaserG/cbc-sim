"""Feedback control algorithms for coherent beam combining.

All controllers share a common interface: ``step(simulator)`` is called
once per simulator substep, reads the current state, and updates
``simulator.channels.correction`` in place.

Provided controllers
--------------------
- :class:`OpenLoop` — no feedback (baseline)
- :class:`SPGD`     — Stochastic Parallel Gradient Descent (blind)
- :class:`LOCSET`   — Locking of Optical Coherence by Single-detector
                      Electronic-frequency Tagging (multitone lock-in)
- :class:`NearFieldPhaseSensor` — direct per-channel phase measurement,
                      e.g. from a heterodyne/off-axis near-field
                      interferogram
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Controller(ABC):
    """Abstract base controller. Subclasses implement :meth:`step`."""

    name: str = "abstract"

    @abstractmethod
    def step(self, simulator) -> None:
        """One controller substep. Modify ``simulator.channels.correction``."""

    def reset(self) -> None:
        """Reset any internal controller state (e.g. accumulators)."""


# --------------------------------------------------------------------------
class OpenLoop(Controller):
    """No feedback; corrections remain at whatever value they started."""

    name = "off"

    def step(self, simulator):
        pass


# --------------------------------------------------------------------------
class SPGD(Controller):
    """Two-sided Stochastic Parallel Gradient Descent.

    At each substep, all channels are dithered simultaneously by ±δ
    (random Rademacher signs). The forward (J⁺) and backward (J⁻)
    intensities are measured, and the per-channel correction is updated
    by an unbiased one-shot gradient estimate:

        φ_n ← φ_n + (γ/N) · (J⁺ − J⁻) · d_n

    SPGD is *blind*: it requires no phase sensor and converges using only
    the bucket detector signal, at the cost of slower scaling with N.
    """

    name = "spgd"

    def __init__(self, dither: float = 0.2, gain: float = 0.05,
                 rng: Optional[np.random.Generator] = None):
        self.dither = dither
        self.gain = gain
        self.rng = rng or np.random.default_rng()

    def step(self, simulator):
        ch = simulator.channels
        N = ch.N
        d = self.rng.choice([-1.0, 1.0], size=N) * self.dither
        ch.correction += d
        Jp = simulator.on_axis_intensity()
        ch.correction -= 2 * d
        Jm = simulator.on_axis_intensity()
        ch.correction += d  # restore
        dJ = Jp - Jm
        ch.correction += (self.gain / N) * dJ * d


# --------------------------------------------------------------------------
class LOCSET(Controller):
    """Locking of Optical Coherence by Single-detector Electronic-freq. Tagging.

    Each channel is dithered at a unique frequency f_n = (n+1)/W cycles per
    substep. The detector signal is correlated against each sinusoid over a
    window of W substeps (lock-in demodulation). The N gradient components
    are extracted in parallel from one detector signal.

    LOCSET converges faster than SPGD for large N because the gradient
    information is recovered per channel rather than scrambled across
    channels in a single scalar measurement.
    """

    name = "locset"

    def __init__(self, dither: float = 0.1, gain: float = 0.02,
                 window: int = 64):
        self.dither = dither
        self.gain = gain
        self.W = window
        self._t = 0
        self._acc: Optional[np.ndarray] = None

    def reset(self):
        self._t = 0
        self._acc = None

    def step(self, simulator):
        ch = simulator.channels
        N = ch.N
        if self._acc is None or len(self._acc) != N:
            self._acc = np.zeros(N)
            self._t = 0
        freqs = (np.arange(N) + 1) / self.W
        sines = np.sin(2 * np.pi * freqs * self._t)
        # Apply dithers, measure, accumulate, remove
        ch.correction += self.dither * sines
        J = simulator.on_axis_intensity()
        self._acc += J * sines
        ch.correction -= self.dither * sines
        self._t += 1
        if self._t >= self.W:
            ch.correction += (self.gain / self.W) * self._acc
            self._acc = np.zeros(N)
            self._t = 0


# --------------------------------------------------------------------------
class NearFieldPhaseSensor(Controller):
    """Idealised near-field interferogram phase sensor.

    Models the result of demodulating a heterodyne or off-axis
    interferogram captured near the aperture plane: a direct,
    noise-limited measurement of each channel's residual phase, used to
    drive a first-order servo loop.

    The achievable Strehl ratio is limited by the sensor noise: with gain
    g and measurement noise σ, the steady-state residual variance is
    σ² · g / (2 − g), giving Strehl ≈ exp(-residual_variance).

    This controller demonstrates the limiting case of model-based
    feedback, against which SPGD and LOCSET can be benchmarked.
    """

    name = "nf_sensor"

    def __init__(self, noise_rad: float = 0.05, gain: float = 0.4,
                 rng: Optional[np.random.Generator] = None):
        self.noise = noise_rad
        self.gain = gain
        self.rng = rng or np.random.default_rng()

    def step(self, simulator):
        ch = simulator.channels
        residual = ch.residual_phase + self.rng.normal(0, self.noise, ch.N)
        ch.correction += self.gain * residual
