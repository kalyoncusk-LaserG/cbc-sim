"""Per-channel state and disturbance models.

The state of an N-channel CBC system at any instant is represented by a
:class:`ChannelArray` containing four (N,) vectors:

- ``true_phase``     — actual instantaneous phase including all disturbances
- ``correction``     — phase correction applied by the feedback controller
- ``pol_angle``      — polarization rotation relative to the source (rad)
- ``tilt_x``, ``tilt_y`` — per-channel beam tilt at the aperture (rad)

The residual phase that drives the focal-plane interference is the
difference ``true_phase - correction``.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ChannelArray:
    """Container for the per-channel state of N parallel beam paths."""

    N: int
    true_phase: np.ndarray = None
    correction: np.ndarray = None
    pol_angle: np.ndarray = None
    tilt_x: np.ndarray = None
    tilt_y: np.ndarray = None

    def __post_init__(self):
        if self.true_phase is None:
            self.true_phase = np.zeros(self.N)
        if self.correction is None:
            self.correction = np.zeros(self.N)
        if self.pol_angle is None:
            self.pol_angle = np.zeros(self.N)
        if self.tilt_x is None:
            self.tilt_x = np.zeros(self.N)
        if self.tilt_y is None:
            self.tilt_y = np.zeros(self.N)

    @property
    def residual_phase(self) -> np.ndarray:
        """Phase remaining after applying correction."""
        return self.true_phase - self.correction

    def jones_x(self) -> np.ndarray:
        return np.cos(self.pol_angle)

    def jones_y(self) -> np.ndarray:
        return np.sin(self.pol_angle)


def initialize_disturbances(N: int,
                            phase_sigma: float = 0.0,
                            pol_jitter_rad: float = 0.0,
                            tilt_jitter_rad: float = 0.0,
                            rng: Optional[np.random.Generator] = None
                            ) -> ChannelArray:
    """Generate a fresh random disturbance realization for N channels.

    Parameters
    ----------
    phase_sigma : float
        Standard deviation of the initial Gaussian phase per channel (rad).
    pol_jitter_rad : float
        Standard deviation of the per-channel polarization rotation (rad).
    tilt_jitter_rad : float
        Standard deviation of per-channel x and y tilt at the aperture (rad).
    """
    if rng is None:
        rng = np.random.default_rng()
    ch = ChannelArray(N=N)
    ch.true_phase = rng.normal(0, phase_sigma, N) if phase_sigma > 0 else np.zeros(N)
    ch.pol_angle = (rng.normal(0, pol_jitter_rad, N)
                    if pol_jitter_rad > 0 else np.zeros(N))
    ch.tilt_x = (rng.normal(0, tilt_jitter_rad, N)
                 if tilt_jitter_rad > 0 else np.zeros(N))
    ch.tilt_y = (rng.normal(0, tilt_jitter_rad, N)
                 if tilt_jitter_rad > 0 else np.zeros(N))
    return ch


def apply_linewidth_step(ch: ChannelArray, dt: float, linewidth: float,
                         rng: np.random.Generator) -> None:
    """In-place Wiener-process phase update modelling Lorentzian linewidth.

    For a laser with FWHM linewidth Δν, the phase performs a random walk
    with variance 2π·Δν·dt per step. This is the standard Schawlow-Townes /
    Lorentzian phase-noise model.
    """
    if linewidth <= 0 or dt <= 0:
        return
    sigma = np.sqrt(2 * np.pi * linewidth * dt)
    ch.true_phase += rng.normal(0, sigma, ch.N)


def tilt_piston_coupling(ch: ChannelArray, positions: np.ndarray,
                         wavenumber: float) -> np.ndarray:
    """Piston phase induced by tilt × aperture-position coupling.

    A linear tilt α applied at aperture position (x_n, y_n) produces an
    extra piston phase k·(x_n·α_x + y_n·α_y) in the focal plane. This is
    the correctable part of the tilt-induced error (the uncorrectable part
    is the envelope shift handled by :func:`tilt_amplitude_loss`).
    """
    return wavenumber * (positions[:, 0] * ch.tilt_x +
                         positions[:, 1] * ch.tilt_y)


def tilt_amplitude_loss(ch: ChannelArray, focal_length: float,
                        focal_waist: float) -> np.ndarray:
    """On-axis amplitude reduction per channel from tilt-induced focal shift.

    A linear tilt α_n shifts the focal-plane envelope by α_n·f, so the
    on-axis amplitude contribution is reduced by the Gaussian envelope
    evaluated at the shift distance:

        A_n = exp(-((α_x f)² + (α_y f)²) / w_f²)
    """
    dx = ch.tilt_x * focal_length
    dy = ch.tilt_y * focal_length
    return np.exp(-(dx ** 2 + dy ** 2) / focal_waist ** 2)
