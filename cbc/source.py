"""Laser source models for coherent beam combining.

The master oscillator is a narrow-linewidth CW laser whose output is split
into N nominally identical beams. The relevant properties for CBC are:

- wavelength λ (sets the wavenumber k = 2π/λ and the focal envelope size)
- power (normalized to unity per channel in this simulator)
- linewidth Δν (FWHM; drives Wiener phase noise per channel)
- polarization (Jones vector; nominal pre-jitter polarization)
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LaserSource:
    """Master oscillator parameters.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength in meters. Default 1064 nm (Yb-doped fiber).
    power : float
        Total output power in watts (normalized; physics is amplitude-based).
    linewidth : float
        Lorentzian FWHM linewidth in Hz. Drives the Wiener-process phase
        noise model: per step of duration dt the phase increments by a
        Gaussian with variance 2π·Δν·dt.
    polarization : array_like, shape (2,)
        Pre-split Jones vector, default linear horizontal [1, 0].
    """

    wavelength: float = 1064e-9
    power: float = 1.0
    linewidth: float = 0.0
    polarization: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j, 0.0 + 0j]))

    def __post_init__(self):
        self.polarization = np.asarray(self.polarization, dtype=complex)

    @property
    def wavenumber(self) -> float:
        """Vacuum wavenumber k = 2π/λ in rad/m."""
        return 2 * np.pi / self.wavelength

    @property
    def coherence_time(self) -> float:
        """Lorentzian coherence time τ_c = 1 / (π·Δν) in seconds."""
        if self.linewidth <= 0:
            return np.inf
        return 1.0 / (np.pi * self.linewidth)
