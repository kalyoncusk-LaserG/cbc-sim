"""Sub-aperture amplitude profiles and their focal-plane representations.

For a tiled aperture array, the focal-plane field of each sub-aperture is
the Fraunhofer transform of its amplitude profile, shifted by the aperture
position. We provide two standard profiles:

- Gaussian beam: a(r) = exp(-r²/w₀²)  →  focal envelope is Gaussian with
  waist w_f = λf / (π·w₀)
- Hard circular aperture of radius R: a(r) = 1[r ≤ R]  →  focal envelope
  is the jinc function 2 J₁(kRr_f/f) / (kRr_f/f), giving the Airy pattern
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.special import j1


class SubAperture(ABC):
    """Abstract base for sub-aperture amplitude profiles."""

    @abstractmethod
    def focal_envelope(self, x, y, wavelength: float,
                       focal_length: float) -> np.ndarray:
        """Field amplitude in the focal plane (real-valued, may be signed).

        Parameters
        ----------
        x, y : array_like
            Focal-plane coordinates in meters.
        wavelength, focal_length : float
            Optical parameters in meters.
        """
        ...

    @abstractmethod
    def focal_waist_estimate(self, wavelength: float,
                             focal_length: float) -> float:
        """Approximate focal-plane spot size (for grid sizing and tilt loss)."""
        ...


class GaussianAperture(SubAperture):
    """Gaussian sub-aperture with 1/e² intensity radius w₀.

    Focal-plane field amplitude: exp(-(x² + y²) / w_f²)
    with focal waist w_f = λ f / (π w₀).
    """

    def __init__(self, w0: float):
        self.w0 = w0

    def focal_waist_estimate(self, wavelength, focal_length):
        return wavelength * focal_length / (np.pi * self.w0)

    def focal_envelope(self, x, y, wavelength, focal_length):
        wf = self.focal_waist_estimate(wavelength, focal_length)
        return np.exp(-(x ** 2 + y ** 2) / wf ** 2)


class HardCircularAperture(SubAperture):
    """Hard circular aperture of radius R.

    Focal-plane amplitude: 2 J₁(α) / α, where α = 2π R r_f / (λ f).
    """

    def __init__(self, radius: float):
        self.R = radius

    def focal_waist_estimate(self, wavelength, focal_length):
        # Use the Airy 1st-zero radius as a characteristic size: 1.22 λf / (2R).
        return 1.22 * wavelength * focal_length / (2 * self.R)

    def focal_envelope(self, x, y, wavelength, focal_length):
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.hypot(x, y)
        arg = 2 * np.pi * self.R * r / (wavelength * focal_length)
        out = np.ones_like(arg, dtype=float)
        mask = arg > 1e-7
        out[mask] = 2 * j1(arg[mask]) / arg[mask]
        return out
