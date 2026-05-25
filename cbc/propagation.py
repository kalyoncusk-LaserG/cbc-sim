"""Fraunhofer propagation from a tiled aperture to the focal plane.

For a tiled aperture array where sub-aperture *n* sits at position
(x_n, y_n) and emits a beam of complex amplitude c_n with profile a(r),
the focal-plane field after a thin lens of focal length f is

    E(x_f, y_f) = ã(x_f, y_f) · Σ_n c_n · exp(-i k (x_n x_f + y_n y_f) / f)

where ã is the Fourier transform of the single-aperture amplitude profile
(the focal-plane "envelope"). The sum is the array factor.

This module computes that field for both the on-axis point (fast metric
evaluation) and the full focal-plane grid (visualization).
"""
from typing import Tuple
import numpy as np


def focal_plane_field(positions: np.ndarray,
                      coeffs_x: np.ndarray,
                      coeffs_y: np.ndarray,
                      x_grid: np.ndarray,
                      y_grid: np.ndarray,
                      envelope: np.ndarray,
                      wavelength: float,
                      focal_length: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the focal-plane complex field components E_x and E_y.

    Parameters
    ----------
    positions : (N, 2) array
        Sub-aperture centre positions in the aperture plane (m).
    coeffs_x, coeffs_y : (N,) complex arrays
        Per-channel coefficients for the x and y Jones components,
        i.e. ``amp_n · exp(i φ_n) · jones_{x,y}_n``.
    x_grid, y_grid : 1-D arrays
        Focal-plane coordinates in meters.
    envelope : (len(y_grid), len(x_grid)) array
        Pre-computed single-aperture focal envelope amplitude.
    wavelength, focal_length : float
        Optical parameters in meters.

    Returns
    -------
    Ex, Ey : (Ny, Nx) complex arrays
    """
    k = 2 * np.pi / wavelength
    # Separable spatial phases.
    phase_x = -k * positions[:, 0:1] * x_grid[np.newaxis, :] / focal_length  # (N, Nx)
    phase_y = -k * positions[:, 1:2] * y_grid[np.newaxis, :] / focal_length  # (N, Ny)
    sx = np.exp(1j * phase_x)
    sy = np.exp(1j * phase_y)
    # E[j, i] = env[j, i] * Σ_n c_n · sx[n, i] · sy[n, j]
    Ex = envelope * np.einsum("n,ni,nj->ji", coeffs_x, sx, sy)
    Ey = envelope * np.einsum("n,ni,nj->ji", coeffs_y, sx, sy)
    return Ex, Ey


def on_axis_amplitude(coeffs_x: np.ndarray,
                      coeffs_y: np.ndarray,
                      envelope_at_origin: float = 1.0
                      ) -> Tuple[complex, complex]:
    """Closed-form on-axis (x=y=0) field amplitude.

    All spatial phases vanish at the origin, so the focal-plane field
    reduces to ``envelope(0) · Σ_n c_n``. Returns ``(E_x, E_y)``.
    """
    return (envelope_at_origin * coeffs_x.sum(),
            envelope_at_origin * coeffs_y.sum())
