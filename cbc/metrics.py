"""Figures of merit for coherent beam combining."""
import numpy as np


def strehl_ratio(coeffs_x: np.ndarray, coeffs_y: np.ndarray) -> float:
    """Strehl ratio relative to N coherent unit-amplitude beams.

    S = (|Σ c_x|² + |Σ c_y|²) / N²

    With perfect amplitude (all A_n = 1) and perfect phasing this reaches 1.
    Per-channel amplitude loss (e.g. from tilt) and pol jitter reduce the
    maximum attainable Strehl.
    """
    N = len(coeffs_x)
    if N == 0:
        return 0.0
    peak = abs(coeffs_x.sum()) ** 2 + abs(coeffs_y.sum()) ** 2
    return float(peak / N ** 2)


def power_in_bucket(intensity: np.ndarray,
                    x_grid: np.ndarray,
                    y_grid: np.ndarray,
                    bucket_radius: float) -> float:
    """Fraction of total intensity within a circular bucket at the origin."""
    X, Y = np.meshgrid(x_grid, y_grid)
    in_bucket = (X ** 2 + Y ** 2) <= bucket_radius ** 2
    total = float(intensity.sum())
    if total <= 0:
        return 0.0
    return float(intensity[in_bucket].sum()) / total


def residual_phase_rms(phases: np.ndarray) -> float:
    """RMS of phase values after wrapping to (-π, π].

    Wrapping removes the irrelevant absolute offset (any global piston that
    feedback could remove). The result reflects only the relative scatter.
    """
    wrapped = np.arctan2(np.sin(phases), np.cos(phases))
    return float(np.std(wrapped))
