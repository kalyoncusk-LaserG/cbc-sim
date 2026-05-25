"""Tiled-aperture array geometries.

Each generator returns an (N, 2) array of sub-aperture center positions in
meters, centered around the origin. The user supplies the pitch
(center-to-center spacing).
"""
import numpy as np


def linear(N: int, pitch: float) -> np.ndarray:
    """1-D linear array along x, centered at origin."""
    xs = (np.arange(N) - (N - 1) / 2) * pitch
    ys = np.zeros(N)
    return np.column_stack([xs, ys])


def square_grid(N: int, pitch: float) -> np.ndarray:
    """Square grid filled row-major, centered at origin.

    Takes the smallest square that fits N apertures (ceil(√N) per side) and
    fills row-major. If N is not a perfect square the last row is partial.
    """
    side = int(np.ceil(np.sqrt(N)))
    coords = []
    for i in range(side):
        for j in range(side):
            if len(coords) >= N:
                break
            coords.append([(j - (side - 1) / 2) * pitch,
                           (i - (side - 1) / 2) * pitch])
    return np.array(coords)


def hexagonal(N: int, pitch: float) -> np.ndarray:
    """Hexagonal close-packed array of the N points closest to the origin.

    Useful values of N for completed hex rings: 1, 7, 19, 37, 61, 91.
    Non-completed values give an asymmetric outer ring.
    """
    R = int(np.ceil(np.sqrt(N))) + 1
    candidates = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            x = (i + j * 0.5) * pitch
            y = j * pitch * np.sqrt(3) / 2
            candidates.append((np.hypot(x, y), x, y))
    candidates.sort()
    return np.array([[c[1], c[2]] for c in candidates[:N]])


def fill_factor(pitch: float, aperture_radius: float,
                geometry: str = "hexagonal") -> float:
    """Geometric fill factor for hard-circular sub-apertures.

    Fill = A_aperture / A_cell, where the cell area depends on the packing.
    """
    A_aper = np.pi * aperture_radius ** 2
    if geometry == "hexagonal":
        A_cell = pitch ** 2 * np.sqrt(3) / 2
    elif geometry == "square":
        A_cell = pitch ** 2
    else:
        return float("nan")
    return A_aper / A_cell
