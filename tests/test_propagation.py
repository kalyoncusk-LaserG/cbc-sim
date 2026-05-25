"""Tests for cbc.propagation."""
import numpy as np
from cbc import propagation


def _flat_grid(npix=33, half_extent=1e-4):
    grid = np.linspace(-half_extent, half_extent, npix)
    envelope = np.ones((npix, npix))
    return grid, envelope


def test_single_channel_on_axis():
    """A single channel at origin with unit coefficient gives unit on-axis field."""
    pos = np.array([[0.0, 0.0]])
    cx = np.array([1.0 + 0j])
    cy = np.array([0.0 + 0j])
    grid, env = _flat_grid()
    Ex, Ey = propagation.focal_plane_field(pos, cx, cy, grid, grid, env,
                                            1064e-9, 1.0)
    center = len(grid) // 2
    assert np.isclose(Ex[center, center], 1.0)
    assert np.isclose(Ey[center, center], 0.0)


def test_two_channels_constructive():
    """Two in-phase channels: |1+1|² = 4 on-axis intensity."""
    pos = np.array([[1e-3, 0.0], [-1e-3, 0.0]])
    cx = np.array([1 + 0j, 1 + 0j])
    cy = np.zeros(2, dtype=complex)
    grid, env = _flat_grid()
    Ex, _ = propagation.focal_plane_field(pos, cx, cy, grid, grid, env,
                                          1064e-9, 1.0)
    center = len(grid) // 2
    assert np.isclose(np.abs(Ex[center, center]) ** 2, 4.0)


def test_two_channels_destructive():
    """Two anti-phase channels: zero on-axis intensity."""
    pos = np.array([[1e-3, 0.0], [-1e-3, 0.0]])
    cx = np.array([1 + 0j, -1 + 0j])
    cy = np.zeros(2, dtype=complex)
    grid, env = _flat_grid()
    Ex, _ = propagation.focal_plane_field(pos, cx, cy, grid, grid, env,
                                          1064e-9, 1.0)
    center = len(grid) // 2
    assert np.abs(Ex[center, center]) ** 2 < 1e-10


def test_on_axis_closed_form_matches_full_grid():
    """The on-axis closed form should match the full-grid evaluation at the center."""
    pos = np.array([[1e-3, 0.5e-3], [-1e-3, -0.5e-3], [0.0, 1e-3]])
    cx = np.array([1 + 0.5j, 0.7 - 0.3j, 0.4 + 0.4j])
    cy = np.array([0.1j, -0.1, 0.2 + 0.1j])
    grid, env = _flat_grid()
    Ex, Ey = propagation.focal_plane_field(pos, cx, cy, grid, grid, env,
                                            1064e-9, 1.0)
    center = len(grid) // 2
    Ex_oa, Ey_oa = propagation.on_axis_amplitude(cx, cy, envelope_at_origin=1.0)
    assert np.isclose(Ex[center, center], Ex_oa)
    assert np.isclose(Ey[center, center], Ey_oa)


def test_array_factor_envelope_factorization():
    """If envelope is a constant c, intensity scales by c²."""
    pos = np.array([[0.5e-3, 0.0], [-0.5e-3, 0.0]])
    cx = np.array([1 + 0j, 1 + 0j])
    cy = np.zeros(2, dtype=complex)
    grid = np.linspace(-1e-4, 1e-4, 17)
    env_unity = np.ones((17, 17))
    env_half = 0.5 * np.ones((17, 17))
    E1, _ = propagation.focal_plane_field(pos, cx, cy, grid, grid, env_unity,
                                           1064e-9, 1.0)
    E2, _ = propagation.focal_plane_field(pos, cx, cy, grid, grid, env_half,
                                           1064e-9, 1.0)
    assert np.allclose(E2, 0.5 * E1)
