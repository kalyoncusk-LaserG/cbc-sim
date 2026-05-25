"""Tests for cbc.geometry."""
import numpy as np
from cbc import geometry


def test_linear_count_and_centering():
    pos = geometry.linear(7, 5e-3)
    assert pos.shape == (7, 2)
    assert np.allclose(pos[:, 1], 0)
    assert np.isclose(pos[:, 0].mean(), 0)


def test_linear_spacing():
    pos = geometry.linear(5, 1e-3)
    diffs = np.diff(pos[:, 0])
    assert np.allclose(diffs, 1e-3)


def test_hexagonal_count():
    for N in [1, 7, 19, 37]:
        pos = geometry.hexagonal(N, 5e-3)
        assert pos.shape == (N, 2)


def test_hexagonal_first_point_is_origin():
    pos = geometry.hexagonal(19, 5e-3)
    r = np.hypot(pos[:, 0], pos[:, 1])
    assert np.isclose(r.min(), 0)


def test_square_grid_count():
    pos = geometry.square_grid(9, 1e-3)
    assert pos.shape == (9, 2)


def test_square_grid_partial_last_row():
    pos = geometry.square_grid(7, 1e-3)
    assert pos.shape == (7, 2)


def test_fill_factor_hex():
    # Identical w0=R=2.5mm and pitch=5.5mm gives ≈ 0.749
    ff = geometry.fill_factor(pitch=5.5e-3, aperture_radius=2.5e-3,
                              geometry="hexagonal")
    assert 0.7 < ff < 0.8


def test_fill_factor_square_less_than_hex():
    ff_hex = geometry.fill_factor(5.5e-3, 2.5e-3, "hexagonal")
    ff_sq = geometry.fill_factor(5.5e-3, 2.5e-3, "square")
    assert ff_sq < ff_hex
